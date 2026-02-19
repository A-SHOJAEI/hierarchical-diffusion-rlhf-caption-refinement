"""Core model implementations for diffusion caption generation and RLHF refinement."""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Config

from hierarchical_diffusion_rlhf_caption_refinement.models.components import (
    CLIPAlignmentLoss,
    ConfidenceEstimator,
    DiffusionScheduler,
    HierarchicalRewardShaper,
    PreferenceHead,
)

logger = logging.getLogger(__name__)


class DiffusionCaptionModel(nn.Module):
    """Diffusion-based caption generation model.

    This model generates captions using a diffusion process over the
    text embedding space, conditioned on image features.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 128,
        num_timesteps: int = 1000,
        dropout: float = 0.1,
        image_embedding_dim: int = 768,
    ) -> None:
        """Initialize diffusion caption model.

        Args:
            vocab_size: Vocabulary size.
            hidden_dim: Hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_length: Maximum sequence length.
            num_timesteps: Number of diffusion timesteps.
            dropout: Dropout probability.
            image_embedding_dim: Dimension of image embeddings.
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.num_timesteps = num_timesteps

        # Initialize transformer backbone
        # Add +1 to n_positions to account for image prefix token
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_length + 1,  # +1 for image token
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2LMHeadModel(config)

        # Image conditioning
        self.image_projection = nn.Sequential(
            nn.Linear(image_embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        # Timestep embedding
        self.timestep_embedding = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Diffusion scheduler
        self.scheduler = DiffusionScheduler(num_timesteps=num_timesteps)

        # Noise prediction head
        self.noise_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeddings: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of diffusion caption model.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            image_embeddings: Image embeddings of shape (batch_size, image_dim).
            timesteps: Diffusion timesteps of shape (batch_size,).

        Returns:
            Dictionary with logits and hidden states.
        """
        batch_size, seq_len = input_ids.shape

        # Get input embeddings
        inputs_embeds = self.transformer.transformer.wte(input_ids)

        # Add image conditioning if provided
        if image_embeddings is not None:
            image_features = self.image_projection(image_embeddings)
            # Add as prefix to sequence
            image_features = image_features.unsqueeze(1)
            inputs_embeds = torch.cat([image_features, inputs_embeds], dim=1)

            # Update attention mask
            image_mask = torch.ones(
                batch_size, 1,
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([image_mask, attention_mask], dim=1)

        # Add timestep embedding if provided
        if timesteps is not None:
            t_emb = self.timestep_embedding(timesteps.unsqueeze(-1).float() / self.num_timesteps)
            t_emb = t_emb.unsqueeze(1)
            inputs_embeds = inputs_embeds + t_emb

        # Forward through transformer
        outputs = self.transformer.transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # Remove image prefix if added
        if image_embeddings is not None:
            hidden_states = hidden_states[:, 1:, :]

        # Generate logits
        logits = self.transformer.lm_head(hidden_states)

        return {
            "logits": logits,
            "hidden_states": hidden_states,
        }

    def generate(
        self,
        image_embeddings: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
    ) -> torch.Tensor:
        """Generate captions using diffusion denoising.

        Args:
            image_embeddings: Image embeddings of shape (batch_size, image_dim).
            max_length: Maximum generation length.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.
            top_p: Nucleus sampling parameter.

        Returns:
            Generated token IDs of shape (batch_size, max_length).
        """
        batch_size = image_embeddings.shape[0]
        device = image_embeddings.device

        # Start with random tokens
        current_ids = torch.randint(
            0, self.transformer.config.vocab_size,
            (batch_size, max_length),
            device=device,
        )

        # Denoise over timesteps (simplified - using fewer steps for efficiency)
        num_inference_steps = 50
        timestep_sequence = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps,
            device=device,
        ).long()

        for t in timestep_sequence:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)

            # Create attention mask
            attention_mask = torch.ones_like(current_ids)

            # Predict noise
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=current_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embeddings,
                    timesteps=timesteps,
                )

                logits = outputs["logits"]

                # Sample next tokens (simplified)
                if t > 0:
                    # Apply temperature
                    logits = logits / temperature

                    # Top-k sampling
                    if top_k > 0:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')

                    # Sample
                    probs = F.softmax(logits, dim=-1)
                    next_ids = torch.multinomial(
                        probs.view(-1, probs.shape[-1]),
                        num_samples=1,
                    ).view(batch_size, max_length)

                    # Update
                    current_ids = next_ids

        return current_ids


class RewardModel(nn.Module):
    """Reward model for RLHF preference learning.

    Predicts reward scores for generated captions based on human preferences.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        max_length: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """Initialize reward model.

        Args:
            vocab_size: Vocabulary size.
            hidden_dim: Hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            max_length: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()

        # Initialize transformer backbone (smaller than generation model)
        config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=hidden_dim,
            n_layer=num_layers,
            n_head=num_heads,
            n_positions=max_length,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
        )
        self.transformer = GPT2LMHeadModel(config).transformer

        # Preference head
        self.preference_head = PreferenceHead(hidden_dim, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute reward scores for captions.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Reward scores of shape (batch_size,).
        """
        # Forward through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        # Compute reward score
        reward = self.preference_head(hidden_states)

        return reward


class RLHFCaptionRefiner(nn.Module):
    """RLHF-based caption refinement model.

    This model refines captions using hierarchical reward shaping
    that combines CLIP-based dense rewards with human preference
    sparse rewards.
    """

    def __init__(
        self,
        base_model: DiffusionCaptionModel,
        reward_model: RewardModel,
        hidden_dim: int = 768,
        clip_weight: float = 0.7,
        preference_weight: float = 0.3,
    ) -> None:
        """Initialize RLHF caption refiner.

        Args:
            base_model: Base diffusion caption model.
            reward_model: Reward model for preferences.
            hidden_dim: Hidden dimension.
            clip_weight: Initial weight for CLIP rewards.
            preference_weight: Initial weight for preference rewards.
        """
        super().__init__()

        self.base_model = base_model
        self.reward_model = reward_model

        # Hierarchical reward shaper (our novel component)
        self.reward_shaper = HierarchicalRewardShaper(
            dense_weight_init=clip_weight,
            sparse_weight_init=preference_weight,
        )

        # CLIP alignment loss
        self.clip_loss = CLIPAlignmentLoss()

        # Confidence estimator
        self.confidence_estimator = ConfidenceEstimator(hidden_dim)

    def compute_dense_rewards(
        self,
        hidden_states: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dense CLIP-based rewards at token level.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_dim).
            image_embeddings: Image embeddings of shape (batch_size, hidden_dim).

        Returns:
            Dense rewards of shape (batch_size, seq_len).
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape

        # Expand image embeddings to match sequence length
        image_expanded = image_embeddings.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute token-level similarity to image
        hidden_norm = F.normalize(hidden_states, dim=-1)
        image_norm = F.normalize(image_expanded, dim=-1)

        similarity = torch.sum(hidden_norm * image_norm, dim=-1)

        # Convert to rewards
        rewards = similarity

        return rewards

    def compute_sparse_rewards(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sparse sequence-level preference rewards.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).

        Returns:
            Sparse rewards of shape (batch_size,).
        """
        with torch.no_grad():
            rewards = self.reward_model(input_ids, attention_mask)

        return rewards

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_embeddings: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with hierarchical reward computation.

        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len).
            attention_mask: Attention mask of shape (batch_size, seq_len).
            image_embeddings: Image embeddings of shape (batch_size, image_dim).
            timesteps: Optional diffusion timesteps.

        Returns:
            Dictionary with logits, rewards, and other outputs.
        """
        # Forward through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeddings=image_embeddings,
            timesteps=timesteps,
        )

        logits = outputs["logits"]
        hidden_states = outputs["hidden_states"]

        # Compute confidence
        confidence = self.confidence_estimator(hidden_states)

        # Compute dense rewards (CLIP-based, token-level)
        dense_rewards = self.compute_dense_rewards(hidden_states, image_embeddings)

        # Compute sparse rewards (preference-based, sequence-level)
        sparse_rewards = self.compute_sparse_rewards(input_ids, attention_mask)

        # Combine rewards using hierarchical shaping
        combined_rewards = self.reward_shaper(
            dense_rewards=dense_rewards,
            sparse_rewards=sparse_rewards,
            confidence=confidence,
        )

        return {
            "logits": logits,
            "hidden_states": hidden_states,
            "dense_rewards": dense_rewards,
            "sparse_rewards": sparse_rewards,
            "combined_rewards": combined_rewards,
            "confidence": confidence,
        }

    def generate_and_refine(
        self,
        image_embeddings: torch.Tensor,
        max_length: int = 50,
        num_refinement_steps: int = 3,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate and refine captions.

        Args:
            image_embeddings: Image embeddings.
            max_length: Maximum generation length.
            num_refinement_steps: Number of RLHF refinement steps.

        Returns:
            Tuple of (generated_ids, reward_scores).
        """
        # Initial generation
        generated_ids = self.base_model.generate(
            image_embeddings=image_embeddings,
            max_length=max_length,
        )

        # Refinement loop
        for step in range(num_refinement_steps):
            attention_mask = torch.ones_like(generated_ids)

            # Compute rewards
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                image_embeddings=image_embeddings,
            )

            rewards = outputs["combined_rewards"]

            # Use rewards to guide refinement (simplified)
            # In full implementation, would use PPO or similar RL algorithm

        # Compute final rewards
        attention_mask = torch.ones_like(generated_ids)
        final_rewards = self.compute_sparse_rewards(generated_ids, attention_mask)

        return generated_ids, final_rewards
