"""Custom loss functions, layers, and components for hierarchical RLHF.

This module contains the novel hierarchical reward shaping mechanism
that combines dense CLIP-based rewards with sparse human preferences.
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class HierarchicalRewardShaper(nn.Module):
    """Novel hierarchical reward shaping mechanism.

    This is the key innovation: combines dense token-level CLIP rewards
    with sparse sequence-level human preferences using adaptive weighting
    based on generation confidence.

    The adaptive weighting allows the model to rely more on CLIP rewards
    when confidence is low (early training) and shift to human preferences
    when confidence is high (later training).
    """

    def __init__(
        self,
        dense_weight_init: float = 0.7,
        sparse_weight_init: float = 0.3,
        confidence_temperature: float = 0.1,
        min_dense_weight: float = 0.3,
        max_dense_weight: float = 0.9,
    ) -> None:
        """Initialize hierarchical reward shaper.

        Args:
            dense_weight_init: Initial weight for dense (CLIP) rewards.
            sparse_weight_init: Initial weight for sparse (human) rewards.
            confidence_temperature: Temperature for confidence scaling.
            min_dense_weight: Minimum weight for dense rewards.
            max_dense_weight: Maximum weight for dense rewards.
        """
        super().__init__()

        # Learnable weight parameters
        self.dense_weight = nn.Parameter(torch.tensor(dense_weight_init))
        self.sparse_weight = nn.Parameter(torch.tensor(sparse_weight_init))

        self.confidence_temperature = confidence_temperature
        self.min_dense_weight = min_dense_weight
        self.max_dense_weight = max_dense_weight

    def compute_adaptive_weights(
        self,
        confidence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute adaptive weights based on generation confidence.

        Args:
            confidence: Confidence scores of shape (batch_size,).

        Returns:
            Tuple of (dense_weight, sparse_weight) with shape (batch_size,).
        """
        # Scale confidence with temperature
        scaled_conf = torch.sigmoid(confidence / self.confidence_temperature)

        # Compute adaptive dense weight (decreases with confidence)
        dense_w = self.max_dense_weight - (self.max_dense_weight - self.min_dense_weight) * scaled_conf

        # Sparse weight is complement
        sparse_w = 1.0 - dense_w

        return dense_w, sparse_w

    def forward(
        self,
        dense_rewards: torch.Tensor,
        sparse_rewards: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute hierarchical reward by combining dense and sparse rewards.

        Args:
            dense_rewards: Token-level CLIP rewards of shape (batch_size, seq_len).
            sparse_rewards: Sequence-level preference rewards of shape (batch_size,).
            confidence: Optional confidence scores of shape (batch_size,).

        Returns:
            Combined rewards of shape (batch_size, seq_len).
        """
        batch_size = dense_rewards.shape[0]

        if confidence is not None:
            # Adaptive weighting based on confidence
            dense_w, sparse_w = self.compute_adaptive_weights(confidence)
        else:
            # Use fixed weights
            dense_w = torch.sigmoid(self.dense_weight).expand(batch_size)
            sparse_w = torch.sigmoid(self.sparse_weight).expand(batch_size)

        # Normalize weights
        total_w = dense_w + sparse_w + 1e-8
        dense_w = dense_w / total_w
        sparse_w = sparse_w / total_w

        # Expand sparse rewards to sequence length
        sparse_rewards_expanded = sparse_rewards.unsqueeze(1).expand_as(dense_rewards)

        # Combine rewards
        combined = (
            dense_w.unsqueeze(1) * dense_rewards +
            sparse_w.unsqueeze(1) * sparse_rewards_expanded
        )

        return combined


class CLIPAlignmentLoss(nn.Module):
    """Custom loss function for CLIP-based alignment.

    Computes alignment between generated captions and images using
    simulated CLIP scores. This provides dense, token-level feedback.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.2,
    ) -> None:
        """Initialize CLIP alignment loss.

        Args:
            temperature: Temperature for contrastive loss.
            margin: Margin for ranking loss.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def compute_similarity(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute similarity between text and image embeddings.

        Args:
            text_embeddings: Text embeddings of shape (batch_size, hidden_dim).
            image_embeddings: Image embeddings of shape (batch_size, hidden_dim).

        Returns:
            Similarity scores of shape (batch_size,).
        """
        # Normalize embeddings
        text_norm = F.normalize(text_embeddings, dim=-1)
        image_norm = F.normalize(image_embeddings, dim=-1)

        # Compute cosine similarity
        similarity = torch.sum(text_norm * image_norm, dim=-1)

        return similarity

    def forward(
        self,
        text_embeddings: torch.Tensor,
        image_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute CLIP alignment loss.

        Args:
            text_embeddings: Text embeddings of shape (batch_size, hidden_dim).
            image_embeddings: Image embeddings of shape (batch_size, hidden_dim).

        Returns:
            Scalar loss value.
        """
        batch_size = text_embeddings.shape[0]

        # Compute similarity matrix
        text_norm = F.normalize(text_embeddings, dim=-1)
        image_norm = F.normalize(image_embeddings, dim=-1)

        logits = torch.matmul(text_norm, image_norm.t()) / self.temperature

        # Contrastive loss (both directions)
        labels = torch.arange(batch_size, device=logits.device)

        loss_t2i = F.cross_entropy(logits, labels)
        loss_i2t = F.cross_entropy(logits.t(), labels)

        loss = (loss_t2i + loss_i2t) / 2.0

        return loss


class DiffusionScheduler(nn.Module):
    """Diffusion noise scheduler for caption generation.

    Implements a cosine noise schedule for the diffusion process.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = "cosine",
    ) -> None:
        """Initialize diffusion scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            schedule_type: Type of noise schedule ('linear' or 'cosine').
        """
        super().__init__()

        self.num_timesteps = num_timesteps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Register as buffers so they move to correct device
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def _cosine_beta_schedule(self, num_timesteps: int) -> torch.Tensor:
        """Compute cosine noise schedule.

        Args:
            num_timesteps: Number of timesteps.

        Returns:
            Beta values for each timestep.
        """
        steps = num_timesteps + 1
        x = torch.linspace(0, num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def add_noise(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Add noise to input according to diffusion schedule.

        Args:
            x: Input tensor of shape (batch_size, ...).
            timesteps: Timestep for each sample of shape (batch_size,).
            noise: Optional noise tensor (generated if None).

        Returns:
            Noisy tensor of same shape as x.
        """
        if noise is None:
            noise = torch.randn_like(x)

        # Get schedule values for these timesteps
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        # Add noise
        noisy_x = sqrt_alpha * x + sqrt_one_minus_alpha * noise

        return noisy_x

    def get_posterior_mean(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        predicted_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute posterior mean for denoising step.

        Args:
            x_t: Noisy input at timestep t.
            timesteps: Current timesteps.
            predicted_noise: Predicted noise from model.

        Returns:
            Posterior mean for previous timestep.
        """
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[timesteps]

        # Reshape for broadcasting
        while len(sqrt_alpha.shape) < len(x_t.shape):
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        # Compute x_0 prediction
        x_0_pred = (x_t - sqrt_one_minus_alpha * predicted_noise) / sqrt_alpha

        return x_0_pred


class ConfidenceEstimator(nn.Module):
    """Estimate generation confidence for adaptive reward weighting.

    This component analyzes model outputs to compute confidence scores
    used in hierarchical reward shaping.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        num_layers: int = 2,
    ) -> None:
        """Initialize confidence estimator.

        Args:
            hidden_dim: Hidden dimension.
            num_layers: Number of layers.
        """
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
        layers.append(nn.Linear(hidden_dim, 1))

        self.estimator = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Estimate confidence from hidden states.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Confidence scores of shape (batch_size,).
        """
        # Pool over sequence dimension (mean pooling)
        pooled = hidden_states.mean(dim=1)

        # Estimate confidence
        confidence = self.estimator(pooled).squeeze(-1)

        return confidence


class PreferenceHead(nn.Module):
    """Preference prediction head for reward modeling.

    Predicts preference scores for ranking caption pairs.
    """

    def __init__(
        self,
        hidden_dim: int = 768,
        dropout: float = 0.1,
    ) -> None:
        """Initialize preference head.

        Args:
            hidden_dim: Hidden dimension.
            dropout: Dropout probability.
        """
        super().__init__()

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict preference score.

        Args:
            hidden_states: Hidden states of shape (batch_size, seq_len, hidden_dim).

        Returns:
            Preference scores of shape (batch_size,).
        """
        # Pool over sequence (use last token)
        pooled = hidden_states[:, -1, :]

        # Predict score
        score = self.head(pooled).squeeze(-1)

        return score
