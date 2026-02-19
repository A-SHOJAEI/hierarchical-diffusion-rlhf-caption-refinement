"""Training loop with RLHF, learning rate scheduling, and early stopping."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from hierarchical_diffusion_rlhf_caption_refinement.data.preprocessing import ImageProcessor
from hierarchical_diffusion_rlhf_caption_refinement.models.model import (
    DiffusionCaptionModel,
    RewardModel,
    RLHFCaptionRefiner,
)

logger = logging.getLogger(__name__)


def train_reward_model(
    reward_model: RewardModel,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
    save_dir: Path,
    patience: int = 3,
) -> RewardModel:
    """Train reward model on preference data.

    Args:
        reward_model: Reward model to train.
        train_loader: Training dataloader with preference pairs.
        val_loader: Validation dataloader.
        num_epochs: Number of training epochs.
        learning_rate: Learning rate.
        device: Device to train on.
        save_dir: Directory to save checkpoints.
        patience: Early stopping patience.

    Returns:
        Trained reward model.
    """
    reward_model = reward_model.to(device)
    optimizer = AdamW(reward_model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training
        reward_model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            # Move to device
            chosen_ids = batch["chosen_input_ids"].to(device)
            chosen_mask = batch["chosen_attention_mask"].to(device)
            rejected_ids = batch["rejected_input_ids"].to(device)
            rejected_mask = batch["rejected_attention_mask"].to(device)

            # Forward pass
            chosen_rewards = reward_model(chosen_ids, chosen_mask)
            rejected_rewards = reward_model(rejected_ids, rejected_mask)

            # Preference loss (chosen should have higher reward)
            loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8).mean()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reward_model.parameters(), max_norm=1.0)
            optimizer.step()

            # Metrics
            train_loss += loss.item()
            acc = (chosen_rewards > rejected_rewards).float().mean().item()
            train_acc += acc
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})

        train_loss /= num_batches
        train_acc /= num_batches

        # Validation
        if val_loader is not None:
            reward_model.eval()
            val_loss = 0.0
            val_acc = 0.0
            num_val_batches = 0

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    chosen_ids = batch["chosen_input_ids"].to(device)
                    chosen_mask = batch["chosen_attention_mask"].to(device)
                    rejected_ids = batch["rejected_input_ids"].to(device)
                    rejected_mask = batch["rejected_attention_mask"].to(device)

                    chosen_rewards = reward_model(chosen_ids, chosen_mask)
                    rejected_rewards = reward_model(rejected_ids, rejected_mask)

                    loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards) + 1e-8).mean()

                    val_loss += loss.item()
                    val_acc += (chosen_rewards > rejected_rewards).float().mean().item()
                    num_val_batches += 1

            val_loss /= num_val_batches
            val_acc /= num_val_batches

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                save_path = save_dir / "best_reward_model.pt"
                torch.save(reward_model.state_dict(), save_path)
                logger.info(f"Saved best reward model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
        else:
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}"
            )

        scheduler.step()

    # Load best model if validation was used
    if val_loader is not None:
        best_model_path = save_dir / "best_reward_model.pt"
        if best_model_path.exists():
            reward_model.load_state_dict(torch.load(best_model_path))
            logger.info(f"Loaded best reward model from {best_model_path}")

    return reward_model


class HierarchicalRLHFTrainer:
    """Trainer for hierarchical RLHF caption refinement.

    This trainer implements the full pipeline:
    1. Pre-train diffusion caption model
    2. Train reward model on preferences
    3. Fine-tune with RLHF using hierarchical rewards
    """

    def __init__(
        self,
        config: Dict[str, Any],
        device: torch.device,
        save_dir: Path,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Configuration dictionary.
            device: Device to train on.
            save_dir: Directory to save checkpoints.
        """
        self.config = config
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Initialize image processor
        self.image_processor = ImageProcessor(
            embedding_dim=config["model"]["image_embedding_dim"],
        )

        # Initialize models
        self.base_model = DiffusionCaptionModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            max_length=config["model"]["max_length"],
            num_timesteps=config["model"]["num_timesteps"],
            dropout=config["model"]["dropout"],
            image_embedding_dim=config["model"]["image_embedding_dim"],
        ).to(device)

        self.reward_model = RewardModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["reward_num_layers"],
            num_heads=config["model"]["num_heads"],
            max_length=config["model"]["max_length"],
            dropout=config["model"]["dropout"],
        ).to(device)

        self.rlhf_model = RLHFCaptionRefiner(
            base_model=self.base_model,
            reward_model=self.reward_model,
            hidden_dim=config["model"]["hidden_dim"],
            clip_weight=config["training"]["clip_weight"],
            preference_weight=config["training"]["preference_weight"],
        ).to(device)

        # Training state
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_base_model(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
    ) -> None:
        """Pre-train base diffusion caption model.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            num_epochs: Number of epochs.
        """
        logger.info("Starting base model pre-training")

        optimizer = AdamW(
            self.base_model.parameters(),
            lr=self.config["training"]["base_learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        for epoch in range(num_epochs):
            # Training
            self.base_model.train()
            train_loss = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Base Train]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Process images
                image_urls = batch["image_url"]
                image_embeddings = self.image_processor.process_batch(image_urls).to(self.device)

                # Sample random timesteps for diffusion
                batch_size = input_ids.shape[0]
                timesteps = torch.randint(
                    0, self.base_model.num_timesteps,
                    (batch_size,),
                    device=self.device,
                )

                # Forward pass
                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embeddings,
                    timesteps=timesteps,
                )

                logits = outputs["logits"]

                # Compute loss (use labels with pad positions set to -100)
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.base_model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= num_batches

            # Validation
            if val_loader is not None:
                val_loss = self._validate_base_model(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    save_path = self.save_dir / "best_base_model.pt"
                    torch.save(self.base_model.state_dict(), save_path)
                    logger.info(f"Saved best base model to {save_path}")
                else:
                    self.patience_counter += 1
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}")

            scheduler.step()

            # Early stopping
            patience = self.config["training"]["patience"]
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    def _validate_base_model(self, val_loader: DataLoader) -> float:
        """Validate base model.

        Args:
            val_loader: Validation dataloader.

        Returns:
            Average validation loss.
        """
        self.base_model.eval()
        val_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                image_urls = batch["image_url"]
                image_embeddings = self.image_processor.process_batch(image_urls).to(self.device)

                batch_size = input_ids.shape[0]
                timesteps = torch.randint(
                    0, self.base_model.num_timesteps,
                    (batch_size,),
                    device=self.device,
                )

                outputs = self.base_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embeddings,
                    timesteps=timesteps,
                )

                logits = outputs["logits"]
                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )

                val_loss += loss.item()
                num_batches += 1

        return val_loss / num_batches

    def train_with_rlhf(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        num_epochs: int,
    ) -> None:
        """Fine-tune with RLHF using hierarchical rewards.

        Args:
            train_loader: Training dataloader.
            val_loader: Validation dataloader.
            num_epochs: Number of epochs.
        """
        logger.info("Starting RLHF fine-tuning")

        optimizer = AdamW(
            self.rlhf_model.parameters(),
            lr=self.config["training"]["rlhf_learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)

        self.best_val_loss = float("inf")
        self.patience_counter = 0

        for epoch in range(num_epochs):
            # Training
            self.rlhf_model.train()
            train_loss = 0.0
            train_reward = 0.0
            num_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [RLHF]")
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                image_urls = batch["image_url"]
                image_embeddings = self.image_processor.process_batch(image_urls).to(self.device)

                # Forward pass with hierarchical rewards
                outputs = self.rlhf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embeddings,
                )

                logits = outputs["logits"]
                combined_rewards = outputs["combined_rewards"]

                # Compute policy gradient loss
                log_probs = F.log_softmax(logits, dim=-1)
                selected_log_probs = torch.gather(
                    log_probs, -1, input_ids.unsqueeze(-1)
                ).squeeze(-1)

                # Weight by rewards and attention mask
                policy_loss = -(selected_log_probs * combined_rewards * attention_mask.float()).sum() / attention_mask.sum()

                # Language modeling loss for stability
                lm_loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )

                # Combined loss
                loss = policy_loss + 0.1 * lm_loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.rlhf_model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_reward += combined_rewards.mean().item()
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "reward": f"{combined_rewards.mean().item():.4f}",
                })

            train_loss /= num_batches
            train_reward /= num_batches

            # Validation
            if val_loader is not None:
                val_loss, val_reward = self._validate_rlhf(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Reward: {train_reward:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Reward: {val_reward:.4f}"
                )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    save_path = self.save_dir / "best_rlhf_model.pt"
                    torch.save(self.rlhf_model.state_dict(), save_path)
                    logger.info(f"Saved best RLHF model to {save_path}")
                else:
                    self.patience_counter += 1
            else:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Reward: {train_reward:.4f}"
                )

            scheduler.step()

            # Early stopping
            patience = self.config["training"]["patience"]
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    def _validate_rlhf(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate RLHF model.

        Args:
            val_loader: Validation dataloader.

        Returns:
            Tuple of (average validation loss, average reward).
        """
        self.rlhf_model.eval()
        val_loss = 0.0
        val_reward = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="RLHF Validation"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                image_urls = batch["image_url"]
                image_embeddings = self.image_processor.process_batch(image_urls).to(self.device)

                outputs = self.rlhf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_embeddings=image_embeddings,
                )

                logits = outputs["logits"]
                combined_rewards = outputs["combined_rewards"]

                loss = F.cross_entropy(
                    logits.view(-1, logits.shape[-1]),
                    labels.view(-1),
                    ignore_index=-100,
                )

                val_loss += loss.item()
                val_reward += combined_rewards.mean().item()
                num_batches += 1

        return val_loss / num_batches, val_reward / num_batches
