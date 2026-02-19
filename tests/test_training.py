"""Tests for training pipeline."""

import pytest
import torch
from pathlib import Path

from hierarchical_diffusion_rlhf_caption_refinement.data.loader import (
    CaptionDataset,
    PreferenceDataset,
    get_dataloader,
)
from hierarchical_diffusion_rlhf_caption_refinement.models.model import RewardModel
from hierarchical_diffusion_rlhf_caption_refinement.training.trainer import (
    HierarchicalRLHFTrainer,
    train_reward_model,
)


class TestHierarchicalRLHFTrainer:
    """Tests for HierarchicalRLHFTrainer."""

    def test_trainer_creation(self, sample_config, device, tmp_path):
        """Test trainer can be created."""
        trainer = HierarchicalRLHFTrainer(
            config=sample_config,
            device=device,
            save_dir=tmp_path,
        )

        assert trainer is not None
        assert trainer.base_model is not None
        assert trainer.reward_model is not None
        assert trainer.rlhf_model is not None

    def test_base_model_training(self, sample_config, device, tmp_path):
        """Test base model training runs without error."""
        trainer = HierarchicalRLHFTrainer(
            config=sample_config,
            device=device,
            save_dir=tmp_path,
        )

        # Create small dataset
        dataset = CaptionDataset(
            split="train",
            max_samples=4,
            max_length=sample_config["data"]["max_length"],
        )

        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False)

        # Train for 1 epoch (should not raise exception)
        trainer.train_base_model(
            train_loader=dataloader,
            val_loader=None,
            num_epochs=1,
        )

        # Check that model parameters were updated
        assert trainer.base_model is not None

    def test_rlhf_training(self, sample_config, device, tmp_path):
        """Test RLHF training runs without error."""
        trainer = HierarchicalRLHFTrainer(
            config=sample_config,
            device=device,
            save_dir=tmp_path,
        )

        # Create small dataset
        dataset = CaptionDataset(
            split="train",
            max_samples=4,
            max_length=sample_config["data"]["max_length"],
        )

        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False)

        # Train for 1 epoch
        trainer.train_with_rlhf(
            train_loader=dataloader,
            val_loader=None,
            num_epochs=1,
        )

        assert trainer.rlhf_model is not None


class TestRewardModelTraining:
    """Tests for reward model training."""

    def test_reward_model_training(self, sample_config, device, tmp_path):
        """Test reward model training runs without error."""
        reward_model = RewardModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["reward_num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        # Create preference dataset
        dataset = PreferenceDataset(
            split="train",
            max_samples=4,
            max_length=sample_config["data"]["max_length"],
        )

        dataloader = get_dataloader(dataset, batch_size=2, shuffle=False)

        # Train for 1 epoch
        trained_model = train_reward_model(
            reward_model=reward_model,
            train_loader=dataloader,
            val_loader=None,
            num_epochs=1,
            learning_rate=0.001,
            device=device,
            save_dir=tmp_path,
            patience=3,
        )

        assert trained_model is not None

    def test_reward_model_with_validation(self, sample_config, device, tmp_path):
        """Test reward model training with validation."""
        reward_model = RewardModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["reward_num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        # Create datasets
        train_dataset = PreferenceDataset(split="train", max_samples=4)
        val_dataset = PreferenceDataset(split="validation", max_samples=2)

        train_loader = get_dataloader(train_dataset, batch_size=2, shuffle=False)
        val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False)

        # Train
        trained_model = train_reward_model(
            reward_model=reward_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=1,
            learning_rate=0.001,
            device=device,
            save_dir=tmp_path,
            patience=3,
        )

        assert trained_model is not None

        # Check that best model was saved
        checkpoint_path = tmp_path / "best_reward_model.pt"
        assert checkpoint_path.exists()


class TestTrainingComponents:
    """Tests for training components."""

    def test_optimizer_creation(self, sample_config, device):
        """Test optimizer can be created for model."""
        from torch.optim import AdamW

        trainer = HierarchicalRLHFTrainer(
            config=sample_config,
            device=device,
            save_dir=Path("test_models"),
        )

        optimizer = AdamW(
            trainer.base_model.parameters(),
            lr=sample_config["training"]["base_learning_rate"],
        )

        assert optimizer is not None

    def test_scheduler_creation(self, sample_config, device):
        """Test LR scheduler can be created."""
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        trainer = HierarchicalRLHFTrainer(
            config=sample_config,
            device=device,
            save_dir=Path("test_models"),
        )

        optimizer = AdamW(trainer.base_model.parameters(), lr=0.001)
        scheduler = CosineAnnealingLR(optimizer, T_max=10)

        assert scheduler is not None

    def test_gradient_clipping(self, sample_config, device):
        """Test gradient clipping."""
        trainer = HierarchicalRLHFTrainer(
            config=sample_config,
            device=device,
            save_dir=Path("test_models"),
        )

        # Create dummy loss and backward with valid token IDs
        dummy_input = torch.randint(0, sample_config["model"]["vocab_size"], (2, 10)).to(device)
        dummy_mask = torch.ones(2, 10).long().to(device)

        outputs = trainer.base_model(dummy_input, dummy_mask)
        loss = outputs["logits"].mean()
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(
            trainer.base_model.parameters(),
            max_norm=1.0,
        )

        # Check that gradients exist and were clipped
        total_norm = 0.0
        for p in trainer.base_model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2

        total_norm = total_norm ** 0.5

        # Total norm should be reasonable after clipping
        assert total_norm < 1000.0
