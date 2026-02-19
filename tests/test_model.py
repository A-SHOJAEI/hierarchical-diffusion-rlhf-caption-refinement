"""Tests for model implementations."""

import pytest
import torch

from hierarchical_diffusion_rlhf_caption_refinement.models.components import (
    CLIPAlignmentLoss,
    ConfidenceEstimator,
    DiffusionScheduler,
    HierarchicalRewardShaper,
    PreferenceHead,
)
from hierarchical_diffusion_rlhf_caption_refinement.models.model import (
    DiffusionCaptionModel,
    RewardModel,
    RLHFCaptionRefiner,
)


class TestDiffusionCaptionModel:
    """Tests for DiffusionCaptionModel."""

    def test_model_creation(self, sample_config, device):
        """Test model can be created."""
        model = DiffusionCaptionModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        assert model is not None

    def test_forward_pass(self, sample_config, sample_batch, device):
        """Test forward pass."""
        model = DiffusionCaptionModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["num_layers"],
            num_heads=sample_config["model"]["num_heads"],
            image_embedding_dim=sample_config["model"]["image_embedding_dim"],
        ).to(device)

        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)
        batch_size = input_ids.shape[0]

        # Create image embeddings
        image_embeddings = torch.randn(
            batch_size, sample_config["model"]["image_embedding_dim"]
        ).to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeddings=image_embeddings,
        )

        assert "logits" in outputs
        assert "hidden_states" in outputs
        assert outputs["logits"].shape[0] == batch_size

    def test_generation(self, sample_config, device):
        """Test caption generation."""
        model = DiffusionCaptionModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["num_layers"],
            num_heads=sample_config["model"]["num_heads"],
            image_embedding_dim=sample_config["model"]["image_embedding_dim"],
            num_timesteps=10,  # Use fewer steps for testing
        ).to(device)

        batch_size = 2
        image_embeddings = torch.randn(
            batch_size, sample_config["model"]["image_embedding_dim"]
        ).to(device)

        generated_ids = model.generate(
            image_embeddings=image_embeddings,
            max_length=20,
        )

        assert generated_ids.shape[0] == batch_size
        assert generated_ids.shape[1] == 20


class TestRewardModel:
    """Tests for RewardModel."""

    def test_reward_model_creation(self, sample_config, device):
        """Test reward model can be created."""
        model = RewardModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["reward_num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        assert model is not None

    def test_reward_prediction(self, sample_config, sample_batch, device):
        """Test reward prediction."""
        model = RewardModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["reward_num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)

        rewards = model(input_ids, attention_mask)

        assert rewards.shape[0] == input_ids.shape[0]
        assert len(rewards.shape) == 1


class TestRLHFCaptionRefiner:
    """Tests for RLHFCaptionRefiner."""

    def test_rlhf_model_creation(self, sample_config, device):
        """Test RLHF model can be created."""
        base_model = DiffusionCaptionModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["num_layers"],
            num_heads=sample_config["model"]["num_heads"],
            image_embedding_dim=sample_config["model"]["image_embedding_dim"],
        ).to(device)

        reward_model = RewardModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["reward_num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        rlhf_model = RLHFCaptionRefiner(
            base_model=base_model,
            reward_model=reward_model,
        ).to(device)

        assert rlhf_model is not None

    def test_forward_with_rewards(self, sample_config, sample_batch, device):
        """Test forward pass with reward computation."""
        base_model = DiffusionCaptionModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["num_layers"],
            num_heads=sample_config["model"]["num_heads"],
            image_embedding_dim=sample_config["model"]["image_embedding_dim"],
        ).to(device)

        reward_model = RewardModel(
            vocab_size=sample_config["model"]["vocab_size"],
            hidden_dim=sample_config["model"]["hidden_dim"],
            num_layers=sample_config["model"]["reward_num_layers"],
            num_heads=sample_config["model"]["num_heads"],
        ).to(device)

        rlhf_model = RLHFCaptionRefiner(
            base_model=base_model,
            reward_model=reward_model,
            hidden_dim=sample_config["model"]["hidden_dim"],
        ).to(device)

        input_ids = sample_batch["input_ids"].to(device)
        attention_mask = sample_batch["attention_mask"].to(device)
        batch_size = input_ids.shape[0]

        image_embeddings = torch.randn(
            batch_size, sample_config["model"]["image_embedding_dim"]
        ).to(device)

        outputs = rlhf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_embeddings=image_embeddings,
        )

        assert "logits" in outputs
        assert "dense_rewards" in outputs
        assert "sparse_rewards" in outputs
        assert "combined_rewards" in outputs
        assert "confidence" in outputs


class TestHierarchicalRewardShaper:
    """Tests for HierarchicalRewardShaper."""

    def test_reward_shaper_creation(self):
        """Test reward shaper can be created."""
        shaper = HierarchicalRewardShaper()

        assert shaper is not None

    def test_combine_rewards(self):
        """Test reward combination."""
        shaper = HierarchicalRewardShaper()

        batch_size, seq_len = 2, 10

        dense_rewards = torch.randn(batch_size, seq_len)
        sparse_rewards = torch.randn(batch_size)

        combined = shaper(dense_rewards, sparse_rewards)

        assert combined.shape == (batch_size, seq_len)

    def test_adaptive_weighting(self):
        """Test adaptive weight computation."""
        shaper = HierarchicalRewardShaper()

        confidence = torch.tensor([0.2, 0.8])

        dense_w, sparse_w = shaper.compute_adaptive_weights(confidence)

        assert dense_w.shape == confidence.shape
        assert sparse_w.shape == confidence.shape
        # Weights should be normalized
        assert torch.allclose(dense_w + sparse_w, torch.ones_like(dense_w), atol=0.01)


class TestCLIPAlignmentLoss:
    """Tests for CLIPAlignmentLoss."""

    def test_loss_creation(self):
        """Test loss can be created."""
        loss_fn = CLIPAlignmentLoss()

        assert loss_fn is not None

    def test_compute_loss(self):
        """Test loss computation."""
        loss_fn = CLIPAlignmentLoss()

        batch_size = 4
        hidden_dim = 256

        text_embeddings = torch.randn(batch_size, hidden_dim)
        image_embeddings = torch.randn(batch_size, hidden_dim)

        loss = loss_fn(text_embeddings, image_embeddings)

        assert isinstance(loss.item(), float)
        assert loss.item() >= 0


class TestDiffusionScheduler:
    """Tests for DiffusionScheduler."""

    def test_scheduler_creation(self):
        """Test scheduler can be created."""
        scheduler = DiffusionScheduler(num_timesteps=100)

        assert scheduler.num_timesteps == 100

    def test_add_noise(self):
        """Test noise addition."""
        scheduler = DiffusionScheduler(num_timesteps=100)

        x = torch.randn(2, 10, 768)
        timesteps = torch.tensor([10, 50])

        noisy_x = scheduler.add_noise(x, timesteps)

        assert noisy_x.shape == x.shape
        assert not torch.allclose(noisy_x, x)

    def test_posterior_mean(self):
        """Test posterior mean computation."""
        scheduler = DiffusionScheduler(num_timesteps=100)

        x_t = torch.randn(2, 10, 768)
        timesteps = torch.tensor([10, 50])
        predicted_noise = torch.randn_like(x_t)

        x_0_pred = scheduler.get_posterior_mean(x_t, timesteps, predicted_noise)

        assert x_0_pred.shape == x_t.shape


class TestConfidenceEstimator:
    """Tests for ConfidenceEstimator."""

    def test_estimator_creation(self):
        """Test confidence estimator can be created."""
        estimator = ConfidenceEstimator(hidden_dim=256)

        assert estimator is not None

    def test_confidence_estimation(self):
        """Test confidence estimation."""
        estimator = ConfidenceEstimator(hidden_dim=256)

        batch_size, seq_len, hidden_dim = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        confidence = estimator(hidden_states)

        assert confidence.shape == (batch_size,)


class TestPreferenceHead:
    """Tests for PreferenceHead."""

    def test_preference_head_creation(self):
        """Test preference head can be created."""
        head = PreferenceHead(hidden_dim=256)

        assert head is not None

    def test_preference_prediction(self):
        """Test preference score prediction."""
        head = PreferenceHead(hidden_dim=256)

        batch_size, seq_len, hidden_dim = 2, 10, 256
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)

        scores = head(hidden_states)

        assert scores.shape == (batch_size,)
