"""Tests for data loading and preprocessing."""

import pytest
import torch

from hierarchical_diffusion_rlhf_caption_refinement.data.loader import (
    CaptionDataset,
    PreferenceDataset,
    get_dataloader,
)
from hierarchical_diffusion_rlhf_caption_refinement.data.preprocessing import (
    ImageProcessor,
    TextTokenizer,
    add_noise_schedule,
    compute_token_level_rewards,
    preprocess_batch,
)


class TestCaptionDataset:
    """Tests for CaptionDataset."""

    def test_dataset_creation(self):
        """Test dataset can be created."""
        dataset = CaptionDataset(
            split="train",
            max_samples=10,
            tokenizer_name="gpt2",
        )

        assert len(dataset) == 10

    def test_dataset_getitem(self):
        """Test dataset __getitem__ returns correct format."""
        dataset = CaptionDataset(split="train", max_samples=5)

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "caption" in item
        assert "image_url" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)

    def test_dataset_different_splits(self):
        """Test dataset works with different splits."""
        for split in ["train", "validation", "test"]:
            dataset = CaptionDataset(split=split, max_samples=5)
            assert len(dataset) == 5


class TestPreferenceDataset:
    """Tests for PreferenceDataset."""

    def test_preference_dataset_creation(self):
        """Test preference dataset can be created."""
        dataset = PreferenceDataset(split="train", max_samples=10)

        assert len(dataset) == 10

    def test_preference_dataset_getitem(self):
        """Test preference dataset returns correct format."""
        dataset = PreferenceDataset(split="train", max_samples=5)

        item = dataset[0]

        assert "chosen_input_ids" in item
        assert "chosen_attention_mask" in item
        assert "rejected_input_ids" in item
        assert "rejected_attention_mask" in item
        assert "image_url" in item


class TestDataLoader:
    """Tests for get_dataloader."""

    def test_dataloader_creation(self):
        """Test dataloader can be created."""
        dataset = CaptionDataset(split="train", max_samples=10)

        dataloader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=True,
        )

        assert dataloader is not None
        assert len(dataloader) == 5  # 10 samples / batch_size 2

    def test_dataloader_iteration(self):
        """Test dataloader can be iterated."""
        dataset = CaptionDataset(split="train", max_samples=10)
        dataloader = get_dataloader(dataset, batch_size=2)

        for batch in dataloader:
            assert batch["input_ids"].shape[0] == 2
            break


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_image_processor_creation(self):
        """Test image processor can be created."""
        processor = ImageProcessor(image_size=224, embedding_dim=768)

        assert processor.image_size == 224
        assert processor.embedding_dim == 768

    def test_process_single_image(self):
        """Test processing single image."""
        processor = ImageProcessor(embedding_dim=512)

        embedding = processor.process("test_image.jpg")

        assert embedding.shape == (512,)
        assert isinstance(embedding, torch.Tensor)

    def test_process_batch(self):
        """Test processing batch of images."""
        processor = ImageProcessor(embedding_dim=512)

        image_urls = ["img1.jpg", "img2.jpg", "img3.jpg"]
        embeddings = processor.process_batch(image_urls)

        assert embeddings.shape == (3, 512)


class TestTextTokenizer:
    """Tests for TextTokenizer."""

    def test_tokenizer_creation(self):
        """Test tokenizer can be created."""
        tokenizer = TextTokenizer(model_name="gpt2", max_length=128)

        assert tokenizer.max_length == 128

    def test_encode_texts(self):
        """Test encoding texts."""
        tokenizer = TextTokenizer(model_name="gpt2", max_length=64)

        texts = ["A dog in the park", "A cat on a chair"]
        encoding = tokenizer.encode(texts)

        assert "input_ids" in encoding
        assert "attention_mask" in encoding
        assert encoding["input_ids"].shape[0] == 2

    def test_decode_tokens(self):
        """Test decoding tokens."""
        tokenizer = TextTokenizer(model_name="gpt2")

        texts = ["Hello world"]
        encoding = tokenizer.encode(texts)
        decoded = tokenizer.decode(encoding["input_ids"])

        assert len(decoded) == 1
        assert isinstance(decoded[0], str)


class TestPreprocessing:
    """Tests for preprocessing functions."""

    def test_preprocess_batch(self, sample_batch, device):
        """Test batch preprocessing."""
        image_processor = ImageProcessor()

        processed = preprocess_batch(
            sample_batch,
            device=device,
            image_processor=image_processor,
        )

        assert "input_ids" in processed
        assert "attention_mask" in processed
        assert "image_embeddings" in processed

    def test_add_noise_schedule(self):
        """Test noise schedule."""
        x = torch.randn(2, 10, 768)
        timestep = 500

        noisy_x = add_noise_schedule(x, timestep, num_timesteps=1000)

        assert noisy_x.shape == x.shape
        assert not torch.allclose(noisy_x, x)

    def test_compute_token_level_rewards(self):
        """Test token-level reward computation."""
        logits = torch.randn(2, 10, 50257)
        target_ids = torch.randint(0, 50257, (2, 10))
        attention_mask = torch.ones(2, 10)

        rewards = compute_token_level_rewards(logits, target_ids, attention_mask)

        assert rewards.shape == (2, 10)
        assert not torch.isnan(rewards).any()
