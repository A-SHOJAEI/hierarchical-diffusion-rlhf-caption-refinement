"""Pytest configuration and fixtures."""

import sys
from pathlib import Path

# Add src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
from transformers import AutoTokenizer


@pytest.fixture
def device() -> torch.device:
    """Provide device for testing.

    Returns:
        CPU device for testing.
    """
    return torch.device("cpu")


@pytest.fixture
def tokenizer():
    """Provide tokenizer for testing.

    Returns:
        GPT2 tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_config() -> dict:
    """Provide sample configuration for testing.

    Returns:
        Configuration dictionary.
    """
    return {
        "seed": 42,
        "device": "cpu",
        "model": {
            "vocab_size": 50257,
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 4,
            "max_length": 64,
            "num_timesteps": 100,
            "dropout": 0.1,
            "image_embedding_dim": 256,
            "reward_num_layers": 2,
        },
        "data": {
            "tokenizer_name": "gpt2",
            "max_length": 64,
            "train_max_samples": 10,
            "val_max_samples": 5,
        },
        "training": {
            "base_pretrain_epochs": 1,
            "base_learning_rate": 0.001,
            "base_batch_size": 2,
            "reward_epochs": 1,
            "reward_learning_rate": 0.001,
            "reward_batch_size": 2,
            "rlhf_epochs": 1,
            "rlhf_learning_rate": 0.0001,
            "rlhf_batch_size": 2,
            "weight_decay": 0.01,
            "patience": 3,
            "clip_weight": 0.7,
            "preference_weight": 0.3,
        },
        "paths": {
            "save_dir": "test_models",
            "results_dir": "test_results",
        },
    }


@pytest.fixture
def sample_batch(tokenizer) -> dict:
    """Provide sample batch for testing.

    Args:
        tokenizer: Tokenizer fixture.

    Returns:
        Sample batch dictionary.
    """
    captions = [
        "A dog playing in the park",
        "A cat sitting on a chair",
    ]

    encoding = tokenizer(
        captions,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "caption": captions,
        "image_url": ["img1.jpg", "img2.jpg"],
    }


@pytest.fixture
def sample_preference_batch(tokenizer) -> dict:
    """Provide sample preference batch for testing.

    Args:
        tokenizer: Tokenizer fixture.

    Returns:
        Sample preference batch dictionary.
    """
    chosen = ["A detailed photo of a dog", "A beautiful sunset"]
    rejected = ["A dog", "Sunset"]

    chosen_enc = tokenizer(
        chosen,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    rejected_enc = tokenizer(
        rejected,
        max_length=64,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
        "image_url": ["img1.jpg", "img2.jpg"],
    }
