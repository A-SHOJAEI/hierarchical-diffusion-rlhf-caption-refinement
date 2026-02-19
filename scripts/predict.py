#!/usr/bin/env python
"""Inference script for generating captions on new images."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hierarchical_diffusion_rlhf_caption_refinement.data.preprocessing import (
    ImageProcessor,
    TextTokenizer,
)
from hierarchical_diffusion_rlhf_caption_refinement.models.model import (
    DiffusionCaptionModel,
    RewardModel,
    RLHFCaptionRefiner,
)
from hierarchical_diffusion_rlhf_caption_refinement.utils.config import (
    get_device,
    load_config,
    set_random_seeds,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate captions for images using trained model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--model-type",
        type=str,
        choices=["base", "rlhf"],
        default="rlhf",
        help="Type of model to use",
    )

    parser.add_argument(
        "--image-url",
        type=str,
        default=None,
        help="Single image URL to caption",
    )

    parser.add_argument(
        "--image-list",
        type=str,
        default=None,
        help="Path to file with list of image URLs",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=50,
        help="Maximum caption length",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter",
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )

    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show confidence scores (RLHF model only)",
    )

    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    model_type: str,
    config: dict,
    device: torch.device,
) -> torch.nn.Module:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model_type: Type of model ('base' or 'rlhf').
        config: Configuration dictionary.
        device: Device to load model on.

    Returns:
        Loaded model.
    """
    logger.info(f"Loading {model_type} model from {checkpoint_path}")

    if model_type == "base":
        model = DiffusionCaptionModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            max_length=config["model"]["max_length"],
            num_timesteps=config["model"]["num_timesteps"],
            dropout=config["model"]["dropout"],
            image_embedding_dim=config["model"]["image_embedding_dim"],
        ).to(device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    elif model_type == "rlhf":
        base_model = DiffusionCaptionModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            max_length=config["model"]["max_length"],
            num_timesteps=config["model"]["num_timesteps"],
            dropout=config["model"]["dropout"],
            image_embedding_dim=config["model"]["image_embedding_dim"],
        ).to(device)

        reward_model = RewardModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["reward_num_layers"],
            num_heads=config["model"]["num_heads"],
            max_length=config["model"]["max_length"],
            dropout=config["model"]["dropout"],
        ).to(device)

        model = RLHFCaptionRefiner(
            base_model=base_model,
            reward_model=reward_model,
            hidden_dim=config["model"]["hidden_dim"],
            clip_weight=config["training"]["clip_weight"],
            preference_weight=config["training"]["preference_weight"],
        ).to(device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    logger.info("Model loaded successfully")

    return model


def load_image_urls(args: argparse.Namespace) -> List[str]:
    """Load image URLs from arguments.

    Args:
        args: Command line arguments.

    Returns:
        List of image URLs.

    Raises:
        ValueError: If no images specified.
    """
    image_urls = []

    if args.image_url:
        image_urls.append(args.image_url)

    if args.image_list:
        list_path = Path(args.image_list)
        if not list_path.exists():
            raise FileNotFoundError(f"Image list file not found: {list_path}")

        with open(list_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    image_urls.append(line)

    if not image_urls:
        raise ValueError("No images specified. Use --image-url or --image-list")

    return image_urls


def generate_captions(
    model: torch.nn.Module,
    image_urls: List[str],
    image_processor: ImageProcessor,
    tokenizer: TextTokenizer,
    device: torch.device,
    args: argparse.Namespace,
) -> List[dict]:
    """Generate captions for images.

    Args:
        model: Trained model.
        image_urls: List of image URLs.
        image_processor: Image processor.
        tokenizer: Text tokenizer.
        device: Device to run on.
        args: Command line arguments.

    Returns:
        List of prediction dictionaries.
    """
    predictions = []

    logger.info(f"Generating captions for {len(image_urls)} images")

    with torch.no_grad():
        for i, image_url in enumerate(image_urls):
            logger.info(f"Processing image {i+1}/{len(image_urls)}: {image_url}")

            # Process image
            image_embedding = image_processor.process(image_url).unsqueeze(0).to(device)

            # Generate caption
            if isinstance(model, RLHFCaptionRefiner):
                generated_ids, rewards = model.generate_and_refine(
                    image_embeddings=image_embedding,
                    max_length=args.max_length,
                    num_refinement_steps=3,
                )
                reward_score = rewards[0].item()
            else:
                generated_ids = model.generate(
                    image_embeddings=image_embedding,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                )
                reward_score = None

            # Decode caption
            caption = tokenizer.decode(generated_ids)[0]

            # Create prediction
            prediction = {
                "image_url": image_url,
                "caption": caption,
            }

            if reward_score is not None and args.show_confidence:
                prediction["confidence_score"] = reward_score

            predictions.append(prediction)

            # Print to console
            print(f"\nImage: {image_url}")
            print(f"Caption: {caption}")
            if reward_score is not None and args.show_confidence:
                print(f"Confidence: {reward_score:.4f}")

    return predictions


def main() -> None:
    """Main prediction pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(log_level="INFO")

    logger.info("Starting Caption Generation")

    # Set random seeds for reproducibility
    set_random_seeds(config["seed"])

    # Get device
    device = get_device(config["device"])

    try:
        # Load model
        model = load_model(
            checkpoint_path=args.checkpoint,
            model_type=args.model_type,
            config=config,
            device=device,
        )

        # Load image URLs
        image_urls = load_image_urls(args)
        logger.info(f"Loaded {len(image_urls)} image URLs")

        # Initialize processors
        image_processor = ImageProcessor(
            embedding_dim=config["model"]["image_embedding_dim"],
        )

        tokenizer = TextTokenizer(
            model_name=config["data"]["tokenizer_name"],
            max_length=config["data"]["max_length"],
        )

        # Generate captions
        predictions = generate_captions(
            model=model,
            image_urls=image_urls,
            image_processor=image_processor,
            tokenizer=tokenizer,
            device=device,
            args=args,
        )

        # Save predictions if output specified
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(predictions, f, indent=2)

            logger.info(f"Saved predictions to {output_path}")

        logger.info("=" * 80)
        logger.info(f"Generated {len(predictions)} captions successfully")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Prediction failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
