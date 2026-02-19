#!/usr/bin/env python
"""Evaluation script for trained caption refinement models."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root and src/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch

from hierarchical_diffusion_rlhf_caption_refinement.data.loader import (
    CaptionDataset,
    get_dataloader,
)
from hierarchical_diffusion_rlhf_caption_refinement.data.preprocessing import (
    ImageProcessor,
    TextTokenizer,
)
from hierarchical_diffusion_rlhf_caption_refinement.evaluation.analysis import (
    analyze_results,
    generate_evaluation_report,
)
from hierarchical_diffusion_rlhf_caption_refinement.evaluation.metrics import (
    evaluate_model,
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
        description="Evaluate trained caption refinement model"
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
        help="Type of model to evaluate",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None for all)",
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

        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    elif model_type == "rlhf":
        # Initialize base model
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

        # Initialize reward model
        reward_model = RewardModel(
            vocab_size=config["model"]["vocab_size"],
            hidden_dim=config["model"]["hidden_dim"],
            num_layers=config["model"]["reward_num_layers"],
            num_heads=config["model"]["num_heads"],
            max_length=config["model"]["max_length"],
            dropout=config["model"]["dropout"],
        ).to(device)

        # Initialize RLHF model
        model = RLHFCaptionRefiner(
            base_model=base_model,
            reward_model=reward_model,
            hidden_dim=config["model"]["hidden_dim"],
            clip_weight=config["training"]["clip_weight"],
            preference_weight=config["training"]["preference_weight"],
        ).to(device)

        # Load checkpoint
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.eval()
    logger.info(f"Model loaded successfully")

    return model


def main() -> None:
    """Main evaluation pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(log_level=config["logging"]["level"])

    logger.info("Starting Model Evaluation")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Set random seeds
    set_random_seeds(config["seed"])

    # Get device
    device = get_device(config["device"])

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        config=config,
        device=device,
    )

    # Load test dataset
    logger.info("Loading test dataset")

    num_samples = args.num_samples or config["evaluation"].get("num_samples")

    test_dataset = CaptionDataset(
        split="test",
        max_samples=num_samples,
        tokenizer_name=config["data"]["tokenizer_name"],
        max_length=config["data"]["max_length"],
    )

    test_loader = get_dataloader(
        test_dataset,
        batch_size=config["evaluation"]["batch_size"],
        shuffle=False,
    )

    logger.info(f"Loaded {len(test_dataset)} test samples")

    # Initialize tokenizer and image processor
    tokenizer = TextTokenizer(
        model_name=config["data"]["tokenizer_name"],
        max_length=config["data"]["max_length"],
    )

    image_processor = ImageProcessor(
        embedding_dim=config["model"]["image_embedding_dim"],
    )

    # Run evaluation
    logger.info("=" * 80)
    logger.info("Running Model Evaluation")
    logger.info("=" * 80)

    try:
        metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            tokenizer=tokenizer,
            image_processor=image_processor,
            device=device,
            compute_all_metrics=config["evaluation"]["compute_all_metrics"],
        )

        # Analyze results
        logger.info("=" * 80)
        logger.info("Analyzing Results")
        logger.info("=" * 80)

        analysis = analyze_results(metrics)

        # Determine output directory
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(config["paths"]["results_dir"])

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_path = output_dir / "evaluation_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Saved metrics to {metrics_path}")

        # Save analysis
        analysis_path = output_dir / "evaluation_analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Saved analysis to {analysis_path}")

        # Generate report
        report_path = output_dir / "evaluation_report.txt"
        generate_evaluation_report(metrics, analysis, report_path)

        # Print summary
        logger.info("=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        for metric, value in sorted(metrics.items()):
            logger.info(f"{metric:.<40} {value:.4f}")

        logger.info("")
        logger.info("Key Insights:")
        for i, insight in enumerate(analysis.get("insights", []), 1):
            logger.info(f"{i}. {insight}")

        if "overall_assessment" in analysis:
            logger.info("")
            logger.info(f"Overall Assessment: {analysis['overall_assessment']}")

        # Compare to target metrics
        target_metrics = config.get("target_metrics", {})
        if target_metrics:
            logger.info("")
            logger.info("Target Metric Comparison:")
            for metric, target in target_metrics.items():
                if metric in metrics:
                    actual = metrics[metric]
                    status = "✓" if actual >= target * 0.9 else "✗"
                    logger.info(
                        f"  {status} {metric}: {actual:.4f} (target: {target:.4f})"
                    )

        logger.info("=" * 80)
        logger.info("EVALUATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
