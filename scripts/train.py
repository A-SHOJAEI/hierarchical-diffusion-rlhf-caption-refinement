#!/usr/bin/env python
"""Full training pipeline for hierarchical diffusion RLHF caption refinement."""

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
    PreferenceDataset,
    get_dataloader,
)
from hierarchical_diffusion_rlhf_caption_refinement.models.model import RewardModel
from hierarchical_diffusion_rlhf_caption_refinement.training.trainer import (
    HierarchicalRLHFTrainer,
    train_reward_model,
)
from hierarchical_diffusion_rlhf_caption_refinement.utils.config import (
    get_device,
    load_config,
    set_random_seeds,
    setup_logging,
    count_parameters,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train hierarchical diffusion RLHF caption refinement model"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "base", "reward", "rlhf"],
        default="all",
        help="Training stage to run",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow tracking",
    )

    return parser.parse_args()


def setup_mlflow(config: dict, args: argparse.Namespace) -> any:
    """Setup MLflow tracking.

    Args:
        config: Configuration dictionary.
        args: Command line arguments.

    Returns:
        MLflow run object or None.
    """
    mlflow_run = None

    if not args.no_mlflow and config.get("logging", {}).get("mlflow_tracking", False):
        try:
            import mlflow

            # Use a local file-based tracking URI to avoid requiring a running
            # MLflow server on localhost (which causes ConnectionRefusedError).
            tracking_uri = config.get("logging", {}).get(
                "mlflow_tracking_uri", "file:./mlruns"
            )
            mlflow.set_tracking_uri(tracking_uri)

            # Set experiment
            experiment_name = config["logging"].get(
                "mlflow_experiment_name",
                "hierarchical_rlhf_caption",
            )
            mlflow.set_experiment(experiment_name)

            # Start run
            mlflow_run = mlflow.start_run()

            # Log parameters
            mlflow.log_params({
                "model_hidden_dim": config["model"]["hidden_dim"],
                "model_num_layers": config["model"]["num_layers"],
                "base_learning_rate": config["training"]["base_learning_rate"],
                "rlhf_learning_rate": config["training"]["rlhf_learning_rate"],
                "clip_weight": config["training"]["clip_weight"],
                "preference_weight": config["training"]["preference_weight"],
                "seed": config["seed"],
            })

            logger.info(f"MLflow tracking enabled: {experiment_name}")
        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
        except Exception as e:
            logger.warning(f"Failed to setup MLflow: {e}. Continuing without tracking.")

    return mlflow_run


def train_stage_base(
    trainer: HierarchicalRLHFTrainer,
    train_loader: any,
    val_loader: any,
    config: dict,
) -> None:
    """Train base diffusion caption model.

    Args:
        trainer: HierarchicalRLHFTrainer instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        config: Configuration dictionary.
    """
    logger.info("=" * 80)
    logger.info("STAGE 1: Pre-training Base Diffusion Caption Model")
    logger.info("=" * 80)

    num_params = count_parameters(trainer.base_model)
    logger.info(f"Base model parameters: {num_params:,}")

    num_epochs = config["training"]["base_pretrain_epochs"]

    try:
        trainer.train_base_model(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
        )

        logger.info("Base model pre-training completed successfully")
    except Exception as e:
        logger.error(f"Base model training failed: {e}")
        raise


def train_stage_reward(
    trainer: HierarchicalRLHFTrainer,
    train_loader: any,
    val_loader: any,
    config: dict,
    device: torch.device,
) -> None:
    """Train reward model on preference data.

    Args:
        trainer: HierarchicalRLHFTrainer instance.
        train_loader: Training dataloader with preferences.
        val_loader: Validation dataloader.
        config: Configuration dictionary.
        device: Device to train on.
    """
    logger.info("=" * 80)
    logger.info("STAGE 2: Training Reward Model on Preferences")
    logger.info("=" * 80)

    num_params = count_parameters(trainer.reward_model)
    logger.info(f"Reward model parameters: {num_params:,}")

    save_dir = Path(config["paths"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = config["training"]["reward_epochs"]
    learning_rate = config["training"]["reward_learning_rate"]
    patience = config["training"]["patience"]

    try:
        trained_reward_model = train_reward_model(
            reward_model=trainer.reward_model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            device=device,
            save_dir=save_dir,
            patience=patience,
        )

        trainer.reward_model = trained_reward_model
        trainer.rlhf_model.reward_model = trained_reward_model

        logger.info("Reward model training completed successfully")
    except Exception as e:
        logger.error(f"Reward model training failed: {e}")
        raise


def train_stage_rlhf(
    trainer: HierarchicalRLHFTrainer,
    train_loader: any,
    val_loader: any,
    config: dict,
) -> None:
    """Fine-tune with RLHF using hierarchical rewards.

    Args:
        trainer: HierarchicalRLHFTrainer instance.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        config: Configuration dictionary.
    """
    logger.info("=" * 80)
    logger.info("STAGE 3: RLHF Fine-tuning with Hierarchical Rewards")
    logger.info("=" * 80)

    num_params = count_parameters(trainer.rlhf_model)
    logger.info(f"RLHF model parameters: {num_params:,}")

    logger.info(
        f"Hierarchical reward shaping - "
        f"CLIP weight: {config['training']['clip_weight']}, "
        f"Preference weight: {config['training']['preference_weight']}"
    )

    num_epochs = config["training"]["rlhf_epochs"]

    try:
        trainer.train_with_rlhf(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
        )

        logger.info("RLHF fine-tuning completed successfully")
    except Exception as e:
        logger.error(f"RLHF training failed: {e}")
        raise


def save_final_results(trainer: HierarchicalRLHFTrainer, config: dict) -> None:
    """Save final training results and model checkpoints.

    Args:
        trainer: HierarchicalRLHFTrainer instance.
        config: Configuration dictionary.
    """
    results_dir = Path(config["paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save final model
    save_dir = Path(config["paths"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    final_model_path = save_dir / "final_model.pt"
    torch.save(trainer.rlhf_model.state_dict(), final_model_path)
    logger.info(f"Saved final model to {final_model_path}")

    # Save training summary
    summary = {
        "config": config,
        "model_parameters": count_parameters(trainer.rlhf_model),
        "best_val_loss": float(trainer.best_val_loss),
    }

    summary_path = results_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Saved training summary to {summary_path}")


def main() -> None:
    """Main training pipeline."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Setup logging
    setup_logging(
        log_level=config["logging"]["level"],
        log_file=config["logging"].get("log_file") if config["logging"]["log_to_file"] else None,
    )

    logger.info("Starting Hierarchical Diffusion RLHF Caption Refinement Training")
    logger.info(f"Configuration: {args.config}")

    # Set random seeds
    set_random_seeds(config["seed"])

    # Get device
    device = get_device(config["device"])

    # Setup MLflow
    mlflow_run = setup_mlflow(config, args)

    try:
        # Initialize trainer
        save_dir = Path(config["paths"]["save_dir"])
        save_dir.mkdir(parents=True, exist_ok=True)

        trainer = HierarchicalRLHFTrainer(
            config=config,
            device=device,
            save_dir=save_dir,
        )

        # Load datasets based on training stage
        if args.stage in ["all", "base", "rlhf"]:
            logger.info("Loading caption datasets")

            train_dataset = CaptionDataset(
                split="train",
                max_samples=config["data"]["train_max_samples"],
                tokenizer_name=config["data"]["tokenizer_name"],
                max_length=config["data"]["max_length"],
            )

            val_dataset = CaptionDataset(
                split="validation",
                max_samples=config["data"]["val_max_samples"],
                tokenizer_name=config["data"]["tokenizer_name"],
                max_length=config["data"]["max_length"],
            )

            caption_train_loader = get_dataloader(
                train_dataset,
                batch_size=config["training"]["base_batch_size"],
                shuffle=True,
            )

            caption_val_loader = get_dataloader(
                val_dataset,
                batch_size=config["evaluation"]["batch_size"],
                shuffle=False,
            )

        # Train base model
        if args.stage in ["all", "base"]:
            train_stage_base(
                trainer=trainer,
                train_loader=caption_train_loader,
                val_loader=caption_val_loader,
                config=config,
            )

        # Train reward model
        if args.stage in ["all", "reward"]:
            logger.info("Loading preference datasets")

            pref_train_dataset = PreferenceDataset(
                split="train",
                max_samples=config["data"]["train_max_samples"] // 2,
                tokenizer_name=config["data"]["tokenizer_name"],
                max_length=config["data"]["max_length"],
            )

            pref_val_dataset = PreferenceDataset(
                split="validation",
                max_samples=config["data"]["val_max_samples"] // 2,
                tokenizer_name=config["data"]["tokenizer_name"],
                max_length=config["data"]["max_length"],
            )

            pref_train_loader = get_dataloader(
                pref_train_dataset,
                batch_size=config["training"]["reward_batch_size"],
                shuffle=True,
            )

            pref_val_loader = get_dataloader(
                pref_val_dataset,
                batch_size=config["evaluation"]["batch_size"],
                shuffle=False,
            )

            train_stage_reward(
                trainer=trainer,
                train_loader=pref_train_loader,
                val_loader=pref_val_loader,
                config=config,
                device=device,
            )

        # Train with RLHF
        if args.stage in ["all", "rlhf"]:
            # Use smaller batch size for RLHF
            rlhf_train_loader = get_dataloader(
                train_dataset,
                batch_size=config["training"]["rlhf_batch_size"],
                shuffle=True,
            )

            rlhf_val_loader = get_dataloader(
                val_dataset,
                batch_size=config["evaluation"]["batch_size"],
                shuffle=False,
            )

            train_stage_rlhf(
                trainer=trainer,
                train_loader=rlhf_train_loader,
                val_loader=rlhf_val_loader,
                config=config,
            )

        # Save final results
        save_final_results(trainer, config)

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Models saved to: {save_dir}")
        logger.info(f"Results saved to: {config['paths']['results_dir']}")

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    finally:
        # End MLflow run
        if mlflow_run is not None:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    main()
