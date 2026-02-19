"""Results analysis and visualization utilities."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def analyze_results(
    results: Dict[str, float],
    save_path: Optional[Path] = None,
) -> Dict[str, any]:
    """Analyze evaluation results and generate insights.

    Args:
        results: Dictionary of metric names to values.
        save_path: Optional path to save analysis.

    Returns:
        Dictionary of analysis insights.
    """
    analysis = {
        "metrics": results,
        "insights": [],
    }

    # CLIP score analysis
    if "clip_score" in results:
        clip_score = results["clip_score"]
        if clip_score >= 0.30:
            analysis["insights"].append(
                "Strong image-text alignment (CLIP score >= 0.30)"
            )
        elif clip_score >= 0.25:
            analysis["insights"].append(
                "Good image-text alignment (CLIP score >= 0.25)"
            )
        else:
            analysis["insights"].append(
                "Image-text alignment needs improvement"
            )

    # CIDEr score analysis
    if "cider" in results:
        cider = results["cider"]
        if cider >= 1.0:
            analysis["insights"].append(
                "High caption quality (CIDEr >= 1.0)"
            )
        elif cider >= 0.7:
            analysis["insights"].append(
                "Moderate caption quality (CIDEr >= 0.7)"
            )
        else:
            analysis["insights"].append(
                "Caption quality needs improvement"
            )

    # Specificity analysis
    if "specificity_score" in results:
        spec = results["specificity_score"]
        if spec >= 0.7:
            analysis["insights"].append(
                "Captions are specific and detailed"
            )
        elif spec >= 0.5:
            analysis["insights"].append(
                "Captions have moderate specificity"
            )
        else:
            analysis["insights"].append(
                "Captions are too generic"
            )

    # Human preference analysis
    if "human_preference_win_rate" in results:
        win_rate = results["human_preference_win_rate"]
        if win_rate >= 0.65:
            analysis["insights"].append(
                f"Strong human preference (win rate: {win_rate:.2%})"
            )
        elif win_rate >= 0.55:
            analysis["insights"].append(
                f"Moderate human preference (win rate: {win_rate:.2%})"
            )
        else:
            analysis["insights"].append(
                "Human preference below baseline"
            )

    # Caption length analysis
    if "avg_caption_length" in results:
        avg_len = results["avg_caption_length"]
        std_len = results.get("std_caption_length", 0.0)

        if avg_len < 5:
            analysis["insights"].append(
                "Captions are too short (avg < 5 words)"
            )
        elif avg_len > 20:
            analysis["insights"].append(
                "Captions are very long (avg > 20 words)"
            )
        else:
            analysis["insights"].append(
                f"Caption length is appropriate (avg: {avg_len:.1f} words)"
            )

        if std_len > 5:
            analysis["insights"].append(
                "High variability in caption length"
            )

    # Overall assessment
    target_metrics = {
        "clip_score": 0.32,
        "cider": 1.2,
        "human_preference_win_rate": 0.68,
        "specificity_score": 0.75,
    }

    targets_met = 0
    total_targets = 0

    for metric, target in target_metrics.items():
        if metric in results:
            total_targets += 1
            if results[metric] >= target * 0.9:  # Within 90% of target
                targets_met += 1

    if total_targets > 0:
        target_rate = targets_met / total_targets
        analysis["target_achievement_rate"] = target_rate
        if target_rate >= 0.75:
            analysis["overall_assessment"] = "Excellent performance"
        elif target_rate >= 0.5:
            analysis["overall_assessment"] = "Good performance"
        else:
            analysis["overall_assessment"] = "Needs improvement"

    # Save analysis if path provided
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(analysis, f, indent=2)
        logger.info(f"Saved analysis to {save_path}")

    return analysis


def plot_training_curves(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    train_rewards: Optional[List[float]] = None,
    val_rewards: Optional[List[float]] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot training curves for loss and rewards.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: Optional list of validation losses.
        train_rewards: Optional list of training rewards.
        val_rewards: Optional list of validation rewards.
        save_path: Optional path to save plot.
    """
    num_plots = 1 + (1 if train_rewards else 0)
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o")

    if val_losses:
        axes[0].plot(epochs, val_losses, label="Val Loss", marker="s")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot rewards if available
    if train_rewards:
        axes[1].plot(epochs, train_rewards, label="Train Reward", marker="o")

        if val_rewards:
            axes[1].plot(epochs, val_rewards, label="Val Reward", marker="s")

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Reward")
        axes[1].set_title("Training and Validation Reward")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved training curves to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_reward_distribution(
    dense_rewards: List[float],
    sparse_rewards: List[float],
    combined_rewards: List[float],
    save_path: Optional[Path] = None,
) -> None:
    """Plot distribution of different reward types.

    Args:
        dense_rewards: List of dense (CLIP) reward values.
        sparse_rewards: List of sparse (preference) reward values.
        combined_rewards: List of combined reward values.
        save_path: Optional path to save plot.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Dense rewards
    axes[0].hist(dense_rewards, bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0].set_xlabel("Dense Reward (CLIP)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title(f"Dense Rewards\nMean: {np.mean(dense_rewards):.3f}")
    axes[0].grid(True, alpha=0.3)

    # Sparse rewards
    axes[1].hist(sparse_rewards, bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[1].set_xlabel("Sparse Reward (Preference)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Sparse Rewards\nMean: {np.mean(sparse_rewards):.3f}")
    axes[1].grid(True, alpha=0.3)

    # Combined rewards
    axes[2].hist(combined_rewards, bins=30, alpha=0.7, color="red", edgecolor="black")
    axes[2].set_xlabel("Combined Reward (Hierarchical)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title(f"Combined Rewards\nMean: {np.mean(combined_rewards):.3f}")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved reward distribution to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_metric_comparison(
    baseline_metrics: Dict[str, float],
    proposed_metrics: Dict[str, float],
    save_path: Optional[Path] = None,
) -> None:
    """Plot comparison between baseline and proposed method.

    Args:
        baseline_metrics: Metrics from baseline method.
        proposed_metrics: Metrics from proposed method.
        save_path: Optional path to save plot.
    """
    # Get common metrics
    common_metrics = set(baseline_metrics.keys()) & set(proposed_metrics.keys())
    metric_names = sorted(common_metrics)

    if not metric_names:
        logger.warning("No common metrics to compare")
        return

    baseline_values = [baseline_metrics[m] for m in metric_names]
    proposed_values = [proposed_metrics[m] for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(x - width/2, baseline_values, width, label="Baseline", alpha=0.8)
    ax.bar(x + width/2, proposed_values, width, label="Proposed (Hierarchical RLHF)", alpha=0.8)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Value")
    ax.set_title("Baseline vs Proposed Method Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved metric comparison to {save_path}")
    else:
        plt.show()

    plt.close()


def generate_evaluation_report(
    metrics: Dict[str, float],
    analysis: Dict[str, any],
    save_path: Path,
) -> None:
    """Generate a comprehensive evaluation report.

    Args:
        metrics: Evaluation metrics.
        analysis: Analysis results.
        save_path: Path to save report.
    """
    report_lines = [
        "=" * 80,
        "HIERARCHICAL DIFFUSION RLHF CAPTION REFINEMENT - EVALUATION REPORT",
        "=" * 80,
        "",
        "METRICS:",
        "-" * 80,
    ]

    # Add metrics
    for metric, value in sorted(metrics.items()):
        report_lines.append(f"{metric:.<40} {value:.4f}")

    report_lines.extend([
        "",
        "INSIGHTS:",
        "-" * 80,
    ])

    # Add insights
    for i, insight in enumerate(analysis.get("insights", []), 1):
        report_lines.append(f"{i}. {insight}")

    # Add overall assessment
    if "overall_assessment" in analysis:
        report_lines.extend([
            "",
            "OVERALL ASSESSMENT:",
            "-" * 80,
            analysis["overall_assessment"],
        ])

    if "target_achievement_rate" in analysis:
        rate = analysis["target_achievement_rate"]
        report_lines.append(f"Target Achievement Rate: {rate:.1%}")

    report_lines.extend([
        "",
        "=" * 80,
    ])

    # Save report
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        f.write("\n".join(report_lines))

    logger.info(f"Generated evaluation report at {save_path}")

    # Also print to console
    print("\n".join(report_lines))
