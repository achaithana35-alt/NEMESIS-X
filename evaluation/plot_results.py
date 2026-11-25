"""Publication-quality result visualization.

Generates plots for evaluation results including:
  - ROC curve with AUC annotation
  - Precision-Recall curve
  - Anomaly score distribution
  - Confusion matrix heatmap
  - Temporal anomaly score timeline

Usage:
    python evaluation/plot_results.py --results results/metrics.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)


# ── Style Configuration ───────────────────────────────────

plt.rcParams.update({
    "figure.figsize": (10, 7),
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "primary": "#2196F3",
    "secondary": "#FF5722",
    "success": "#4CAF50",
    "warning": "#FF9800",
    "danger": "#F44336",
    "neutral": "#607D8B",
}


def plot_roc_curve(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    save_path: str = "results/roc_curve.png",
    title: str = "Frame-Level ROC Curve — UCF-Crime Dataset",
) -> None:
    """Generate a publication-quality ROC curve.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_scores: Predicted anomaly scores.
        save_path:   Path to save the plot.
        title:       Plot title.
    """
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 8))

    # ROC curve
    ax.plot(
        fpr,
        tpr,
        color=COLORS["primary"],
        linewidth=2.5,
        label=f"Our Pipeline (AUC = {roc_auc:.4f})",
    )

    # Reference lines for comparison
    ax.plot(
        [0, 1],
        [0, 1],
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label="Random Classifier (AUC = 0.5000)",
    )

    # Shade the AUC area
    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])

    # Mark the optimal operating point (Youden's J)
    j_stat = tpr - fpr
    optimal_idx = np.argmax(j_stat)
    ax.scatter(
        fpr[optimal_idx],
        tpr[optimal_idx],
        marker="o",
        s=100,
        color=COLORS["secondary"],
        zorder=5,
        label=f"Optimal Threshold ({thresholds[optimal_idx]:.3f})",
    )

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"ROC curve saved to {save_path}")


def plot_precision_recall(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    save_path: str = "results/pr_curve.png",
) -> None:
    """Generate a Precision-Recall curve.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_scores: Predicted anomaly scores.
        save_path:   Path to save the plot.
    """
    precision, recall, thresholds = precision_recall_curve(gt_labels, pred_scores)
    ap = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(
        recall,
        precision,
        color=COLORS["success"],
        linewidth=2.5,
        label=f"Our Pipeline (AP = {ap:.4f})",
    )
    ax.fill_between(recall, precision, alpha=0.15, color=COLORS["success"])

    # Baseline (random)
    anomaly_ratio = gt_labels.mean()
    ax.axhline(
        y=anomaly_ratio,
        color=COLORS["neutral"],
        linestyle="--",
        linewidth=1,
        alpha=0.7,
        label=f"Random (P = {anomaly_ratio:.3f})",
    )

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Curve — UCF-Crime Dataset", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"PR curve saved to {save_path}")


def plot_score_distribution(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    save_path: str = "results/score_distribution.png",
) -> None:
    """Plot the distribution of anomaly scores for normal vs anomalous frames.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_scores: Predicted anomaly scores.
        save_path:   Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    normal_scores = pred_scores[gt_labels == 0]
    anomaly_scores = pred_scores[gt_labels == 1]

    ax.hist(
        normal_scores,
        bins=50,
        alpha=0.6,
        color=COLORS["success"],
        label=f"Normal (n={len(normal_scores)})",
        density=True,
    )
    ax.hist(
        anomaly_scores,
        bins=50,
        alpha=0.6,
        color=COLORS["danger"],
        label=f"Anomalous (n={len(anomaly_scores)})",
        density=True,
    )

    ax.set_xlabel("Anomaly Score", fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.set_title("Anomaly Score Distribution", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Score distribution saved to {save_path}")


def plot_confusion_matrix(
    gt_labels: np.ndarray,
    pred_binary: np.ndarray,
    save_path: str = "results/confusion_matrix.png",
) -> None:
    """Plot a confusion matrix heatmap.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_binary: Binary predictions.
        save_path:   Path to save the plot.
    """
    cm = confusion_matrix(gt_labels, pred_binary, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Anomalous"],
        yticklabels=["Normal", "Anomalous"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("Actual", fontsize=14)
    ax.set_title("Confusion Matrix", fontsize=16, fontweight="bold")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_temporal_scores(
    pred_scores: np.ndarray,
    gt_labels: np.ndarray | None = None,
    threshold: float = 0.65,
    save_path: str = "results/temporal_scores.png",
    fps: float = 30.0,
) -> None:
    """Plot anomaly scores over time with ground-truth overlay.

    Args:
        pred_scores: Predicted anomaly scores per frame.
        gt_labels:   Optional ground truth labels.
        threshold:   Decision threshold.
        save_path:   Path to save the plot.
        fps:         Video FPS for time axis.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    time_axis = np.arange(len(pred_scores)) / fps

    ax.plot(
        time_axis,
        pred_scores,
        color=COLORS["primary"],
        linewidth=1,
        alpha=0.8,
        label="Anomaly Score",
    )

    # Threshold line
    ax.axhline(
        y=threshold,
        color=COLORS["danger"],
        linestyle="--",
        linewidth=1.5,
        alpha=0.7,
        label=f"Threshold ({threshold})",
    )

    # Ground truth shading
    if gt_labels is not None:
        gt_time = np.arange(len(gt_labels)) / fps
        ax.fill_between(
            gt_time[:len(pred_scores)],
            0,
            gt_labels[:len(pred_scores)],
            alpha=0.15,
            color=COLORS["danger"],
            label="Ground Truth Anomaly",
        )

    ax.set_xlabel("Time (seconds)", fontsize=14)
    ax.set_ylabel("Anomaly Score", fontsize=14)
    ax.set_title("Temporal Anomaly Detection", fontsize=16, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim([-0.05, 1.05])

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Temporal scores saved to {save_path}")


def generate_all_plots(results_path: str = "results/metrics.json") -> None:
    """Generate all evaluation plots from saved results.

    Args:
        results_path: Path to the metrics JSON file.
    """
    results_path = Path(results_path)
    if not results_path.exists():
        logger.warning(f"Results file not found: {results_path}")
        return

    with open(results_path, "r") as f:
        data = json.load(f)

    output_dir = results_path.parent
    logger.info(f"Generating plots in {output_dir}")

    # If we have raw scores, generate plots from them
    if "gt_labels" in data and "pred_scores" in data:
        gt = np.array(data["gt_labels"])
        pred = np.array(data["pred_scores"])

        plot_roc_curve(gt, pred, str(output_dir / "roc_curve.png"))
        plot_precision_recall(gt, pred, str(output_dir / "pr_curve.png"))
        plot_score_distribution(gt, pred, str(output_dir / "score_distribution.png"))
        plot_temporal_scores(pred, gt, save_path=str(output_dir / "temporal_scores.png"))

        threshold = data.get("metrics", {}).get("optimal_threshold", 0.5)
        pred_binary = (pred >= threshold).astype(int)
        plot_confusion_matrix(gt, pred_binary, str(output_dir / "confusion_matrix.png"))

    logger.success("All plots generated")


def main():
    parser = argparse.ArgumentParser(description="Generate Evaluation Plots")
    parser.add_argument(
        "--results",
        type=str,
        default="results/metrics.json",
        help="Path to results JSON",
    )
    args = parser.parse_args()
    generate_all_plots(args.results)


if __name__ == "__main__":
    main()
