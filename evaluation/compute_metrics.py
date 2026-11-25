"""Metrics computation for anomaly detection evaluation.

Computes standard video anomaly detection metrics:
  - Frame-level ROC-AUC (primary metric)
  - Precision, Recall, F1-Score
  - Average Precision (AP)
  - False Alarm Rate
  - Equal Error Rate (EER)

Usage:
    python evaluation/compute_metrics.py --results results/metrics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_roc_auc(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the frame-level ROC-AUC score.

    Args:
        gt_labels:   Binary ground truth labels (0 = normal, 1 = anomalous).
        pred_scores: Predicted anomaly scores in [0, 1].

    Returns:
        Tuple of (auc_score, fpr, tpr, thresholds).
    """
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    auc_score = auc(fpr, tpr)
    return auc_score, fpr, tpr, thresholds


def compute_precision_recall(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Average Precision and PR curve data.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_scores: Predicted anomaly scores.

    Returns:
        Tuple of (ap_score, precision, recall, thresholds).
    """
    precision, recall, thresholds = precision_recall_curve(gt_labels, pred_scores)
    ap = average_precision_score(gt_labels, pred_scores)
    return ap, precision, recall, thresholds


def compute_eer(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
) -> tuple[float, float]:
    """Compute Equal Error Rate (EER).

    The EER is the point where the false positive rate equals the
    false negative rate. Lower EER = better discrimination.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_scores: Predicted anomaly scores.

    Returns:
        Tuple of (eer, eer_threshold).
    """
    fpr, tpr, thresholds = roc_curve(gt_labels, pred_scores)
    fnr = 1 - tpr

    # Find the point where FPR ≈ FNR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2)
    eer_threshold = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5

    return eer, eer_threshold


def compute_false_alarm_rate(
    gt_labels: np.ndarray,
    pred_binary: np.ndarray,
) -> float:
    """Compute False Alarm Rate (FAR).

    FAR = FP / (FP + TN) — fraction of normal frames incorrectly flagged.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_binary: Binary predictions.

    Returns:
        False alarm rate.
    """
    cm = confusion_matrix(gt_labels, pred_binary, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0


def compute_all_metrics(
    gt_labels: np.ndarray,
    pred_scores: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all evaluation metrics.

    Args:
        gt_labels:   Binary ground truth labels.
        pred_scores: Predicted anomaly scores in [0, 1].
        threshold:   Decision threshold for binary classification.

    Returns:
        Dictionary with all computed metrics.
    """
    # Ensure numpy arrays
    gt_labels = np.asarray(gt_labels, dtype=np.int32)
    pred_scores = np.asarray(pred_scores, dtype=np.float32)

    # Binary predictions
    pred_binary = (pred_scores >= threshold).astype(np.int32)

    # ROC-AUC
    roc_auc_val, fpr, tpr, roc_thresholds = compute_roc_auc(gt_labels, pred_scores)

    # Precision-Recall
    ap, pr_precision, pr_recall, pr_thresholds = compute_precision_recall(
        gt_labels, pred_scores
    )

    # Classification metrics at the given threshold
    precision = float(precision_score(gt_labels, pred_binary, zero_division=0))
    recall = float(recall_score(gt_labels, pred_binary, zero_division=0))
    f1 = float(f1_score(gt_labels, pred_binary, zero_division=0))

    # EER
    eer, eer_threshold = compute_eer(gt_labels, pred_scores)

    # False alarm rate
    far = compute_false_alarm_rate(gt_labels, pred_binary)

    # Optimal threshold (Youden's J statistic)
    j_statistic = tpr - fpr
    optimal_idx = np.argmax(j_statistic)
    optimal_threshold = float(roc_thresholds[optimal_idx]) if optimal_idx < len(roc_thresholds) else threshold

    metrics = {
        "roc_auc": round(float(roc_auc_val), 4),
        "average_precision": round(float(ap), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "false_alarm_rate": round(far, 4),
        "optimal_threshold": round(optimal_threshold, 4),
        "decision_threshold": threshold,
        "total_frames": int(len(gt_labels)),
        "anomaly_frames": int(gt_labels.sum()),
        "normal_frames": int((1 - gt_labels).sum()),
        "anomaly_ratio": round(float(gt_labels.mean()), 4),
    }

    logger.info("Evaluation Metrics:")
    logger.info(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    logger.info(f"  Average Precision: {metrics['average_precision']:.4f}")
    logger.info(f"  Precision:         {metrics['precision']:.4f}")
    logger.info(f"  Recall:            {metrics['recall']:.4f}")
    logger.info(f"  F1-Score:          {metrics['f1_score']:.4f}")
    logger.info(f"  EER:               {metrics['eer']:.4f}")
    logger.info(f"  False Alarm Rate:  {metrics['false_alarm_rate']:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compute Evaluation Metrics")
    parser.add_argument(
        "--results",
        type=str,
        default="results/metrics.json",
        help="Path to results JSON file",
    )

    args = parser.parse_args()

    results_path = Path(args.results)
    if results_path.exists():
        with open(results_path, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded results from {results_path}")
        logger.info(json.dumps(data.get("metrics", data), indent=2))
    else:
        logger.info("No results file found. Run evaluate_ucf_crime.py first.")


if __name__ == "__main__":
    main()
