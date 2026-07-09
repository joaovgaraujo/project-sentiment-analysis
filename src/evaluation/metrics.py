"""Model evaluation: metrics computation and reporting."""

from typing import Sequence

import numpy as np


def evaluate_model(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Compute evaluation metrics for the classifier using NumPy.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing accuracy, f1_score, precision, and recall.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    true_positives = int(np.sum((y_pred_arr == 1) & (y_true_arr == 1)))
    true_negatives = int(np.sum((y_pred_arr == 0) & (y_true_arr == 0)))
    false_positives = int(np.sum((y_pred_arr == 1) & (y_true_arr == 0)))
    false_negatives = int(np.sum((y_pred_arr == 0) & (y_true_arr == 1)))

    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0.0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "accuracy": accuracy,
        "f1_score": f1_score,
        "precision": precision,
        "recall": recall,
    }


def print_report(metrics: dict[str, float]) -> None:
    """Display evaluation metrics in a human-readable format.

    Args:
        metrics: Dictionary returned by evaluate_model.
    """
    print("\n" + "=" * 40)
    print("  Classification Metrics Report")
    print("=" * 40)
    for name, value in metrics.items():
        print(f"  {name:<12}: {value:.4f}")
    print("=" * 40 + "\n")
