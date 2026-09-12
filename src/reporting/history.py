"""Versioned metrics history (new module, no changes to existing code)."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np


def _f1_for_class(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    tp = int(np.sum((y_pred == label) & (y_true == label)))
    fp = int(np.sum((y_pred == label) & (y_true != label)))
    fn = int(np.sum((y_pred != label) & (y_true == label)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if (precision + recall) <= 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Mean of per-class F1 (class 0 and class 1), NumPy only."""
    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)
    return float((_f1_for_class(yt, yp, 0) + _f1_for_class(yt, yp, 1)) / 2)


def save_history(
    metrics: dict[str, float],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    out_dir: str = "results/metrics",
    run_at: datetime | str | None = None,
) -> Path:
    """Persist totals + metrics as metrics-YYYYMMDDHHMMSS.json (stdlib only)."""
    if run_at is None:
        run_at = datetime.now()
    if isinstance(run_at, str):
        run_at = datetime.fromisoformat(run_at)
    stamp = run_at.strftime("%Y%m%d%H%M%S")

    yt = np.asarray(list(y_true), dtype=int)
    yp = np.asarray(list(y_pred), dtype=int)

    payload = {
        "timestamp": run_at.isoformat(timespec="seconds"),
        "totals": {
            "n_test": int(len(yt)),
            "n_true_pos": int(np.sum(yt == 1)),
            "n_true_neg": int(np.sum(yt == 0)),
            "n_pred_pos": int(np.sum(yp == 1)),
            "n_pred_neg": int(np.sum(yp == 0)),
        },
        "metrics": {
            "accuracy": float(metrics.get("accuracy", 0.0)),
            "precision": float(metrics.get("precision", 0.0)),
            "recall": float(metrics.get("recall", 0.0)),
            "f1_score": float(metrics.get("f1_score", 0.0)),
            "macro_f1": compute_macro_f1(yt, yp),
        },
    }

    out_path = Path(out_dir) / f"metrics-{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
