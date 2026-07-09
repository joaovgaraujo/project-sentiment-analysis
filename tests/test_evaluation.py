"""Unit tests for the evaluation metrics module."""

from src.evaluation.metrics import evaluate_model


def test_evaluate_model_perfect_predictions() -> None:
    """evaluate_model should return 1.0 for all metrics with perfect predictions."""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 1, 0, 0]
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0


def test_evaluate_model_all_wrong() -> None:
    """evaluate_model should return 0.0 accuracy when all predictions are wrong."""
    y_true = [1, 1, 0, 0]
    y_pred = [0, 0, 1, 1]
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["accuracy"] == 0.0


def test_evaluate_model_mixed() -> None:
    """evaluate_model should compute correct values for mixed predictions."""
    y_true = [1, 1, 0, 0]
    y_pred = [1, 0, 0, 1]
    metrics = evaluate_model(y_true, y_pred)
    assert metrics["accuracy"] == 0.5
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_evaluate_model_returns_expected_keys() -> None:
    """evaluate_model should return dict with accuracy, f1_score, precision, recall."""
    metrics = evaluate_model([1, 0], [1, 0])
    assert set(metrics.keys()) == {"accuracy", "f1_score", "precision", "recall"}


def test_evaluate_model_division_by_zero() -> None:
    """evaluate_model should return 0.0 when division by zero occurs."""
    # All predicted as 0 -> precision denominator is 0
    metrics = evaluate_model([1, 1], [0, 0])
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1_score"] == 0.0
