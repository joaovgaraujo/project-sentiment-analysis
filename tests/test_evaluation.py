"""Tests for the evaluation metrics module."""

import unittest

from src.evaluation.metrics import evaluate_model


class TestEvaluateModel(unittest.TestCase):
    """Tests for evaluate_model."""

    def test_perfect_predictions(self) -> None:
        metrics = evaluate_model([1, 1, 0, 0], [1, 1, 0, 0])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1_score"], 1.0)

    def test_all_wrong(self) -> None:
        metrics = evaluate_model([1, 1, 0, 0], [0, 0, 1, 1])
        self.assertEqual(metrics["accuracy"], 0.0)

    def test_mixed(self) -> None:
        metrics = evaluate_model([1, 1, 0, 0], [1, 0, 0, 1])
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertTrue(0.0 <= metrics["f1_score"] <= 1.0)
        self.assertTrue(0.0 <= metrics["precision"] <= 1.0)
        self.assertTrue(0.0 <= metrics["recall"] <= 1.0)

    def test_returns_expected_keys(self) -> None:
        metrics = evaluate_model([1, 0], [1, 0])
        self.assertEqual(
            set(metrics.keys()),
            {"accuracy", "f1_score", "precision", "recall"},
        )

    def test_division_by_zero(self) -> None:
        # All predicted as 0 -> precision and recall denominators are 0.
        metrics = evaluate_model([1, 1], [0, 0])
        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
