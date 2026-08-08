"""Property tests: invariants that should hold across many inputs.

Each case runs many inputs through subTest so a single failure reports the
exact input that broke the property.
"""

import unittest

import numpy as np
import pandas as pd

from src.data.loader import validate_columns
from src.evaluation.metrics import evaluate_model
from src.models.model import predict, train_model
from src.preprocessing.transform import (
    build_vocabulary,
    clean_text,
    normalize_label,
    texts_to_matrix,
)
from src.training.train import split_dataset


class TestValidateColumnsProperty(unittest.TestCase):
    """validate_columns raises exactly when a required column is missing."""

    CASES = [
        (["a", "b", "c"], ["a", "b"], False),
        (["a", "b"], ["a", "b", "c"], True),
        (["x", "y"], ["a"], True),
        (["reviews.text", "reviews.rating"], ["reviews.text", "reviews.rating"], False),
        (["reviews.text"], ["reviews.text", "reviews.rating"], True),
        ([], ["a"], True),
        (["a", "b", "c"], [], False),
    ]

    def test_raises_iff_missing(self) -> None:
        for columns, required, should_raise in self.CASES:
            with self.subTest(columns=columns, required=required):
                df = (
                    pd.DataFrame({col: [1] for col in columns})
                    if columns
                    else pd.DataFrame()
                )
                if should_raise:
                    with self.assertRaises(ValueError):
                        validate_columns(df, required)
                else:
                    validate_columns(df, required)


class TestCleanTextProperties(unittest.TestCase):
    """clean_text is idempotent and its output is normalized."""

    SAMPLE_TEXTS = [
        "Hello, World!",
        "  spaces   here  ",
        "UPPERCASE TEXT!!!",
        "already clean",
        "MiXeD CaSe 123",
        "no-punctuation-here",
        "tabs\there\tand\tthere",
        "",
        "a",
        "Hello!!! @#$% World???",
        "price: $100.00!!",
        "don't stop!",
        "123 456 789",
        "  \t\n  ",
        "café résumé naïve",
    ]

    def test_idempotency(self) -> None:
        for text in self.SAMPLE_TEXTS:
            with self.subTest(text=text):
                once = clean_text(text)
                self.assertEqual(clean_text(once), once)

    def test_output_format(self) -> None:
        for text in self.SAMPLE_TEXTS:
            with self.subTest(text=text):
                result = clean_text(text)
                self.assertEqual(result, result.lower())
                self.assertEqual(result, result.strip())
                self.assertNotIn("  ", result)
                for ch in result:
                    self.assertTrue(ch.isalnum() or ch == " ")


class TestNormalizeLabelProperties(unittest.TestCase):
    """normalize_label maps ratings deterministically to labels."""

    CASES = [(1, 0), (2, 0), (3, None), (4, 1), (5, 1)]

    def test_correctness(self) -> None:
        for rating, expected in self.CASES:
            with self.subTest(rating=rating):
                self.assertEqual(normalize_label(rating), expected)

    def test_determinism(self) -> None:
        for rating in [1, 2, 3, 4, 5]:
            with self.subTest(rating=rating):
                self.assertEqual(normalize_label(rating), normalize_label(rating))


class TestVocabMatrixProperties(unittest.TestCase):
    """Vocabulary and word-count matrix invariants."""

    SAMPLE_TEXT_LISTS = [
        ["hello world", "hello"],
        ["the cat sat", "the dog sat", "the cat"],
        ["a b c d e", "a a a b b"],
        ["single"],
        ["word word word", "other other"],
    ]

    def test_vocabulary_completeness_and_contiguity(self) -> None:
        for texts in self.SAMPLE_TEXT_LISTS:
            with self.subTest(texts=texts):
                vocab = build_vocabulary(pd.Series(texts))
                all_words = set()
                for text in texts:
                    all_words.update(text.split())
                self.assertEqual(set(vocab.keys()), all_words)
                self.assertEqual(sorted(vocab.values()), list(range(len(vocab))))

    def test_matrix_shape(self) -> None:
        for texts in self.SAMPLE_TEXT_LISTS:
            with self.subTest(texts=texts):
                series = pd.Series(texts)
                vocab = build_vocabulary(series)
                matrix = texts_to_matrix(series, vocab)
                self.assertEqual(matrix.shape, (len(texts), len(vocab)))

    def test_matrix_non_negativity(self) -> None:
        for texts in self.SAMPLE_TEXT_LISTS:
            with self.subTest(texts=texts):
                series = pd.Series(texts)
                vocab = build_vocabulary(series)
                matrix = texts_to_matrix(series, vocab)
                self.assertTrue(np.all(matrix >= 0))

    def test_matrix_count_accuracy(self) -> None:
        for texts in self.SAMPLE_TEXT_LISTS:
            with self.subTest(texts=texts):
                series = pd.Series(texts)
                vocab = build_vocabulary(series)
                matrix = texts_to_matrix(series, vocab)
                for i, text in enumerate(texts):
                    words = text.split()
                    for word, idx in vocab.items():
                        self.assertEqual(matrix[i, idx], words.count(word))


class TestSplitDatasetProperties(unittest.TestCase):
    """split_dataset conserves rows and is reproducible."""

    def test_conservation(self) -> None:
        for n in [2, 5, 10, 20, 50, 100]:
            with self.subTest(n=n):
                df = pd.DataFrame({
                    "text": [f"t{i}" for i in range(n)],
                    "label": [i % 2 for i in range(n)],
                })
                x_train, x_test, y_train, y_test = split_dataset(
                    df, "text", "label", 0.2
                )
                self.assertEqual(len(x_train) + len(x_test), n)
                self.assertEqual(len(y_train) + len(y_test), n)

    def test_reproducibility(self) -> None:
        for n in [5, 10, 20]:
            with self.subTest(n=n):
                df = pd.DataFrame({
                    "text": [f"t{i}" for i in range(n)],
                    "label": [i % 2 for i in range(n)],
                })
                split1 = split_dataset(df, "text", "label", 0.3)
                split2 = split_dataset(df, "text", "label", 0.3)
                for s1, s2 in zip(split1, split2):
                    self.assertEqual(list(s1), list(s2))


class TestPredictProperties(unittest.TestCase):
    """predict returns a binary integer array matching the input length."""

    @classmethod
    def setUpClass(cls) -> None:
        X_train = np.array([
            [3, 0, 0, 0],
            [2, 0, 1, 0],
            [0, 0, 0, 3],
            [0, 1, 0, 2],
        ], dtype=np.float64)
        y_train = np.array([1, 1, 0, 0])
        cls.model = train_model(X_train, y_train)

    def test_output_invariant(self) -> None:
        rng = np.random.default_rng(42)
        for n_test in [1, 3, 5, 10]:
            with self.subTest(n_test=n_test):
                X_test = rng.random((n_test, 4))
                result = predict(self.model, X_test)
                self.assertEqual(result.shape, (n_test,))
                self.assertTrue(all(val in (0, 1) for val in result))
                self.assertTrue(np.issubdtype(result.dtype, np.integer))


class TestMetricsProperties(unittest.TestCase):
    """Every metric stays within [0.0, 1.0] for any binary label pair."""

    BINARY_PAIRS = [
        ([1, 1, 1, 1], [1, 1, 1, 1]),
        ([0, 0, 0, 0], [0, 0, 0, 0]),
        ([1, 0, 1, 0], [0, 1, 0, 1]),
        ([1, 1, 0, 0], [1, 0, 1, 0]),
        ([1], [0]),
        ([0], [1]),
        ([1, 0], [1, 0]),
        ([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1]),
        ([0, 0, 0, 0, 0], [1, 1, 1, 1, 1]),
    ]

    def test_metrics_in_valid_range(self) -> None:
        for y_true, y_pred in self.BINARY_PAIRS:
            with self.subTest(y_true=y_true, y_pred=y_pred):
                metrics = evaluate_model(y_true, y_pred)
                self.assertEqual(
                    set(metrics.keys()),
                    {"accuracy", "f1_score", "precision", "recall"},
                )
                for value in metrics.values():
                    self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
