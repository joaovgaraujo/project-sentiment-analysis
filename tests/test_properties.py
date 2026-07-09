"""Property-based tests using pytest parametrize (no hypothesis).

Tests universal properties that should hold across many inputs.
"""

import numpy as np
import pandas as pd
import pytest

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


# --- Property 1.2: validate_columns raises ValueError iff required column missing ---
# **Validates: Requirements 1.3**

class TestValidateColumnsProperty:
    """validate_columns raises ValueError if and only if a required column is missing."""

    @pytest.mark.parametrize("columns,required,should_raise", [
        (["a", "b", "c"], ["a", "b"], False),
        (["a", "b"], ["a", "b", "c"], True),
        (["x", "y"], ["a"], True),
        (["reviews.text", "reviews.rating"], ["reviews.text", "reviews.rating"], False),
        (["reviews.text"], ["reviews.text", "reviews.rating"], True),
        ([], ["a"], True),
        (["a", "b", "c"], [], False),
    ])
    def test_validate_columns_property(self, columns, required, should_raise) -> None:
        df = pd.DataFrame({col: [1] for col in columns}) if columns else pd.DataFrame()
        if should_raise:
            with pytest.raises(ValueError):
                validate_columns(df, required)
        else:
            validate_columns(df, required)  # should not raise


# --- Property 2.2: Text cleaning properties ---
# **Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5**

class TestCleanTextProperties:
    """Property 1: Idempotency. Property 2: Output format."""

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

    @pytest.mark.parametrize("text", SAMPLE_TEXTS)
    def test_idempotency(self, text: str) -> None:
        """clean_text(clean_text(t)) == clean_text(t)"""
        once = clean_text(text)
        twice = clean_text(once)
        assert once == twice

    @pytest.mark.parametrize("text", SAMPLE_TEXTS)
    def test_output_format(self, text: str) -> None:
        """Result is lowercase, no leading/trailing whitespace, no consecutive spaces."""
        result = clean_text(text)
        # Lowercase
        assert result == result.lower()
        # No leading/trailing whitespace
        assert result == result.strip()
        # No consecutive spaces
        assert "  " not in result
        # Only alphanumeric and spaces
        for ch in result:
            assert ch.isalnum() or ch == " "


# --- Property 2.3: Label normalization ---
# **Validates: Requirements 3.1, 3.2, 3.3, 3.4**

class TestNormalizeLabelProperties:
    """Property 3: Label determinism and correctness."""

    @pytest.mark.parametrize("rating,expected", [
        (1, 0),
        (2, 0),
        (3, None),
        (4, 1),
        (5, 1),
    ])
    def test_label_correctness(self, rating: int, expected) -> None:
        """rating >= 4 -> 1, rating <= 2 -> 0, rating == 3 -> None"""
        assert normalize_label(rating) == expected

    @pytest.mark.parametrize("rating", [1, 2, 3, 4, 5])
    def test_label_determinism(self, rating: int) -> None:
        """Same input always produces same output."""
        result1 = normalize_label(rating)
        result2 = normalize_label(rating)
        assert result1 == result2


# --- Property 2.4: Vocabulary and vectorization ---
# **Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5, 5.6**

class TestVocabMatrixProperties:
    """Properties 6-9: Vocabulary completeness, matrix shape, non-negativity, count accuracy."""

    SAMPLE_TEXT_LISTS = [
        ["hello world", "hello"],
        ["the cat sat", "the dog sat", "the cat"],
        ["a b c d e", "a a a b b"],
        ["single"],
        ["word word word", "other other"],
    ]

    @pytest.mark.parametrize("texts", SAMPLE_TEXT_LISTS)
    def test_vocabulary_completeness_and_contiguity(self, texts) -> None:
        """Property 6: All words get unique contiguous indices."""
        series = pd.Series(texts)
        vocab = build_vocabulary(series)

        # All words present
        all_words = set()
        for text in texts:
            all_words.update(text.split())
        assert set(vocab.keys()) == all_words

        # Contiguous indices [0, len(vocab))
        indices = sorted(vocab.values())
        assert indices == list(range(len(vocab)))

    @pytest.mark.parametrize("texts", SAMPLE_TEXT_LISTS)
    def test_matrix_shape(self, texts) -> None:
        """Property 7: shape == (len(texts), len(vocab))"""
        series = pd.Series(texts)
        vocab = build_vocabulary(series)
        matrix = texts_to_matrix(series, vocab)
        assert matrix.shape == (len(texts), len(vocab))

    @pytest.mark.parametrize("texts", SAMPLE_TEXT_LISTS)
    def test_matrix_non_negativity(self, texts) -> None:
        """Property 8: All values >= 0."""
        series = pd.Series(texts)
        vocab = build_vocabulary(series)
        matrix = texts_to_matrix(series, vocab)
        assert np.all(matrix >= 0)

    @pytest.mark.parametrize("texts", SAMPLE_TEXT_LISTS)
    def test_matrix_count_accuracy(self, texts) -> None:
        """Property 9: Cell equals word occurrence count."""
        series = pd.Series(texts)
        vocab = build_vocabulary(series)
        matrix = texts_to_matrix(series, vocab)
        for i, text in enumerate(texts):
            words = text.split()
            for word, idx in vocab.items():
                expected_count = words.count(word)
                assert matrix[i, idx] == expected_count


# --- Property 3.2: Split dataset ---
# **Validates: Requirements 6.1, 6.3, 6.4**

class TestSplitDatasetProperties:
    """Properties 10-11: Split conservation and reproducibility."""

    @pytest.mark.parametrize("n", [2, 5, 10, 20, 50, 100])
    def test_split_conservation(self, n: int) -> None:
        """Property 10: len(x_train) + len(x_test) == len(data)"""
        df = pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": [i % 2 for i in range(n)]})
        x_train, x_test, y_train, y_test = split_dataset(df, "text", "label", 0.2)
        assert len(x_train) + len(x_test) == n
        assert len(y_train) + len(y_test) == n

    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_split_reproducibility(self, n: int) -> None:
        """Property 11: Same seed produces identical splits."""
        df = pd.DataFrame({"text": [f"t{i}" for i in range(n)], "label": [i % 2 for i in range(n)]})
        split1 = split_dataset(df, "text", "label", 0.3)
        split2 = split_dataset(df, "text", "label", 0.3)
        for s1, s2 in zip(split1, split2):
            assert list(s1) == list(s2)


# --- Property 5.2: Prediction output invariant ---
# **Validates: Requirements 8.1, 8.2, 8.3**

class TestPredictProperties:
    """Property 12: Prediction returns array of correct length with binary values."""

    @pytest.mark.parametrize("n_test", [1, 3, 5, 10])
    def test_prediction_output_invariant(self, n_test: int) -> None:
        """predict returns array of length len(X) with all values in {0, 1}."""
        # Train a simple model
        X_train = np.array([
            [3, 0, 0, 0],
            [2, 0, 1, 0],
            [0, 0, 0, 3],
            [0, 1, 0, 2],
        ], dtype=np.float64)
        y_train = np.array([1, 1, 0, 0])
        model = train_model(X_train, y_train)

        # Generate random test data
        rng = np.random.default_rng(42)
        X_test = rng.random((n_test, 4))
        result = predict(model, X_test)

        assert len(result) == n_test
        assert result.shape == (n_test,)
        assert all(val in (0, 1) for val in result)
        assert np.issubdtype(result.dtype, np.integer)


# --- Property 6.2: Metrics output validity ---
# **Validates: Requirements 9.5, 9.7**

class TestMetricsProperties:
    """Property 13: All metric values in [0.0, 1.0] for any binary arrays."""

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

    @pytest.mark.parametrize("y_true,y_pred", BINARY_PAIRS)
    def test_metrics_in_valid_range(self, y_true, y_pred) -> None:
        """All metrics should be in [0.0, 1.0]."""
        metrics = evaluate_model(y_true, y_pred)
        assert set(metrics.keys()) == {"accuracy", "f1_score", "precision", "recall"}
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key} = {value} is out of range"
