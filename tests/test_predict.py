"""Unit tests for the model prediction module."""

import numpy as np

from src.models.model import predict, train_model


def _get_trained_model():
    """Helper: train a small model on synthetic count matrix data."""
    # 6 samples, 4 features (words: great, terrible, good, bad)
    X_train = np.array([
        [2, 0, 1, 0],  # positive
        [1, 0, 2, 0],  # positive
        [3, 0, 0, 0],  # positive
        [0, 2, 0, 1],  # negative
        [0, 1, 0, 2],  # negative
        [0, 3, 0, 0],  # negative
    ], dtype=np.float64)
    y_train = np.array([1, 1, 1, 0, 0, 0])
    return train_model(X_train, y_train)


def test_predict_returns_array() -> None:
    """predict should return a numpy array of labels."""
    model = _get_trained_model()
    X_test = np.array([[2, 0, 1, 0], [0, 2, 0, 1]], dtype=np.float64)
    result = predict(model, X_test)
    assert isinstance(result, np.ndarray)


def test_predict_output_is_binary() -> None:
    """predict should return only 0 or 1 as label values."""
    model = _get_trained_model()
    X_test = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [1, 1, 0, 0]], dtype=np.float64)
    result = predict(model, X_test)
    assert all(val in (0, 1) for val in result)


def test_predict_batch_length_matches_input() -> None:
    """predict output length should match the number of input samples."""
    model = _get_trained_model()
    X_test = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    result = predict(model, X_test)
    assert len(result) == 4
    assert result.shape == (4,)


def test_predict_dtype_is_int() -> None:
    """predict should return array with integer dtype."""
    model = _get_trained_model()
    X_test = np.array([[1, 0, 1, 0]], dtype=np.float64)
    result = predict(model, X_test)
    assert np.issubdtype(result.dtype, np.integer)
