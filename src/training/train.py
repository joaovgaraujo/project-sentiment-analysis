"""Training pipeline: dataset splitting and model training orchestration."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate_model
from src.models.model import predict, train_model
from src.preprocessing.transform import build_vocabulary, texts_to_matrix
from src.utils.config import LABEL_COLUMN, RANDOM_SEED, TEST_SIZE, TEXT_COLUMN


def split_dataset(
    data: pd.DataFrame,
    text_column: str,
    label_column: str,
    test_size: float,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Split the dataset into training and test sets using NumPy-based shuffling.

    Args:
        data: Preprocessed DataFrame.
        text_column: Name of the text column.
        label_column: Name of the label column.
        test_size: Proportion of the dataset to use for testing (0-1).

    Returns:
        Tuple of (x_train, x_test, y_train, y_test).

    Raises:
        ValueError: If data has fewer than 2 rows.
    """
    if len(data) < 2:
        raise ValueError("Dataset must have at least 2 rows")
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1 (exclusive)")

    rng = np.random.default_rng(RANDOM_SEED)
    n = len(data)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = max(1, min(int(np.ceil(n * test_size)), n - 1))
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    x_train = data[text_column].iloc[train_indices].reset_index(drop=True)
    x_test = data[text_column].iloc[test_indices].reset_index(drop=True)
    y_train = data[label_column].iloc[train_indices].reset_index(drop=True)
    y_test = data[label_column].iloc[test_indices].reset_index(drop=True)

    return x_train, x_test, y_train, y_test


def run_training(data: pd.DataFrame) -> dict[str, object]:
    """Execute the full training pipeline.

    Args:
        data: Preprocessed DataFrame with text and sentiment columns.

    Returns:
        Dictionary with keys: model, predictions, y_test, metrics, vocab.

    Raises:
        ValueError: If data has fewer than 2 rows.
    """
    x_train, x_test, y_train, y_test = split_dataset(
        data, TEXT_COLUMN, LABEL_COLUMN, TEST_SIZE
    )

    vocab = build_vocabulary(x_train)
    X_train = texts_to_matrix(x_train, vocab)
    X_test = texts_to_matrix(x_test, vocab)

    model = train_model(X_train, y_train.to_numpy())
    y_pred = predict(model, X_test)

    metrics = evaluate_model(y_test, y_pred)

    return {
        "model": model,
        "predictions": y_pred,
        "y_test": y_test,
        "metrics": metrics,
        "vocab": vocab,
    }
