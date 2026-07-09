"""Sentiment classification model definition."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.utils.config import RANDOM_SEED


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a LogisticRegression classifier on count matrix data.

    Args:
        X_train: Training feature matrix (word counts).
        y_train: Training labels (0 or 1).

    Returns:
        Fitted LogisticRegression model.
    """
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    return model


def predict(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    """Generate predictions using the trained model.

    Args:
        model: Fitted LogisticRegression model.
        X: Feature matrix (word counts).

    Returns:
        Array of predicted labels (0 or 1).
    """
    return np.asarray(model.predict(X), dtype=int)
