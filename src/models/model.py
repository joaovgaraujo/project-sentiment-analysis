"""Sentiment classification model built with PyTorch."""

import numpy as np
import torch
from torch import nn

from src.utils.config import EPOCHS, LEARNING_RATE, RANDOM_SEED

# Device-agnostic: use the GPU when it is available, otherwise the CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SentimentClassifier(nn.Module):
    """Logistic regression: a single linear layer that outputs one logit."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(1)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a float32 tensor on the active device."""
    return torch.tensor(np.asarray(array), dtype=torch.float32).to(DEVICE)


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
) -> SentimentClassifier:
    """Train the classifier with gradient descent.

    If test data is given, the training and test error are printed every
    10 epochs so the learning progress can be followed.
    """
    torch.manual_seed(RANDOM_SEED)

    x = _to_tensor(X_train)
    y = _to_tensor(y_train)

    model = SentimentClassifier(x.shape[1]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    has_test = X_test is not None and y_test is not None
    if has_test:
        x_test = _to_tensor(X_test)
        y_test_t = _to_tensor(y_test)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_loss = loss_fn(model(x), y)
        train_loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == 1:
            message = f"Epoch {epoch:3d} | train error: {train_loss.item():.4f}"
            if has_test:
                model.eval()
                with torch.inference_mode():
                    test_loss = loss_fn(model(x_test), y_test_t)
                message += f" | test error: {test_loss.item():.4f}"
            print(message)

    return model


def predict(model: SentimentClassifier, X: np.ndarray) -> np.ndarray:
    """Return predicted labels (0 or 1) for the given feature matrix."""
    model.eval()
    with torch.inference_mode():
        logits = model(_to_tensor(X))
        labels = (torch.sigmoid(logits) >= 0.5).int()
    return labels.cpu().numpy().astype(int)


def save_model(model: SentimentClassifier, path: str) -> None:
    """Save the trained model weights to disk."""
    torch.save(model.state_dict(), path)
