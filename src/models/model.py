"""Sentiment classification model built with PyTorch."""

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.utils.config import BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_SEED

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
    batch_size: int = BATCH_SIZE,
) -> SentimentClassifier:
    """Train the classifier with mini-batch gradient descent.

    Training samples are wrapped in a ``TensorDataset`` and iterated over with
    a ``DataLoader`` so batching and shuffling follow the standard PyTorch
    workflow. If test data is given, the training and test error are printed
    every 10 epochs so the learning progress can be followed.
    """
    torch.manual_seed(RANDOM_SEED)

    x = _to_tensor(X_train)
    y = _to_tensor(y_train)
    assert x.shape[0] == y.shape[0], "X_train and y_train must have the same length"

    train_dataset = TensorDataset(x, y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    model = SentimentClassifier(x.shape[1]).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    has_test = X_test is not None and y_test is not None
    if has_test:
        x_test = _to_tensor(X_test)
        y_test_t = _to_tensor(y_test)
        assert x_test.shape[0] == y_test_t.shape[0], "X_test and y_test must have the same length"
        test_loader = DataLoader(TensorDataset(x_test, y_test_t), batch_size=batch_size)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            batch_loss = loss_fn(model(x_batch), y_batch)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss.item() * x_batch.size(0)
        train_loss = total_loss / len(train_dataset)

        if epoch % 10 == 0 or epoch == 1:
            message = f"Epoch {epoch:3d} | train error: {train_loss:.4f}"
            if has_test:
                model.eval()
                total_test_loss = 0.0
                with torch.inference_mode():
                    for x_batch, y_batch in test_loader:
                        total_test_loss += loss_fn(model(x_batch), y_batch).item() * x_batch.size(0)
                test_loss = total_test_loss / x_test.shape[0]
                message += f" | test error: {test_loss:.4f}"
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


def load_model(path: str, input_dim: int) -> SentimentClassifier:
    """Load a trained model's weights from disk.

    Args:
        path: Path to the file produced by save_model.
        input_dim: Size of the feature vector the model was trained with
            (i.e. the vocabulary size).

    Returns:
        The reconstructed model in evaluation mode.
    """
    model = SentimentClassifier(input_dim).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model
