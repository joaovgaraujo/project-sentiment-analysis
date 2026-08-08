"""Tests for the model: tensor shape, accuracy, and save/load round-trip."""

import os
import tempfile
import unittest

import numpy as np
import torch

from src.evaluation.metrics import evaluate_model
from src.models.model import (
    DEVICE,
    SentimentClassifier,
    predict,
    save_model,
    train_model,
)

# Linearly separable data: first two words mark positive, last two negative.
X_TRAIN = np.array([
    [2, 1, 0, 0],
    [1, 2, 0, 0],
    [3, 0, 0, 0],
    [0, 0, 2, 1],
    [0, 0, 1, 2],
    [0, 0, 3, 0],
], dtype=np.float64)
Y_TRAIN = np.array([1, 1, 1, 0, 0, 0])


class TestModelForward(unittest.TestCase):
    """The model turns a feature matrix into one logit per sample."""

    def test_output_shape(self) -> None:
        model = SentimentClassifier(4).to(DEVICE)
        x = torch.rand(3, 4).to(DEVICE)
        output = model(x)
        self.assertEqual(output.shape, (3,))


class TestModelAccuracy(unittest.TestCase):
    """A trained model reaches the expected accuracy on separable data."""

    def test_accuracy_above_threshold(self) -> None:
        model = train_model(X_TRAIN, Y_TRAIN)
        y_pred = predict(model, X_TRAIN)
        accuracy = evaluate_model(Y_TRAIN, y_pred)["accuracy"]
        self.assertGreaterEqual(accuracy, 0.8)


class TestModelSaveLoad(unittest.TestCase):
    """Saving and reloading the weights reproduces the same predictions."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp_dir.name, "model.pt")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_save_creates_file(self) -> None:
        model = train_model(X_TRAIN, Y_TRAIN)
        save_model(model, self.path)
        self.assertTrue(os.path.isfile(self.path))

    def test_reload_reproduces_predictions(self) -> None:
        model = train_model(X_TRAIN, Y_TRAIN)
        save_model(model, self.path)

        reloaded = SentimentClassifier(X_TRAIN.shape[1]).to(DEVICE)
        reloaded.load_state_dict(torch.load(self.path, map_location=DEVICE))

        self.assertTrue(
            np.array_equal(predict(model, X_TRAIN), predict(reloaded, X_TRAIN))
        )


if __name__ == "__main__":
    unittest.main()
