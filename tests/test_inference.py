"""Unit tests for model persistence (load_model) and the inference script."""

import numpy as np

from src.inference.predict import predict_sentiment
from src.models.model import load_model, predict, save_model, train_model
from src.preprocessing.transform import load_vocabulary, save_vocabulary

VOCAB = {"great": 0, "terrible": 1, "good": 2, "bad": 3}

X_TRAIN = np.array(
    [
        [2, 0, 1, 0],
        [1, 0, 2, 0],
        [3, 0, 0, 0],
        [0, 2, 0, 1],
        [0, 1, 0, 2],
        [0, 3, 0, 0],
    ],
    dtype=np.float64,
)
Y_TRAIN = np.array([1, 1, 1, 0, 0, 0])


def test_load_model_matches_saved_predictions(tmp_path) -> None:
    """A model reloaded from disk should predict identically to the original."""
    model = train_model(X_TRAIN, Y_TRAIN, epochs=5)
    model_path = tmp_path / "model.pt"
    save_model(model, str(model_path))

    loaded = load_model(str(model_path), input_dim=X_TRAIN.shape[1])

    assert np.array_equal(predict(model, X_TRAIN), predict(loaded, X_TRAIN))


def test_save_and_load_vocabulary_roundtrip(tmp_path) -> None:
    """Saving and loading a vocabulary should return the same mapping."""
    vocab_path = tmp_path / "vocab.json"
    save_vocabulary(VOCAB, str(vocab_path))

    assert load_vocabulary(str(vocab_path)) == VOCAB


def test_predict_sentiment_positive(tmp_path) -> None:
    """predict_sentiment should classify a clearly positive review as positive."""
    model = train_model(X_TRAIN, Y_TRAIN, epochs=50)
    model_path = tmp_path / "model.pt"
    vocab_path = tmp_path / "vocab.json"
    save_model(model, str(model_path))
    save_vocabulary(VOCAB, str(vocab_path))

    result = predict_sentiment(
        "great and good", model_path=str(model_path), vocab_path=str(vocab_path)
    )

    assert result == "positive"


def test_predict_sentiment_negative(tmp_path) -> None:
    """predict_sentiment should classify a clearly negative review as negative."""
    model = train_model(X_TRAIN, Y_TRAIN, epochs=50)
    model_path = tmp_path / "model.pt"
    vocab_path = tmp_path / "vocab.json"
    save_model(model, str(model_path))
    save_vocabulary(VOCAB, str(vocab_path))

    result = predict_sentiment(
        "terrible and bad", model_path=str(model_path), vocab_path=str(vocab_path)
    )

    assert result == "negative"
