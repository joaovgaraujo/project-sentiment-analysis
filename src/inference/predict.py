"""Simple inference script: classify new review text with a trained model."""

import sys

import pandas as pd

from src.models.model import load_model, predict
from src.preprocessing.transform import clean_text, load_vocabulary, texts_to_matrix
from src.utils.config import MODEL_PATH, VOCAB_PATH

LABELS: dict[int, str] = {0: "negative", 1: "positive"}


def predict_sentiment(
    text: str,
    model_path: str = MODEL_PATH,
    vocab_path: str = VOCAB_PATH,
) -> str:
    """Classify the sentiment of a single review text.

    Loads the trained model and the vocabulary produced during training,
    then vectorizes and classifies the given text the same way training
    data was processed.

    Args:
        text: Raw review text.
        model_path: Path to the trained model weights (see save_model).
        vocab_path: Path to the vocabulary saved during training (see
            save_vocabulary).

    Returns:
        "positive" or "negative".
    """
    vocab = load_vocabulary(vocab_path)
    model = load_model(model_path, input_dim=len(vocab))

    cleaned = clean_text(text)
    features = texts_to_matrix(pd.Series([cleaned]), vocab)
    label = predict(model, features)[0]

    return LABELS[int(label)]


if __name__ == "__main__":
    review_text = " ".join(sys.argv[1:]) or input("Review text: ")
    print(predict_sentiment(review_text))
