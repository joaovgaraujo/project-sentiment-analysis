"""Text preprocessing and label normalization transformations."""

import re

import numpy as np
import pandas as pd

from src.utils.config import (
    LABEL_COLUMN,
    MAX_NEGATIVE_RATING,
    MIN_POSITIVE_RATING,
    RATING_COLUMN,
    TEXT_COLUMN,
)


def clean_text(text: str) -> str:
    """Remove noise and normalize a review text.

    Applies lowercasing, punctuation removal, and whitespace collapsing.
    Only alphanumeric characters (letters and digits) and spaces are retained.

    Args:
        text: Raw review text.

    Returns:
        Cleaned and normalized text.
    """
    # Step 1: Lowercase
    text = text.lower()
    # Step 2: Keep only alphanumeric characters and whitespace
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    # Step 3: Remove underscores (since \w includes _)
    text = text.replace("_", "")
    # Step 4: Remove any character that is not alphanumeric or space
    text = "".join(ch for ch in text if ch.isalnum() or ch == " ")
    # Step 5: Collapse whitespace and strip
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(rating: int) -> int | None:
    """Convert a numeric rating into a binary sentiment label.

    Args:
        rating: Review rating from 1 to 5.

    Returns:
        1 for positive sentiment, 0 for negative, None for neutral (discarded).
    """
    if rating >= MIN_POSITIVE_RATING:
        return 1
    elif rating <= MAX_NEGATIVE_RATING:
        return 0
    else:
        return None


def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply text cleaning and label normalization to the full dataset.

    Drops rows with missing text, missing rating, or neutral rating (3).

    Args:
        df: DataFrame with text and rating columns.

    Returns:
        Processed DataFrame with a binary sentiment column added.
    """
    # Step 1: Drop rows with missing text or rating
    df = df.dropna(subset=[TEXT_COLUMN, RATING_COLUMN]).copy()

    # Step 2: Ensure rating is integer
    df[RATING_COLUMN] = df[RATING_COLUMN].astype(int)

    # Step 3: Apply text cleaning
    df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_text)

    # Step 4: Apply label normalization
    df[LABEL_COLUMN] = df[RATING_COLUMN].apply(normalize_label)

    # Step 5: Drop neutral reviews (None labels)
    df = df.dropna(subset=[LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    # Step 6: Drop rows where cleaned text is empty
    df = df[df[TEXT_COLUMN].str.len() > 0]

    return df.reset_index(drop=True)


def build_vocabulary(texts: pd.Series) -> dict[str, int]:
    """Build a word-to-index mapping from training texts.

    Args:
        texts: Series of cleaned text strings.

    Returns:
        Dictionary mapping each unique word to a contiguous integer index.
    """
    vocab: dict[str, int] = {}
    idx = 0
    for text in texts:
        for word in text.split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1
    return vocab


def texts_to_matrix(texts: pd.Series, vocab: dict[str, int]) -> np.ndarray:
    """Convert texts to a word-count matrix using NumPy.

    Args:
        texts: Series of cleaned text strings.
        vocab: Vocabulary dictionary from build_vocabulary.

    Returns:
        NumPy array of shape (len(texts), len(vocab)) with word counts.
    """
    matrix = np.zeros((len(texts), len(vocab)), dtype=np.float64)
    for i, text in enumerate(texts):
        for word in text.split():
            if word in vocab:
                matrix[i, vocab[word]] += 1
    return matrix
