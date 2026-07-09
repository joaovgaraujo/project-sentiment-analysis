"""Data loading and validation for the product reviews dataset."""

import os

import pandas as pd

from src.utils.config import RATING_COLUMN, TEXT_COLUMN


def load_data(path: str) -> pd.DataFrame:
    """Load the product reviews dataset from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame containing the loaded reviews.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        ValueError: If required columns are missing from the dataset.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    validate_columns(df, [TEXT_COLUMN, RATING_COLUMN])

    if len(df) == 0:
        raise ValueError("Dataset is empty: no rows found.")

    return df


def validate_columns(df: pd.DataFrame, required: list[str]) -> None:
    """Verify that all required columns are present in the DataFrame.

    Args:
        df: DataFrame to validate.
        required: List of required column names.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
