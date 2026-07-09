"""Unit tests for the data loader module."""

import pandas as pd
import pytest

from src.data.loader import load_data, validate_columns


def test_load_data_returns_dataframe(tmp_path) -> None:
    """load_data should return a DataFrame when given a valid CSV file."""
    csv_file = tmp_path / "reviews.csv"
    csv_file.write_text(
        "reviews.text,reviews.rating\n"
        "Great product,5\n"
        "Terrible quality,1\n"
    )
    df = load_data(str(csv_file))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "reviews.text" in df.columns
    assert "reviews.rating" in df.columns


def test_load_data_raises_file_not_found() -> None:
    """load_data should raise FileNotFoundError for a non-existent path."""
    with pytest.raises(FileNotFoundError, match="Dataset file not found"):
        load_data("nonexistent/path/to/file.csv")


def test_validate_columns_passes_with_required_columns() -> None:
    """validate_columns should not raise when all required columns are present."""
    df = pd.DataFrame({"reviews.text": ["hello"], "reviews.rating": [5]})
    # Should not raise
    validate_columns(df, ["reviews.text", "reviews.rating"])


def test_validate_columns_raises_on_missing_column() -> None:
    """validate_columns should raise ValueError when a column is missing."""
    df = pd.DataFrame({"reviews.text": ["hello"], "other_col": [5]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_columns(df, ["reviews.text", "reviews.rating"])
