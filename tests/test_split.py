"""Unit tests for the split_dataset function."""

import pandas as pd
import pytest

from src.training.train import split_dataset


def _make_dataframe(n: int) -> pd.DataFrame:
    """Create a simple test DataFrame with n rows."""
    return pd.DataFrame({
        "text": [f"review {i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
    })


def test_split_conservation() -> None:
    """Split should preserve total number of rows."""
    df = _make_dataframe(10)
    x_train, x_test, y_train, y_test = split_dataset(df, "text", "label", 0.2)
    assert len(x_train) + len(x_test) == 10
    assert len(y_train) + len(y_test) == 10


def test_split_reproducibility() -> None:
    """Same data should produce identical splits."""
    df = _make_dataframe(20)
    split1 = split_dataset(df, "text", "label", 0.3)
    split2 = split_dataset(df, "text", "label", 0.3)
    for s1, s2 in zip(split1, split2):
        assert list(s1) == list(s2)


def test_split_raises_on_small_dataset() -> None:
    """split_dataset should raise ValueError for datasets with < 2 rows."""
    df = _make_dataframe(1)
    with pytest.raises(ValueError, match="at least 2 rows"):
        split_dataset(df, "text", "label", 0.2)


def test_split_two_rows() -> None:
    """split_dataset should handle exactly 2 rows."""
    df = _make_dataframe(2)
    x_train, x_test, y_train, y_test = split_dataset(df, "text", "label", 0.5)
    assert len(x_train) + len(x_test) == 2
