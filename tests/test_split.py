"""Tests for the split_dataset function."""

import unittest

import pandas as pd

from src.training.train import split_dataset


def _make_dataframe(n: int) -> pd.DataFrame:
    """Build a simple DataFrame with n rows."""
    return pd.DataFrame({
        "text": [f"review {i}" for i in range(n)],
        "label": [i % 2 for i in range(n)],
    })


class TestSplitDataset(unittest.TestCase):
    """Tests for split_dataset."""

    def test_conservation(self) -> None:
        df = _make_dataframe(10)
        x_train, x_test, y_train, y_test = split_dataset(df, "text", "label", 0.2)
        self.assertEqual(len(x_train) + len(x_test), 10)
        self.assertEqual(len(y_train) + len(y_test), 10)

    def test_reproducibility(self) -> None:
        df = _make_dataframe(20)
        split1 = split_dataset(df, "text", "label", 0.3)
        split2 = split_dataset(df, "text", "label", 0.3)
        for s1, s2 in zip(split1, split2):
            self.assertEqual(list(s1), list(s2))

    def test_raises_on_small_dataset(self) -> None:
        df = _make_dataframe(1)
        with self.assertRaises(ValueError):
            split_dataset(df, "text", "label", 0.2)

    def test_two_rows(self) -> None:
        df = _make_dataframe(2)
        x_train, x_test, y_train, y_test = split_dataset(df, "text", "label", 0.5)
        self.assertEqual(len(x_train) + len(x_test), 2)


if __name__ == "__main__":
    unittest.main()
