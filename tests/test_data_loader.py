"""Tests for the data loader module."""

import os
import tempfile
import unittest

import pandas as pd

from src.data.loader import load_data, validate_columns


class TestLoadData(unittest.TestCase):
    """Tests for load_data."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmp_dir.name, "reviews.csv")

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _write_csv(self, content: str) -> None:
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(content)

    def test_returns_dataframe(self) -> None:
        self._write_csv(
            "reviews.text,reviews.rating\n"
            "Great product,5\n"
            "Terrible quality,1\n"
        )
        df = load_data(self.csv_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("reviews.text", df.columns)
        self.assertIn("reviews.rating", df.columns)

    def test_raises_file_not_found(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_data("nonexistent/path/to/file.csv")

    def test_raises_on_empty_dataset(self) -> None:
        self._write_csv("reviews.text,reviews.rating\n")
        with self.assertRaises(ValueError):
            load_data(self.csv_path)


class TestValidateColumns(unittest.TestCase):
    """Tests for validate_columns."""

    def test_passes_with_required_columns(self) -> None:
        df = pd.DataFrame({"reviews.text": ["hello"], "reviews.rating": [5]})
        validate_columns(df, ["reviews.text", "reviews.rating"])

    def test_raises_on_missing_column(self) -> None:
        df = pd.DataFrame({"reviews.text": ["hello"], "other_col": [5]})
        with self.assertRaises(ValueError):
            validate_columns(df, ["reviews.text", "reviews.rating"])


if __name__ == "__main__":
    unittest.main()
