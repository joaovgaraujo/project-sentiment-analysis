"""Tests for the preprocessing module."""

import unittest

from src.preprocessing.transform import clean_text, normalize_label


class TestCleanText(unittest.TestCase):
    """Tests for clean_text."""

    def test_converts_to_lowercase(self) -> None:
        self.assertEqual(clean_text("Hello World"), "hello world")
        self.assertEqual(clean_text("UPPERCASE TEXT"), "uppercase text")
        self.assertEqual(clean_text("MiXeD CaSe"), "mixed case")

    def test_removes_extra_spaces(self) -> None:
        self.assertEqual(clean_text("  spaces   here  "), "spaces here")
        self.assertEqual(clean_text("a    b"), "a b")
        self.assertEqual(clean_text("  leading"), "leading")
        self.assertEqual(clean_text("trailing  "), "trailing")

    def test_removes_punctuation(self) -> None:
        self.assertEqual(clean_text("Hello, World!"), "hello world")
        self.assertEqual(clean_text("price: $100.00!!"), "price 10000")
        self.assertEqual(clean_text("a...b---c"), "abc")
        self.assertEqual(clean_text("don't stop!"), "dont stop")

    def test_empty_string(self) -> None:
        self.assertEqual(clean_text(""), "")

    def test_idempotent(self) -> None:
        texts = ["Hello, World!", "  spaces   here  ", "UPPER!!", "already clean"]
        for text in texts:
            with self.subTest(text=text):
                once = clean_text(text)
                self.assertEqual(clean_text(once), once)


class TestNormalizeLabel(unittest.TestCase):
    """Tests for normalize_label."""

    def test_positive(self) -> None:
        self.assertEqual(normalize_label(4), 1)
        self.assertEqual(normalize_label(5), 1)

    def test_negative(self) -> None:
        self.assertEqual(normalize_label(1), 0)
        self.assertEqual(normalize_label(2), 0)

    def test_neutral(self) -> None:
        self.assertIsNone(normalize_label(3))


if __name__ == "__main__":
    unittest.main()
