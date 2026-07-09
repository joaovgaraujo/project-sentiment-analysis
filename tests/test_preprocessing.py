"""Unit tests for the preprocessing module."""

from src.preprocessing.transform import clean_text, normalize_label


def test_clean_text_converts_to_lowercase() -> None:
    """clean_text should convert all characters to lowercase."""
    assert clean_text("Hello World") == "hello world"
    assert clean_text("UPPERCASE TEXT") == "uppercase text"
    assert clean_text("MiXeD CaSe") == "mixed case"


def test_clean_text_removes_extra_spaces() -> None:
    """clean_text should collapse multiple spaces into one."""
    assert clean_text("  spaces   here  ") == "spaces here"
    assert clean_text("a    b") == "a b"
    assert clean_text("  leading") == "leading"
    assert clean_text("trailing  ") == "trailing"


def test_clean_text_removes_punctuation() -> None:
    """clean_text should remove punctuation and special characters."""
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("price: $100.00!!") == "price 10000"
    assert clean_text("a...b---c") == "abc"
    assert clean_text("don't stop!") == "dont stop"


def test_clean_text_empty_string() -> None:
    """clean_text should return empty string for empty input."""
    assert clean_text("") == ""


def test_clean_text_idempotent() -> None:
    """clean_text applied twice should equal clean_text applied once."""
    texts = ["Hello, World!", "  spaces   here  ", "UPPER!!", "already clean"]
    for text in texts:
        once = clean_text(text)
        twice = clean_text(once)
        assert once == twice


def test_normalize_label_positive() -> None:
    """normalize_label should return 1 for ratings >= 4."""
    assert normalize_label(4) == 1
    assert normalize_label(5) == 1


def test_normalize_label_negative() -> None:
    """normalize_label should return 0 for ratings <= 2."""
    assert normalize_label(1) == 0
    assert normalize_label(2) == 0


def test_normalize_label_neutral() -> None:
    """normalize_label should return None for rating == 3."""
    assert normalize_label(3) is None
