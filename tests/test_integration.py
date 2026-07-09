"""Integration test for the full sentiment analysis pipeline."""

import pandas as pd

from src.data.loader import load_data
from src.preprocessing.transform import preprocess_dataset
from src.training.train import run_training


def test_full_pipeline(tmp_path) -> None:
    """Full pipeline: load -> preprocess -> train -> evaluate."""
    csv_file = tmp_path / "reviews.csv"
    csv_file.write_text(
        "reviews.text,reviews.rating\n"
        "This product is absolutely amazing and wonderful,5\n"
        "I love this so much it is great,5\n"
        "Excellent quality highly recommend to everyone,4\n"
        "Great purchase very happy with it,4\n"
        "Really good product works perfectly fine,5\n"
        "Terrible product do not buy ever,1\n"
        "Awful quality waste of money completely,1\n"
        "Horrible experience worst purchase ever made,2\n"
        "Bad product broke after one day,2\n"
        "Very disappointed terrible quality overall bad,1\n"
    )

    df = load_data(str(csv_file))
    assert len(df) == 10

    df = preprocess_dataset(df)
    assert len(df) > 0
    assert "sentiment" in df.columns

    results = run_training(df)

    assert "model" in results
    assert "predictions" in results
    assert "metrics" in results
    assert "vocab" in results

    metrics = results["metrics"]
    assert set(metrics.keys()) == {"accuracy", "f1_score", "precision", "recall"}
    for value in metrics.values():
        assert 0.0 <= value <= 1.0
