"""Integration test for the full sentiment analysis pipeline."""

import os
import tempfile
import unittest

from src.data.loader import load_data
from src.preprocessing.transform import preprocess_dataset
from src.training.train import run_training


class TestFullPipeline(unittest.TestCase):
    """End-to-end test: load -> preprocess -> train -> evaluate."""

    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmp_dir.name, "reviews.csv")
        with open(self.csv_path, "w", encoding="utf-8") as f:
            f.write(
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

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def test_full_pipeline(self) -> None:
        df = load_data(self.csv_path)
        self.assertEqual(len(df), 10)

        df = preprocess_dataset(df)
        self.assertGreater(len(df), 0)
        self.assertIn("sentiment", df.columns)

        results = run_training(df)
        for key in ("model", "predictions", "metrics", "vocab"):
            self.assertIn(key, results)

        metrics = results["metrics"]
        self.assertEqual(
            set(metrics.keys()),
            {"accuracy", "f1_score", "precision", "recall"},
        )
        for value in metrics.values():
            self.assertTrue(0.0 <= value <= 1.0)


if __name__ == "__main__":
    unittest.main()
