"""Central entry point for the sentiment analysis pipeline."""

from src.data.loader import load_data
from src.evaluation.metrics import print_report
from src.preprocessing.transform import preprocess_dataset
from src.training.train import run_training


def main() -> None:
    """Run the full sentiment analysis pipeline."""
    df = load_data("data/raw/reviews.csv")
    df = preprocess_dataset(df)
    results = run_training(df)
    print_report(results["metrics"])


if __name__ == "__main__":
    main()
