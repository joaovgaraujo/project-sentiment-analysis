"""Central entry point for the sentiment analysis pipeline."""

from src.data.loader import load_data
from src.evaluation.metrics import print_report
from src.preprocessing.transform import preprocess_dataset
from src.reporting.history import save_history
from src.reporting.negatives import append_daily_report, build_negative_report
from src.training.train import run_training


def main() -> None:
    """Run the full sentiment analysis pipeline."""
    df = load_data("data/raw/reviews.csv")
    df = preprocess_dataset(df)
    results = run_training(df)
    print_report(results["metrics"])
    negatives = build_negative_report(df, results["model"], results["vocab"])
    print(f"Daily negatives: {append_daily_report(negatives)}")
    print(
        f"Metrics history: "
        f"{save_history(results['metrics'], results['y_test'], results['predictions'])}"
    )


if __name__ == "__main__":
    main()
