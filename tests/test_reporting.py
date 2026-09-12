"""Tests for the new reporting modules (stdlib unittest only)."""

import json
import os
import tempfile
import unittest
from datetime import date, datetime

import pandas as pd

from src.reporting.history import compute_macro_f1, save_history
from src.reporting.negatives import append_daily_report, build_negative_report


def _df():
    return pd.DataFrame(
        {
            "reviews.text": ["ruim", "otimo", "pessimo", "bom"],
            "reviews.rating": [1, 5, 1, 4],
            "asins": ["B2", "B1", "B2", "B1"],
            "reviews.date": ["2026-01-02", "2026-01-05", "2026-01-03", "2026-01-01"],
        }
    )


class TestNegatives(unittest.TestCase):
    def test_filters_sorts_and_groups(self):
        df = _df()
        out = build_negative_report(df, y_all=[0, 1, 0, 1])
        self.assertEqual(list(out.columns), ["produto", "data", "texto", "rating"])
        self.assertEqual(len(out), 2)
        # grouped by product, most recent first within product
        self.assertEqual(list(out["produto"]), ["B2", "B2"])
        self.assertEqual(list(out["data"]), ["2026-01-03", "2026-01-02"])

    def test_fallback_without_product_date_columns(self):
        df = pd.DataFrame(
            {"reviews.text": ["ruim", "otimo"], "reviews.rating": [1, 5]}
        )
        out = build_negative_report(df, y_all=[0, 1])
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["produto"], "desconhecido")

    def test_append_same_day_new_file_next_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            df = _df().head(2)
            neg = build_negative_report(df, y_all=[0, 1])
            p1 = append_daily_report(neg, out_dir=tmp, run_date=date(2026, 9, 12))
            self.assertTrue(os.path.basename(p1) == "result-20260912.csv")
            append_daily_report(neg, out_dir=tmp, run_date=date(2026, 9, 12))
            content = pd.read_csv(p1)
            self.assertEqual(len(content), 2)  # appended, header not duplicated
            p2 = append_daily_report(neg, out_dir=tmp, run_date=date(2026, 9, 13))
            self.assertTrue(os.path.basename(p2) == "result-20260913.csv")


class TestHistory(unittest.TestCase):
    def test_macro_f1(self):
        self.assertEqual(compute_macro_f1([1, 1, 0, 0], [1, 1, 0, 0]), 1.0)
        self.assertEqual(compute_macro_f1([1, 1, 0, 0], [0, 0, 1, 1]), 0.0)
        self.assertTrue(
            0.0 <= compute_macro_f1([1, 1, 0, 0], [1, 0, 0, 1]) <= 1.0
        )

    def test_versioned_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            metrics = {"accuracy": 0.9, "precision": 0.8, "recall": 1.0, "f1_score": 0.88}
            p1 = save_history(
                metrics, [1, 1, 0, 0], [1, 0, 0, 1],
                out_dir=tmp, run_at=datetime(2026, 9, 12, 10, 0, 0),
            )
            self.assertTrue(os.path.basename(p1) == "metrics-20260912100000.json")
            with open(p1, encoding="utf-8") as f:
                payload = json.loads(f.read())
            self.assertEqual(
                set(payload["metrics"]),
                {"accuracy", "precision", "recall", "f1_score", "macro_f1"},
            )
            self.assertEqual(payload["totals"]["n_test"], 4)
            p2 = save_history(
                metrics, [1, 0], [1, 0],
                out_dir=tmp, run_at=datetime(2026, 9, 12, 10, 0, 1),
            )
            self.assertNotEqual(p1, p2)  # always versioned, never overwrites


if __name__ == "__main__":
    unittest.main()
