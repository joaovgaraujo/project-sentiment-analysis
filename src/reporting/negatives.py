"""Daily report of negative reviews (new module, no changes to existing code)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd

from src.utils.config import RATING_COLUMN, TEXT_COLUMN

PRODUCT_CANDIDATES: tuple[str, ...] = (
    "asins",
    "asin",
    "product_id",
    "product",
    "name",
    "brand",
    "reviews.asins",
)
DATE_CANDIDATES: tuple[str, ...] = (
    "reviews.date",
    "reviews.dateAdded",
    "date",
    "dateAdded",
    "review_date",
    "created_at",
)


def detect_column(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    """Return the first candidate column present in df (case-insensitive fallback)."""
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def build_negative_report(
    df: pd.DataFrame,
    model=None,
    vocab: dict[str, int] | None = None,
    y_all: Sequence[int] | None = None,
    product_col: str | None = None,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Score the full df and return only predicted negatives, grouped by product.

    Sorted by product ASC then recency DESC. Works without knowing the CSV
    schema: product/date columns are auto-detected, with graceful fallback.
    Pass ``y_all`` in tests to skip model inference.
    """
    if y_all is None:
        if model is None or vocab is None:
            raise ValueError("Provide (model + vocab) or y_all.")
        from src.models.model import predict
        from src.preprocessing.transform import texts_to_matrix

        y_all = predict(model, texts_to_matrix(df[TEXT_COLUMN], vocab))

    y_all = pd.Series(list(y_all), index=df.index)
    neg = df[y_all == 0].copy()
    if neg.empty:
        return pd.DataFrame(columns=["produto", "data", "texto", "rating"])

    prod_col = product_col or detect_column(df, PRODUCT_CANDIDATES)
    dt_col = date_col or detect_column(df, DATE_CANDIDATES)

    neg["_produto"] = neg[prod_col].astype(str) if prod_col else "desconhecido"
    if dt_col:
        neg["_data_sort"] = pd.to_datetime(neg[dt_col], errors="coerce")
    else:
        neg["_data_sort"] = pd.NaT
    neg["_data_out"] = neg["_data_sort"].dt.strftime("%Y-%m-%d").fillna("")

    out = pd.DataFrame(
        {
            "produto": neg["_produto"].values,
            "data": neg["_data_out"].values,
            "texto": neg[TEXT_COLUMN].values,
            "rating": neg[RATING_COLUMN].values
            if RATING_COLUMN in neg.columns
            else "",
        }
    )
    out["_sort"] = neg["_data_sort"].values
    out = out.sort_values(
        by=["produto", "_sort"], ascending=[True, False], na_position="last"
    ).drop(columns=["_sort"]).reset_index(drop=True)
    return out


def append_daily_report(
    df_negatives: pd.DataFrame,
    out_dir: str = "results/daily",
    run_date: date | datetime | str | None = None,
) -> Path:
    """Append negatives to result-YYYYMMDD.csv (new file each day, stdlib only)."""
    if run_date is None:
        run_date = date.today()
    if isinstance(run_date, datetime):
        run_date = run_date.date()
    if isinstance(run_date, str):
        run_date = datetime.fromisoformat(run_date).date()
    stamp = run_date.strftime("%Y%m%d")

    out_path = Path(out_dir) / f"result-{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df_negatives.empty:
        if not out_path.exists():
            pd.DataFrame(columns=["produto", "data", "texto", "rating"]).to_csv(
                out_path, index=False
            )
        return out_path

    df_negatives.to_csv(
        out_path,
        mode="a" if out_path.exists() else "w",
        header=not out_path.exists(),
        index=False,
    )
    return out_path
