#!/usr/bin/env python3
"""Train spot long-entry and exit classifiers using shared pipeline settings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from spot_pipeline_settings import END_DATE, START_DATE, SYMBOL, TIMEFRAME, cache_file, summary_path

FEATURES = ["price_speed_5", "distance_pct", "angle_ema20_deg", "length"]

TREND_DOWN = "downtrend"
TREND_UP = "uptrend"
TREND_SIDE = "sideways"

ENTRY_MODEL_PATH = summary_path("models/spot_entry_pipeline.pkl")
EXIT_MODEL_PATH = summary_path("models/spot_exit_pipeline.pkl")
REPORT_PATH = summary_path("models/spot_model_report.json")


@dataclass
class Segment:
    start_index: int
    end_index: int
    trend: str
    length: int


def load_price_data() -> pd.DataFrame:
    cache = cache_file()
    if not cache.exists():
        raise FileNotFoundError(
            f"Cached OHLC data not found at {cache}. Run count_downward_impulses.py first."
        )
    df = pd.read_parquet(cache)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_100"] = df["close"].ewm(span=100).mean()

    df["ema20_slope"] = df["ema_20"].diff()
    df["ema50_slope"] = df["ema_50"].diff()

    trend_labels: List[str] = []
    for _, row in df.iterrows():
        ema20 = row["ema_20"]
        ema50 = row["ema_50"]
        ema100 = row["ema_100"]
        slope20 = row["ema20_slope"]
        slope50 = row["ema50_slope"]
        if pd.isna(ema100):
            trend_labels.append(TREND_SIDE)
        elif ema20 < ema50 < ema100 and slope20 < 0 and slope50 < 0:
            trend_labels.append(TREND_DOWN)
        elif ema20 > ema50 > ema100 and slope20 > 0 and slope50 > 0:
            trend_labels.append(TREND_UP)
        else:
            trend_labels.append(TREND_SIDE)
    df["trend_type"] = trend_labels

    df["close_diff"] = df["close"].diff()
    df["price_speed_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["distance_pct"] = (df["close"] - df["ema_20"]) / df["ema_20"]
    slope_ema20 = (df["ema_20"] - df["ema_20"].shift(5)) / 5
    df["angle_ema20_deg"] = np.degrees(np.arctan(slope_ema20))

    df["length"] = 0
    current_len = 0
    for idx in range(1, len(df)):
        diff = df.loc[idx, "close_diff"]
        prev_diff = df.loc[idx - 1, "close_diff"]
        if diff < 0 <= prev_diff:
            current_len = 1
        elif diff < 0:
            current_len += 1
        else:
            current_len = 0
        df.at[idx, "length"] = current_len if current_len > 0 else 0

    return df


def find_segments(df: pd.DataFrame, direction: str) -> List[Segment]:
    segments: List[Segment] = []
    start_idx: Optional[int] = None
    comparator = (lambda diff: diff < 0) if direction == "down" else (lambda diff: diff > 0)

    for idx in range(1, len(df)):
        diff = df.loc[idx, "close_diff"]
        if comparator(diff):
            if start_idx is None:
                start_idx = idx - 1
        else:
            if start_idx is not None and idx - 1 > start_idx:
                end_idx = idx - 1
                trend = df.loc[start_idx, "trend_type"]
                segments.append(
                    Segment(
                        start_index=start_idx,
                        end_index=end_idx,
                        trend=trend,
                        length=end_idx - start_idx + 1,
                    )
                )
                start_idx = None
            else:
                start_idx = None
    if start_idx is not None and start_idx < len(df) - 1:
        end_idx = len(df) - 1
        trend = df.loc[start_idx, "trend_type"]
        segments.append(
            Segment(
                start_index=start_idx,
                end_index=end_idx,
                trend=trend,
                length=end_idx - start_idx + 1,
            )
        )
    return segments


def build_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    downward_segments = find_segments(df, direction="down")
    upward_segments = find_segments(df, direction="up")

    df["entry_label"] = 0
    df["exit_label"] = 0

    for seg in downward_segments:
        df.loc[seg.end_index, "entry_label"] = 1
        df.loc[seg.end_index, "length"] = seg.length
    for seg in upward_segments:
        df.loc[seg.end_index, "exit_label"] = 1
        df.loc[seg.end_index, "length"] = seg.length

    dataset = df.dropna(subset=FEATURES).reset_index(drop=True)
    X = dataset[FEATURES]
    y_entry = dataset["entry_label"].astype(int)
    y_exit = dataset["exit_label"].astype(int)
    return X, y_entry, y_exit


def train_classifier(X: pd.DataFrame, y: pd.Series) -> Tuple[Pipeline, Dict[str, float]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_proba)

    metrics = {
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "roc_auc": roc_auc,
        "support": report["1"]["support"],
    }

    return pipeline, metrics


def main() -> None:
    price_df = load_price_data()
    feature_df = compute_features(price_df)
    X, y_entry, y_exit = build_dataset(feature_df)

    entry_pipeline, entry_metrics = train_classifier(X, y_entry)
    exit_pipeline, exit_metrics = train_classifier(X, y_exit)

    ENTRY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(entry_pipeline, ENTRY_MODEL_PATH)
    joblib.dump(exit_pipeline, EXIT_MODEL_PATH)

    report = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "period": f"{START_DATE.strftime('%Y-%m-%d')} -> {END_DATE.strftime('%Y-%m-%d')}",
        "features": FEATURES,
        "entry_metrics": entry_metrics,
        "exit_metrics": exit_metrics,
        "samples": {
            "total": int(len(X)),
            "entry_positives": int(y_entry.sum()),
            "exit_positives": int(y_exit.sum()),
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Entry metrics:", entry_metrics)
    print("Exit metrics:", exit_metrics)
    print(f"Report saved to {REPORT_PATH}")
    print(f"Pipelines saved to {ENTRY_MODEL_PATH} and {EXIT_MODEL_PATH}")


if __name__ == "__main__":
    main()
