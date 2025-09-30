#!/usr/bin/env python3
"""Extract upward impulse profiles (speed, distance, angle, length) using shared settings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from spot_pipeline_settings import END_DATE, START_DATE, SYMBOL, TIMEFRAME, cache_file, summary_path

TREND_DOWN = "downtrend"
TREND_UP = "uptrend"
TREND_SIDE = "sideways"

OUTPUT_CSV = summary_path("upward_profiles.csv")
OUTPUT_JSON = summary_path("upward_profiles_summary.json")


def load_price_data() -> pd.DataFrame:
    cache = cache_file()
    if not cache.exists():
        raise FileNotFoundError(
            f"Cached OHLC data not found at {cache}. Run count_downward_impulses.py first."
        )
    df = pd.read_parquet(cache)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_trend_and_features(df: pd.DataFrame) -> pd.DataFrame:
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

    return df


def find_upward_segments(df: pd.DataFrame) -> List[Dict[str, int]]:
    segments: List[Dict[str, int]] = []
    start_idx: Optional[int] = None
    for idx in range(1, len(df)):
        diff = df.loc[idx, "close_diff"]
        if diff > 0:
            if start_idx is None:
                start_idx = idx - 1
        else:
            if start_idx is not None and idx - 1 > start_idx:
                end_idx = idx - 1
                trend = df.loc[start_idx, "trend_type"]
                start_price = df.loc[start_idx, "close"]
                end_price = df.loc[end_idx, "close"]
                change_pct = (end_price / start_price - 1) * 100
                segments.append(
                    {
                        "start_index": start_idx,
                        "end_index": end_idx,
                        "length": end_idx - start_idx + 1,
                        "trend": trend,
                        "price_change_pct": change_pct,
                    }
                )
                start_idx = None
            else:
                start_idx = None
    if start_idx is not None and start_idx < len(df) - 1:
        end_idx = len(df) - 1
        start_price = df.loc[start_idx, "close"]
        end_price = df.loc[end_idx, "close"]
        change_pct = (end_price / start_price - 1) * 100
        trend = df.loc[start_idx, "trend_type"]
        segments.append(
            {
                "start_index": start_idx,
                "end_index": end_idx,
                "length": end_idx - start_idx + 1,
                "trend": trend,
                "price_change_pct": change_pct,
            }
        )
    return segments


def build_profiles(df: pd.DataFrame, segments: List[Dict[str, int]]) -> pd.DataFrame:
    records: List[Dict[str, float]] = []
    for seg in segments:
        end_idx = seg["end_index"]
        if end_idx < 5:
            continue
        row = df.loc[end_idx]
        if any(pd.isna(row[feature]) for feature in ["price_speed_5", "distance_pct", "angle_ema20_deg"]):
            continue
        records.append(
            {
                "trend": seg["trend"],
                "start_time": df.loc[seg["start_index"], "datetime"].isoformat(),
                "end_time": row["datetime"].isoformat(),
                "length": seg["length"],
                "price_change_pct": seg["price_change_pct"],
                "price_speed_5": row["price_speed_5"],
                "distance_pct": row["distance_pct"],
                "angle_ema20_deg": row["angle_ema20_deg"],
            }
        )
    return pd.DataFrame(records)


def summarize_profiles(df_profiles: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for trend in [TREND_UP, TREND_DOWN, TREND_SIDE]:
        subset = df_profiles[df_profiles["trend"] == trend]
        if subset.empty:
            summary[trend] = {
                "count": 0,
                "price_speed_mean": 0.0,
                "price_speed_std": 0.0,
                "distance_mean": 0.0,
                "distance_std": 0.0,
                "angle_mean": 0.0,
                "angle_std": 0.0,
                "length_mean": 0.0,
                "length_std": 0.0,
            }
            continue
        summary[trend] = {
            "count": int(len(subset)),
            "price_speed_mean": float(subset["price_speed_5"].mean()),
            "price_speed_std": float(subset["price_speed_5"].std(ddof=0)),
            "distance_mean": float(subset["distance_pct"].mean()),
            "distance_std": float(subset["distance_pct"].std(ddof=0)),
            "angle_mean": float(subset["angle_ema20_deg"].mean()),
            "angle_std": float(subset["angle_ema20_deg"].std(ddof=0)),
            "length_mean": float(subset["length"].mean()),
            "length_std": float(subset["length"].std(ddof=0)),
        }
    return summary


def main() -> None:
    price_df = load_price_data()
    price_df = compute_trend_and_features(price_df)
    segments = find_upward_segments(price_df)
    profiles_df = build_profiles(price_df, segments)

    profiles_df.to_csv(OUTPUT_CSV, index=False)
    summary = summarize_profiles(profiles_df)
    payload = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "period": f"{START_DATE.strftime('%Y-%m-%d')} -> {END_DATE.strftime('%Y-%m-%d')}",
        "profiles_count": int(len(profiles_df)),
        "summary": summary,
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved {len(profiles_df)} profiles to {OUTPUT_CSV}")
    print(json.dumps(payload, indent=2))
    print(f"Summary saved to {OUTPUT_JSON.resolve()}")


if __name__ == "__main__":
    main()
