#!/usr/bin/env python3
"""Count downward impulses per trend type for BTC/USDT using shared settings."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import ccxt
import numpy as np
import pandas as pd

from spot_pipeline_settings import END_DATE, START_DATE, SYMBOL, TIMEFRAME, cache_file, summary_path

TREND_DOWN = "downtrend"
TREND_UP = "uptrend"
TREND_SIDE = "sideways"

@dataclass
class Segment:
    start_index: int
    end_index: int
    length: int
    trend: str
    price_change_pct: float


def fetch_ohlcv() -> pd.DataFrame:
    cache = cache_file()
    if cache.exists():
        return pd.read_parquet(cache)

    exchange = ccxt.binance({"enableRateLimit": True, "options": {"adjustForTimeDifference": True}})
    since = int(START_DATE.timestamp() * 1000)
    end_ms = int(END_DATE.timestamp() * 1000)
    timeframe_ms = int(exchange.parse_timeframe(TIMEFRAME) * 1000)

    all_rows: List[List[float]] = []
    while since <= end_ms:
        batch = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since, limit=1000)
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since:
            since += timeframe_ms
        else:
            since = last_ts + timeframe_ms
        if last_ts >= end_ms:
            break

    if not all_rows:
        raise RuntimeError("No OHLCV data downloaded")

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates(subset="timestamp")
    df = df[(df["timestamp"] >= int(START_DATE.timestamp() * 1000)) & (df["timestamp"] <= int(END_DATE.timestamp() * 1000))]
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def compute_trend_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_20"] = df["close"].ewm(span=20).mean()
    df["ema_50"] = df["close"].ewm(span=50).mean()
    df["ema_100"] = df["close"].ewm(span=100).mean()

    df["ema20_slope"] = df["ema_20"].diff()
    df["ema50_slope"] = df["ema_50"].diff()

    conditions = []
    for _, row in df.iterrows():
        ema20 = row["ema_20"]
        ema50 = row["ema_50"]
        ema100 = row["ema_100"]
        slope20 = row["ema20_slope"]
        slope50 = row["ema50_slope"]
        if pd.isna(ema100):
            conditions.append(TREND_SIDE)
            continue
        if ema20 < ema50 < ema100 and slope20 < 0 and slope50 < 0:
            conditions.append(TREND_DOWN)
        elif ema20 > ema50 > ema100 and slope20 > 0 and slope50 > 0:
            conditions.append(TREND_UP)
        else:
            conditions.append(TREND_SIDE)
    df["trend_type"] = conditions
    df["close_diff"] = df["close"].diff()
    return df


def find_downward_segments(df: pd.DataFrame) -> List[Segment]:
    segments: List[Segment] = []
    start_idx: Optional[int] = None
    for idx in range(1, len(df)):
        diff = df.loc[idx, "close_diff"]
        if diff < 0:
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
                    Segment(
                        start_index=start_idx,
                        end_index=end_idx,
                        length=end_idx - start_idx + 1,
                        trend=trend,
                        price_change_pct=change_pct,
                    )
                )
                start_idx = None
            else:
                start_idx = None
    if start_idx is not None and start_idx < len(df) - 1:
        end_idx = len(df) - 1
        trend = df.loc[start_idx, "trend_type"]
        start_price = df.loc[start_idx, "close"]
        end_price = df.loc[end_idx, "close"]
        change_pct = (end_price / start_price - 1) * 100
        segments.append(
            Segment(
                start_index=start_idx,
                end_index=end_idx,
                length=end_idx - start_idx + 1,
                trend=trend,
                price_change_pct=change_pct,
            )
        )
    return segments


def categorize_segments(segments: List[Segment]) -> Counter:
    counts = Counter()
    details = {
        TREND_DOWN: [],
        TREND_UP: [],
        TREND_SIDE: [],
    }
    for seg in segments:
        if seg.trend == TREND_DOWN:
            counts["downtrend_impulses"] += 1
            details[TREND_DOWN].append(seg.price_change_pct)
        elif seg.trend == TREND_UP:
            counts["uptrend_corrections"] += 1
            details[TREND_UP].append(seg.price_change_pct)
        else:
            counts["sideways_down_moves"] += 1
            details[TREND_SIDE].append(seg.price_change_pct)
    stats = {
        key: {
            "count": counts.get(name, 0) if isinstance(name, str) else counts.get(key, 0),
            "avg_change_pct": float(np.mean(vals)) if vals else 0.0,
            "median_change_pct": float(np.median(vals)) if vals else 0.0,
        }
        for key, vals, name in [
            ("downtrend_impulses", details[TREND_DOWN], "downtrend_impulses"),
            ("uptrend_corrections", details[TREND_UP], "uptrend_corrections"),
            ("sideways_down_moves", details[TREND_SIDE], "sideways_down_moves"),
        ]
    }
    return counts, stats


def main() -> None:
    df = fetch_ohlcv()
    df = compute_trend_labels(df)
    segments = find_downward_segments(df)
    counts, stats = categorize_segments(segments)

    summary = {
        "symbol": SYMBOL,
        "timeframe": TIMEFRAME,
        "period": f"{START_DATE.strftime('%Y-%m-%d')} -> {END_DATE.strftime('%Y-%m-%d')}",
        "total_segments": len(segments),
        "counts": counts,
        "stats": stats,
    }

    output_path = summary_path("downward_impulses_summary.json")
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Summary saved to {output_path.resolve()}")


if __name__ == "__main__":
    main()
