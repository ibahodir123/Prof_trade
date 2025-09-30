#!/usr/bin/env python3
"""Compute z-score weights for downward trend profiles using shared settings."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from spot_pipeline_settings import summary_path

PROFILES_CSV = summary_path("downward_profiles.csv")
OUTPUT_JSON = summary_path("downtrend_weights.json")

PROFILE_FEATURES = ["price_speed_5", "distance_pct", "angle_ema20_deg", "length"]


def compute_weights() -> dict:
    if not PROFILES_CSV.exists():
        raise FileNotFoundError("downward_profiles.csv not found. Run extract_downward_profiles.py first.")

    profiles = pd.read_csv(PROFILES_CSV)
    downtrend = profiles[profiles["trend"] == "downtrend"].copy()

    if downtrend.empty:
        raise RuntimeError("No downtrend profiles found in dataset.")

    stats: dict[str, dict[str, float]] = {}
    for feature in PROFILE_FEATURES:
        mean = downtrend[feature].mean()
        std = downtrend[feature].std(ddof=0)
        stats[feature] = {"mean": float(mean), "std": float(std)}

    profile_records = []
    for _, row in downtrend.iterrows():
        z_scores: dict[str, float] = {}
        for feature in PROFILE_FEATURES:
            mean = stats[feature]["mean"]
            std = stats[feature]["std"] or 1e-9
            z_scores[feature] = float((row[feature] - mean) / std)
        composite = float(np.linalg.norm([z_scores[f] for f in PROFILE_FEATURES]) / len(PROFILE_FEATURES))
        profile_records.append(
            {
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "length": int(row["length"]),
                "price_change_pct": float(row["price_change_pct"]),
                "z_scores": z_scores,
                "composite_z": composite,
            }
        )

    composites = [item["composite_z"] for item in profile_records]
    percentiles = np.percentile(composites, [25, 50, 75])

    for record in profile_records:
        score = record["composite_z"]
        if score <= percentiles[0]:
            record["confidence_label"] = "high"
        elif score <= percentiles[1]:
            record["confidence_label"] = "medium"
        elif score <= percentiles[2]:
            record["confidence_label"] = "low"
        else:
            record["confidence_label"] = "very_low"

    return {
        "features": stats,
        "composite_percentiles": {
            "p25": float(percentiles[0]),
            "p50": float(percentiles[1]),
            "p75": float(percentiles[2]),
        },
        "profiles": profile_records,
    }


def main() -> None:
    result = compute_weights()
    OUTPUT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Computed weights for {len(result['profiles'])} downtrend profiles -> {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
