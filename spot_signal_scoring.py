#!/usr/bin/env python3
"""Utility for scoring long-entry minima and long-exit maxima using four impulse features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from spot_pipeline_settings import summary_path

FeatureMap = Mapping[str, float]


@dataclass(frozen=True)
class ScoreResult:
    composite_z: float
    z_scores: Dict[str, float]
    confidence: str
    numeric_weight: float


@dataclass(frozen=True)
class _ProfileWeights:
    means: Dict[str, float]
    stds: Dict[str, float]
    thresholds: Tuple[float, float, float]


class ImpulseScorer:
    """Scores current impulse state against historical long-only patterns."""

    REQUIRED_FEATURES = (
        "price_speed_5",
        "distance_pct",
        "angle_ema20_deg",
        "length",
    )

    def __init__(self, downtrend_weights: Path, uptrend_weights: Path) -> None:
        self._downtrend = self._load_profile(downtrend_weights)
        self._uptrend = self._load_profile(uptrend_weights)

    def score_long_entry(self, features: FeatureMap) -> ScoreResult:
        """Return confidence that the current state is a long entry minimum."""
        return self._score(features, self._downtrend)

    def score_long_exit(self, features: FeatureMap) -> ScoreResult:
        """Return confidence that the current state is a long exit maximum."""
        return self._score(features, self._uptrend)

    def _score(self, features: FeatureMap, weights: _ProfileWeights) -> ScoreResult:
        values = self._extract_features(features)
        z_scores: Dict[str, float] = {}
        for name, value in values.items():
            mean = weights.means[name]
            std = weights.stds[name] or 1e-9
            z_scores[name] = (value - mean) / std

        norm = math.sqrt(sum(z * z for z in z_scores.values()))
        composite = norm / len(self.REQUIRED_FEATURES)
        confidence = self._label_from_thresholds(composite, weights.thresholds)
        weight = self._confidence_to_weight(confidence)
        return ScoreResult(composite, z_scores, confidence, weight)

    def _extract_features(self, features: FeatureMap) -> Dict[str, float]:
        missing = [name for name in self.REQUIRED_FEATURES if name not in features]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        return {name: float(features[name]) for name in self.REQUIRED_FEATURES}

    @staticmethod
    def _label_from_thresholds(value: float, thresholds: Tuple[float, float, float]) -> str:
        p25, p50, p75 = thresholds
        if value <= p25:
            return "high"
        if value <= p50:
            return "medium"
        if value <= p75:
            return "low"
        return "very_low"

    @staticmethod
    def _confidence_to_weight(confidence: str) -> float:
        mapping = {
            "high": 1.0,
            "medium": 0.66,
            "low": 0.33,
            "very_low": 0.0,
        }
        return mapping.get(confidence, 0.0)

    def _load_profile(self, path: Path) -> _ProfileWeights:
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        features = data.get("features", {})
        composite = data.get("composite_percentiles", {})

        means: Dict[str, float] = {}
        stds: Dict[str, float] = {}
        for name in self.REQUIRED_FEATURES:
            if name not in features:
                raise KeyError(f"Feature '{name}' missing in weights file {path}")
            feature_info = features[name]
            means[name] = float(feature_info.get("mean", 0.0))
            stds[name] = float(feature_info.get("std", 0.0))

        thresholds = (
            float(composite.get("p25", 0.0)),
            float(composite.get("p50", 0.0)),
            float(composite.get("p75", 0.0)),
        )
        return _ProfileWeights(means, stds, thresholds)


def load_default_scorer(root: Path | None = None) -> ImpulseScorer:
    if root is not None:
        downtrend_path = Path(root) / "downtrend_weights.json"
        uptrend_path = Path(root) / "uptrend_weights.json"
    else:
        downtrend_path = summary_path("downtrend_weights.json")
        uptrend_path = summary_path("uptrend_weights.json")
    return ImpulseScorer(downtrend_path, uptrend_path)


__all__ = ["ImpulseScorer", "ScoreResult", "load_default_scorer"]
