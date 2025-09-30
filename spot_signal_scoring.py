#!/usr/bin/env python3
"""Utility for scoring long-entry minima and long-exit maxima using four impulse features."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

from spot_pipeline_settings import FeatureStats, summary_path

FeatureMap = Mapping[str, float]


@dataclass(frozen=True)
class ScoreResult:
    composite_z: float
    z_scores: Dict[str, float]
    confidence: str
    numeric_weight: float
    stop_loss: float | None = None
    take_profit: float | None = None


@dataclass(frozen=True)
class _ProfileWeights:
    means: Dict[str, float]
    stds: Dict[str, float]
    thresholds: Tuple[float, float, float]

    def feature_stats(self, name: str) -> FeatureStats:
        return FeatureStats(self.means[name], self.stds[name])


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

    def score_long_entry(self, features: FeatureMap, *, price: float | None = None, swing_low: float | None = None) -> ScoreResult:
        """Return confidence that the current state is a long entry minimum."""
        base = self._score(features, self._downtrend)
        stop_loss = self._compute_stop_loss(base, price, swing_low)
        take_profit = self._compute_take_profit(base, price)
        return ScoreResult(base.composite_z, base.z_scores, base.confidence, base.numeric_weight, stop_loss, take_profit)

    def score_long_exit(self, features: FeatureMap, *, price: float | None = None, swing_high: float | None = None) -> ScoreResult:
        """Return confidence that the current state is a long exit maximum."""
        base = self._score(features, self._uptrend)
        stop_loss = self._compute_stop_loss(base, price, swing_high, is_exit=True)
        take_profit = self._compute_take_profit(base, price, is_exit=True)
        return ScoreResult(base.composite_z, base.z_scores, base.confidence, base.numeric_weight, stop_loss, take_profit)

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

    def _compute_stop_loss(
        self,
        score: ScoreResult,
        price: float | None,
        swing: float | None,
        *,
        is_exit: bool = False,
        distance_multiplier: float = 1.0,
    ) -> float | None:
        if price is None or swing is None:
            return None
        stats = self._downtrend.feature_stats("distance_pct") if not is_exit else self._uptrend.feature_stats("distance_pct")
        abs_distance = abs(score.z_scores.get("distance_pct", 0.0)) * stats.std + abs(stats.mean)
        length_z = score.z_scores.get("length", 0.0)
        adjustment = abs_distance * (1.0 + 0.2 * length_z)
        if not is_exit:
            level = min(price, swing)
            stop = level * (1 - adjustment * distance_multiplier)
        else:
            level = max(price, swing)
            stop = level * (1 + adjustment * 0.5)
        return max(stop, 0.0)

    def _compute_take_profit(
        self,
        score: ScoreResult,
        price: float | None,
        *,
        is_exit: bool = False,
    ) -> float | None:
        if price is None:
            return None
        stats = self._uptrend.feature_stats("distance_pct") if not is_exit else self._downtrend.feature_stats("distance_pct")
        abs_distance = abs(score.z_scores.get("distance_pct", 0.0)) * stats.std + abs(stats.mean)
        length_z = score.z_scores.get("length", 0.0)
        gain = abs_distance * (1.2 - 0.1 * length_z)
        if not is_exit:
            return price * (1 + gain)
        return price * (1 - gain * 0.5)


def load_default_scorer(root: Path | None = None) -> ImpulseScorer:
    if root is not None:
        downtrend_path = Path(root) / "downtrend_weights.json"
        uptrend_path = Path(root) / "uptrend_weights.json"
    else:
        downtrend_path = summary_path("downtrend_weights.json")
        uptrend_path = summary_path("uptrend_weights.json")
    return ImpulseScorer(downtrend_path, uptrend_path)


__all__ = ["ImpulseScorer", "ScoreResult", "load_default_scorer"]
