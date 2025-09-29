#!/usr/bin/env python3
"""Utility for training, loading and evaluating ML models used by optimized_ml_bot."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class BacktestTrade:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    return_pct: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_price": round(self.exit_price, 4),
            "return_pct": round(self.return_pct, 4),
        }


class AdvancedMLTrainer:
    """Handle loading, training and evaluation for ML models."""

    def __init__(self) -> None:
        self.entry_model: Optional[RandomForestClassifier] = None
        self.exit_model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None

    # ------------------------------------------------------------------
    # Model loading / inference helpers
    # ------------------------------------------------------------------
    def load_models(self) -> bool:
        """Load pre-trained models and scaler from disk."""
        try:
            self.entry_model = joblib.load("models/entry_model.pkl")
            self.exit_model = joblib.load("models/exit_model.pkl")
            self.scaler = joblib.load("models/ema_scaler.pkl")
            self.feature_names = joblib.load("models/feature_names.pkl")
            logger.info("ML models and scaler loaded successfully")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load ML artefacts: %s", exc)
            return False

    def _ensure_models_loaded(self) -> bool:
        if self.entry_model is None or self.exit_model is None or self.scaler is None:
            return self.load_models()
        return True

    def predict_entry_exit(self, features: np.ndarray) -> Tuple[float, float]:
        """Return entry/exit probabilities for a feature vector."""
        if not self._ensure_models_loaded():
            return 0.0, 0.0

        try:
            if features.ndim == 1:
                features = features.reshape(1, -1)

            expected = getattr(self.scaler, "n_features_in_", features.shape[1])
            if features.shape[1] != expected:
                logger.warning(
                    "Feature vector size mismatch: expected %s, got %s",
                    expected,
                    features.shape[1],
                )
                return 0.0, 0.0

            features_scaled = self.scaler.transform(features)

            if hasattr(self.entry_model, "predict_proba") and hasattr(self.exit_model, "predict_proba"):
                entry_prob = float(self.entry_model.predict_proba(features_scaled)[0][1])
                exit_prob = float(self.exit_model.predict_proba(features_scaled)[0][1])
            else:  # pragma: no cover - legacy fallback
                entry_prob = float(self.entry_model.predict(features_scaled)[0])
                exit_prob = float(self.exit_model.predict(features_scaled)[0])

            return max(0.0, min(1.0, entry_prob)), max(0.0, min(1.0, exit_prob))
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Inference failed: %s", exc)
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # Exchange helpers
    # ------------------------------------------------------------------
    def _create_exchange(self):
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover
            logger.error("ccxt is required for market data operations: %s", exc)
            return None

        try:
            return ccxt.binance(
                {
                    "apiKey": "",
                    "secret": "",
                    "enableRateLimit": True,
                    "options": {"adjustForTimeDifference": True},
                }
            )
        except Exception as exc:
            logger.error("Failed to initialise Binance exchange: %s", exc)
            return None

    def _download_symbol_dataframe(
        self,
        exchange,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str,
    ) -> Optional[pd.DataFrame]:
        if exchange is None:
            return None

        timeframe_ms = int(exchange.parse_timeframe(timeframe) * 1000)
        start_ms = int(start_date.timestamp() * 1000)
        end_ms = int(end_date.timestamp() * 1000)

        candles: List[List[Any]] = []
        since = start_ms
        last_timestamp = None
        rate_limit = getattr(exchange, "rateLimit", 1200) / 1000.0

        while since < end_ms:
            try:
                batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            except Exception as exc:  # pragma: no cover - network errors
                logger.warning("Fetch failed for %s @ %s: %s", symbol, since, exc)
                time.sleep(rate_limit)
                continue

            if not batch:
                break

            candles.extend(batch)
            fetched_last = batch[-1][0]

            if last_timestamp is not None and fetched_last <= last_timestamp:
                since = last_timestamp + timeframe_ms
            else:
                since = fetched_last + timeframe_ms
                last_timestamp = fetched_last

            if fetched_last >= end_ms:
                break

            time.sleep(rate_limit)

        if not candles:
            logger.warning("No candles downloaded for %s", symbol)
            return None

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="timestamp")
        df = df[(df["timestamp"] >= start_ms) & (df["timestamp"] <= end_ms)]
        df = df.sort_values("timestamp").reset_index(drop=True)

        if df.empty:
            logger.warning("No candles remain for %s after filtering", symbol)
            return None

        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_100"] = df["close"].ewm(span=100).mean()
        return df

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def collect_historical_data(
        self,
        symbols: List[str],
        days: int = 30,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        timeframe: str = "1h",
    ) -> Optional[List[np.ndarray]]:
        """Fetch historical candles and convert them into feature vectors."""

        end_date = end_date or datetime.utcnow()
        start_date = start_date or (end_date - timedelta(days=days))

        if start_date >= end_date:
            logger.error("Start date must be earlier than end date")
            return None

        exchange = self._create_exchange()
        if exchange is None:
            return None

        logger.info(
            "Collecting %s candles for up to %d symbols between %s and %s",
            timeframe,
            min(len(symbols), 5),
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )

        all_features: List[np.ndarray] = []

        for symbol in symbols[:5]:
            try:
                df = self._download_symbol_dataframe(exchange, symbol, start_date, end_date, timeframe)
                if df is None or len(df) < 60:
                    continue

                symbol_features = 0
                for idx in range(50, len(df)):
                    slice_df = df.iloc[: idx + 1]
                    feature_vector = self.generate_features_from_data(slice_df)
                    if feature_vector is not None:
                        all_features.append(feature_vector)
                        symbol_features += 1

                logger.info("Prepared %d samples for %s", symbol_features, symbol)
            except Exception as exc:
                logger.warning("Failed to process %s: %s", symbol, exc)
                continue

        if not all_features:
            logger.error("Historical feature collection returned no samples")
            return None

        logger.info("Collected %d feature vectors in total", len(all_features))
        return all_features

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def generate_features_from_data(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Transform the most recent row of a price slice into features."""
        try:
            if len(df) < 50:
                return None

            latest = df.iloc[-1]
            ema20 = float(latest.get("ema_20", 0.0))
            ema50 = float(latest.get("ema_50", 0.0))
            ema100 = float(latest.get("ema_100", 0.0))
            close_price = float(latest.get("close", 0.0))

            ema20_prev5 = float(df["ema_20"].iloc[-5])
            ema50_prev5 = float(df["ema_50"].iloc[-5])
            ema100_prev5 = float(df["ema_100"].iloc[-5])
            ema20_prev10 = float(df["ema_20"].iloc[-10])
            close_prev5 = float(df["close"].iloc[-5])
            ema20_prev6 = float(df["ema_20"].iloc[-6])
            ema50_prev6 = float(df["ema_50"].iloc[-6])

            def safe_div(numerator: float, denominator: float) -> float:
                return numerator / denominator if denominator not in (0.0, -0.0) else 0.0

            if ema20 > ema50 > ema100:
                trend_state = 2.0
                trend_direction = 1.0
            elif ema20 < ema50 < ema100:
                trend_state = 0.0
                trend_direction = -1.0
            else:
                trend_state = 1.0
                trend_direction = 0.0

            slope_ema20 = safe_div(ema20 - ema20_prev5, 5.0)
            slope_ema50 = safe_div(ema50 - ema50_prev5, 5.0)
            slope_ema100 = safe_div(ema100 - ema100_prev5, 5.0)
            slope_price = safe_div(close_price - close_prev5, 5.0)

            angle_ema20 = float(np.degrees(np.arctan(slope_ema20)))
            angle_ema50 = float(np.degrees(np.arctan(slope_ema50)))
            angle_ema100 = float(np.degrees(np.arctan(slope_ema100)))

            diff_now = ema20 - ema50
            diff_prev = ema20_prev6 - ema50_prev6
            if trend_direction > 0:
                impulse = angle_ema20 > 0 and diff_now > diff_prev
            elif trend_direction < 0:
                impulse = angle_ema20 < 0 and diff_now < diff_prev
            else:
                impulse = False
            phase_state = 1.0 if impulse else 0.0

            total_ema = ema20 + ema50 + ema100 if (ema20 + ema50 + ema100) != 0 else 1.0
            ema20_weight = ema20 / total_ema
            ema50_weight = ema50 / total_ema
            ema100_weight = ema100 / total_ema

            features = np.array(
                [
                    ema20,
                    ema50,
                    ema100,
                    safe_div(ema20 - ema20_prev5, abs(ema20_prev5) if ema20_prev5 else 1.0),
                    safe_div(ema50 - ema50_prev5, abs(ema50_prev5) if ema50_prev5 else 1.0),
                    safe_div(ema100 - ema100_prev5, abs(ema100_prev5) if ema100_prev5 else 1.0),
                    safe_div(ema20 - ema50, abs(ema50) if ema50 else 1.0),
                    safe_div(ema50 - ema100, abs(ema100) if ema100 else 1.0),
                    safe_div(ema20 - ema100, abs(ema100) if ema100 else 1.0),
                    safe_div(ema20 - ema20_prev10, abs(ema20_prev10) if ema20_prev10 else 1.0),
                    trend_state,
                    safe_div(close_price - close_prev5, abs(close_prev5) if close_prev5 else 1.0),
                    safe_div(close_price - ema20, abs(ema20) if ema20 else 1.0),
                    angle_ema20,
                    angle_ema50,
                    angle_ema100,
                    ema20_weight,
                    ema50_weight,
                    ema100_weight,
                    trend_direction,
                    phase_state,
                ]
            )
            return features
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to build feature vector: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Training pipeline
    # ------------------------------------------------------------------
    def train_models(self, symbols: List[str], start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, timeframe: str = "1h") -> bool:
        """Train entry/exit classifiers on historical data."""
        try:
            start_date = start_date or datetime(2020, 1, 1)
            end_date = end_date or datetime(2024, 12, 31, 23, 59, 59)

            logger.info("Starting model training for period %s -> %s (timeframe=%s)", start_date, end_date, timeframe)
            historical_data = self.collect_historical_data(
                symbols,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
            )
            if historical_data is None:
                logger.error("Training aborted: no historical data collected")
                return False

            X = np.array(historical_data)
            n_samples, n_features = X.shape
            logger.info("Training dataset: %d samples, %d features", n_samples, n_features)
            # --- collect phase weights ---
            impulse_weights = []
            correction_weights = []
            for row in X:
                speed = float(row[11])
                distance = float(row[12])
                angle = float(row[13])
                phase = float(row[20])

                total = abs(speed) + abs(distance) + abs(angle)
                if total == 0.0:
                    weight = (0.0, 0.0, 0.0)
                else:
                    weight = (abs(speed) / total, abs(distance) / total, abs(angle) / total)

                if phase >= 0.5:
                    impulse_weights.append(weight)
                else:
                    correction_weights.append(weight)

            def aggregate(weights):
                if not weights:
                    return {"count": 0}
                import numpy as np
                arr = np.array(weights)
                return {
                    "count": len(weights),
                    "mean": arr.mean(axis=0).tolist(),
                    "median": np.median(arr, axis=0).tolist(),
                    "min": arr.min(axis=0).tolist(),
                    "max": arr.max(axis=0).tolist(),
                }

            impulse_stats = aggregate(impulse_weights)
            correction_stats = aggregate(correction_weights)


            entry_condition = (
                (X[:, 10] == 2)
                & (X[:, 20] == 1)
                & (X[:, 11] > 0.0)
                & (X[:, 12] > 0.0)
                & (X[:, 13] > 0.0)
            )
            y_entry = entry_condition.astype(int)
            y_entry = (y_entry + np.random.choice([0, 1], n_samples, p=[0.3, 0.7])).clip(0, 1)

            exit_condition = (
                (X[:, 10] != 2)
                | (X[:, 20] == 0)
                | (X[:, 11] <= 0.0)
                | (X[:, 12] <= 0.0)
                | (X[:, 13] <= 0.0)
            )
            y_exit = exit_condition.astype(int)
            y_exit = (y_exit + np.random.choice([0, 1], n_samples, p=[0.2, 0.8])).clip(0, 1)

            self.entry_model = RandomForestClassifier(n_estimators=200, random_state=42)
            self.exit_model = RandomForestClassifier(n_estimators=200, random_state=42)
            self.scaler = StandardScaler()

            X_scaled = self.scaler.fit_transform(X)
            self.entry_model.fit(X_scaled, y_entry)
            self.exit_model.fit(X_scaled, y_exit)

            entry_score = float(self.entry_model.score(X_scaled, y_entry))
            exit_score = float(self.exit_model.score(X_scaled, y_exit))
            logger.info("Entry model score: %.4f", entry_score)
            logger.info("Exit model score: %.4f", exit_score)

            training_period = f"{start_date.date()}_{end_date.date()}"
            self.save_models_with_metadata(symbols, n_samples, entry_score, exit_score, training_period, timeframe)
            self.save_phase_stats(impulse_stats, correction_stats, timeframe)
            logger.info("Training finished successfully")
            return True
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Model training failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------
    def backtest_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1h",
        entry_threshold: float = 0.6,
        exit_threshold: float = 0.6,
        initial_capital: float = 1000.0,
    ) -> Optional[Dict[str, Any]]:
        """Run a simple long-only backtest for a symbol using trained models."""
        if start_date >= end_date:
            logger.error("Backtest start date must be earlier than end date")
            return None

        if not self._ensure_models_loaded():
            logger.error("Models must be trained/loaded before backtesting")
            return None

        exchange = self._create_exchange()
        df = self._download_symbol_dataframe(exchange, symbol, start_date, end_date, timeframe)
        if df is None or len(df) < 60:
            logger.error("Not enough data to backtest %s", symbol)
            return None

        capital = initial_capital
        equity_curve: List[Dict[str, Any]] = []
        trades: List[BacktestTrade] = []

        position_open = False
        entry_price = 0.0
        entry_time: Optional[datetime] = None

        for idx in range(50, len(df)):
            slice_df = df.iloc[: idx + 1]
            features = self.generate_features_from_data(slice_df)
            if features is None:
                continue

            entry_prob, exit_prob = self.predict_entry_exit(features)
            price = float(df.iloc[idx]["close"])
            timestamp = datetime.utcfromtimestamp(float(df.iloc[idx]["timestamp"]) / 1000.0)

            current_equity = capital if not position_open else capital * (price / entry_price)
            equity_curve.append({"timestamp": timestamp.isoformat(), "equity": round(current_equity, 2)})

            if not position_open and entry_prob >= entry_threshold:
                position_open = True
                entry_price = price
                entry_time = timestamp
                continue

            if position_open and exit_prob >= exit_threshold:
                trade_return = (price - entry_price) / entry_price
                capital *= 1 + trade_return
                trades.append(
                    BacktestTrade(
                        entry_time=entry_time or timestamp,
                        exit_time=timestamp,
                        entry_price=entry_price,
                        exit_price=price,
                        return_pct=trade_return * 100,
                    )
                )
                position_open = False
                entry_price = 0.0
                entry_time = None
                equity_curve[-1]["equity"] = round(capital, 2)

        if position_open:
            price = float(df.iloc[-1]["close"])
            timestamp = datetime.utcfromtimestamp(float(df.iloc[-1]["timestamp"]) / 1000.0)
            trade_return = (price - entry_price) / entry_price
            capital *= 1 + trade_return
            trades.append(
                BacktestTrade(
                    entry_time=entry_time or timestamp,
                    exit_time=timestamp,
                    entry_price=entry_price,
                    exit_price=price,
                    return_pct=trade_return * 100,
                )
            )
            equity_curve.append({"timestamp": timestamp.isoformat(), "equity": round(capital, 2)})

        if not equity_curve:
            logger.error("Backtest produced no equity data")
            return None

        equity_values = [point["equity"] for point in equity_curve]
        peak = equity_values[0]
        max_drawdown = 0.0
        for value in equity_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        total_return_pct = (capital / initial_capital - 1) * 100
        trade_returns = [trade.return_pct for trade in trades]
        wins = sum(1 for r in trade_returns if r > 0)
        losses = sum(1 for r in trade_returns if r < 0)
        win_rate = (wins / len(trade_returns) * 100) if trade_returns else 0.0
        avg_trade = float(np.mean(trade_returns)) if trade_returns else 0.0

        return {
            "symbol": symbol,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "timeframe": timeframe,
            "entry_threshold": entry_threshold,
            "exit_threshold": exit_threshold,
            "initial_capital": initial_capital,
            "final_capital": round(capital, 2),
            "total_return_pct": round(total_return_pct, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "trades": [trade.to_dict() for trade in trades],
            "trades_count": len(trades),
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(win_rate, 2),
            "average_trade_return_pct": round(avg_trade, 2),
            "equity_curve": equity_curve,
        }

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def save_phase_stats(self, impulse_stats: dict, correction_stats: dict, timeframe: str) -> None:
        """Persist aggregated weights for impulse and correction phases."""
        try:
            os.makedirs("models", exist_ok=True)
            payload = {
                "impulse": impulse_stats,
                "correction": correction_stats,
                "timeframe": timeframe,
            }
            with open("models/phase_weight_stats.json", "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to save phase weight statistics: %s", exc)

    def save_models_with_metadata(
        self,
        symbols: List[str],
        n_samples: int,
        entry_score: float,
        exit_score: float,
        training_period: Optional[str] = None,
        timeframe: str = "1h",
    ) -> None:
        """Persist trained artefacts alongside metadata."""
        try:
            os.makedirs("models", exist_ok=True)

            joblib.dump(self.entry_model, "models/entry_model.pkl")
            joblib.dump(self.exit_model, "models/exit_model.pkl")
            joblib.dump(self.scaler, "models/ema_scaler.pkl")

            feature_names = [
                "ema_20",
                "ema_50",
                "ema_100",
                "ema20_change_5",
                "ema50_change_5",
                "ema100_change_5",
                "ema20_vs_ema50_pct",
                "ema50_vs_ema100_pct",
                "ema20_vs_ema100_pct",
                "ema20_change_10",
                "trend_state",
                "price_speed_5",
                "price_vs_ema20_pct",
                "angle_ema20_deg",
                "angle_ema50_deg",
                "angle_ema100_deg",
                "ema20_weight",
                "ema50_weight",
                "ema100_weight",
                "trend_direction",
                "phase_state",
            ]
            joblib.dump(feature_names, "models/feature_names.pkl")

            metadata: Dict[str, Any] = {
                "training_date": datetime.utcnow().isoformat(),
                "symbols_used": symbols[:5],
                "samples_count": n_samples,
                "features_count": len(feature_names),
                "entry_model_score": entry_score,
                "exit_model_score": exit_score,
                "data_source": "real_binance_historical",
                "training_period": training_period or "unspecified",
                "timeframe": timeframe,
                "model_type": "RandomForestClassifier",
                "scaler_type": "StandardScaler",
            }

            with open("models/training_metadata.json", "w", encoding="utf-8") as handle:
                json.dump(metadata, handle, ensure_ascii=False, indent=2)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to save trained artefacts: %s", exc)

    def load_training_metadata(self) -> Optional[Dict[str, Any]]:
        """Return metadata about the last training run if available."""
        try:
            metadata_path = "models/training_metadata.json"
            if not os.path.exists(metadata_path):
                logger.warning("Training metadata file not found")
                return None

            with open(metadata_path, "r", encoding="utf-8") as handle:
                metadata = json.load(handle)

            logger.info("Training metadata loaded: %s", metadata.get("training_date", "unknown"))
            return metadata
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to load training metadata: %s", exc)
            return None

    def is_model_trained_on_real_data(self) -> bool:
        """Check whether current models were trained on real binance data."""
        metadata = self.load_training_metadata()
        if metadata:
            return metadata.get("data_source") == "real_binance_historical"
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trainer = AdvancedMLTrainer()
    if trainer.load_models():
        print("Models loaded successfully")
    else:
        print("Failed to load models")
