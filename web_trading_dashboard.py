#!/usr/bin/env python3
"""Web dashboard for monitoring and controlling OptimizedMLBot."""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import ccxt
import numpy as np
import pandas as pd
import psutil
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from advanced_ml_trainer import AdvancedMLTrainer
from spot_signal_scoring import load_default_scorer

logger = logging.getLogger(__name__)

try:
    from optimized_ml_bot import OptimizedMLBot as _RealOptimizedMLBot
except Exception:  # pragma: no cover - fallback for development
    class _RealOptimizedMLBot:  # type: ignore[override]
        """Minimal fallback implementation used when the real bot is unavailable."""

        def run_cycle(self) -> None:
            print("Fallback OptimizedMLBot cycle executed")


try:
    from bot_config import load_config as _load_config  # type: ignore
except Exception:  # pragma: no cover - fallback for development
    def _load_config() -> Dict[str, Any]:
        return {
            "binance_api": {
                "api_key": "",
                "secret_key": "",
            },
            "risk_management": {
                "max_daily_loss": 5,
                "max_open_positions": 5,
                "min_position_size": 0.001,
                "max_position_size": 0.03,
                "emergency_stop": True,
                "emergency_drawdown": 15,
            },
        }


class WebTradingDashboard:
    """Flask + Socket.IO dashboard for the OptimizedML bot."""

    def __init__(self) -> None:
        self.app = Flask(__name__)
        self.app.secret_key = "replace-me"
        self.socketio = SocketIO(self.app, async_mode="threading")

        self.bot: Optional[_RealOptimizedMLBot] = None
        self.bot_thread: Optional[threading.Thread] = None
        self.bot_running = False

        self.config = _load_config()
        self.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT", "BNB/USDT"]
        self.update_interval = 5

        try:
            self.scorer = load_default_scorer()
        except Exception as exc:
            logger.warning("Failed to load impulse scorer: %s", exc)
            self.scorer = None

        self.ml_trainer = AdvancedMLTrainer()
        if not self.ml_trainer._ensure_models_loaded():  # type: ignore[attr-defined]
            logger.warning("ML models are not loaded; backtests will fail until training runs")

        self.setup_routes()
        self.setup_socket_events()
    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_iso_datetime(value: Optional[str], end_of_day: bool = False) -> datetime:
        if not value:
            raise ValueError("Date value is required")
        raw_value = value.strip()
        iso_value = raw_value[:-1] + "+00:00" if raw_value.endswith("Z") else raw_value
        try:
            parsed = datetime.fromisoformat(iso_value)
        except ValueError as exc:
            raise ValueError(f"Invalid date format: {value}") from exc
        if parsed.tzinfo is not None:
            parsed = parsed.replace(tzinfo=None)
        if len(raw_value) == 10 and end_of_day:
            parsed = parsed.replace(hour=23, minute=59, second=59)
        return parsed

    # ------------------------------------------------------------------
    # Flask routes
    # ------------------------------------------------------------------
    def setup_routes(self) -> None:
        @self.app.route("/")
        def index() -> str:
            return render_template("dashboard.html", symbols=self.symbols, config=self.config)

        @self.app.route("/api/bot/status")
        def bot_status() -> Any:
            return jsonify(
                {
                    "running": self.bot_running,
                    "uptime": self.get_bot_uptime() if self.bot_running else 0,
                    "cpu_usage": psutil.cpu_percent(),
                    "memory_usage": psutil.virtual_memory().percent,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        @self.app.route("/api/bot/start", methods=["POST"])
        def start_bot() -> Any:
            if not self.bot_running:
                try:
                    self.start_bot_thread()
                    return jsonify({"success": True, "message": "Bot started"})
                except Exception as exc:  # pragma: no cover - safety net
                    return jsonify({"success": False, "error": str(exc)})
            return jsonify({"success": False, "error": "Bot already running"})

        @self.app.route("/api/bot/stop", methods=["POST"])
        def stop_bot() -> Any:
            if self.bot_running:
                try:
                    self.stop_bot_thread()
                    return jsonify({"success": True, "message": "Bot stopped"})
                except Exception as exc:  # pragma: no cover - safety net
                    return jsonify({"success": False, "error": str(exc)})
            return jsonify({"success": False, "error": "Bot is not running"})

        @self.app.route("/api/market/data")
        def market_data() -> Any:
            try:
                config = _load_config()
                exchange = ccxt.binance(
                    {
                        "apiKey": config["binance_api"].get("api_key", ""),
                        "secret": config["binance_api"].get("secret_key", ""),
                        "enableRateLimit": True,
                    }
                )
                symbol = request.args.get("symbol", "BTC/USDT")
                timeframe = request.args.get("timeframe", "1h")
                limit = int(request.args.get("limit", 200))

                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv:
                    return jsonify({"success": False, "error": "No market data"})

                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df["ema20"] = df["close"].ewm(span=20).mean()

                return jsonify(
                    {
                        "success": True,
                        "symbol": symbol,
                        "data": df.to_dict("records"),
                    }
                )
            except Exception as exc:  # pragma: no cover - defensive
                return jsonify({"success": False, "error": str(exc)})

        @self.app.route("/api/backtest", methods=["POST"])
        def run_backtest() -> Any:
            payload = request.get_json(silent=True) or {}
            symbol = payload.get("symbol")
            start_str = payload.get("start_date")
            end_str = payload.get("end_date")
            timeframe = payload.get("timeframe", "1h")

            if not symbol or not start_str or not end_str:
                return jsonify({"success": False, "error": "symbol, start_date and end_date are required"}), 400

            try:
                start_date = self._parse_iso_datetime(start_str, end_of_day=False)
                end_date = self._parse_iso_datetime(end_str, end_of_day=True)
            except ValueError as exc:
                return jsonify({"success": False, "error": str(exc)}), 400

            try:
                entry_threshold = float(payload.get("entry_threshold", 0.6))
                exit_threshold = float(payload.get("exit_threshold", 0.6))
            except (TypeError, ValueError):
                return jsonify({"success": False, "error": "Invalid threshold values"}), 400

            entry_threshold = min(max(entry_threshold, 0.0), 1.0)
            exit_threshold = min(max(exit_threshold, 0.0), 1.0)

            result = self.ml_trainer.backtest_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                timeframe=timeframe,
                entry_threshold=entry_threshold,
                exit_threshold=exit_threshold,
            )

            if not result:
                return jsonify({"success": False, "error": "Backtest failed or returned no data"}), 500

            phase_records = result.get('phase_records', [])
            impulse_total = sum(1 for record in phase_records if record.get('phase') == 'impulse')
            impulse_ok = sum(1 for record in phase_records if record.get('phase') == 'impulse' and record.get('match'))
            correction_total = sum(1 for record in phase_records if record.get('phase') == 'correction')
            correction_ok = sum(1 for record in phase_records if record.get('phase') == 'correction' and record.get('match'))
            result['phase_summary'] = {
                'impulse_total': impulse_total,
                'impulse_ok': impulse_ok,
                'correction_total': correction_total,
                'correction_ok': correction_ok,
            }

            return jsonify({"success": True, "result": result})

        @self.app.route("/api/signals/current")
        def current_signals() -> Any:
            try:
                if self.scorer is None:
                    raise RuntimeError("Impulse scorer not initialized")

                config = _load_config()
                exchange = ccxt.binance(
                    {
                        "apiKey": config["binance_api"].get("api_key", ""),
                        "secret": config["binance_api"].get("secret_key", ""),
                        "enableRateLimit": True,
                    }
                )

                signals: Dict[str, Dict[str, Any]] = {}
                for symbol in self.symbols:
                    try:
                        ohlcv = exchange.fetch_ohlcv(symbol, "1h", limit=200)
                        if not ohlcv or len(ohlcv) < 60:
                            raise ValueError("Not enough candles")

                        df = pd.DataFrame(
                            ohlcv,
                            columns=["timestamp", "open", "high", "low", "close", "volume"],
                        )
                        df["ema20"] = df["close"].ewm(span=20).mean()
                        df["price_speed_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
                        df["distance_pct"] = (df["close"] - df["ema20"]) / df["ema20"]
                        slope = (df["ema20"] - df["ema20"].shift(5)) / 5
                        df["angle_ema20_deg"] = np.degrees(np.arctan(slope))
                        df["close_diff"] = df["close"].diff()

                        down_length = 0
                        up_length = 0
                        for diff in df["close_diff"].iloc[:-1]:
                            if diff < 0:
                                down_length = down_length + 1 if down_length > 0 else 1
                                up_length = 0
                            elif diff > 0:
                                up_length = up_length + 1 if up_length > 0 else 1
                                down_length = 0
                            else:
                                down_length = down_length + 1 if down_length > 0 else 0
                                up_length = up_length + 1 if up_length > 0 else 0
                        last_diff = df["close_diff"].iloc[-1]
                        if last_diff < 0:
                            down_length = down_length + 1 if down_length > 0 else 1
                            up_length = 0
                        elif last_diff > 0:
                            up_length = up_length + 1 if up_length > 0 else 1
                            down_length = 0
                        else:
                            down_length = down_length + 1 if down_length > 0 else 0
                            up_length = up_length + 1 if up_length > 0 else 0

                        row = df.iloc[-1]
                        price = float(row["close"])
                        swing_low = float(row["low"])
                        swing_high = float(row["high"])

                        entry_features = {
                            "price_speed_5": float(row["price_speed_5"]),
                            "distance_pct": float(row["distance_pct"]),
                            "angle_ema20_deg": float(row["angle_ema20_deg"]),
                            "length": float(down_length),
                        }
                        exit_features = {
                            "price_speed_5": float(row["price_speed_5"]),
                            "distance_pct": float(row["distance_pct"]),
                            "angle_ema20_deg": float(row["angle_ema20_deg"]),
                            "length": float(up_length),
                        }

                        entry_score = self.scorer.score_long_entry(entry_features, price=price, swing_low=swing_low)
                        exit_score = self.scorer.score_long_exit(exit_features, price=price, swing_high=swing_high)

                        signal_type = "HOLD"
                        probability = max(entry_score.numeric_weight, exit_score.numeric_weight)
                        if entry_score.numeric_weight >= 0.66 and entry_score.numeric_weight >= exit_score.numeric_weight:
                            signal_type = "BUY"
                            probability = entry_score.numeric_weight
                        elif exit_score.numeric_weight >= 0.66 and exit_score.numeric_weight > entry_score.numeric_weight:
                            signal_type = "SELL"
                            probability = exit_score.numeric_weight

                        signals[symbol] = {
                            "signal": signal_type,
                            "probability": round(probability, 3),
                            "price": round(price, 2),
                            "entry": {
                                "confidence": entry_score.confidence,
                                "numeric_weight": entry_score.numeric_weight,
                                "composite_z": entry_score.composite_z,
                                "stop_loss": float(entry_score.stop_loss) if entry_score.stop_loss is not None else None,
                                "take_profit": float(entry_score.take_profit) if entry_score.take_profit is not None else None,
                                "features": {k: float(v) for k, v in entry_features.items()},
                                "z_scores": {k: float(v) for k, v in entry_score.z_scores.items()},
                            },
                            "exit": {
                                "confidence": exit_score.confidence,
                                "numeric_weight": exit_score.numeric_weight,
                                "composite_z": exit_score.composite_z,
                                "stop_loss": float(exit_score.stop_loss) if exit_score.stop_loss is not None else None,
                                "take_profit": float(exit_score.take_profit) if exit_score.take_profit is not None else None,
                                "features": {k: float(v) for k, v in exit_features.items()},
                                "z_scores": {k: float(v) for k, v in exit_score.z_scores.items()},
                            },
                        }
                    except Exception as symbol_error:
                        logger.warning("Failed to build signal for %s: %s", symbol, symbol_error)
                        signals[symbol] = {
                            "signal": "ERROR",
                            "probability": 0.0,
                            "price": 0.0,
                            "error": str(symbol_error),
                        }

                return jsonify({"success": True, "signals": signals})
            except Exception as exc:
                logger.exception("Failed to fetch signals: %s", exc)
                return jsonify({"success": False, "error": str(exc)})

        @self.app.route("/api/statistics")
        def statistics() -> Any:
            try:
                config = _load_config()
                exchange = ccxt.binance(
                    {
                        "apiKey": config["binance_api"].get("api_key", ""),
                        "secret": config["binance_api"].get("secret_key", ""),
                        "enableRateLimit": True,
                    }
                )

                total_trades = 0
                total_profit = 0.0
                winning_trades = 0
                current_positions = 0
                total_balance = 0.0

                try:
                    balances = exchange.fetch_balance()
                    total_balance = float(balances.get("total", {}).get("USDT", 0.0))
                    for symbol in self.symbols:
                        base = symbol.split("/")[0]
                        if float(balances.get("total", {}).get(base, 0.0)) > 0:
                            current_positions += 1
                except Exception:
                    pass

                for symbol in self.symbols[:3]:
                    try:
                        ohlcv = exchange.fetch_ohlcv(symbol, "1d", limit=60)
                        if not ohlcv:
                            continue
                        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
                        df["ema20"] = df["close"].ewm(span=20).mean()
                        df["signal"] = "HOLD"
                        df.loc[df["close"] > df["ema20"], "signal"] = "BUY"
                        df.loc[df["close"] < df["ema20"], "signal"] = "SELL"

                        in_position = False
                        entry_price = 0.0
                        for _, row in df.iterrows():
                            if row["signal"] == "BUY" and not in_position:
                                in_position = True
                                entry_price = float(row["close"])
                            elif row["signal"] == "SELL" and in_position:
                                in_position = False
                                exit_price = float(row["close"])
                                profit = (exit_price - entry_price) / max(entry_price, 1e-8)
                                total_profit += profit
                                total_trades += 1
                                if profit > 0:
                                    winning_trades += 1
                    except Exception:
                        continue

                win_rate = (winning_trades / total_trades) * 100 if total_trades else 0.0
                stats = {
                    "total_trades": total_trades,
                    "win_rate": round(win_rate, 2),
                    "total_profit": round(total_profit * 100, 2),
                    "current_positions": current_positions,
                    "total_balance": round(total_balance, 2),
                }

                return jsonify({"success": True, "statistics": stats})
            except Exception as exc:  # pragma: no cover
                return jsonify({"success": False, "error": str(exc)})

        @self.app.route("/api/risk/settings")
        def risk_settings() -> Any:
            try:
                config = _load_config()
                return jsonify({"success": True, "risk_settings": config.get("risk_management", {})})
            except Exception as exc:
                return jsonify({"success": False, "error": str(exc)})
    # ------------------------------------------------------------------
    # Socket.IO events
    # ------------------------------------------------------------------
    def setup_socket_events(self) -> None:
        @self.socketio.on("connect")
        def handle_connect() -> None:
            emit("status", {"message": "Connected to WebTradingDashboard"})
            self.emit_dashboard_update()

        @self.socketio.on("disconnect")
        def handle_disconnect() -> None:  # pragma: no cover - logging only
            print("Client disconnected from dashboard")

        @self.socketio.on("request_update")
        def handle_request_update(_: Any) -> None:
            self.emit_dashboard_update()

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def emit_dashboard_update(self) -> None:
        try:
            payload = {
                "running": self.bot_running,
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "timestamp": datetime.now().isoformat(),
            }
            self.socketio.emit("dashboard_update", payload)
        except Exception as exc:  # pragma: no cover
            print(f"Failed to emit dashboard update: {exc}")

    def start_bot_thread(self) -> None:
        if self.bot_thread and self.bot_thread.is_alive():
            return

        self.bot_running = True
        self.bot = _RealOptimizedMLBot()
        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start_time = time.time()  # type: ignore[attr-defined]
        self.bot_thread.start()

    def stop_bot_thread(self) -> None:
        self.bot_running = False
        if self.bot_thread and self.bot_thread.is_alive():
            self.bot_thread.join(timeout=10)
            if self.bot_thread.is_alive():
                print("Dashboard was unable to stop bot thread cleanly")

    def _run_bot(self) -> None:
        try:
            while self.bot_running and self.bot is not None:
                try:
                    self.bot.run_cycle()
                    self.emit_dashboard_update()
                    time.sleep(self.update_interval)
                except Exception as exc:  # pragma: no cover - runtime guard
                    print(f"Bot cycle error: {exc}")
                    time.sleep(5)
        finally:
            self.bot_running = False

    def get_bot_uptime(self) -> float:
        if self.bot_thread and hasattr(self.bot_thread, "start_time"):
            return time.time() - getattr(self.bot_thread, "start_time")
        return 0.0
    def create_templates(self) -> None:
        base_dir = Path(__file__).resolve().parent
        templates_dir = base_dir / "templates"
        static_dir = base_dir / "static"
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)

        symbols_json = json.dumps(self.symbols)
        dashboard_html = """<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
    <title>Optimized ML Bot Dashboard</title>
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css\">
    <script src=\"https://cdn.jsdelivr.net/npm/axios@1.6.7/dist/axios.min.js\" defer></script>
    <script src=\"https://cdn.socket.io/4.7.5/socket.io.min.js\" defer></script>
    <script src=\"https://cdn.plot.ly/plotly-2.32.0.min.js\" defer></script>
</head>
<body class=\"bg-light\">
<div class=\"container py-4\">
    <h1 class=\"mb-4\">Optimized ML Bot Dashboard</h1>

    <div class=\"mb-3\">
        <button id=\"startBot\" class=\"btn btn-success me-2\">Start Bot</button>
        <button id=\"stopBot\" class=\"btn btn-danger\">Stop Bot</button>
        <span id=\"botStatus\" class=\"ms-3 fw-bold\">Status: unknown</span>
    </div>

    <div class=\"row\" id=\"statisticsContent\"></div>

    <div class=\"mt-4\">
        <h3>Signals</h3>
        <div id=\"signalsList\"></div>
    </div>

    <div class=\"mt-4\">
        <h3>Price Chart</h3>
        <div id=\"priceChart\"></div>
    </div>

    <div class=\"mt-5\">
        <h3>Run Backtest</h3>
        <form id=\"backtestForm\" class=\"row gy-2 gx-3 align-items-end\">
            <div class=\"col-sm-4 col-md-3\">
                <label for=\"backtestSymbol\" class=\"form-label\">Symbol</label>
                <select id=\"backtestSymbol\" class=\"form-select\"></select>
            </div>
            <div class=\"col-sm-4 col-md-3\">
                <label for=\"backtestStart\" class=\"form-label\">Start date</label>
                <input type=\"date\" id=\"backtestStart\" class=\"form-control\" required>
            </div>
            <div class=\"col-sm-4 col-md-3\">
                <label for=\"backtestEnd\" class=\"form-label\">End date</label>
                <input type=\"date\" id=\"backtestEnd\" class=\"form-control\" required>
            </div>
            <div class=\"col-sm-4 col-md-2\">
                <label for=\"backtestTimeframe\" class=\"form-label\">Timeframe</label>
                <select id=\"backtestTimeframe\" class=\"form-select\">
                    <option value=\"1h\">1h</option>
                    <option value=\"4h\">4h</option>
                    <option value=\"1d\">1d</option>
                </select>
            </div>
            <div class=\"col-sm-4 col-md-2\">
                <label for=\"entryThreshold\" class=\"form-label\">Entry >=</label>
                <input type=\"number\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.6\" id=\"entryThreshold\" class=\"form-control\">
            </div>
            <div class=\"col-sm-4 col-md-2\">
                <label for=\"exitThreshold\" class=\"form-label\">Exit >=</label>
                <input type=\"number\" min=\"0\" max=\"1\" step=\"0.05\" value=\"0.6\" id=\"exitThreshold\" class=\"form-control\">
            </div>
            <div class=\"col-sm-4 col-md-2\">
                <button type=\"submit\" class=\"btn btn-primary w-100 mt-2\">Run</button>
            </div>
        </form>
        <div id=\"backtestResult\" class=\"mt-3\"></div>
        <div id=\"equityChart\" class=\"mt-3\"></div>
    </div>
</div>

<script>
document.addEventListener('DOMContentLoaded', function () {
    const SYMBOLS = __SYMBOLS_JSON__;
    const socket = io();

    const symbolSelect = document.getElementById('backtestSymbol');
    symbolSelect.innerHTML = SYMBOLS.map(symbol => `<option value="${symbol}">${symbol}</option>`).join('');

    socket.on('dashboard_update', function (payload) {
        const statusText = payload.running ? 'running' : 'stopped';
        document.getElementById('botStatus').textContent = `Status: ${statusText}`;
    });

    document.getElementById('startBot').addEventListener('click', function () {
        fetch('/api/bot/start', { method: 'POST' });
    });

    document.getElementById('stopBot').addEventListener('click', function () {
        fetch('/api/bot/stop', { method: 'POST' });
    });

    document.getElementById('backtestForm').addEventListener('submit', function (event) {
        event.preventDefault();
        runBacktest();
    });

    function runBacktest() {
        const payload = {
            symbol: document.getElementById('backtestSymbol').value,
            start_date: document.getElementById('backtestStart').value,
            end_date: document.getElementById('backtestEnd').value,
            timeframe: document.getElementById('backtestTimeframe').value,
            entry_threshold: parseFloat(document.getElementById('entryThreshold').value),
            exit_threshold: parseFloat(document.getElementById('exitThreshold').value)
        };

        const resultContainer = document.getElementById('backtestResult');
        resultContainer.innerHTML = '<div class="alert alert-info">Running backtest...</div>';

        fetch('/api/backtest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        })
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    resultContainer.innerHTML = `<div class="alert alert-danger">${data.error || 'Backtest failed'}</div>`;
                    document.getElementById('equityChart').innerHTML = '';
                    return;
                }
                renderBacktestResult(data.result);
            })
            .catch(error => {
                resultContainer.innerHTML = `<div class="alert alert-danger">${error}</div>`;
                document.getElementById('equityChart').innerHTML = '';
            });
    }

    function renderBacktestResult(result) {
        const container = document.getElementById('backtestResult');
        const trades = result.trades || [];
        container.innerHTML = `
            <div class="card">
                <div class="card-body">
                    <h5 class="card-title">${result.symbol} (${result.timeframe})</h5>
                    <p class="card-text">Period: ${result.start_date} -> ${result.end_date}</p>
                    <div class="row">
                        <div class="col-md-3"><strong>Total return:</strong> ${result.total_return_pct}%</div>
                        <div class="col-md-3"><strong>Max drawdown:</strong> ${result.max_drawdown_pct}%</div>
                        <div class="col-md-3"><strong>Trades:</strong> ${result.trades_count}</div>
                        <div class="col-md-3"><strong>Win rate:</strong> ${result.win_rate_pct}%</div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-3"><strong>Final capital:</strong> $${result.final_capital}</div>
                        <div class="col-md-3"><strong>Average trade:</strong> ${result.average_trade_return_pct}%</div>
                        <div class="col-md-3"><strong>Entry >=</strong> ${result.entry_threshold}</div>
                        <div class="col-md-3"><strong>Exit >=</strong> ${result.exit_threshold}</div>
                    </div>
                    <div class="row mt-2">
                        <div class="col-md-6"><strong>Impulse matches:</strong> ${result.phase_summary ? `${result.phase_summary.impulse_ok}/${result.phase_summary.impulse_total}` : 'n/a'}</div>
                        <div class="col-md-6"><strong>Correction matches:</strong> ${result.phase_summary ? `${result.phase_summary.correction_ok}/${result.phase_summary.correction_total}` : 'n/a'}</div>
                    </div>
                </div>
            </div>
            <div class="mt-3">
                ${trades.length ? `<h5>Trades</h5>` : '<div class="alert alert-warning">No trades generated</div>'}
                ${trades.length ? `
                    <div class="table-responsive">
                        <table class="table table-sm table-striped">
                            <thead>
                                <tr>
                                    <th>#</th>
                                    <th>Entry</th>
                                    <th>Exit</th>
                                    <th>Entry price</th>
                                    <th>Exit price</th>
                                    <th>Return %</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${trades.map((trade, idx) => `
                                    <tr>
                                        <td>${idx + 1}</td>
                                        <td>${trade.entry_time}</td>
                                        <td>${trade.exit_time}</td>
                                        <td>${trade.entry_price}</td>
                                        <td>${trade.exit_price}</td>
                                        <td class="${trade.return_pct >= 0 ? 'text-success' : 'text-danger'}">${trade.return_pct.toFixed(2)}</td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                ` : ''}
            </div>
        `;

        if (result.equity_curve && result.equity_curve.length > 1) {
            const timestamps = result.equity_curve.map(point => point.timestamp);
            const equity = result.equity_curve.map(point => point.equity);
            Plotly.newPlot('equityChart', [
                {
                    x: timestamps,
                    y: equity,
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Equity'
                }
            ], {
                title: 'Equity Curve',
                xaxis: { title: 'Timestamp' },
                yaxis: { title: 'Equity (USD)' },
                height: 350
            });
        } else {
            document.getElementById('equityChart').innerHTML = '';
        }
    }

    function refreshStatistics() {
        fetch('/api/statistics')
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    return;
                }
                const stats = data.statistics;
                document.getElementById('statisticsContent').innerHTML = `
                    <div class="col-md-3">
                        <div class="card text-bg-primary mb-3">
                            <div class="card-body">
                                <h5 class="card-title">Trades</h5>
                                <p class="card-text h3">${stats.total_trades}</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-bg-success mb-3">
                            <div class="card-body">
                                <h5 class="card-title">Win Rate</h5>
                                <p class="card-text h3">${stats.win_rate}%</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-bg-info mb-3">
                            <div class="card-body">
                                <h5 class="card-title">Profit</h5>
                                <p class="card-text h3">${stats.total_profit}%</p>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card text-bg-warning mb-3">
                            <div class="card-body">
                                <h5 class="card-title">Positions</h5>
                                <p class="card-text h3">${stats.current_positions}</p>
                            </div>
                        </div>
                    </div>`;
            });
    }

    function refreshSignals() {
        fetch('/api/signals/current')
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    return;
                }
                const list = document.getElementById('signalsList');
                const confidenceChart = document.getElementById('confidenceChart');
                list.innerHTML = '';

                const labels = [];
                const entryWeights = [];
                const exitWeights = [];

                Object.entries(data.signals).forEach(([symbol, payload]) => {
                    if (payload.signal === 'ERROR') {
                        list.innerHTML += `<div class="alert alert-warning"><strong>${symbol}</strong>: ${payload.error || 'Unavailable'}</div>`;
                        return;
                    }

                    const entry = payload.entry || {};
                    const exit = payload.exit || {};
                    const features = entry.features || {};
                    const zScores = entry.z_scores || {};

                    const featureRows = Object.keys(features).map((name) => {
                        const value = typeof features[name] === 'number' ? features[name] : 0;
                        const z = typeof zScores[name] === 'number' ? zScores[name] : 0;
                        return `<tr><td>${name}</td><td>${value.toFixed(6)}</td><td>${z.toFixed(3)}</td></tr>`;
                    }).join('');

                    const stopLoss = typeof entry.stop_loss === 'number' ? `$${entry.stop_loss.toFixed(2)}` : '—';
                    const takeProfit = typeof entry.take_profit === 'number' ? `$${entry.take_profit.toFixed(2)}` : '—';
                    const exitStop = typeof exit.stop_loss === 'number' ? `$${exit.stop_loss.toFixed(2)}` : '—';
                    const exitTake = typeof exit.take_profit === 'number' ? `$${exit.take_profit.toFixed(2)}` : '—';

                    const badgeClass = payload.signal === 'BUY' ? 'bg-success' : (payload.signal === 'SELL' ? 'bg-danger' : 'bg-secondary');
                    const entryComposite = typeof entry.composite_z === 'number' ? entry.composite_z : 0;
                    const exitComposite = typeof exit.composite_z === 'number' ? exit.composite_z : 0;
                    const entryWeight = typeof entry.numeric_weight === 'number' ? entry.numeric_weight : 0;
                    const exitWeight = typeof exit.numeric_weight === 'number' ? exit.numeric_weight : 0;

                    list.innerHTML += `
                        <div class="card mb-3">
                            <div class="card-header d-flex justify-content-between align-items-center">
                                <span><strong>${symbol}</strong> — ${payload.signal}</span>
                                <span class="badge ${badgeClass}">Weight ${(payload.probability * 100).toFixed(1)}%</span>
                            </div>
                            <div class="card-body">
                                <p class="mb-2">
                                    Price: $${payload.price.toFixed(2)} |
                                    Entry confidence: ${entry.confidence || 'n/a'} |
                                    Exit confidence: ${exit.confidence || 'n/a'}
                                </p>
                                <p class="mb-2">
                                    Entry stop: ${stopLoss} | Entry take: ${takeProfit}<br/>
                                    Exit stop: ${exitStop} | Exit take: ${exitTake}
                                </p>
                                <div class="table-responsive">
                                    <table class="table table-sm table-bordered">
                                        <thead><tr><th>Feature</th><th>Value</th><th>Z-score</th></tr></thead>
                                        <tbody>${featureRows}</tbody>
                                    </table>
                                </div>
                                <p class="text-muted mb-0">Entry composite Z: ${entryComposite.toFixed(3)} | Exit composite Z: ${exitComposite.toFixed(3)}</p>
                            </div>
                        </div>
                    `;

                    labels.push(symbol);
                    entryWeights.push(entryWeight);
                    exitWeights.push(exitWeight);
                });

                if (labels.length && typeof Plotly !== 'undefined') {
                    Plotly.newPlot('confidenceChart', [
                        { x: labels, y: entryWeights, name: 'Entry Weight', type: 'bar' },
                        { x: labels, y: exitWeights, name: 'Exit Weight', type: 'bar' }
                    ], {
                        title: 'Signal Weights',
                        barmode: 'group',
                        yaxis: { title: 'Weight' },
                        xaxis: { title: 'Symbol' },
                        height: 350
                    });
                } else if (confidenceChart) {
                    confidenceChart.innerHTML = '<p class="text-muted">No signals available</p>';
                }
            });
    }

    function refreshChart() {
        fetch('/api/market/data?symbol=BTC/USDT&timeframe=1h&limit=200')
            .then(response => response.json())
            .then(data => {
                if (!data.success) {
                    return;
                }
                const timestamps = data.data.map(row => row.timestamp);
                const prices = data.data.map(row => row.close);
                const ema20 = data.data.map(row => row.ema20);

                const layout = {
                    title: `${data.symbol} price and EMA20`,
                    xaxis: { title: 'Timestamp' },
                    yaxis: { title: 'Price (USDT)' },
                    height: 400
                };

                Plotly.newPlot('priceChart', [
                    { x: timestamps, y: prices, type: 'scatter', mode: 'lines', name: 'Close' },
                    { x: timestamps, y: ema20, type: 'scatter', mode: 'lines', name: 'EMA20' }
                ], layout);
            });
    }

    refreshStatistics();
    refreshSignals();
    refreshChart();
    setInterval(refreshStatistics, 15000);
    setInterval(refreshSignals, 15000);
    setInterval(refreshChart, 60000);
});
</script>
</body>
</html>
"""
        dashboard_html = dashboard_html.replace("__SYMBOLS_JSON__", symbols_json)
        (templates_dir / "dashboard.html").write_text(dashboard_html, encoding="utf-8")

    def run(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False) -> None:
        print(f"Starting WebTradingDashboard on {host}:{port}")
        self.create_templates()
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


def main() -> None:
    dashboard = WebTradingDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()


