#!/usr/bin/env python3
"""Utility helpers for reading and writing the trading bot configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

CONFIG_PATH = Path("bot_config.json")

DEFAULT_CONFIG: Dict[str, Any] = {
    "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT"],
    "update_interval": 5,
    "max_positions": 5,
    "max_drawdown_percent": 20,
}


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from JSON and ensure required defaults are present."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} is missing.")

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to load configuration: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Configuration root must be a JSON object.")

    merged = {**DEFAULT_CONFIG, **data}
    return merged


def save_config(config: Dict[str, Any], path: Path = CONFIG_PATH) -> None:
    """Persist configuration to disk in UTF-8 JSON format."""
    try:
        payload = json.dumps(config, indent=2, ensure_ascii=False)
        path.write_text(payload, encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(f"Failed to persist configuration: {exc}") from exc


def get_config_value(key: str, default: Any = None, *, path: Path = CONFIG_PATH) -> Any:
    """Return a configuration value while swallowing load errors."""
    try:
        config = load_config(path)
    except Exception:
        return default
    return config.get(key, default)
