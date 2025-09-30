#!/usr/bin/env python3
"""Shared configuration helpers for the spot long pipeline."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Final

_DEFAULT_START: Final = datetime(2020, 1, 1)
_DEFAULT_END: Final = datetime(2024, 12, 31, 23, 59, 59)
_DEFAULT_TIMEFRAME: Final = "4h"
_DEFAULT_SYMBOL: Final = "BTC/USDT"


def _parse_datetime(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    try:
        if value.endswith("Z"):
            value = value[:-1]
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid datetime value: {value}") from exc


START_DATE: Final = _parse_datetime(os.getenv("SPOT_PIPELINE_START"), _DEFAULT_START)
END_DATE: Final = _parse_datetime(os.getenv("SPOT_PIPELINE_END"), _DEFAULT_END)
TIMEFRAME: Final = os.getenv("SPOT_PIPELINE_TIMEFRAME", _DEFAULT_TIMEFRAME)
SYMBOL: Final = os.getenv("SPOT_PIPELINE_SYMBOL", _DEFAULT_SYMBOL)

_DATA_ROOT = Path(os.getenv("SPOT_PIPELINE_DATA_DIR", ".")).resolve()


def cache_file() -> Path:
    """Return the parquet cache path for the configured symbol/timeframe."""
    start_tag = START_DATE.strftime("%Y%m%d")
    end_tag = END_DATE.strftime("%Y%m%d")
    symbol_tag = SYMBOL.replace("/", "").lower()
    filename = f"data_{symbol_tag}_{start_tag}_{end_tag}_{TIMEFRAME}.parquet"
    return _DATA_ROOT / filename


def summary_path(name: str) -> Path:
    """Return an absolute path for a summary JSON/CSV file."""
    return _DATA_ROOT / name


__all__ = [
    "START_DATE",
    "END_DATE",
    "TIMEFRAME",
    "SYMBOL",
    "cache_file",
    "summary_path",
]
