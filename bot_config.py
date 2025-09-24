#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
⚙️ Конфигурация торгового бота
Загрузка настроек из JSON файла
"""

import json
import os
from pathlib import Path
from typing import Dict, Any

def load_config() -> Dict[str, Any]:
    """Загрузка конфигурации из файла"""
    config_file = Path("bot_config.json")

    if not config_file.exists():
        raise FileNotFoundError("Файл bot_config.json не найден!")

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Добавляем значения по умолчанию
        defaults = {
            'symbols': ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT'],
            'update_interval': 5,
            'max_positions': 5,
            'max_drawdown_percent': 20
        }

        for key, value in defaults.items():
            if key not in config:
                config[key] = value

        return config

    except Exception as e:
        raise Exception(f"Ошибка загрузки конфигурации: {e}")

def save_config(config: Dict[str, Any]) -> None:
    """Сохранение конфигурации в файл"""
    try:
        with open("bot_config.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except Exception as e:
        raise Exception(f"Ошибка сохранения конфигурации: {e}")

def get_config_value(key: str, default: Any = None) -> Any:
    """Получение значения из конфигурации"""
    try:
        config = load_config()
        return config.get(key, default)
    except:
        return default
