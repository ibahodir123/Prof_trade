#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 Тест реальных данных
Проверяет получение данных с Binance API
"""

import json
import sys
import os
from datetime import datetime
import pandas as pd
import ccxt
import warnings
warnings.filterwarnings('ignore')

def test_binance_connection():
    """Тест подключения к Binance"""
    print("🔗 Тестирование подключения к Binance...")

    try:
        exchange = ccxt.binance()
        print("✅ Подключение к Binance успешно")

        # Тест получения данных
        symbol = 'BTC/USDT'
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=10)

        if ohlcv:
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

            print(f"✅ Данные получены для {symbol}")
            print(f"📊 Последняя цена: {df['close'].iloc[-1]}")
            return True
        else:
            print("❌ Не удалось получить данные")
            return False

    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return False

def main():
    """Главная функция"""
    print("🧪 ТЕСТИРОВАНИЕ РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 50)

    # Тест подключения к Binance
    if test_binance_connection():
        print("✅ Тест пройден! Данные доступны")
    else:
        print("❌ Тест не пройден")

    print("\n" + "=" * 50)
    print("🎉 Тестирование завершено!")

if __name__ == "__main__":
    main()
