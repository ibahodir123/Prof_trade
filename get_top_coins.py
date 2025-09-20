#!/usr/bin/env python3
"""
Скрипт для получения топ-50 самых популярных USDT пар с Binance
по объему торгов за 24 часа
"""

import ccxt
import pandas as pd
from datetime import datetime

def get_top_usdt_pairs(count=50):
    """Получить топ USDT пары по объему торгов"""
    try:
        print(f"🔍 Получаю топ-{count} USDT пар с Binance...")
        
        # Инициализируем Binance без API ключей (публичные данные)
        exchange = ccxt.binance({
            'rateLimit': 1200,
            'enableRateLimit': True,
        })
        
        # Загружаем все рынки
        markets = exchange.load_markets()
        
        # Получаем тикеры (объемы торгов)
        tickers = exchange.fetch_tickers()
        
        # Фильтруем только USDT пары с объемом > 0
        usdt_pairs = []
        for symbol, ticker in tickers.items():
            if (symbol.endswith('/USDT') and 
                ticker['quoteVolume'] is not None and 
                ticker['quoteVolume'] > 0):
                
                usdt_pairs.append({
                    'symbol': symbol,
                    'volume_24h': ticker['quoteVolume'],
                    'price': ticker['last'],
                    'change_24h': ticker['percentage']
                })
        
        # Сортируем по объему торгов (по убыванию)
        usdt_pairs.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        # Берем топ N пар
        top_pairs = usdt_pairs[:count]
        
        print(f"✅ Найдено {len(top_pairs)} активных USDT пар")
        print("\n📊 Топ-10 по объему торгов:")
        for i, pair in enumerate(top_pairs[:10], 1):
            volume_m = pair['volume_24h'] / 1_000_000
            print(f"{i:2d}. {pair['symbol']:<12} - ${volume_m:,.0f}M ({pair['change_24h']:+.1f}%)")
        
        # Сохраняем в файл
        symbols = [pair['symbol'] for pair in top_pairs]
        
        with open('top_coins_list.txt', 'w') as f:
            f.write(f"# Топ-{count} USDT пар по объему торгов на Binance\n")
            f.write(f"# Сгенерировано: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for symbol in symbols:
                f.write(f"{symbol}\n")
        
        print(f"\n💾 Список сохранен в top_coins_list.txt")
        
        return symbols
        
    except Exception as e:
        print(f"❌ Ошибка получения данных: {e}")
        return None

if __name__ == "__main__":
    # Получаем топ-50 монет
    top_coins = get_top_usdt_pairs(50)
    
    if top_coins:
        print(f"\n🚀 Готово! Получено {len(top_coins)} монет для обучения")
        print("Теперь можно запустить: python retrain_models.py")
    else:
        print("❌ Не удалось получить список монет")