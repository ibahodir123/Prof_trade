#!/usr/bin/env python3
"""
Тест адаптивного обучения для монет не из топ-50
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_bot_binance import is_coin_in_top50, adaptive_retrain_for_coin, analyze_coin_signal_advanced_ema

def test_adaptive_learning():
    """Тестируем адаптивное обучение"""
    
    print("🧪 Тест адаптивного обучения")
    print("=" * 50)
    
    # Тестируем монету из топ-50
    print("\n1️⃣ Тест монеты из топ-50:")
    btc_in_top50 = is_coin_in_top50("BTC/USDT")
    print(f"   BTC/USDT в топ-50: {btc_in_top50}")
    
    # Тестируем монету НЕ из топ-50
    print("\n2️⃣ Тест монеты НЕ из топ-50:")
    shib_in_top50 = is_coin_in_top50("SHIB/USDT")
    print(f"   SHIB/USDT в топ-50: {shib_in_top50}")
    
    if not shib_in_top50:
        print("\n🔄 Запускаю адаптивное переобучение для SHIB/USDT...")
        success = adaptive_retrain_for_coin("SHIB/USDT")
        print(f"   Результат переобучения: {success}")
        
        if success:
            print("\n📊 Тестирую анализ SHIB/USDT с новыми моделями...")
            try:
                signal_data = analyze_coin_signal_advanced_ema("SHIB/USDT")
                if signal_data and not signal_data.get('error'):
                    print(f"   ✅ Анализ успешен!")
                    print(f"   📈 Сигнал: {signal_data.get('signal_type', 'Неизвестно')}")
                    print(f"   🎯 ML статус: {signal_data.get('ml_status', 'Неизвестно')}")
                else:
                    print(f"   ❌ Ошибка анализа: {signal_data.get('error', 'Неизвестная ошибка')}")
            except Exception as e:
                print(f"   ❌ Ошибка анализа: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Тест завершен!")

if __name__ == "__main__":
    test_adaptive_learning()
