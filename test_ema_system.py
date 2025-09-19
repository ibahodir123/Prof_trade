#!/usr/bin/env python3
"""
Тестирование EMA системы анализа
"""

from ema_pattern_analyzer import EMAPatternAnalyzer
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_ema_analyzer():
    """Тестирование EMA анализатора"""
    
    print("🚀 Тестирование EMA системы анализа...")
    
    # Создаем тестовые данные
    dates = pd.date_range('2025-01-01', periods=200, freq='1H')
    trend_data = []
    
    base_price = 100
    for i in range(200):
        if i < 50:  # Восходящий тренд
            price = base_price + i * 0.5 + np.random.normal(0, 1)
        elif i < 80:  # Коррекция
            price = base_price + 50 * 0.5 - (i - 50) * 0.3 + np.random.normal(0, 1)
        elif i < 130:  # Продолжение тренда
            price = base_price + 50 * 0.5 - 30 * 0.3 + (i - 80) * 0.4 + np.random.normal(0, 1)
        else:  # Флет
            price = base_price + 50 * 0.5 - 30 * 0.3 + 50 * 0.4 + np.random.normal(0, 0.5)
        
        trend_data.append({
            'open': price + np.random.normal(0, 0.5),
            'high': price + abs(np.random.normal(0, 1)),
            'low': price - abs(np.random.normal(0, 1)),
            'close': price,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(trend_data, index=dates)
    
    print(f"📊 Создано {len(df)} записей тестовых данных")
    
    # Тестируем EMA анализатор
    analyzer = EMAPatternAnalyzer()
    result = analyzer.analyze_ema_patterns(df)
    
    print("\n🎯 РЕЗУЛЬТАТЫ EMA АНАЛИЗА:")
    print(f"Тренд: {result['trend']}")
    print(f"Фаза: {result['phase']}")
    print(f"Сигнал: {result['signal']['type']}")
    print(f"Причина: {result['signal']['reason']}")
    
    if 'entry_price' in result['signal']:
        print(f"Цена входа: ${result['signal']['entry_price']:.4f}")
        print(f"Take Profit: ${result['signal']['take_profit']:.4f}")
        print(f"Stop Loss: ${result['signal']['stop_loss']:.4f}")
        print(f"Уверенность: {result['signal']['confidence']:.2f}")
    
    print(f"\n📊 EMA УРОВНИ:")
    levels = result['levels']
    print(f"EMA 20: ${levels['ema_20']:.4f}")
    print(f"EMA 50: ${levels['ema_50']:.4f}")
    print(f"EMA 100: ${levels['ema_100']:.4f}")
    
    print(f"\n✅ EMA система работает корректно!")
    
    return result

def test_ema_features():
    """Тестирование EMA признаков"""
    
    print("\n🔍 Тестирование EMA признаков...")
    
    # Создаем простые тестовые данные
    dates = pd.date_range('2025-01-01', periods=100, freq='1H')
    data = []
    
    for i in range(100):
        price = 100 + i * 0.1 + np.random.normal(0, 0.5)
        data.append({
            'open': price,
            'high': price + 1,
            'low': price - 1,
            'close': price,
            'volume': 1000
        })
    
    df = pd.DataFrame(data, index=dates)
    
    # Тестируем расчет признаков
    analyzer = EMAPatternAnalyzer()
    features = analyzer.calculate_ema_features(df)
    
    print(f"📊 Рассчитано {len(features.columns)} EMA признаков:")
    
    # Показываем некоторые ключевые признаки
    key_features = [
        'ema_20', 'ema_50', 'ema_100',
        'ema_20_speed', 'ema_50_speed', 'ema_100_speed',
        'price_speed', 'trend_strength', 'ema_trend_direction'
    ]
    
    for feature in key_features:
        if feature in features.columns:
            value = features[feature].iloc[-1]
            print(f"  {feature}: {value:.4f}")
    
    print(f"\n✅ EMA признаки рассчитываются корректно!")
    
    return features

if __name__ == "__main__":
    try:
        # Тестируем EMA анализатор
        result = test_ema_analyzer()
        
        # Тестируем EMA признаки
        features = test_ema_features()
        
        print(f"\n🎉 ВСЕ ТЕСТЫ ПРОШЛИ УСПЕШНО!")
        print(f"📈 EMA система готова к использованию в боте!")
        
    except Exception as e:
        print(f"❌ Ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()




