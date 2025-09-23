#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
📈 РЕАЛЬНЫЙ ДЕТЕКТОР РЫНКА
============================

Подключается к Binance API и анализирует реальные данные:
- Получает реальные цены
- Рассчитывает EMA20
- Применяет наши 4 критерия
- Детектирует минимумы и максимумы

Автор: AI Assistant
Дата: 2025-01-22
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

class RealMarketDetector:
    """📈 Реальный детектор рынка"""
    
    def __init__(self):
        """Инициализация детектора"""
        print("📈 Реальный детектор рынка инициализирован")
        
        # 🔗 Подключаемся к Binance
        self.exchange = ccxt.binance({
            'apiKey': '',  # Не нужен для публичных данных
            'secret': '',  # Не нужен для публичных данных
            'sandbox': False,  # Реальные данные
            'enableRateLimit': True,
        })
        
        # 🎯 Наши веса (из предыдущих тестов)
        self.weights_minimums = {
            'priceVelocity': 0.037,     # 3.7% - скорость цены
            'ema20Velocity': 0.031,     # 3.1% - скорость EMA20
            'ema20Angle': 0.217,        # 21.7% - угол EMA20
            'priceDistance': 0.715      # 71.5% - расстояние до EMA20
        }
        
        self.weights_maximums = {
            'priceVelocity': 0.715,     # 71.5% - зеркальная логика
            'ema20Velocity': 0.217,     # 21.7% - зеркальная логика
            'ema20Angle': 0.031,        # 3.1% - зеркальная логика
            'priceDistance': 0.037      # 3.7% - зеркальная логика
        }
        
        # 📊 Пороги
        self.confidence_threshold = 0.25  # 25%
        
        print(f"⚖️ Веса минимумов: Distance={self.weights_minimums['priceDistance']:.1%}")
        print(f"⚖️ Веса максимумов: Distance={self.weights_maximums['priceDistance']:.1%}")
    
    def get_real_data(self, symbol='BTC/USDT', timeframe='1h', limit=100):
        """📊 Получаем реальные данные с Binance"""
        try:
            print(f"📊 Получаем данные {symbol} ({timeframe})...")
            
            # 📈 Получаем свечи
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # 📊 Создаем DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 📈 Рассчитываем EMA20
            df['ema20'] = df['close'].ewm(span=20).mean()
            
            print(f"✅ Получено {len(df)} свечей")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения данных: {e}")
            return None
    
    def calculate_features(self, current, previous):
        """📊 Рассчитываем наши 4 критерия"""
        if previous is None or len(previous) < 2:
            return None
        
        # 📈 Базовые данные
        current_price = current['close']
        current_ema20 = current['ema20']
        prev_price = previous['close'].iloc[-1]
        prev_ema20 = previous['ema20'].iloc[-1]
        
        # 🧮 Наши 4 критерия
        features = {
            'priceVelocity': abs(current_price - prev_price) / prev_price,
            'ema20Velocity': abs(current_ema20 - prev_ema20) / prev_ema20,
            'ema20Angle': abs(np.degrees(np.arctan((current_ema20 - prev_ema20) / prev_ema20))),
            'priceDistance': abs(current_price - current_ema20) / current_ema20
        }
        
        return features
    
    def predict_minimum(self, features):
        """🔻 Предсказываем минимум"""
        if not features:
            return 0.0
        
        # 🧮 Взвешенная оценка
        weighted_score = 0.0
        for feature, weight in self.weights_minimums.items():
            weighted_score += weight * features[feature]
        
        # 🎯 Уверенность
        confidence = min(1.0, weighted_score * 10.0)
        
        return confidence
    
    def predict_maximum(self, features):
        """🔺 Предсказываем максимум"""
        if not features:
            return 0.0
        
        # 🧮 Взвешенная оценка (зеркальная логика)
        weighted_score = 0.0
        for feature, weight in self.weights_maximums.items():
            weighted_score += weight * features[feature]
        
        # 🎯 Уверенность
        confidence = min(1.0, weighted_score * 10.0)
        
        return confidence
    
    def analyze_real_market(self, symbol='BTC/USDT', timeframe='1h'):
        """📊 Анализируем реальный рынок"""
        print(f"\n🚀 АНАЛИЗ РЕАЛЬНОГО РЫНКА: {symbol}")
        print("=" * 60)
        
        # 📊 Получаем данные
        df = self.get_real_data(symbol, timeframe)
        if df is None:
            return
        
        print(f"📊 Анализируем {len(df)} свечей...")
        
        # 🔍 Анализируем каждую свечу
        minimums = []
        maximums = []
        
        for i in range(20, len(df)):  # Начинаем с 20-й свечи (для EMA20)
            current = df.iloc[i]
            previous = df.iloc[:i]
            
            # 📊 Рассчитываем признаки
            features = self.calculate_features(current, previous)
            if not features:
                continue
            
            # 🔻 Проверяем минимум
            min_confidence = self.predict_minimum(features)
            if min_confidence >= self.confidence_threshold:
                minimums.append({
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'confidence': min_confidence,
                    'features': features
                })
                print(f"🔻 МИНИМУМ: ${current['close']:.0f} ({current['timestamp']}) - Уверенность: {min_confidence:.1%}")
            
            # 🔺 Проверяем максимум
            max_confidence = self.predict_maximum(features)
            if max_confidence >= self.confidence_threshold:
                maximums.append({
                    'timestamp': current['timestamp'],
                    'price': current['close'],
                    'confidence': max_confidence,
                    'features': features
                })
                print(f"🔺 МАКСИМУМ: ${current['close']:.0f} ({current['timestamp']}) - Уверенность: {max_confidence:.1%}")
        
        # 📊 Итоговая статистика
        print("\n" + "=" * 50)
        print("📊 ИТОГОВАЯ СТАТИСТИКА РЕАЛЬНОГО РЫНКА")
        print("=" * 50)
        print(f"🔻 Минимумов найдено: {len(minimums)}")
        print(f"🔺 Максимумов найдено: {len(maximums)}")
        
        if minimums:
            avg_min_confidence = np.mean([m['confidence'] for m in minimums])
            print(f"📈 Средняя уверенность минимумов: {avg_min_confidence:.1%}")
        
        if maximums:
            avg_max_confidence = np.mean([m['confidence'] for m in maximums])
            print(f"📉 Средняя уверенность максимумов: {avg_max_confidence:.1%}")
        
        # 📊 Создаем график (отключаем для экономии памяти)
        # self.create_real_chart(df, minimums, maximums)
        
        return minimums, maximums
    
    def create_real_chart(self, df, minimums, maximums):
        """📊 Создаем график реальных данных"""
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # 📈 График цены
        ax.plot(df['timestamp'], df['close'], 'b-', linewidth=1, label='Цена')
        ax.plot(df['timestamp'], df['ema20'], 'orange', linewidth=1, alpha=0.7, label='EMA20')
        
        # 🔻 Минимумы
        if minimums:
            min_times = [m['timestamp'] for m in minimums]
            min_prices = [m['price'] for m in minimums]
            ax.scatter(min_times, min_prices, color='green', s=100, marker='^', 
                      label=f'Минимумы ({len(minimums)})', zorder=5)
        
        # 🔺 Максимумы
        if maximums:
            max_times = [m['timestamp'] for m in maximums]
            max_prices = [m['price'] for m in maximums]
            ax.scatter(max_times, max_prices, color='red', s=100, marker='v', 
                      label=f'Максимумы ({len(maximums)})', zorder=5)
        
        ax.set_title('📈 Реальный Анализ Рынка', fontsize=16, fontweight='bold')
        ax.set_xlabel('Время')
        ax.set_ylabel('Цена')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('real_market_analysis.png', dpi=150, bbox_inches='tight')
        print("📊 График сохранен: real_market_analysis.png")
    
    def test_multiple_symbols(self, symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']):
        """🧪 Тестируем несколько символов"""
        print(f"\n🧪 ТЕСТИРОВАНИЕ НЕСКОЛЬКИХ СИМВОЛОВ")
        print("=" * 60)
        
        results = {}
        
        for symbol in symbols:
            print(f"\n📊 Тестируем {symbol}...")
            minimums, maximums = self.analyze_real_market(symbol)
            
            results[symbol] = {
                'minimums': len(minimums),
                'maximums': len(maximums),
                'total_signals': len(minimums) + len(maximums)
            }
            
            time.sleep(1)  # Пауза между запросами
        
        # 📊 Итоговая статистика
        print("\n" + "=" * 50)
        print("📊 СРАВНИТЕЛЬНАЯ СТАТИСТИКА")
        print("=" * 50)
        
        for symbol, result in results.items():
            print(f"{symbol}: {result['total_signals']} сигналов ({result['minimums']} мин, {result['maximums']} макс)")

if __name__ == "__main__":
    detector = RealMarketDetector()
    
    # 🧪 Тестируем один символ
    # detector.analyze_real_market('BTC/USDT', '1h')
    
    # 🧪 Тестируем несколько символов
    detector.test_multiple_symbols()
