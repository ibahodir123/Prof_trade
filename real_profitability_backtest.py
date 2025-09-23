#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
💰 РЕАЛЬНЫЙ БЭКТЕСТ ПРИБЫЛЬНОСТИ
=====================================

Проверяем реальную прибыльность наших детекторов:
- Симулируем реальные сделки
- Рассчитываем прибыль/убыток
- Учитываем комиссии
- Анализируем риск

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

class RealProfitabilityBacktest:
    """💰 Реальный бэктест прибыльности"""
    
    def __init__(self):
        """Инициализация бэктеста"""
        print("💰 Реальный бэктест прибыльности инициализирован")
        
        # 🔗 Подключаемся к Binance
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # 🎯 Наши веса
        self.weights_minimums = {
            'priceVelocity': 0.037,
            'ema20Velocity': 0.031,
            'ema20Angle': 0.217,
            'priceDistance': 0.715
        }
        
        self.weights_maximums = {
            'priceVelocity': 0.715,
            'ema20Velocity': 0.217,
            'ema20Angle': 0.031,
            'priceDistance': 0.037
        }
        
        # 💰 Параметры торговли
        self.commission = 0.001  # 0.1% комиссия
        self.confidence_threshold = 0.25  # 25% для минимумов
        self.max_confidence_threshold = 0.05  # 5% для максимумов (ЕЩЁ СНИЖАЕМ!)
        self.stop_loss = 0.05  # 5% стоп-лосс
        self.initial_capital = 10000  # $10,000 начальный капитал
        
        print(f"💰 Начальный капитал: ${self.initial_capital:,}")
        print(f"⚖️ Комиссия: {self.commission:.1%}")
        print(f"🎯 Порог уверенности: {self.confidence_threshold:.1%}")
    
    def get_real_data(self, symbol='BTC/USDT', timeframe='1h', limit=200):
        """📊 Получаем реальные данные"""
        try:
            print(f"📊 Получаем данные {symbol} ({timeframe})...")
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
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
        
        current_price = current['close']
        current_ema20 = current['ema20']
        prev_price = previous['close'].iloc[-1]
        prev_ema20 = previous['ema20'].iloc[-1]
        
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
        
        weighted_score = 0.0
        for feature, weight in self.weights_minimums.items():
            weighted_score += weight * features[feature]
        
        confidence = min(1.0, weighted_score * 10.0)
        return confidence
    
    def predict_maximum(self, features):
        """🔺 Предсказываем максимум"""
        if not features:
            return 0.0
        
        weighted_score = 0.0
        for feature, weight in self.weights_maximums.items():
            weighted_score += weight * features[feature]
        
        # 🚀 УВЕЛИЧИВАЕМ ЧУВСТВИТЕЛЬНОСТЬ МАКСИМУМОВ
        confidence = min(1.0, weighted_score * 20.0)  # Увеличиваем с 10.0 до 20.0
        
        # 🎯 БОНУСЫ ДЛЯ МАКСИМУМОВ
        if features['priceDistance'] > 0.02:  # Если цена далеко от EMA20
            confidence += 0.1
        if features['ema20Angle'] > 1.0:  # Если угол EMA20 большой
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def run_backtest(self, symbol='BTC/USDT', timeframe='1h'):
        """💰 Запускаем реальный бэктест"""
        print(f"\n💰 РЕАЛЬНЫЙ БЭКТЕСТ: {symbol}")
        print("=" * 60)
        
        # 📊 Получаем данные
        df = self.get_real_data(symbol, timeframe, limit=200)
        if df is None:
            return
        
        # 💰 Инициализируем торговлю
        capital = self.initial_capital
        position = None
        entry_price = None
        entry_time = None
        trades = []
        
        print(f"📊 Анализируем {len(df)} свечей...")
        
        # 🔍 Анализируем каждую свечу
        for i in range(20, len(df)):
            current = df.iloc[i]
            previous = df.iloc[:i]
            
            # 📊 Рассчитываем признаки
            features = self.calculate_features(current, previous)
            if not features:
                continue
            
            current_price = current['close']
            current_time = current['timestamp']
            
            # 🔻 Проверяем минимум (вход в лонг)
            if position is None:
                min_confidence = self.predict_minimum(features)
                if min_confidence >= self.confidence_threshold:
                    position = 'long'
                    entry_price = current_price
                    entry_time = current_time
                    print(f"🟢 ВХОД В ЛОНГ: ${entry_price:.0f} ({current_time}) - Уверенность: {min_confidence:.1%}")
            
            # 🔺 Проверяем максимум (выход из лонга)
            elif position == 'long':
                max_confidence = self.predict_maximum(features)
                
                # 🛡️ Проверяем стоп-лосс
                current_loss = (entry_price - current_price) / entry_price
                
                # 🔍 ОТЛАДКА МАКСИМУМОВ
                if i % 20 == 0:  # Каждые 20 свечей
                    print(f"🔍 ОТЛАДКА: Свеча {i}, Макс. уверенность: {max_confidence:.1%}, Порог: {self.max_confidence_threshold:.1%}, Убыток: {current_loss:.1%}")
                
                if max_confidence >= self.max_confidence_threshold or current_loss >= self.stop_loss:
                    # 💰 Рассчитываем прибыль
                    profit_pct = (current_price - entry_price) / entry_price
                    profit_pct_after_commission = profit_pct - (2 * self.commission)  # Вход + выход
                    
                    profit_amount = capital * profit_pct_after_commission
                    capital += profit_amount
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': current_time,
                        'entry_price': entry_price,
                        'exit_price': current_price,
                        'profit_pct': profit_pct_after_commission,
                        'profit_amount': profit_amount,
                        'capital_after': capital,
                        'min_confidence': min_confidence if 'min_confidence' in locals() else 0,
                        'max_confidence': max_confidence
                    })
                    
                    exit_reason = "СТОП-ЛОСС" if current_loss >= self.stop_loss else "МАКСИМУМ"
                    print(f"🔴 ВЫХОД ИЗ ЛОНГА: ${current_price:.0f} ({current_time}) - {exit_reason} - Прибыль: {profit_pct_after_commission:.2%} (${profit_amount:.0f})")
                    
                    position = None
                    entry_price = None
                    entry_time = None
        
        # 🚨 Принудительный выход в конце
        if position == 'long':
            final_price = df['close'].iloc[-1]
            profit_pct = (final_price - entry_price) / entry_price
            profit_pct_after_commission = profit_pct - (2 * self.commission)
            profit_amount = capital * profit_pct_after_commission
            capital += profit_amount
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': df['timestamp'].iloc[-1],
                'entry_price': entry_price,
                'exit_price': final_price,
                'profit_pct': profit_pct_after_commission,
                'profit_amount': profit_amount,
                'capital_after': capital,
                'min_confidence': 0,
                'max_confidence': 0
            })
            
            print(f"🚨 ПРИНУДИТЕЛЬНЫЙ ВЫХОД: ${final_price:.0f} - Прибыль: {profit_pct_after_commission:.2%} (${profit_amount:.0f})")
        
        # 📊 Анализируем результаты
        self.analyze_results(trades, capital)
        
        return trades, capital
    
    def analyze_results(self, trades, final_capital):
        """📊 Анализируем результаты бэктеста"""
        print("\n" + "=" * 60)
        print("💰 АНАЛИЗ РЕАЛЬНОЙ ПРИБЫЛЬНОСТИ")
        print("=" * 60)
        
        if not trades:
            print("❌ Сделок не было")
            return
        
        # 📊 Базовая статистика
        total_trades = len(trades)
        profitable_trades = [t for t in trades if t['profit_pct'] > 0]
        losing_trades = [t for t in trades if t['profit_pct'] < 0]
        
        win_rate = (len(profitable_trades) / total_trades) * 100
        total_profit_pct = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        
        print(f"🎯 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных: {len(profitable_trades)} ({win_rate:.1f}%)")
        print(f"❌ Убыточных: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"💰 Общая прибыль: {total_profit_pct:.2f}%")
        print(f"💵 Капитал: ${self.initial_capital:,} → ${final_capital:,.0f}")
        
        # 📊 Детальная статистика
        if profitable_trades:
            avg_profit = np.mean([t['profit_pct'] for t in profitable_trades])
            best_trade = max(profitable_trades, key=lambda x: x['profit_pct'])
            print(f"📈 Средняя прибыль: {avg_profit:.2f}%")
            print(f"🏆 Лучшая сделка: {best_trade['profit_pct']:.2f}%")
        
        if losing_trades:
            avg_loss = np.mean([t['profit_pct'] for t in losing_trades])
            worst_trade = min(losing_trades, key=lambda x: x['profit_pct'])
            print(f"📉 Средний убыток: {avg_loss:.2f}%")
            print(f"💥 Худшая сделка: {worst_trade['profit_pct']:.2f}%")
        
        # 📊 Анализ уверенности
        avg_min_confidence = np.mean([t['min_confidence'] for t in trades if t['min_confidence'] > 0])
        avg_max_confidence = np.mean([t['max_confidence'] for t in trades if t['max_confidence'] > 0])
        
        print(f"🔻 Средняя уверенность входа: {avg_min_confidence:.1f}%")
        print(f"🔺 Средняя уверенность выхода: {avg_max_confidence:.1f}%")
        
        # 📊 Оценка системы
        if total_profit_pct > 10:
            print("🏆 СИСТЕМА ВЫСОКОПРИБЫЛЬНА!")
        elif total_profit_pct > 5:
            print("✅ Система прибыльна!")
        elif total_profit_pct > 0:
            print("⚠️ Система слабоприбыльна")
        else:
            print("❌ Система убыточна")
    
    def test_multiple_symbols(self, symbols=['BTC/USDT', 'ETH/USDT', 'SOL/USDT']):
        """🧪 Тестируем несколько символов"""
        print(f"\n🧪 БЭКТЕСТ НЕСКОЛЬКИХ СИМВОЛОВ")
        print("=" * 60)
        
        results = {}
        
        for symbol in symbols:
            print(f"\n📊 Тестируем {symbol}...")
            trades, final_capital = self.run_backtest(symbol)
            
            if trades:
                total_profit = ((final_capital - self.initial_capital) / self.initial_capital) * 100
                results[symbol] = {
                    'trades': len(trades),
                    'profit_pct': total_profit,
                    'final_capital': final_capital
                }
            else:
                results[symbol] = {
                    'trades': 0,
                    'profit_pct': 0,
                    'final_capital': self.initial_capital
                }
            
            time.sleep(1)  # Пауза между запросами
        
        # 📊 Итоговая статистика
        print("\n" + "=" * 60)
        print("📊 СРАВНИТЕЛЬНАЯ СТАТИСТИКА ПРИБЫЛЬНОСТИ")
        print("=" * 60)
        
        total_profit = 0
        for symbol, result in results.items():
            print(f"{symbol}: {result['trades']} сделок, {result['profit_pct']:.2f}% прибыль")
            total_profit += result['profit_pct']
        
        avg_profit = total_profit / len(symbols)
        print(f"\n📊 Средняя прибыльность: {avg_profit:.2f}%")
        
        if avg_profit > 5:
            print("🏆 СИСТЕМА ВСЕГО ВЫСОКОПРИБЫЛЬНА!")
        elif avg_profit > 0:
            print("✅ Система в целом прибыльна!")
        else:
            print("❌ Система в целом убыточна")

if __name__ == "__main__":
    backtest = RealProfitabilityBacktest()
    
    # 🧪 Тестируем один символ
    # backtest.run_backtest('BTC/USDT', '1h')
    
    # 🧪 Тестируем несколько символов
    backtest.test_multiple_symbols()
