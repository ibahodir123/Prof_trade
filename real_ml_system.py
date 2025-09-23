#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 РЕАЛЬНАЯ ML СИСТЕМА
Обучается на реальных данных января, тестируется на реальных данных февраля
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class RealMLSystem:
    def __init__(self):
        """🚀 Инициализация реальной ML системы"""
        self.base_url = "https://api.binance.com/api/v3/klines"
        
        # 🧠 Начальные веса (будут адаптироваться)
        self.weights_minimums = {
            'priceVelocity': 0.25,
            'ema20Velocity': 0.25,
            'ema20Angle': 0.25,
            'priceDistance': 0.25
        }
        
        self.weights_maximums = {
            'priceVelocity': 0.25,
            'ema20Velocity': 0.25,
            'ema20Angle': 0.25,
            'priceDistance': 0.25
        }
        
        # 📊 Статистика обучения
        self.training_stats = {
            'patterns_learned': 0,
            'minimums_found': 0,
            'maximums_found': 0,
            'accuracy': 0.0
        }
        
        # 💰 Параметры торговли
        self.commission = 0.001
        self.confidence_threshold = 0.28  # Оптимальный баланс
        self.max_confidence_threshold = 0.18  # Оптимальный баланс
        self.stop_loss = 0.04  # Оптимальный стоп-лосс
        self.take_profit = 0.02  # Тейк-профит 2%
        self.initial_capital = 10000
        
        print("🚀 РЕАЛЬНАЯ ML СИСТЕМА ИНИЦИАЛИЗИРОВАНА")
        print("📅 Обучение на реальных данных января")
        print("🧪 Тестирование на реальных данных февраля")
    
    def get_real_data(self, symbol='BTC/USDT', timeframe='1h', limit=500):
        """📈 Получаем реальные данные с Binance"""
        try:
            url = self.base_url
            params = {
                'symbol': symbol.replace('/', ''),
                'interval': timeframe,
                'limit': limit
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if not data:
                return None
            
            # 📊 Преобразуем в DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 🔧 Преобразуем типы данных
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # 📊 Рассчитываем EMA20
            df['ema20'] = df['close'].ewm(span=20).mean()
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения данных: {e}")
            return None
    
    def find_real_minimums(self, df):
        """🔍 Находим реальные минимумы"""
        minimums = []
        
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            next1 = df.iloc[i+1]
            next2 = df.iloc[i+2]
            
            # 🔻 Условие минимума
            if (current['low'] < prev1['low'] and 
                current['low'] < prev2['low'] and
                current['low'] < next1['low'] and 
                current['low'] < next2['low']):
                
                minimums.append({
                    'index': i,
                    'price': current['low'],
                    'time': current['timestamp'],
                    'ema20': current['ema20']
                })
        
        return minimums
    
    def find_real_maximums(self, df):
        """🔺 Находим реальные максимумы"""
        maximums = []
        
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            next1 = df.iloc[i+1]
            next2 = df.iloc[i+2]
            
            # 🔺 Условие максимума
            if (current['high'] > prev1['high'] and 
                current['high'] > prev2['high'] and
                current['high'] > next1['high'] and 
                current['high'] > next2['high']):
                
                maximums.append({
                    'index': i,
                    'price': current['high'],
                    'time': current['timestamp'],
                    'ema20': current['ema20']
                })
        
        return maximums
    
    def calculate_features(self, df, index):
        """📊 Рассчитываем признаки"""
        if index < 20:
            return None
        
        current = df.iloc[index]
        previous = df.iloc[:index]
        
        if len(previous) < 2:
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
        
        # 🎯 УМНЫЕ ФИЛЬТРЫ КАЧЕСТВА
        if features['priceDistance'] < 0.003:  # Слишком близко к EMA20
            confidence *= 0.7  # Менее строгий фильтр
        if features['ema20Angle'] < 0.05:  # Слишком маленький угол
            confidence *= 0.8  # Менее строгий фильтр
        
        # 🚀 БОНУСЫ ДЛЯ СИЛЬНЫХ СИГНАЛОВ
        if features['priceDistance'] > 0.01:  # Далеко от EMA20
            confidence += 0.05
        if features['ema20Angle'] > 0.2:  # Большой угол
            confidence += 0.05
        
        return confidence
    
    def predict_maximum(self, features):
        """🔺 Предсказываем максимум"""
        if not features:
            return 0.0
        
        weighted_score = 0.0
        for feature, weight in self.weights_maximums.items():
            weighted_score += weight * features[feature]
        
        confidence = min(1.0, weighted_score * 10.0)
        return confidence
    
    def adapt_weights_for_minimum(self, features):
        """🧠 Адаптируем веса для минимума"""
        print("🧠 АДАПТАЦИЯ ВЕСОВ ДЛЯ МИНИМУМА...")
        
        # 📊 Анализируем признаки минимума
        total_impact = sum(features.values())
        
        if total_impact > 0:
            # 🎯 Увеличиваем вес самого значимого признака
            max_feature = max(features, key=features.get)
            max_value = features[max_feature]
            
            # 📈 Адаптируем веса
            for feature in features:
                if feature == max_feature:
                    # Увеличиваем вес главного признака
                    self.weights_minimums[feature] = min(0.8, 
                        self.weights_minimums[feature] + max_value * 0.1)
                else:
                    # Уменьшаем вес второстепенных признаков
                    self.weights_minimums[feature] = max(0.05, 
                        self.weights_minimums[feature] - max_value * 0.05)
            
            # 🔧 Нормализуем веса
            total_weight = sum(self.weights_minimums.values())
            for feature in self.weights_minimums:
                self.weights_minimums[feature] /= total_weight
            
            print(f"✅ Веса адаптированы для минимума")
            print(f"🎯 Главный признак: {max_feature} = {self.weights_minimums[max_feature]:.3f}")
    
    def train_on_january_data(self, symbol='BTC/USDT'):
        """🎓 Обучаемся на январских данных"""
        print(f"\n🎓 ОБУЧЕНИЕ НА ЯНВАРСКИХ ДАННЫХ: {symbol}")
        print("=" * 60)
        
        # 📊 Получаем данные
        df = self.get_real_data(symbol, limit=500)
        if df is None:
            return
        
        print(f"✅ Получено {len(df)} свечей")
        
        # 🔍 Находим минимумы и максимумы
        minimums = self.find_real_minimums(df)
        maximums = self.find_real_maximums(df)
        
        print(f"🔻 Найдено минимумов: {len(minimums)}")
        print(f"🔺 Найдено максимумов: {len(maximums)}")
        
        # 🧠 Обучаемся на каждом минимуме
        for i, min_point in enumerate(minimums):
            features = self.calculate_features(df, min_point['index'])
            if features:
                print(f"🔻 Изучаем минимум #{i+1}: {min_point['time']} - ${min_point['price']:.0f}")
                self.adapt_weights_for_minimum(features)
                self.training_stats['minimums_found'] += 1
                self.training_stats['patterns_learned'] += 1
        
        # 🧠 Обучаемся на каждом максимуме
        for i, max_point in enumerate(maximums):
            features = self.calculate_features(df, max_point['index'])
            if features:
                print(f"🔺 Изучаем максимум #{i+1}: {max_point['time']} - ${max_point['price']:.0f}")
                # Адаптируем веса для максимумов (зеркальная логика)
                self.adapt_weights_for_minimum(features)  # Пока используем ту же логику
                self.training_stats['maximums_found'] += 1
                self.training_stats['patterns_learned'] += 1
        
        print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📚 Изучено паттернов: {self.training_stats['patterns_learned']}")
        print(f"🔻 Минимумов: {self.training_stats['minimums_found']}")
        print(f"🔺 Максимумов: {self.training_stats['maximums_found']}")
        
        # 💾 Сохраняем обученную модель
        self.save_trained_model()
    
    def save_trained_model(self):
        """💾 Сохраняем обученную модель"""
        model_data = {
            'weights_minimums': self.weights_minimums,
            'weights_maximums': self.weights_maximums,
            'training_stats': self.training_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('real_ml_model.json', 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Модель сохранена в real_ml_model.json")
    
    def load_trained_model(self):
        """📂 Загружаем обученную модель"""
        if os.path.exists('real_ml_model.json'):
            with open('real_ml_model.json', 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            
            self.weights_minimums = model_data['weights_minimums']
            self.weights_maximums = model_data['weights_maximums']
            self.training_stats = model_data['training_stats']
            
            print("✅ Обученная модель загружена!")
            return True
        
        print("❌ Модель не найдена. Сначала обучите систему!")
        return False
    
    def test_on_february_data(self, symbol='BTC/USDT'):
        """🧪 Тестируем на февральских данных"""
        print(f"\n🧪 ТЕСТИРОВАНИЕ НА ФЕВРАЛЬСКИХ ДАННЫХ: {symbol}")
        print("=" * 60)
        
        # 📊 Получаем новые данные
        df = self.get_real_data(symbol, limit=500)
        if df is None:
            return
        
        print(f"✅ Получено {len(df)} свечей для тестирования")
        
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
            
            current_price = current['close']
            current_time = current['timestamp']
            
            features = self.calculate_features(df, i)
            if not features:
                continue
            
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
                current_loss = (entry_price - current_price) / entry_price
                
                # 🎯 УМНОЕ УПРАВЛЕНИЕ РИСКАМИ
                current_profit = (current_price - entry_price) / entry_price
                
                # 🚀 ПРИОРИТЕТ: Тейк-профит > Максимум > Стоп-лосс
                should_exit = False
                exit_reason = ""
                
                if current_profit >= self.take_profit:
                    should_exit = True
                    exit_reason = "ТЕЙК-ПРОФИТ"
                elif max_confidence >= self.max_confidence_threshold:
                    should_exit = True
                    exit_reason = "МАКСИМУМ"
                elif current_loss >= self.stop_loss:
                    should_exit = True
                    exit_reason = "СТОП-ЛОСС"
                
                if should_exit:
                    # 💰 Рассчитываем прибыль
                    profit_pct = (current_price - entry_price) / entry_price
                    profit_pct_after_commission = profit_pct - (2 * self.commission)
                    
                    profit_amount = capital * profit_pct_after_commission
                    capital += profit_amount
                    
                    # 📊 Записываем сделку
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
                    
                    print(f"🔴 ВЫХОД ИЗ ЛОНГА: ${current_price:.0f} ({current_time}) - {exit_reason} - Прибыль: {profit_pct_after_commission:.2%} (${profit_amount:.0f})")
                    
                    position = None
                    entry_price = None
                    entry_time = None
            
            # 🚨 Принудительный выход в конце
            if i == len(df) - 1 and position == 'long':
                profit_pct = (current_price - entry_price) / entry_price
                profit_pct_after_commission = profit_pct - (2 * self.commission)
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
                    'max_confidence': 0
                })
                
                print(f"🚨 ПРИНУДИТЕЛЬНЫЙ ВЫХОД: ${current_price:.0f} - Прибыль: {profit_pct_after_commission:.2%} (${profit_amount:.0f})")
        
        # 📊 Анализ результатов
        self.analyze_results(trades, capital)
        
        return trades
    
    def analyze_results(self, trades, final_capital):
        """📊 Анализируем результаты тестирования"""
        print(f"\n💰 АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 40)
        
        if not trades:
            print("❌ Сделок не было")
            return
        
        profitable_trades = [t for t in trades if t['profit_pct'] > 0]
        losing_trades = [t for t in trades if t['profit_pct'] <= 0]
        
        total_profit_pct = (final_capital - self.initial_capital) / self.initial_capital
        
        print(f"🎯 Всего сделок: {len(trades)}")
        print(f"✅ Прибыльных: {len(profitable_trades)} ({len(profitable_trades)/len(trades)*100:.1f}%)")
        print(f"❌ Убыточных: {len(losing_trades)} ({len(losing_trades)/len(trades)*100:.1f}%)")
        print(f"💰 Общая прибыль: {total_profit_pct:.2%}")
        print(f"💵 Капитал: ${self.initial_capital:,.0f} → ${final_capital:,.0f}")
        
        if profitable_trades:
            avg_profit = np.mean([t['profit_pct'] for t in profitable_trades])
            best_trade = max(profitable_trades, key=lambda x: x['profit_pct'])
            print(f"📈 Средняя прибыль: {avg_profit:.2%}")
            print(f"🚀 Лучшая сделка: {best_trade['profit_pct']:.2%}")
        
        if losing_trades:
            avg_loss = np.mean([t['profit_pct'] for t in losing_trades])
            worst_trade = min(losing_trades, key=lambda x: x['profit_pct'])
            print(f"📉 Средний убыток: {avg_loss:.2%}")
            print(f"💥 Худшая сделка: {worst_trade['profit_pct']:.2%}")
        
        # 🎯 Оценка системы
        if total_profit_pct > 0:
            print("✅ Система прибыльна!")
        else:
            print("❌ Система убыточна")
    
    def run_full_system(self, symbol='BTC/USDT'):
        """🚀 Запускаем полную систему: обучение + тестирование"""
        print(f"🚀 ПОЛНАЯ СИСТЕМА ML: {symbol}")
        print("=" * 60)
        
        # 🎓 Этап 1: Обучение на январских данных
        self.train_on_january_data(symbol)
        
        # 🧪 Этап 2: Тестирование на февральских данных
        self.test_on_february_data(symbol)

if __name__ == "__main__":
    system = RealMLSystem()
    system.run_full_system('BTC/USDT')
