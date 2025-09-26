#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ОБУЧЕНИЕ НА РЕАЛЬНЫХ ДАННЫХ
Обучаем ML модель на настоящих рыночных данных с Binance
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class RealDataTrainer:
    def __init__(self):
        """🚀 Инициализация тренера на реальных данных"""
        self.base_url = "https://api.binance.com/api/v3/klines"
        
        # 🧠 Веса для минимумов (из браузера)
        self.weights_minimums = {
            'priceVelocity': 0.037,
            'ema20Velocity': 0.031,
            'ema20Angle': 0.217,
            'priceDistance': 0.715
        }
        
        # 🧠 Веса для максимумов (зеркальная логика)
        self.weights_maximums = {
            'priceVelocity': 0.037,
            'ema20Velocity': 0.031,
            'ema20Angle': 0.217,
            'priceDistance': 0.715
        }
        
        print("🚀 ТРЕНЕР НА РЕАЛЬНЫХ ДАННЫХ ИНИЦИАЛИЗИРОВАН")
        print("📊 Будем обучаться на настоящих рыночных данных!")
    
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
                print(f"❌ Нет данных для {symbol}")
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
        """🔍 Находим реальные минимумы в данных"""
        minimums = []
        
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            next1 = df.iloc[i+1]
            next2 = df.iloc[i+2]
            
            # 🔻 Условие минимума: текущая цена меньше соседних
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
        """🔺 Находим реальные максимумы в данных"""
        maximums = []
        
        for i in range(2, len(df) - 2):
            current = df.iloc[i]
            prev1 = df.iloc[i-1]
            prev2 = df.iloc[i-2]
            next1 = df.iloc[i+1]
            next2 = df.iloc[i+2]
            
            # 🔺 Условие максимума: текущая цена больше соседних
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
        """📊 Рассчитываем признаки для точки"""
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
    
    def train_on_real_data(self, symbol='BTC/USDT'):
        """🎓 Обучаемся на реальных данных"""
        print(f"\n🎓 ОБУЧЕНИЕ НА РЕАЛЬНЫХ ДАННЫХ: {symbol}")
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
        
        # 📚 Собираем обучающие данные
        training_data = []
        
        # 🔻 Добавляем минимумы
        for min_point in minimums:
            features = self.calculate_features(df, min_point['index'])
            if features:
                training_data.append({
                    'type': 'minimum',
                    'features': features,
                    'price': min_point['price'],
                    'time': min_point['time']
                })
        
        # 🔺 Добавляем максимумы
        for max_point in maximums:
            features = self.calculate_features(df, max_point['index'])
            if features:
                training_data.append({
                    'type': 'maximum',
                    'features': features,
                    'price': max_point['price'],
                    'time': max_point['time']
                })
        
        print(f"📚 Всего обучающих примеров: {len(training_data)}")
        
        # 🧠 Анализируем паттерны
        self.analyze_patterns(training_data)
        
        # 💾 Сохраняем обученную модель
        self.save_trained_model(training_data)
        
        return training_data
    
    def analyze_patterns(self, training_data):
        """🧠 Анализируем паттерны в данных"""
        print(f"\n🧠 АНАЛИЗ ПАТТЕРНОВ")
        print("=" * 40)
        
        minimums = [d for d in training_data if d['type'] == 'minimum']
        maximums = [d for d in training_data if d['type'] == 'maximum']
        
        print(f"🔻 Минимумы: {len(minimums)}")
        print(f"🔺 Максимумы: {len(maximums)}")
        
        # 📊 Анализируем признаки минимумов
        if minimums:
            print(f"\n🔻 АНАЛИЗ МИНИМУМОВ:")
            for feature in ['priceVelocity', 'ema20Velocity', 'ema20Angle', 'priceDistance']:
                values = [m['features'][feature] for m in minimums]
                print(f"  {feature}: мин={min(values):.4f}, макс={max(values):.4f}, ср={np.mean(values):.4f}")
        
        # 📊 Анализируем признаки максимумов
        if maximums:
            print(f"\n🔺 АНАЛИЗ МАКСИМУМОВ:")
            for feature in ['priceVelocity', 'ema20Velocity', 'ema20Angle', 'priceDistance']:
                values = [m['features'][feature] for m in maximums]
                print(f"  {feature}: мин={min(values):.4f}, макс={max(values):.4f}, ср={np.mean(values):.4f}")
    
    def save_trained_model(self, training_data):
        """💾 Сохраняем обученную модель"""
        model_data = {
            'weights_minimums': self.weights_minimums,
            'weights_maximums': self.weights_maximums,
            'training_data': training_data,
            'timestamp': datetime.now().isoformat()
        }
        
        with open('real_trained_model.json', 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 Модель сохранена в real_trained_model.json")
    
    def test_multiple_symbols(self):
        """🧪 Тестируем на нескольких символах"""
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
        
        print(f"\n🧪 ОБУЧЕНИЕ НА НЕСКОЛЬКИХ СИМВОЛАХ")
        print("=" * 60)
        
        all_training_data = []
        
        for symbol in symbols:
            print(f"\n📊 Обучаемся на {symbol}...")
            training_data = self.train_on_real_data(symbol)
            if training_data:
                all_training_data.extend(training_data)
        
        # 💾 Сохраняем общую модель
        if all_training_data:
            self.save_trained_model(all_training_data)
            print(f"\n✅ ОБЩАЯ МОДЕЛЬ СОХРАНЕНА!")
            print(f"📚 Всего примеров: {len(all_training_data)}")

if __name__ == "__main__":
    trainer = RealDataTrainer()
    trainer.test_multiple_symbols()






