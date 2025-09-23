#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📚 БЫСТРЫЙ ИСТОРИЧЕСКИЙ ТРЕНЕР
Обучение на данных 2022-2024 с оптимизацией
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FastHistoricalTrainer:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']  # Уменьшаем количество символов
        self.minimums_database = []
        self.maximums_database = []
        
        # Параметры
        self.min_lookback = 6
        self.min_profit_threshold = 2.0
        self.max_duration = 48
        
        print("📚 БЫСТРЫЙ ИСТОРИЧЕСКИЙ ТРЕНЕР")
        print("📊 Период: 2022-2024")
        print("🎯 Символы: BTC, ETH, ADA")
        print("=" * 50)
    
    def get_fast_data(self, symbol: str, year: int) -> pd.DataFrame:
        """Быстрое получение данных за год"""
        try:
            print(f"📊 Загружаю {symbol} за {year} год...")
            exchange = ccxt.binance()
            
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
            
            since = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            all_ohlcv = []
            current_since = since
            
            # Ограничиваем количество запросов
            max_requests = 50
            request_count = 0
            
            while current_since < end_ts and request_count < max_requests:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    request_count += 1
                    
                    if current_since > end_ts:
                        break
                        
                    time.sleep(0.2)  # Увеличиваем паузу
                    
                except Exception as e:
                    print(f"   ⚠️ Ошибка запроса: {e}")
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            # Фильтруем по конечной дате
            all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end_ts]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            print(f"   ✅ {len(df)} свечей загружено")
            return df
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return pd.DataFrame()
    
    def prepare_simple_features(self, df: pd.DataFrame, idx: int, is_minimum: bool = True) -> dict:
        """Упрощенные признаки"""
        try:
            if idx < 50 or idx >= len(df) - 6:
                return None
            
            current = df.iloc[idx]
            prev = df.iloc[idx - 1]
            
            features = {
                'price_velocity': (current['close'] - prev['close']) / prev['close'],
                'ema20_velocity': (current['ema_20'] - prev['ema_20']) / prev['ema_20'],
                'ema50_velocity': (current['ema_50'] - prev['ema_50']) / prev['ema_50'],
                'ema100_velocity': (current['ema_100'] - prev['ema_100']) / prev['ema_100'],
                'price_distance_ema20': (current['close'] - current['ema_20']) / current['ema_20'],
                'price_distance_ema50': (current['close'] - current['ema_50']) / current['ema_50'],
                'price_distance_ema100': (current['close'] - current['ema_100']) / current['ema_100'],
                'ema20_angle': ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) * 100,
                'ema50_angle': ((current['ema_50'] - prev['ema_50']) / prev['ema_50']) * 100,
                'volatility': df['close'].iloc[idx-20:idx].std() / df['close'].iloc[idx-20:idx].mean(),
                'volume_ratio': current['volume'] / df['volume'].iloc[idx-20:idx].mean() if df['volume'].iloc[idx-20:idx].mean() > 0 else 1
            }
            
            return features
            
        except Exception:
            return None
    
    def find_simple_extremes(self, df: pd.DataFrame, symbol: str):
        """Упрощенный поиск экстремумов"""
        print(f"🔍 Ищу экстремумы в {symbol}...")
        
        minimums_found = 0
        maximums_found = 0
        
        # Ищем минимумы
        for i in range(100, len(df) - 6):
            try:
                current_low = df.iloc[i]['low']
                current_time = df.index[i]
                
                # Проверяем локальный минимум
                is_minimum = True
                for j in range(max(0, i-5), min(len(df), i+6)):
                    if j != i and df.iloc[j]['low'] <= current_low:
                        is_minimum = False
                        break
                
                if not is_minimum:
                    continue
                
                # Ищем максимум в будущем
                max_price = current_low
                max_time = current_time
                
                for j in range(i+1, min(len(df), i+49)):
                    if df.iloc[j]['high'] > max_price:
                        max_price = df.iloc[j]['high']
                        max_time = df.index[j]
                
                profit_percent = ((max_price - current_low) / current_low) * 100
                duration_hours = (max_time - current_time).total_seconds() / 3600
                
                if profit_percent >= self.min_profit_threshold and duration_hours <= self.max_duration:
                    features = self.prepare_simple_features(df, i, is_minimum=True)
                    
                    if features:
                        minimum = {
                            'symbol': symbol,
                            'entry_time': current_time,
                            'exit_time': max_time,
                            'entry_price': current_low,
                            'exit_price': max_price,
                            'profit_percent': profit_percent,
                            'duration_hours': duration_hours,
                            'features': features,
                            'is_profitable': profit_percent >= 3.0
                        }
                        
                        self.minimums_database.append(minimum)
                        minimums_found += 1
                        
            except Exception:
                continue
        
        # Ищем максимумы
        for i in range(100, len(df) - 6):
            try:
                current_high = df.iloc[i]['high']
                current_time = df.index[i]
                
                # Проверяем локальный максимум
                is_maximum = True
                for j in range(max(0, i-5), min(len(df), i+6)):
                    if j != i and df.iloc[j]['high'] >= current_high:
                        is_maximum = False
                        break
                
                if not is_maximum:
                    continue
                
                # Ищем минимум в будущем
                min_price = current_high
                min_time = current_time
                
                for j in range(i+1, min(len(df), i+49)):
                    if df.iloc[j]['low'] < min_price:
                        min_price = df.iloc[j]['low']
                        min_time = df.index[j]
                
                drop_percent = ((min_price - current_high) / current_high) * 100
                duration_hours = (min_time - current_time).total_seconds() / 3600
                
                if drop_percent <= -self.min_profit_threshold and duration_hours <= self.max_duration:
                    features = self.prepare_simple_features(df, i, is_minimum=False)
                    
                    if features:
                        maximum = {
                            'symbol': symbol,
                            'entry_time': current_time,
                            'exit_time': min_time,
                            'entry_price': current_high,
                            'exit_price': min_price,
                            'drop_percent': drop_percent,
                            'duration_hours': duration_hours,
                            'features': features,
                            'is_profitable_exit': drop_percent <= -3.0
                        }
                        
                        self.maximums_database.append(maximum)
                        maximums_found += 1
                        
            except Exception:
                continue
        
        print(f"   ✅ {minimums_found} минимумов, {maximums_found} максимумов")
    
    def collect_fast_data(self):
        """Быстрый сбор данных"""
        print("\n📚 БЫСТРЫЙ СБОР ДАННЫХ")
        print("-" * 30)
        
        self.minimums_database = []
        self.maximums_database = []
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol}")
            
            # Загружаем данные по годам
            for year in range(2022, 2025):
                df = self.get_fast_data(symbol, year)
                if not df.empty:
                    self.find_simple_extremes(df, symbol)
                
                time.sleep(2)  # Пауза между годами
        
        print(f"\n📊 ИТОГО НАЙДЕНО:")
        print(f"🎯 Минимумов: {len(self.minimums_database)}")
        print(f"🔺 Максимумов: {len(self.maximums_database)}")
        
        if self.minimums_database:
            profits = [m['profit_percent'] for m in self.minimums_database]
            profitable = [m for m in self.minimums_database if m['is_profitable']]
            print(f"📈 Средняя прибыль: {np.mean(profits):.2f}%")
            print(f"✅ Прибыльных: {len(profitable)} ({len(profitable)/len(self.minimums_database)*100:.1f}%)")
        
        if self.maximums_database:
            drops = [m['drop_percent'] for m in self.maximums_database]
            good_exits = [m for m in self.maximums_database if m['is_profitable_exit']]
            print(f"📉 Среднее падение: {np.mean(drops):.2f}%")
            print(f"✅ Хороших выходов: {len(good_exits)} ({len(good_exits)/len(self.maximums_database)*100:.1f}%)")
    
    def train_fast_models(self):
        """Быстрое обучение моделей"""
        print("\n🧠 БЫСТРОЕ ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("-" * 30)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Обучение модели минимумов
        if len(self.minimums_database) >= 20:
            print("🎯 Обучение модели минимумов...")
            
            X_min = []
            y_min = []
            feature_names_min = list(self.minimums_database[0]['features'].keys())
            
            for minimum in self.minimums_database:
                features_list = [minimum['features'][name] for name in feature_names_min]
                X_min.append(features_list)
                y_min.append(1 if minimum['is_profitable'] else 0)
            
            X_min = np.array(X_min)
            y_min = np.array(y_min)
            
            # Масштабирование
            scaler_min = StandardScaler()
            X_min_scaled = scaler_min.fit_transform(X_min)
            
            # Обучение модели
            model_min = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
            model_min.fit(X_min_scaled, y_min)
            
            # Сохранение
            min_model_file = f'fast_minimum_model_{timestamp}.pkl'
            min_scaler_file = f'fast_minimum_scaler_{timestamp}.pkl'
            min_features_file = f'fast_minimum_features_{timestamp}.pkl'
            
            with open(min_model_file, 'wb') as f:
                pickle.dump(model_min, f)
            
            with open(min_scaler_file, 'wb') as f:
                pickle.dump(scaler_min, f)
            
            with open(min_features_file, 'wb') as f:
                pickle.dump(feature_names_min, f)
            
            print(f"   ✅ Модель минимумов: {min_model_file}")
            print(f"   📊 Данных: {len(X_min)} образцов")
            print(f"   ✅ Прибыльных: {sum(y_min)} ({sum(y_min)/len(y_min)*100:.1f}%)")
        
        # Обучение модели максимумов
        if len(self.maximums_database) >= 20:
            print("\n🔺 Обучение модели максимумов...")
            
            X_max = []
            y_max = []
            feature_names_max = list(self.maximums_database[0]['features'].keys())
            
            for maximum in self.maximums_database:
                features_list = [maximum['features'][name] for name in feature_names_max]
                X_max.append(features_list)
                y_max.append(1 if maximum['is_profitable_exit'] else 0)
            
            X_max = np.array(X_max)
            y_max = np.array(y_max)
            
            # Масштабирование
            scaler_max = StandardScaler()
            X_max_scaled = scaler_max.fit_transform(X_max)
            
            # Обучение модели
            model_max = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
            model_max.fit(X_max_scaled, y_max)
            
            # Сохранение
            max_model_file = f'fast_maximum_model_{timestamp}.pkl'
            max_scaler_file = f'fast_maximum_scaler_{timestamp}.pkl'
            max_features_file = f'fast_maximum_features_{timestamp}.pkl'
            
            with open(max_model_file, 'wb') as f:
                pickle.dump(model_max, f)
            
            with open(max_scaler_file, 'wb') as f:
                pickle.dump(scaler_max, f)
            
            with open(max_features_file, 'wb') as f:
                pickle.dump(feature_names_max, f)
            
            print(f"   ✅ Модель максимумов: {max_model_file}")
            print(f"   📊 Данных: {len(X_max)} образцов")
            print(f"   ✅ Хороших выходов: {sum(y_max)} ({sum(y_max)/len(y_max)*100:.1f}%)")
        
        # Сохранение метаданных
        metadata = {
            'training_period': '2022-2024',
            'testing_period': '2025',
            'symbols': self.symbols,
            'minimums_count': len(self.minimums_database),
            'maximums_count': len(self.maximums_database),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'RandomForestClassifier',
            'scaler_type': 'StandardScaler'
        }
        
        metadata_file = f'fast_training_metadata_{timestamp}.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Метаданные: {metadata_file}")
    
    def run_fast_training(self):
        """Запуск быстрого обучения"""
        print("🚀 ЗАПУСК БЫСТРОГО ОБУЧЕНИЯ")
        print("=" * 50)
        
        # 1. Сбор данных
        self.collect_fast_data()
        
        if not self.minimums_database and not self.maximums_database:
            print("❌ Не удалось собрать данные")
            return
        
        # 2. Обучение моделей
        self.train_fast_models()
        
        print(f"\n✅ БЫСТРОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📊 Минимумов: {len(self.minimums_database)}")
        print(f"📊 Максимумов: {len(self.maximums_database)}")
        print(f"🧠 Модели готовы для бэктестинга на данных 2025 года")

if __name__ == "__main__":
    trainer = FastHistoricalTrainer()
    trainer.run_fast_training()
