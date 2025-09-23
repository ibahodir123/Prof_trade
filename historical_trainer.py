#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
📚 ИСТОРИЧЕСКИЙ ТРЕНЕР ДЛЯ ДОЛГОСРОЧНОГО ОБУЧЕНИЯ
Обучение на данных с 2017 по 2024 год (максимум доступных данных)
Сохранение модели для бэктестинга на новых данных
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

class HistoricalTrainer:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        
        # Базы данных
        self.minimums_database = []
        self.maximums_database = []
        
        # Параметры обучения
        self.min_lookback = 6
        self.min_profit_threshold = 2.0
        self.max_duration = 72  # Увеличиваем для исторических данных
        self.verification_hours = 12
        
        # Периоды обучения
        self.training_start = datetime(2022, 1, 1)  # Начало обучения с 2022
        self.training_end = datetime(2024, 12, 31)  # Конец обучения
        self.testing_start = datetime(2025, 1, 1)   # Начало тестирования
        
        print("📚 ИСТОРИЧЕСКИЙ ТРЕНЕР ДЛЯ ДОЛГОСРОЧНОГО ОБУЧЕНИЯ")
        print("📊 Период обучения: 2022-2024")
        print("🧪 Период тестирования: 2025")
        print("=" * 60)
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение исторических данных"""
        try:
            print(f"📊 Загружаю {symbol} с {start_date.strftime('%Y-%m-%d')} по {end_date.strftime('%Y-%m-%d')}...")
            exchange = ccxt.binance()
            
            since = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            all_ohlcv = []
            current_since = since
            
            while current_since < end_ts:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    if current_since > end_ts:
                        break
                        
                    time.sleep(0.1)  # Ограничение запросов
                    
                except Exception as e:
                    print(f"   ⚠️ Ошибка загрузки: {e}")
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
    
    def prepare_historical_minimum_features(self, df: pd.DataFrame, min_idx: int) -> dict:
        """Подготовка признаков для исторических минимумов"""
        try:
            if min_idx < 50 or min_idx >= len(df) - 6:
                return None
            
            current = df.iloc[min_idx]
            prev = df.iloc[min_idx - 1]
            prev_2 = df.iloc[min_idx - 2]
            
            # Расширенные признаки для исторических данных
            features = {
                # Скорости
                'price_velocity': (current['close'] - prev['close']) / prev['close'],
                'ema20_velocity': (current['ema_20'] - prev['ema_20']) / prev['ema_20'],
                'ema50_velocity': (current['ema_50'] - prev['ema_50']) / prev['ema_50'],
                'ema100_velocity': (current['ema_100'] - prev['ema_100']) / prev['ema_100'],
                
                # Ускорение
                'price_acceleration': ((current['close'] - prev['close']) / prev['close']) - 
                                   ((prev['close'] - prev_2['close']) / prev_2['close']),
                
                # Расстояния от EMA
                'price_distance_ema20': (current['close'] - current['ema_20']) / current['ema_20'],
                'price_distance_ema50': (current['close'] - current['ema_50']) / current['ema_50'],
                'price_distance_ema100': (current['close'] - current['ema_100']) / current['ema_100'],
                
                # Углы
                'ema20_angle': ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) * 100,
                'ema50_angle': ((current['ema_50'] - prev['ema_50']) / prev['ema_50']) * 100,
                'ema100_angle': ((current['ema_100'] - prev['ema_100']) / prev['ema_100']) * 100,
                
                # Волатильность
                'volatility': df['close'].iloc[min_idx-20:min_idx].std() / df['close'].iloc[min_idx-20:min_idx].mean(),
                
                # Объем (если доступен)
                'volume_ratio': current['volume'] / df['volume'].iloc[min_idx-20:min_idx].mean() if df['volume'].iloc[min_idx-20:min_idx].mean() > 0 else 1,
                
                # Изменения расстояний
                'distance_change_ema20': (current['close'] - current['ema_20']) / current['ema_20'] - 
                                       (prev['close'] - prev['ema_20']) / prev['ema_20'],
                'distance_change_ema50': (current['close'] - current['ema_50']) / current['ema_50'] - 
                                       (prev['close'] - prev['ema_50']) / prev['ema_50'],
                
                # Соотношения скоростей
                'velocity_ratio_price_ema20': (current['close'] - prev['close']) / prev['close'] / 
                                            ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) if (current['ema_20'] - prev['ema_20']) / prev['ema_20'] != 0 else 0,
                'velocity_ratio_ema20_ema50': ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) / 
                                            ((current['ema_50'] - prev['ema_50']) / prev['ema_50']) if (current['ema_50'] - prev['ema_50']) / prev['ema_50'] != 0 else 0,
                
                # Расстояния между EMA
                'ema20_to_ema50': (current['ema_20'] - current['ema_50']) / current['ema_50'],
                'ema50_to_ema100': (current['ema_50'] - current['ema_100']) / current['ema_100'],
                'ema20_to_ema100': (current['ema_20'] - current['ema_100']) / current['ema_100']
            }
            
            return features
            
        except Exception:
            return None
    
    def prepare_historical_maximum_features(self, df: pd.DataFrame, max_idx: int) -> dict:
        """Подготовка признаков для исторических максимумов"""
        try:
            if max_idx < 50 or max_idx >= len(df) - 6:
                return None
            
            current = df.iloc[max_idx]
            prev = df.iloc[max_idx - 1]
            prev_2 = df.iloc[max_idx - 2]
            
            # Расширенные признаки для максимумов
            price_velocity = (current['close'] - prev['close']) / prev['close']
            ema20_velocity = (current['ema_20'] - prev['ema_20']) / prev['ema_20']
            ema50_velocity = (current['ema_50'] - prev['ema_50']) / prev['ema_50']
            ema100_velocity = (current['ema_100'] - prev['ema_100']) / prev['ema_100']
            
            features = {
                # Скорости
                'price_velocity': price_velocity,
                'ema20_velocity': ema20_velocity,
                'ema50_velocity': ema50_velocity,
                'ema100_velocity': ema100_velocity,
                
                # Ускорение
                'price_acceleration': price_velocity - ((prev['close'] - prev_2['close']) / prev_2['close']),
                
                # Расстояния от EMA
                'price_distance_ema20': (current['close'] - current['ema_20']) / current['ema_20'],
                'price_distance_ema50': (current['close'] - current['ema_50']) / current['ema_50'],
                'price_distance_ema100': (current['close'] - current['ema_100']) / current['ema_100'],
                
                # Углы
                'ema20_angle': ema20_velocity * 100,
                'ema50_angle': ema50_velocity * 100,
                'ema100_angle': ema100_velocity * 100,
                
                # Волатильность
                'volatility': df['close'].iloc[max_idx-20:max_idx].std() / df['close'].iloc[max_idx-20:max_idx].mean(),
                
                # Объем
                'volume_ratio': current['volume'] / df['volume'].iloc[max_idx-20:max_idx].mean() if df['volume'].iloc[max_idx-20:max_idx].mean() > 0 else 1,
                
                # Изменения расстояний
                'distance_change_ema20': (current['close'] - current['ema_20']) / current['ema_20'] - 
                                       (prev['close'] - prev['ema_20']) / prev['ema_20'],
                'distance_change_ema50': (current['close'] - current['ema_50']) / current['ema_50'] - 
                                       (prev['close'] - prev['ema_50']) / prev['ema_50'],
                
                # Соотношения скоростей
                'velocity_ratio_price_ema20': price_velocity / ema20_velocity if ema20_velocity != 0 else 0,
                'velocity_ratio_ema20_ema50': ema20_velocity / ema50_velocity if ema50_velocity != 0 else 0,
                'velocity_ratio_ema50_ema100': ema50_velocity / ema100_velocity if ema100_velocity != 0 else 0,
                
                # Расстояния между EMA
                'ema20_to_ema50': (current['ema_20'] - current['ema_50']) / current['ema_50'],
                'ema50_to_ema100': (current['ema_50'] - current['ema_100']) / current['ema_100'],
                'ema20_to_ema100': (current['ema_20'] - current['ema_100']) / current['ema_100'],
                
                # Наклоны EMA
                'ema20_slope': (current['ema_20'] - df.iloc[max_idx-5]['ema_20']) / 5,
                'ema50_slope': (current['ema_50'] - df.iloc[max_idx-5]['ema_50']) / 5,
                'ema100_slope': (current['ema_100'] - df.iloc[max_idx-5]['ema_100']) / 5,
                
                # Соотношения EMA
                'ema20_ema50_ratio': current['ema_20'] / current['ema_50'],
                'ema50_ema100_ratio': current['ema_50'] / current['ema_100'],
                'ema20_ema100_ratio': current['ema_20'] / current['ema_100']
            }
            
            return features
            
        except Exception:
            return None
    
    def find_historical_minimums(self, df: pd.DataFrame, symbol: str):
        """Поиск исторических минимумов"""
        print(f"🎯 Ищу минимумы в {symbol}...")
        
        minimums_found = 0
        
        for i in range(100, len(df) - 6):
            try:
                current_low = df.iloc[i]['low']
                current_time = df.index[i]
                
                # Проверяем, что это локальный минимум
                is_minimum = True
                for j in range(max(0, i-self.min_lookback), min(len(df), i+self.min_lookback+1)):
                    if j != i and df.iloc[j]['low'] <= current_low:
                        is_minimum = False
                        break
                
                if not is_minimum:
                    continue
                
                # Ищем максимум в будущем
                max_price = current_low
                max_idx = i
                max_time = current_time
                
                for j in range(i+1, min(len(df), i+73)):  # Увеличиваем поиск для исторических данных
                    if df.iloc[j]['high'] > max_price:
                        max_price = df.iloc[j]['high']
                        max_idx = j
                        max_time = df.index[j]
                
                profit_percent = ((max_price - current_low) / current_low) * 100
                duration_hours = (max_time - current_time).total_seconds() / 3600
                
                # Фильтруем по критериям
                if (profit_percent >= self.min_profit_threshold and 
                    duration_hours <= self.max_duration):
                    
                    features = self.prepare_historical_minimum_features(df, i)
                    
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
                            'is_profitable': profit_percent >= 3.0,
                            'data_age_hours': (datetime.now() - current_time).total_seconds() / 3600
                        }
                        
                        self.minimums_database.append(minimum)
                        minimums_found += 1
                        
            except Exception:
                continue
        
        print(f"   ✅ {minimums_found} минимумов")
    
    def find_historical_maximums(self, df: pd.DataFrame, symbol: str):
        """Поиск исторических максимумов"""
        print(f"🔺 Ищу максимумы в {symbol}...")
        
        maximums_found = 0
        
        for i in range(100, len(df) - 6):
            try:
                current_high = df.iloc[i]['high']
                current_time = df.index[i]
                
                # Проверяем, что это локальный максимум
                is_maximum = True
                for j in range(max(0, i-self.min_lookback), min(len(df), i+self.min_lookback+1)):
                    if j != i and df.iloc[j]['high'] >= current_high:
                        is_maximum = False
                        break
                
                if not is_maximum:
                    continue
                
                # Ищем минимум в будущем
                min_price = current_high
                min_idx = i
                min_time = current_time
                
                for j in range(i+1, min(len(df), i+73)):  # Увеличиваем поиск для исторических данных
                    if df.iloc[j]['low'] < min_price:
                        min_price = df.iloc[j]['low']
                        min_idx = j
                        min_time = df.index[j]
                
                drop_percent = ((min_price - current_high) / current_high) * 100
                duration_hours = (min_time - current_time).total_seconds() / 3600
                
                # Фильтруем по критериям
                if (drop_percent <= -self.min_profit_threshold and 
                    duration_hours <= self.max_duration):
                    
                    features = self.prepare_historical_maximum_features(df, i)
                    
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
                            'is_profitable_exit': drop_percent <= -3.0,
                            'data_age_hours': (datetime.now() - current_time).total_seconds() / 3600
                        }
                        
                        self.maximums_database.append(maximum)
                        maximums_found += 1
                        
            except Exception:
                continue
        
        print(f"   ✅ {maximums_found} максимумов")
    
    def collect_historical_data(self):
        """Сбор исторических данных"""
        print("\n📚 СБОР ИСТОРИЧЕСКИХ ДАННЫХ")
        print("-" * 40)
        
        self.minimums_database = []
        self.maximums_database = []
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol}")
            
            # Загружаем данные по годам для экономии памяти
            for year in range(2022, 2025):
                start_date = datetime(year, 1, 1)
                end_date = datetime(year, 12, 31)
                
                df = self.get_historical_data(symbol, start_date, end_date)
                if not df.empty:
                    self.find_historical_minimums(df, symbol)
                    self.find_historical_maximums(df, symbol)
                
                time.sleep(1)  # Пауза между годами
        
        print(f"\n📊 ИТОГО НАЙДЕНО:")
        print(f"🎯 Минимумов: {len(self.minimums_database)}")
        print(f"🔺 Максимумов: {len(self.maximums_database)}")
        
        if self.minimums_database:
            profits = [m['profit_percent'] for m in self.minimums_database]
            profitable = [m for m in self.minimums_database if m['is_profitable']]
            print(f"📈 Средняя прибыль: {np.mean(profits):.2f}%")
            print(f"✅ Прибыльных минимумов: {len(profitable)} ({len(profitable)/len(self.minimums_database)*100:.1f}%)")
        
        if self.maximums_database:
            drops = [m['drop_percent'] for m in self.maximums_database]
            good_exits = [m for m in self.maximums_database if m['is_profitable_exit']]
            print(f"📉 Среднее падение: {np.mean(drops):.2f}%")
            print(f"✅ Хороших выходов: {len(good_exits)} ({len(good_exits)/len(self.maximums_database)*100:.1f}%)")
    
    def train_historical_models(self):
        """Обучение исторических моделей"""
        print("\n🧠 ОБУЧЕНИЕ ИСТОРИЧЕСКИХ МОДЕЛЕЙ")
        print("-" * 40)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Обучение модели минимумов
        if len(self.minimums_database) >= 50:
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
            model_min = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
            model_min.fit(X_min_scaled, y_min)
            
            # Сохранение
            min_model_file = f'historical_minimum_model_{timestamp}.pkl'
            min_scaler_file = f'historical_minimum_scaler_{timestamp}.pkl'
            min_features_file = f'historical_minimum_features_{timestamp}.pkl'
            
            with open(min_model_file, 'wb') as f:
                pickle.dump(model_min, f)
            
            with open(min_scaler_file, 'wb') as f:
                pickle.dump(scaler_min, f)
            
            with open(min_features_file, 'wb') as f:
                pickle.dump(feature_names_min, f)
            
            print(f"   ✅ Модель минимумов сохранена: {min_model_file}")
            print(f"   📊 Данных: {len(X_min)} образцов, {len(feature_names_min)} признаков")
            print(f"   ✅ Прибыльных: {sum(y_min)} ({sum(y_min)/len(y_min)*100:.1f}%)")
        
        # Обучение модели максимумов
        if len(self.maximums_database) >= 50:
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
            model_max = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
            model_max.fit(X_max_scaled, y_max)
            
            # Сохранение
            max_model_file = f'historical_maximum_model_{timestamp}.pkl'
            max_scaler_file = f'historical_maximum_scaler_{timestamp}.pkl'
            max_features_file = f'historical_maximum_features_{timestamp}.pkl'
            
            with open(max_model_file, 'wb') as f:
                pickle.dump(model_max, f)
            
            with open(max_scaler_file, 'wb') as f:
                pickle.dump(scaler_max, f)
            
            with open(max_features_file, 'wb') as f:
                pickle.dump(feature_names_max, f)
            
            print(f"   ✅ Модель максимумов сохранена: {max_model_file}")
            print(f"   📊 Данных: {len(X_max)} образцов, {len(feature_names_max)} признаков")
            print(f"   ✅ Хороших выходов: {sum(y_max)} ({sum(y_max)/len(y_max)*100:.1f}%)")
        
        # Сохранение метаданных
        metadata = {
            'training_period': f"{self.training_start.strftime('%Y-%m-%d')} - {self.training_end.strftime('%Y-%m-%d')}",
            'testing_period': f"{self.testing_start.strftime('%Y-%m-%d')} - настоящее время",
            'symbols': self.symbols,
            'minimums_count': len(self.minimums_database),
            'maximums_count': len(self.maximums_database),
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_type': 'RandomForestClassifier',
            'scaler_type': 'StandardScaler'
        }
        
        metadata_file = f'historical_training_metadata_{timestamp}.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 Метаданные сохранены: {metadata_file}")
    
    def run_historical_training(self):
        """Запуск исторического обучения"""
        print("🚀 ЗАПУСК ИСТОРИЧЕСКОГО ОБУЧЕНИЯ")
        print("=" * 60)
        
        # 1. Сбор исторических данных
        self.collect_historical_data()
        
        if not self.minimums_database and not self.maximums_database:
            print("❌ Не удалось собрать исторические данные")
            return
        
        # 2. Обучение моделей
        self.train_historical_models()
        
        print(f"\n✅ ИСТОРИЧЕСКОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"📊 Минимумов: {len(self.minimums_database)}")
        print(f"📊 Максимумов: {len(self.maximums_database)}")
        print(f"🧠 Модели готовы для бэктестинга на данных 2025 года")

if __name__ == "__main__":
    trainer = HistoricalTrainer()
    trainer.run_historical_training()
