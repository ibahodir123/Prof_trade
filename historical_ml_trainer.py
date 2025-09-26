#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Правильный ML Trainer для исторических данных
Обучение: 2020-2024 годы
Тестирование: 2025 год
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class HistoricalMLTrainer:
    def __init__(self):
        self.train_start = datetime(2020, 1, 1)
        self.train_end = datetime(2024, 12, 31)
        self.test_start = datetime(2025, 1, 1)
        self.test_end = datetime.now()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.movements_database = []
        
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение исторических данных с Binance"""
        try:
            print(f"📊 Загружаю данные {symbol} с {start_date.date()} по {end_date.date()}")
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
                    
                    # Не превышаем конечную дату
                    if current_since > end_ts:
                        break
                        
                    time.sleep(0.1)  # Пауза для API
                    
                    print(f"   Загружено: {len(all_ohlcv)} свечей...")
                    
                except Exception as e:
                    print(f"❌ Ошибка загрузки: {e}")
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            # Фильтруем по конечной дате
            all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end_ts]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            print(f"✅ Загружено {len(df)} свечей для {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def prepare_ml_features(self, df: pd.DataFrame, symbol: str) -> dict:
        """Подготовка 27 ML признаков"""
        try:
            if len(df) < 100:
                return None
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # Берем последние значения
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 1. Velocity группа (3 признака)
            price_velocity = (latest['close'] - prev['close']) / prev['close']
            ema20_velocity = (latest['ema_20'] - prev['ema_20']) / prev['ema_20']
            ema50_velocity = (latest['ema_50'] - prev['ema_50']) / prev['ema_50']
            
            # 2. Acceleration группа (3 признака)  
            price_accel = price_velocity - ((prev['close'] - df.iloc[-3]['close']) / df.iloc[-3]['close'])
            ema20_accel = ema20_velocity - ((prev['ema_20'] - df.iloc[-3]['ema_20']) / df.iloc[-3]['ema_20'])
            ema50_accel = ema50_velocity - ((prev['ema_50'] - df.iloc[-3]['ema_50']) / df.iloc[-3]['ema_50'])
            
            # 3. Velocity Ratios группа (3 признака)
            velocity_ratio_20_50 = ema20_velocity / ema50_velocity if ema50_velocity != 0 else 0
            velocity_ratio_price_20 = price_velocity / ema20_velocity if ema20_velocity != 0 else 0
            velocity_ratio_price_50 = price_velocity / ema50_velocity if ema50_velocity != 0 else 0
            
            # 4. Distance to EMAs группа (3 признака)
            distance_to_ema20 = (latest['close'] - latest['ema_20']) / latest['ema_20']
            distance_to_ema50 = (latest['close'] - latest['ema_50']) / latest['ema_50']
            distance_to_ema100 = (latest['close'] - latest['ema_100']) / latest['ema_100']
            
            # 5. Distance Changes группа (3 признака)
            prev_dist_20 = (prev['close'] - prev['ema_20']) / prev['ema_20']
            prev_dist_50 = (prev['close'] - prev['ema_50']) / prev['ema_50']
            prev_dist_100 = (prev['close'] - prev['ema_100']) / prev['ema_100']
            
            distance_change_20 = distance_to_ema20 - prev_dist_20
            distance_change_50 = distance_to_ema50 - prev_dist_50
            distance_change_100 = distance_to_ema100 - prev_dist_100
            
            # 6. EMA Angles группа (3 признака)
            ema20_angle = (latest['ema_20'] - df.iloc[-10]['ema_20']) / df.iloc[-10]['ema_20']
            ema50_angle = (latest['ema_50'] - df.iloc[-10]['ema_50']) / df.iloc[-10]['ema_50']
            ema100_angle = (latest['ema_100'] - df.iloc[-10]['ema_100']) / df.iloc[-10]['ema_100']
            
            # 7. Angle Changes группа (3 признака)
            prev_ema20_angle = (prev['ema_20'] - df.iloc[-11]['ema_20']) / df.iloc[-11]['ema_20']
            prev_ema50_angle = (prev['ema_50'] - df.iloc[-11]['ema_50']) / df.iloc[-11]['ema_50']
            prev_ema100_angle = (prev['ema_100'] - df.iloc[-11]['ema_100']) / df.iloc[-11]['ema_100']
            
            ema20_angle_change = ema20_angle - prev_ema20_angle
            ema50_angle_change = ema50_angle - prev_ema50_angle
            ema100_angle_change = ema100_angle - prev_ema100_angle
            
            # 8. EMA Relationships группа (3 признака)
            ema20_vs_50 = (latest['ema_20'] - latest['ema_50']) / latest['ema_50']
            ema50_vs_100 = (latest['ema_50'] - latest['ema_100']) / latest['ema_100']
            ema20_vs_100 = (latest['ema_20'] - latest['ema_100']) / latest['ema_100']
            
            # 9. Synchronization группа (6 признаков)
            price_ema20_sync = 1 if (price_velocity > 0) == (ema20_velocity > 0) else 0
            price_ema50_sync = 1 if (price_velocity > 0) == (ema50_velocity > 0) else 0
            ema20_ema50_sync = 1 if (ema20_velocity > 0) == (ema50_velocity > 0) else 0
            all_up_sync = 1 if all([price_velocity > 0, ema20_velocity > 0, ema50_velocity > 0]) else 0
            all_down_sync = 1 if all([price_velocity < 0, ema20_velocity < 0, ema50_velocity < 0]) else 0
            mixed_signals = 1 if len(set([price_velocity > 0, ema20_velocity > 0, ema50_velocity > 0])) > 1 else 0
            
            features = {
                # Velocity группа
                'price_velocity': price_velocity,
                'ema20_velocity': ema20_velocity,
                'ema50_velocity': ema50_velocity,
                
                # Acceleration группа
                'price_acceleration': price_accel,
                'ema20_acceleration': ema20_accel,
                'ema50_acceleration': ema50_accel,
                
                # Velocity Ratios группа
                'velocity_ratio_20_50': velocity_ratio_20_50,
                'velocity_ratio_price_20': velocity_ratio_price_20,
                'velocity_ratio_price_50': velocity_ratio_price_50,
                
                # Distance to EMAs группа
                'distance_to_ema20': distance_to_ema20,
                'distance_to_ema50': distance_to_ema50,
                'distance_to_ema100': distance_to_ema100,
                
                # Distance Changes группа
                'distance_change_20': distance_change_20,
                'distance_change_50': distance_change_50,
                'distance_change_100': distance_change_100,
                
                # EMA Angles группа
                'ema20_angle': ema20_angle,
                'ema50_angle': ema50_angle,
                'ema100_angle': ema100_angle,
                
                # Angle Changes группа
                'ema20_angle_change': ema20_angle_change,
                'ema50_angle_change': ema50_angle_change,
                'ema100_angle_change': ema100_angle_change,
                
                # EMA Relationships группа
                'ema20_vs_50': ema20_vs_50,
                'ema50_vs_100': ema50_vs_100,
                'ema20_vs_100': ema20_vs_100,
                
                # Synchronization группа
                'price_ema20_sync': price_ema20_sync,
                'price_ema50_sync': price_ema50_sync,
                'ema20_ema50_sync': ema20_ema50_sync,
                'all_up_sync': all_up_sync,
                'all_down_sync': all_down_sync,
                'mixed_signals': mixed_signals
            }
            
            return features
            
        except Exception as e:
            print(f"❌ Ошибка подготовки признаков: {e}")
            return None
    
    def find_movements_in_data(self, df: pd.DataFrame, symbol: str):
        """Поиск всех min->max движений в данных"""
        print(f"🔍 Ищу движения в {symbol}...")
        
        movements_found = 0
        
        for i in range(100, len(df) - 24):  # Оставляем 24 часа для проверки будущего
            try:
                current_price = df.iloc[i]['close']
                
                # Ищем локальный минимум
                lookback = 6
                is_local_min = True
                
                for j in range(max(0, i-lookback), min(len(df), i+lookback+1)):
                    if j != i and df.iloc[j]['low'] <= current_price:
                        is_local_min = False
                        break
                
                if not is_local_min:
                    continue
                
                # Ищем максимум в будущем (следующие 24 часа)
                max_price = current_price
                max_idx = i
                
                for j in range(i+1, min(len(df), i+25)):
                    if df.iloc[j]['high'] > max_price:
                        max_price = df.iloc[j]['high']
                        max_idx = j
                
                # Проверяем прибыльность движения
                movement_percent = ((max_price - current_price) / current_price) * 100
                
                if movement_percent >= 1.0:  # Минимум 1% движение
                    # Получаем признаки на точке входа (минимум)
                    entry_slice = df.iloc[:i+1]
                    features = self.prepare_ml_features(entry_slice, symbol)
                    
                    if features:
                        movement = {
                            'symbol': symbol,
                            'entry_time': df.index[i],
                            'exit_time': df.index[max_idx],
                            'entry_price': current_price,
                            'exit_price': max_price,
                            'movement_percent': movement_percent,
                            'duration_hours': max_idx - i,
                            'features': features
                        }
                        
                        self.movements_database.append(movement)
                        movements_found += 1
                        
                        if movements_found % 50 == 0:
                            print(f"   Найдено движений: {movements_found}")
                            
            except Exception as e:
                continue
        
        print(f"✅ Найдено {movements_found} движений в {symbol}")
    
    def collect_training_data(self):
        """Сбор данных для обучения (2020-2024)"""
        print("📚 СБОР ДАННЫХ ДЛЯ ОБУЧЕНИЯ (2020-2024)")
        print("=" * 50)
        
        self.movements_database = []
        
        for symbol in self.symbols:
            df = self.get_historical_data(symbol, self.train_start, self.train_end)
            if not df.empty:
                self.find_movements_in_data(df, symbol)
            time.sleep(1)  # Пауза между символами
        
        # Сохраняем тренировочные данные
        with open('historical_movements_2020_2024.json', 'w', encoding='utf-8') as f:
            # Конвертируем datetime в строки для JSON
            movements_for_json = []
            for movement in self.movements_database:
                movement_copy = movement.copy()
                movement_copy['entry_time'] = movement['entry_time'].isoformat()
                movement_copy['exit_time'] = movement['exit_time'].isoformat()
                movements_for_json.append(movement_copy)
            
            json.dump(movements_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"\n📊 СТАТИСТИКА ТРЕНИРОВОЧНЫХ ДАННЫХ:")
        print(f"   Всего движений: {len(self.movements_database)}")
        
        if self.movements_database:
            profits = [m['movement_percent'] for m in self.movements_database]
            print(f"   Средняя прибыль: {np.mean(profits):.2f}%")
            print(f"   Медианная прибыль: {np.median(profits):.2f}%")
            print(f"   Максимальная прибыль: {np.max(profits):.2f}%")
            
            # Распределение по группам
            small = len([p for p in profits if 1 <= p < 3])
            medium = len([p for p in profits if 3 <= p < 7])
            large = len([p for p in profits if p >= 7])
            
            print(f"   Малые движения (1-3%): {small}")
            print(f"   Средние движения (3-7%): {medium}")
            print(f"   Крупные движения (7%+): {large}")
    
    def train_model(self):
        """Обучение ML модели"""
        print("\n🧠 ОБУЧЕНИЕ ML МОДЕЛИ")
        print("=" * 30)
        
        if not self.movements_database:
            print("❌ Нет данных для обучения!")
            return
        
        # Подготавливаем данные
        X = []
        y = []
        
        feature_names = list(self.movements_database[0]['features'].keys())
        
        for movement in self.movements_database:
            features_list = [movement['features'][name] for name in feature_names]
            X.append(features_list)
            
            # Создаем метки классов
            profit = movement['movement_percent']
            if profit < 3:
                label = 0  # Малое движение
            elif profit < 7:
                label = 1  # Среднее движение
            else:
                label = 2  # Крупное движение
            
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Подготовлено данных: {len(X)} образцов, {len(feature_names)} признаков")
        
        # Разделяем на train/validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Обучаем модель
        model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=15)
        model.fit(X_train, y_train)
        
        # Оценка модели
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        print(f"✅ Точность на обучении: {train_acc:.3f}")
        print(f"✅ Точность на валидации: {val_acc:.3f}")
        
        # Важность признаков
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 ТОП-10 важных признаков:")
        for name, importance in feature_importance[:10]:
            print(f"   {name}: {importance:.4f}")
        
        # Сохраняем модель
        with open('historical_model_2020_2024.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('historical_feature_names.pkl', 'wb') as f:
            pickle.dump(feature_names, f)
        
        print(f"\n💾 Модель сохранена в 'historical_model_2020_2024.pkl'")
        print(f"💾 Признаки сохранены в 'historical_feature_names.pkl'")
        
        return model, feature_names

if __name__ == "__main__":
    print("🚀 ИСТОРИЧЕСКИЙ ML TRAINER")
    print("📅 Обучение: 2020-2024 годы")
    print("🎯 Тестирование: 2025 год")
    print("=" * 40)
    
    trainer = HistoricalMLTrainer()
    
    print("1️⃣ Сбор тренировочных данных...")
    trainer.collect_training_data()
    
    print("\n2️⃣ Обучение модели...")
    trainer.train_model()
    
    print("\n✅ ГОТОВО! Теперь можно запустить historical_backtest.py")







