#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔺 ЧИСТЫЙ ТРЕНЕР МАКСИМУМОВ
Обучение только на EMA-основанных признаках без RSI
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class CleanMaximumTrainer:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.maximums_database = []
        
        # Параметры для поиска максимумов
        self.min_lookback = 6
        self.min_drop_threshold = 2.0
        self.max_duration = 48
        self.verification_hours = 6
        
        print("🔺 ЧИСТЫЙ ТРЕНЕР МАКСИМУМОВ (БЕЗ RSI)")
        print("📊 Только EMA-основанные признаки")
        print("=" * 50)
        
    def get_recent_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Получение актуальных данных"""
        try:
            print(f"📊 Загружаю {symbol} за {days} дней...")
            exchange = ccxt.binance()
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            since = int(start_time.timestamp() * 1000)
            end_ts = int(end_time.timestamp() * 1000)
            
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
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end_ts]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            print(f"   ✅ {len(df)} свечей")
            return df
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return pd.DataFrame()
    
    def prepare_clean_maximum_features(self, df: pd.DataFrame, max_idx: int) -> dict:
        """Подготовка чистых EMA-признаков для максимума"""
        try:
            if max_idx < 50 or max_idx >= len(df) - 6:
                return None
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            current = df.iloc[max_idx]
            prev = df.iloc[max_idx - 1]
            prev_2 = df.iloc[max_idx - 2]
            
            # 🔺 ТОЛЬКО EMA-ОСНОВАННЫЕ ПРИЗНАКИ
            
            # 1. Скорости (на максимуме)
            price_velocity = (current['close'] - prev['close']) / prev['close']
            ema20_velocity = (current['ema_20'] - prev['ema_20']) / prev['ema_20']
            ema50_velocity = (current['ema_50'] - prev['ema_50']) / prev['ema_50']
            ema100_velocity = (current['ema_100'] - prev['ema_100']) / prev['ema_100']
            
            # 2. Ускорение (критично для максимумов!)
            prev_price_velocity = (prev['close'] - prev_2['close']) / prev_2['close']
            price_acceleration = price_velocity - prev_price_velocity
            
            # 3. Расстояния (цена ВЫШЕ EMA - перекупленность)
            price_distance_ema20 = (current['close'] - current['ema_20']) / current['ema_20']
            price_distance_ema50 = (current['close'] - current['ema_50']) / current['ema_50']
            price_distance_ema100 = (current['close'] - current['ema_100']) / current['ema_100']
            
            # 4. Углы тренда (положительные = перекупленность)
            ema20_angle = ema20_velocity * 100
            ema50_angle = ema50_velocity * 100
            ema100_angle = ema100_velocity * 100
            
            # 5. Объем
            avg_volume = df['volume'].iloc[max_idx-20:max_idx].mean()
            volume_ratio = current['volume'] / avg_volume if avg_volume > 0 else 1
            
            # 6. Волатильность
            volatility = df['close'].iloc[max_idx-20:max_idx].std() / df['close'].iloc[max_idx-20:max_idx].mean()
            
            # 7. Изменения расстояний
            prev_dist_ema20 = (prev['close'] - prev['ema_20']) / prev['ema_20']
            prev_dist_ema50 = (prev['close'] - prev['ema_50']) / prev['ema_50']
            distance_change_ema20 = price_distance_ema20 - prev_dist_ema20
            distance_change_ema50 = price_distance_ema50 - prev_dist_ema50
            
            # 8. Соотношения скоростей
            velocity_ratio_price_ema20 = price_velocity / ema20_velocity if ema20_velocity != 0 else 0
            velocity_ratio_ema20_ema50 = ema20_velocity / ema50_velocity if ema50_velocity != 0 else 0
            velocity_ratio_ema50_ema100 = ema50_velocity / ema100_velocity if ema100_velocity != 0 else 0
            
            # 9. Расстояния между EMA
            ema20_to_ema50 = (current['ema_20'] - current['ema_50']) / current['ema_50']
            ema50_to_ema100 = (current['ema_50'] - current['ema_100']) / current['ema_100']
            ema20_to_ema100 = (current['ema_20'] - current['ema_100']) / current['ema_100']
            
            features = {
                # Скорости
                'price_velocity': price_velocity,
                'ema20_velocity': ema20_velocity,
                'ema50_velocity': ema50_velocity,
                'ema100_velocity': ema100_velocity,
                
                # Ускорение
                'price_acceleration': price_acceleration,
                
                # Расстояния от EMA
                'price_distance_ema20': price_distance_ema20,
                'price_distance_ema50': price_distance_ema50,
                'price_distance_ema100': price_distance_ema100,
                
                # Углы
                'ema20_angle': ema20_angle,
                'ema50_angle': ema50_angle,
                'ema100_angle': ema100_angle,
                
                # Объем и волатильность
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                
                # Изменения расстояний
                'distance_change_ema20': distance_change_ema20,
                'distance_change_ema50': distance_change_ema50,
                
                # Соотношения скоростей
                'velocity_ratio_price_ema20': velocity_ratio_price_ema20,
                'velocity_ratio_ema20_ema50': velocity_ratio_ema20_ema50,
                'velocity_ratio_ema50_ema100': velocity_ratio_ema50_ema100,
                
                # Расстояния между EMA
                'ema20_to_ema50': ema20_to_ema50,
                'ema50_to_ema100': ema50_to_ema100,
                'ema20_to_ema100': ema20_to_ema100
            }
            
            return features
            
        except Exception:
            return None
    
    def find_clean_maximums(self, df: pd.DataFrame, symbol: str):
        """Поиск чистых максимумов"""
        print(f"🔺 Ищу максимумы в {symbol}...")
        
        maximums_found = 0
        
        for i in range(50, len(df) - 6):
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
                
                for j in range(i+1, min(len(df), i+49)):
                    if df.iloc[j]['low'] < min_price:
                        min_price = df.iloc[j]['low']
                        min_idx = j
                        min_time = df.index[j]
                
                drop_percent = ((min_price - current_high) / current_high) * 100
                duration_hours = (min_time - current_time).total_seconds() / 3600
                
                # Фильтруем по критериям
                if (drop_percent <= -self.min_drop_threshold and 
                    duration_hours <= self.max_duration):
                    
                    features = self.prepare_clean_maximum_features(df, i)
                    
                    if features:
                        # Проверяем качество признаков
                        if (features['price_distance_ema20'] > 0.01 and  # Цена выше EMA20
                            features['volume_ratio'] > 0.5 and  # Объем не слишком низкий
                            features['ema20_angle'] > 0):  # EMA20 направлена вверх
                            
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
    
    def collect_clean_data(self):
        """Сбор чистых данных"""
        print("\n📚 СБОР ЧИСТЫХ ДАННЫХ МАКСИМУМОВ")
        print("-" * 40)
        
        self.maximums_database = []
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol}")
            df = self.get_recent_data(symbol, days=30)
            if not df.empty:
                self.find_clean_maximums(df, symbol)
            time.sleep(0.5)
        
        print(f"\n📊 ИТОГО НАЙДЕНО: {len(self.maximums_database)} максимумов")
        
        if self.maximums_database:
            drops = [m['drop_percent'] for m in self.maximums_database]
            durations = [m['duration_hours'] for m in self.maximums_database]
            good_exits = [m for m in self.maximums_database if m['is_profitable_exit']]
            
            print(f"📉 Среднее падение: {np.mean(drops):.2f}%")
            print(f"⏱️ Средняя длительность: {np.mean(durations):.1f} часов")
            print(f"🎯 Максимальное падение: {np.min(drops):.2f}%")
            print(f"✅ Хороших выходов: {len(good_exits)} ({len(good_exits)/len(self.maximums_database)*100:.1f}%)")
            
            # Статистика по символам
            symbol_stats = {}
            for maximum in self.maximums_database:
                symbol = maximum['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'count': 0, 'avg_drop': 0, 'good_exits': 0, 'max_drop': 0}
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['avg_drop'] += maximum['drop_percent']
                if maximum['is_profitable_exit']:
                    symbol_stats[symbol]['good_exits'] += 1
                if maximum['drop_percent'] < symbol_stats[symbol]['max_drop']:
                    symbol_stats[symbol]['max_drop'] = maximum['drop_percent']
            
            print(f"\n🏆 По символам:")
            for symbol, stats in sorted(symbol_stats.items(), key=lambda x: abs(x[1]['avg_drop']/x[1]['count']), reverse=True):
                avg_drop = stats['avg_drop'] / stats['count']
                exit_rate = stats['good_exits'] / stats['count'] * 100
                print(f"   {symbol}: {stats['count']} максимумов, {avg_drop:.2f}% среднее падение, {exit_rate:.1f}% хороших выходов")
    
    def analyze_clean_patterns(self):
        """Анализ чистых закономерностей"""
        print("\n🔍 АНАЛИЗ ЧИСТЫХ ЗАКОНОМЕРНОСТЕЙ")
        print("-" * 40)
        
        if len(self.maximums_database) < 5:
            print("❌ Недостаточно данных для анализа")
            return
        
        # Анализ признаков
        feature_stats = {}
        feature_names = list(self.maximums_database[0]['features'].keys())
        
        for feature_name in feature_names:
            good_exit_values = []
            bad_exit_values = []
            
            for maximum in self.maximums_database:
                value = maximum['features'][feature_name]
                if maximum['is_profitable_exit']:
                    good_exit_values.append(value)
                else:
                    bad_exit_values.append(value)
            
            if good_exit_values and bad_exit_values:
                feature_stats[feature_name] = {
                    'good_exit_mean': np.mean(good_exit_values),
                    'good_exit_std': np.std(good_exit_values),
                    'bad_exit_mean': np.mean(bad_exit_values),
                    'bad_exit_std': np.std(bad_exit_values),
                    'difference': abs(np.mean(good_exit_values) - np.mean(bad_exit_values))
                }
        
        # Сортируем по разности
        feature_importance = sorted(feature_stats.items(), key=lambda x: x[1]['difference'], reverse=True)
        
        print("🏆 ВАЖНОСТЬ ПРИЗНАКОВ (только EMA-основанные):")
        for feature_name, stats in feature_importance:
            print(f"   {feature_name}:")
            print(f"     Хорошие выходы: {stats['good_exit_mean']:.4f} ± {stats['good_exit_std']:.4f}")
            print(f"     Плохие выходы: {stats['bad_exit_mean']:.4f} ± {stats['bad_exit_std']:.4f}")
            print(f"     Разность: {stats['difference']:.4f}")
            print()
    
    def train_clean_model(self):
        """Обучение чистой модели"""
        print("\n🧠 ОБУЧЕНИЕ ЧИСТОЙ МОДЕЛИ")
        print("-" * 30)
        
        if len(self.maximums_database) < 10:
            print("❌ Недостаточно данных для обучения")
            return None
        
        # Подготавливаем данные
        X = []
        y = []
        
        feature_names = list(self.maximums_database[0]['features'].keys())
        
        for maximum in self.maximums_database:
            features_list = [maximum['features'][name] for name in feature_names]
            X.append(features_list)
            y.append(1 if maximum['is_profitable_exit'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Данных: {len(X)} образцов, {len(feature_names)} признаков")
        print(f"✅ Хороших выходов: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
        # Обучаем модель
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X, y)
        
        # Важность признаков
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 ВАЖНОСТЬ ПРИЗНАКОВ (чистая модель):")
        for name, importance in feature_importance:
            print(f"   {name}: {importance:.4f}")
        
        # Сохраняем модель
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'clean_maximum_model_{timestamp}.pkl'
        feature_filename = f'clean_maximum_features_{timestamp}.pkl'
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        with open(feature_filename, 'wb') as f:
            pickle.dump(feature_names, f)
        
        print(f"\n💾 Чистая модель сохранена: {model_filename}")
        print(f"💾 Признаки сохранены: {feature_filename}")
        
        return model, feature_names
    
    def run_clean_analysis(self):
        """Запуск чистого анализа"""
        print("🚀 ЧИСТЫЙ АНАЛИЗ МАКСИМУМОВ (БЕЗ RSI)")
        print("=" * 50)
        
        # 1. Сбор данных
        self.collect_clean_data()
        
        if not self.maximums_database:
            print("❌ Не удалось собрать данные")
            return
        
        # 2. Анализ закономерностей
        self.analyze_clean_patterns()
        
        # 3. Обучение модели
        model_result = self.train_clean_model()
        
        print(f"\n✅ ЧИСТЫЙ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📊 Найдено {len(self.maximums_database)} максимумов")
        
        if model_result:
            print(f"🧠 Чистая модель обучена и готова к использованию")

if __name__ == "__main__":
    trainer = CleanMaximumTrainer()
    trainer.run_clean_analysis()
