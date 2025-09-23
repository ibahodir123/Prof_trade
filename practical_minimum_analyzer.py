#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 ПРАКТИЧЕСКИЙ АНАЛИЗАТОР МИНИМУМОВ
Анализ реальных минимумов с расширенным периодом данных
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
from sklearn.ensemble import RandomForestClassifier
import pickle
import warnings
warnings.filterwarnings('ignore')

class PracticalMinimumAnalyzer:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.minimums_database = []
        
        # Более мягкие критерии для большего количества данных
        self.min_lookback = 6
        self.min_profit_threshold = 1.5  # Снижаем до 1.5%
        self.max_duration = 48
        self.verification_hours = 6
        
        print("🎯 ПРАКТИЧЕСКИЙ АНАЛИЗАТОР МИНИМУМОВ")
        print("📊 Расширенный период для большего количества данных")
        print("=" * 50)
    
    def get_extended_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Получение расширенных данных"""
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
    
    def analyze_minimum_features(self, df: pd.DataFrame, min_idx: int) -> dict:
        """Анализ признаков минимума"""
        try:
            if min_idx < 50 or min_idx >= len(df) - 6:
                return None
            
            # EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            current = df.iloc[min_idx]
            prev = df.iloc[min_idx - 1]
            
            # Ключевые признаки
            price_velocity = (current['close'] - prev['close']) / prev['close']
            ema20_velocity = (current['ema_20'] - prev['ema_20']) / prev['ema_20']
            ema50_velocity = (current['ema_50'] - prev['ema_50']) / prev['ema_50']
            
            # Расстояния (самое важное!)
            price_distance_ema20 = (current['close'] - current['ema_20']) / current['ema_20']
            price_distance_ema50 = (current['close'] - current['ema_50']) / current['ema_50']
            
            # Углы
            ema20_angle = ema20_velocity * 100
            ema50_angle = ema50_velocity * 100
            
            # Объем
            avg_volume = df['volume'].iloc[min_idx-20:min_idx].mean()
            volume_ratio = current['volume'] / avg_volume if avg_volume > 0 else 1
            
            # Волатильность
            volatility = df['close'].iloc[min_idx-20:min_idx].std() / df['close'].iloc[min_idx-20:min_idx].mean()
            
            features = {
                'price_velocity': price_velocity,
                'ema20_velocity': ema20_velocity,
                'ema50_velocity': ema50_velocity,
                'price_distance_ema20': price_distance_ema20,
                'price_distance_ema50': price_distance_ema50,
                'ema20_angle': ema20_angle,
                'ema50_angle': ema50_angle,
                'volume_ratio': volume_ratio,
                'volatility': volatility
            }
            
            return features
            
        except Exception:
            return None
    
    def find_practical_minimums(self, df: pd.DataFrame, symbol: str):
        """Поиск практических минимумов"""
        print(f"🔍 Ищу минимумы в {symbol}...")
        
        minimums_found = 0
        
        for i in range(50, len(df) - 6):
            try:
                current_low = df.iloc[i]['low']
                current_time = df.index[i]
                
                # Поиск локального минимума
                is_minimum = True
                for j in range(max(0, i-self.min_lookback), min(len(df), i+self.min_lookback+1)):
                    if j != i and df.iloc[j]['low'] <= current_low:
                        is_minimum = False
                        break
                
                if not is_minimum:
                    continue
                
                # Поиск максимума в будущем
                max_price = current_low
                max_idx = i
                max_time = current_time
                
                for j in range(i+1, min(len(df), i+49)):
                    if df.iloc[j]['high'] > max_price:
                        max_price = df.iloc[j]['high']
                        max_idx = j
                        max_time = df.index[j]
                
                movement_percent = ((max_price - current_low) / current_low) * 100
                duration_hours = (max_time - current_time).total_seconds() / 3600
                
                # Мягкие критерии
                if movement_percent >= self.min_profit_threshold and duration_hours <= self.max_duration:
                    features = self.analyze_minimum_features(df, i)
                    
                    if features:
                        minimum = {
                            'symbol': symbol,
                            'entry_time': current_time,
                            'exit_time': max_time,
                            'entry_price': current_low,
                            'exit_price': max_price,
                            'movement_percent': movement_percent,
                            'duration_hours': duration_hours,
                            'features': features,
                            'is_profitable': movement_percent >= 2.0,
                            'data_age_hours': (datetime.now() - current_time).total_seconds() / 3600
                        }
                        
                        self.minimums_database.append(minimum)
                        minimums_found += 1
                        
            except Exception:
                continue
        
        print(f"   ✅ {minimums_found} минимумов")
    
    def collect_practical_data(self):
        """Сбор практических данных"""
        print("\n📚 СБОР ПРАКТИЧЕСКИХ ДАННЫХ")
        print("-" * 35)
        
        self.minimums_database = []
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol}")
            df = self.get_extended_data(symbol, days=30)  # 30 дней
            if not df.empty:
                self.find_practical_minimums(df, symbol)
            time.sleep(0.5)
        
        print(f"\n📊 ИТОГО НАЙДЕНО: {len(self.minimums_database)} минимумов")
        
        if self.minimums_database:
            profits = [m['movement_percent'] for m in self.minimums_database]
            durations = [m['duration_hours'] for m in self.minimums_database]
            profitable = [m for m in self.minimums_database if m['is_profitable']]
            
            print(f"📈 Средняя прибыль: {np.mean(profits):.2f}%")
            print(f"⏱️ Средняя длительность: {np.mean(durations):.1f} часов")
            print(f"🎯 Максимальная прибыль: {np.max(profits):.2f}%")
            print(f"✅ Прибыльных минимумов: {len(profitable)} ({len(profitable)/len(self.minimums_database)*100:.1f}%)")
            
            # Статистика по символам
            symbol_stats = {}
            for minimum in self.minimums_database:
                symbol = minimum['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'count': 0, 'avg_profit': 0, 'profitable': 0, 'max_profit': 0}
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['avg_profit'] += minimum['movement_percent']
                if minimum['is_profitable']:
                    symbol_stats[symbol]['profitable'] += 1
                if minimum['movement_percent'] > symbol_stats[symbol]['max_profit']:
                    symbol_stats[symbol]['max_profit'] = minimum['movement_percent']
            
            print(f"\n🏆 ДЕТАЛЬНАЯ СТАТИСТИКА ПО СИМВОЛАМ:")
            for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['avg_profit'], reverse=True):
                avg_profit = stats['avg_profit'] / stats['count']
                profit_rate = stats['profitable'] / stats['count'] * 100
                print(f"   {symbol}:")
                print(f"     Количество: {stats['count']} минимумов")
                print(f"     Средняя прибыль: {avg_profit:.2f}%")
                print(f"     Максимальная прибыль: {stats['max_profit']:.2f}%")
                print(f"     Процент прибыльных: {profit_rate:.1f}%")
                print()
    
    def analyze_patterns(self):
        """Анализ закономерностей"""
        print("\n🔍 АНАЛИЗ ЗАКОНОМЕРНОСТЕЙ")
        print("-" * 30)
        
        if len(self.minimums_database) < 5:
            print("❌ Недостаточно данных для анализа закономерностей")
            return
        
        # Анализ признаков
        feature_stats = {}
        feature_names = list(self.minimums_database[0]['features'].keys())
        
        for feature_name in feature_names:
            profitable_values = []
            unprofitable_values = []
            
            for minimum in self.minimums_database:
                value = minimum['features'][feature_name]
                if minimum['is_profitable']:
                    profitable_values.append(value)
                else:
                    unprofitable_values.append(value)
            
            if profitable_values and unprofitable_values:
                feature_stats[feature_name] = {
                    'profitable_mean': np.mean(profitable_values),
                    'profitable_std': np.std(profitable_values),
                    'unprofitable_mean': np.mean(unprofitable_values),
                    'unprofitable_std': np.std(unprofitable_values),
                    'difference': abs(np.mean(profitable_values) - np.mean(unprofitable_values))
                }
        
        # Сортируем по разности (важности)
        feature_importance = sorted(feature_stats.items(), key=lambda x: x[1]['difference'], reverse=True)
        
        print("🏆 ВАЖНОСТЬ ПРИЗНАКОВ (по разности между прибыльными и убыточными):")
        for feature_name, stats in feature_importance:
            print(f"   {feature_name}:")
            print(f"     Прибыльные минимумы: {stats['profitable_mean']:.4f} ± {stats['profitable_std']:.4f}")
            print(f"     Убыточные минимумы: {stats['unprofitable_mean']:.4f} ± {stats['unprofitable_std']:.4f}")
            print(f"     Разность: {stats['difference']:.4f}")
            print()
        
        # Анализ временных закономерностей
        print("⏰ ВРЕМЕННЫЕ ЗАКОНОМЕРНОСТИ:")
        
        # По дням недели
        day_stats = {}
        for minimum in self.minimums_database:
            day = minimum['entry_time'].strftime('%A')
            if day not in day_stats:
                day_stats[day] = {'count': 0, 'total_profit': 0, 'profitable': 0}
            day_stats[day]['count'] += 1
            day_stats[day]['total_profit'] += minimum['movement_percent']
            if minimum['is_profitable']:
                day_stats[day]['profitable'] += 1
        
        print("   По дням недели:")
        for day, stats in sorted(day_stats.items(), key=lambda x: x[1]['total_profit']/x[1]['count'], reverse=True):
            avg_profit = stats['total_profit'] / stats['count']
            profit_rate = stats['profitable'] / stats['count'] * 100
            print(f"     {day}: {stats['count']} минимумов, {avg_profit:.2f}% средняя прибыль, {profit_rate:.1f}% прибыльных")
        
        # По времени суток
        hour_stats = {}
        for minimum in self.minimums_database:
            hour = minimum['entry_time'].hour
            hour_range = f"{hour:02d}:00-{(hour+1)%24:02d}:00"
            if hour_range not in hour_stats:
                hour_stats[hour_range] = {'count': 0, 'total_profit': 0, 'profitable': 0}
            hour_stats[hour_range]['count'] += 1
            hour_stats[hour_range]['total_profit'] += minimum['movement_percent']
            if minimum['is_profitable']:
                hour_stats[hour_range]['profitable'] += 1
        
        print("\n   По времени суток (топ-5):")
        top_hours = sorted(hour_stats.items(), key=lambda x: x[1]['total_profit']/x[1]['count'], reverse=True)[:5]
        for hour_range, stats in top_hours:
            avg_profit = stats['total_profit'] / stats['count']
            profit_rate = stats['profitable'] / stats['count'] * 100
            print(f"     {hour_range}: {stats['count']} минимумов, {avg_profit:.2f}% средняя прибыль, {profit_rate:.1f}% прибыльных")
    
    def train_practical_model(self):
        """Обучение практической модели"""
        print("\n🧠 ОБУЧЕНИЕ ПРАКТИЧЕСКОЙ МОДЕЛИ")
        print("-" * 35)
        
        if len(self.minimums_database) < 10:
            print("❌ Недостаточно данных для обучения модели")
            print(f"   Найдено: {len(self.minimums_database)} минимумов")
            print(f"   Нужно минимум: 10 минимумов")
            return None
        
        # Подготавливаем данные
        X = []
        y = []
        
        feature_names = list(self.minimums_database[0]['features'].keys())
        
        for minimum in self.minimums_database:
            features_list = [minimum['features'][name] for name in feature_names]
            X.append(features_list)
            y.append(1 if minimum['is_profitable'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Данных: {len(X)} образцов, {len(feature_names)} признаков")
        print(f"✅ Прибыльных: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
        # Обучаем модель
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X, y)
        
        # Важность признаков
        importances = model.feature_importances_
        feature_importance = list(zip(feature_names, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n🏆 ВАЖНОСТЬ ПРИЗНАКОВ (ML модель):")
        for name, importance in feature_importance:
            print(f"   {name}: {importance:.4f}")
        
        # Сохраняем модель
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f'practical_model_{timestamp}.pkl'
        feature_filename = f'practical_features_{timestamp}.pkl'
        
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        with open(feature_filename, 'wb') as f:
            pickle.dump(feature_names, f)
        
        print(f"\n💾 Модель сохранена: {model_filename}")
        print(f"💾 Признаки сохранены: {feature_filename}")
        
        return model, feature_names
    
    def run_practical_analysis(self):
        """Запуск практического анализа"""
        print("🚀 ПРАКТИЧЕСКИЙ АНАЛИЗ МИНИМУМОВ")
        print("=" * 50)
        
        # 1. Сбор данных
        self.collect_practical_data()
        
        if not self.minimums_database:
            print("❌ Не удалось собрать данные")
            return
        
        # 2. Анализ закономерностей
        self.analyze_patterns()
        
        # 3. Обучение модели
        model_result = self.train_practical_model()
        
        # 4. Сохранение результатов
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Сохраняем базу минимумов
        minimums_for_json = []
        for minimum in self.minimums_database:
            minimum_copy = minimum.copy()
            minimum_copy['entry_time'] = minimum['entry_time'].isoformat()
            minimum_copy['exit_time'] = minimum['exit_time'].isoformat()
            minimums_for_json.append(minimum_copy)
        
        with open(f'practical_minimums_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(minimums_for_json, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ ПРАКТИЧЕСКИЙ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"📊 Найдено {len(self.minimums_database)} минимумов")
        print(f"💾 Результаты сохранены в practical_minimums_{timestamp}.json")
        
        if model_result:
            print(f"🧠 Модель обучена и готова к использованию")
        else:
            print(f"⚠️ Модель не обучена (недостаточно данных)")

if __name__ == "__main__":
    analyzer = PracticalMinimumAnalyzer()
    analyzer.run_practical_analysis()
