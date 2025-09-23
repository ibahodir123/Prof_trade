#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 РЕАЛЬНЫЙ ТРЕНЕР МИНИМУМОВ
Обучение на актуальных данных с проверкой в реальном времени
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
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class RealTimeMinimumTrainer:
    def __init__(self):
        # 🎯 Актуальные символы для реальной торговли
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.minimums_database = []
        
        # 📊 Параметры для реального времени
        self.min_lookback = 8
        self.min_profit_threshold = 2.0  # Минимум 2% для реальной торговли
        self.max_duration = 24  # Максимум 24 часа
        self.verification_hours = 6  # Проверяем результат через 6 часов
        
        print("🎯 РЕАЛЬНЫЙ ТРЕНЕР МИНИМУМОВ")
        print("📊 Обучение на актуальных данных")
        print("⏰ Проверка результатов в реальном времени")
        print("=" * 50)
        
    def get_recent_data(self, symbol: str, hours: int = 168) -> pd.DataFrame:  # 7 дней
        """Получение актуальных данных за последние часы"""
        try:
            print(f"📊 Загружаю актуальные данные {symbol} за {hours} часов...")
            exchange = ccxt.binance()
            
            # Получаем данные за последние часы
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
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
                    print(f"   ❌ Ошибка API: {e}")
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end_ts]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            print(f"   ✅ {len(df)} свечей (последняя: {df.index[-1]})")
            return df
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return pd.DataFrame()
    
    def prepare_real_features(self, df: pd.DataFrame, min_idx: int) -> dict:
        """Подготовка признаков для реального минимума"""
        try:
            if min_idx < 50 or min_idx >= len(df) - 6:  # Оставляем 6 часов для проверки
                return None
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            current = df.iloc[min_idx]
            prev = df.iloc[min_idx - 1]
            prev_2 = df.iloc[min_idx - 2]
            
            # 🎯 Ключевые признаки для реального времени
            price_velocity = (current['close'] - prev['close']) / prev['close']
            ema20_velocity = (current['ema_20'] - prev['ema_20']) / prev['ema_20']
            ema50_velocity = (current['ema_50'] - prev['ema_50']) / prev['ema_50']
            
            # Углы тренда
            ema20_angle = ema20_velocity * 100
            ema50_angle = ema50_velocity * 100
            
            # Расстояния до EMA (КРИТИЧНО!)
            price_distance_ema20 = (current['close'] - current['ema_20']) / current['ema_20']
            price_distance_ema50 = (current['close'] - current['ema_50']) / current['ema_50']
            price_distance_ema100 = (current['close'] - current['ema_100']) / current['ema_100']
            
            # Объем и волатильность
            avg_volume = df['volume'].iloc[min_idx-20:min_idx].mean()
            volume_ratio = current['volume'] / avg_volume if avg_volume > 0 else 1
            
            volatility = df['close'].iloc[min_idx-20:min_idx].std() / df['close'].iloc[min_idx-20:min_idx].mean()
            
            # Изменения расстояний
            prev_dist_ema20 = (prev['close'] - prev['ema_20']) / prev['ema_20']
            distance_change_ema20 = price_distance_ema20 - prev_dist_ema20
            
            # Ускорение цены
            prev_price_velocity = (prev['close'] - prev_2['close']) / prev_2['close']
            price_acceleration = price_velocity - prev_price_velocity
            
            features = {
                'price_velocity': price_velocity,
                'ema20_velocity': ema20_velocity,
                'ema50_velocity': ema50_velocity,
                'ema20_angle': ema20_angle,
                'ema50_angle': ema50_angle,
                'price_distance_ema20': price_distance_ema20,
                'price_distance_ema50': price_distance_ema50,
                'price_distance_ema100': price_distance_ema100,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'distance_change_ema20': distance_change_ema20,
                'price_acceleration': price_acceleration
            }
            
            return features
            
        except Exception as e:
            return None
    
    def find_real_minimums(self, df: pd.DataFrame, symbol: str):
        """Поиск реальных минимумов с проверкой"""
        print(f"🔍 Ищу реальные минимумы в {symbol}...")
        
        minimums_found = 0
        
        # Ищем минимумы, оставляя время для проверки
        for i in range(100, len(df) - self.verification_hours):
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
                
                # Ищем максимум в будущем (следующие 24 часа)
                max_price = current_low
                max_idx = i
                max_time = current_time
                
                for j in range(i+1, min(len(df), i+25)):
                    if df.iloc[j]['high'] > max_price:
                        max_price = df.iloc[j]['high']
                        max_idx = j
                        max_time = df.index[j]
                
                # Проверяем прибыльность
                movement_percent = ((max_price - current_low) / current_low) * 100
                duration_hours = (max_time - current_time).total_seconds() / 3600
                
                # Фильтруем по реальным критериям
                if (movement_percent >= self.min_profit_threshold and 
                    duration_hours <= self.max_duration):
                    
                    features = self.prepare_real_features(df, i)
                    
                    if features:
                        # Проверяем качество признаков
                        if (abs(features['price_distance_ema20']) > 0.01 and  # Цена далеко от EMA20
                            features['volume_ratio'] > 0.5 and  # Объем не слишком низкий
                            features['volatility'] < 0.1):  # Волатильность не экстремальная
                            
                            minimum = {
                                'symbol': symbol,
                                'entry_time': current_time,
                                'exit_time': max_time,
                                'entry_price': current_low,
                                'exit_price': max_price,
                                'movement_percent': movement_percent,
                                'duration_hours': duration_hours,
                                'features': features,
                                'is_profitable': movement_percent >= 3.0,  # 3%+ = прибыльный
                                'data_freshness': (datetime.now() - current_time).total_seconds() / 3600  # Часы назад
                            }
                            
                            self.minimums_database.append(minimum)
                            minimums_found += 1
                            
            except Exception:
                continue
        
        print(f"   ✅ {minimums_found} реальных минимумов")
    
    def collect_real_data(self):
        """Сбор реальных данных"""
        print("\n📚 СБОР РЕАЛЬНЫХ ДАННЫХ")
        print("-" * 30)
        
        self.minimums_database = []
        
        for i, symbol in enumerate(self.symbols):
            print(f"[{i+1}/{len(self.symbols)}] {symbol}")
            df = self.get_recent_data(symbol, hours=168)  # 7 дней данных
            if not df.empty:
                self.find_real_minimums(df, symbol)
            time.sleep(0.5)
        
        print(f"\n📊 ИТОГО НАЙДЕНО: {len(self.minimums_database)} реальных минимумов")
        
        if self.minimums_database:
            profits = [m['movement_percent'] for m in self.minimums_database]
            durations = [m['duration_hours'] for m in self.minimums_database]
            freshness = [m['data_freshness'] for m in self.minimums_database]
            profitable = [m for m in self.minimums_database if m['is_profitable']]
            
            print(f"📈 Средняя прибыль: {np.mean(profits):.2f}%")
            print(f"⏱️ Средняя длительность: {np.mean(durations):.1f} часов")
            print(f"🎯 Максимальная прибыль: {np.max(profits):.2f}%")
            print(f"✅ Прибыльных минимумов: {len(profitable)} ({len(profitable)/len(self.minimums_database)*100:.1f}%)")
            print(f"🕐 Средний возраст данных: {np.mean(freshness):.1f} часов")
            
            # Статистика по символам
            symbol_stats = {}
            for minimum in self.minimums_database:
                symbol = minimum['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'count': 0, 'avg_profit': 0, 'profitable': 0}
                symbol_stats[symbol]['count'] += 1
                symbol_stats[symbol]['avg_profit'] += minimum['movement_percent']
                if minimum['is_profitable']:
                    symbol_stats[symbol]['profitable'] += 1
            
            print(f"\n🏆 По символам:")
            for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['count'], reverse=True):
                avg_profit = stats['avg_profit'] / stats['count']
                profit_rate = stats['profitable'] / stats['count'] * 100
                print(f"   {symbol}: {stats['count']} минимумов, {avg_profit:.2f}% средняя прибыль, {profit_rate:.1f}% прибыльных")
    
    def train_real_model(self):
        """Обучение модели на реальных данных"""
        print("\n🧠 ОБУЧЕНИЕ МОДЕЛИ НА РЕАЛЬНЫХ ДАННЫХ")
        print("-" * 40)
        
        if not self.minimums_database:
            print("❌ Нет данных для обучения!")
            return None
        
        # Подготавливаем данные
        X = []
        y = []
        
        feature_names = list(self.minimums_database[0]['features'].keys())
        
        for minimum in self.minimums_database:
            features_list = [minimum['features'][name] for name in feature_names]
            X.append(features_list)
            
            # Бинарная классификация: прибыльный или нет
            y.append(1 if minimum['is_profitable'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"📊 Данных: {len(X)} образцов, {len(feature_names)} признаков")
        print(f"✅ Прибыльных: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        print(f"❌ Убыточных: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
        
        # Разделяем данные
        if len(X) > 10:  # Минимум данных для обучения
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            # Обучаем модель
            model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
            model.fit(X_train, y_train)
            
            # Оценка
            train_acc = accuracy_score(y_train, model.predict(X_train))
            val_acc = accuracy_score(y_val, model.predict(X_val))
            
            print(f"✅ Точность обучения: {train_acc:.3f}")
            print(f"✅ Точность валидации: {val_acc:.3f}")
            
            # Важность признаков
            importances = model.feature_importances_
            feature_importance = list(zip(feature_names, importances))
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n🏆 ВАЖНОСТЬ ПРИЗНАКОВ (РЕАЛЬНЫЕ ДАННЫЕ):")
            for name, importance in feature_importance:
                print(f"   {name}: {importance:.4f}")
            
            # Сохраняем модель
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f'real_minimum_model_{timestamp}.pkl'
            feature_filename = f'real_minimum_features_{timestamp}.pkl'
            
            with open(model_filename, 'wb') as f:
                pickle.dump(model, f)
            
            with open(feature_filename, 'wb') as f:
                pickle.dump(feature_names, f)
            
            print(f"\n💾 Модель сохранена: {model_filename}")
            print(f"💾 Признаки сохранены: {feature_filename}")
            
            return model, feature_names, feature_importance
        else:
            print("❌ Недостаточно данных для обучения модели")
            return None
    
    def test_on_current_data(self, model, feature_names):
        """Тестирование на текущих данных"""
        print("\n🧪 ТЕСТИРОВАНИЕ НА ТЕКУЩИХ ДАННЫХ")
        print("-" * 35)
        
        current_signals = []
        
        for symbol in self.symbols:
            print(f"📊 Анализирую {symbol}...")
            df = self.get_recent_data(symbol, hours=48)  # Последние 2 дня
            
            if len(df) > 50:
                # Берем последнюю точку как потенциальный минимум
                current_idx = len(df) - 6  # 6 часов назад для проверки
                current_price = df.iloc[current_idx]['close']
                current_time = df.index[current_idx]
                
                features = self.prepare_real_features(df, current_idx)
                
                if features:
                    # Предсказание
                    feature_vector = [features[name] for name in feature_names]
                    prediction = model.predict([feature_vector])[0]
                    probability = model.predict_proba([feature_vector])[0]
                    
                    # Проверяем, что произошло за 6 часов
                    if current_idx + 6 < len(df):
                        future_price = df.iloc[current_idx + 6]['high']
                        actual_profit = ((future_price - current_price) / current_price) * 100
                        
                        signal = {
                            'symbol': symbol,
                            'time': current_time,
                            'predicted_profitable': bool(prediction),
                            'probability': float(probability[1]),  # Вероятность прибыльности
                            'actual_profit': actual_profit,
                            'actual_profitable': actual_profit >= 3.0,
                            'correct_prediction': bool(prediction) == (actual_profit >= 3.0),
                            'features': features
                        }
                        
                        current_signals.append(signal)
                        
                        status = "✅ ПРАВИЛЬНО" if signal['correct_prediction'] else "❌ НЕПРАВИЛЬНО"
                        print(f"   Предсказание: {signal['predicted_profitable']} (вероятность: {signal['probability']:.3f})")
                        print(f"   Реальность: {actual_profit:.2f}% ({status})")
        
        # Статистика тестирования
        if current_signals:
            correct = sum(1 for s in current_signals if s['correct_prediction'])
            total = len(current_signals)
            accuracy = correct / total * 100
            
            print(f"\n📊 РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
            print(f"   Всего сигналов: {total}")
            print(f"   Правильных предсказаний: {correct}")
            print(f"   Точность на текущих данных: {accuracy:.1f}%")
            
            # Сохраняем результаты тестирования
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'real_test_results_{timestamp}.json', 'w', encoding='utf-8') as f:
                # Конвертируем datetime для JSON
                signals_for_json = []
                for signal in current_signals:
                    signal_copy = signal.copy()
                    signal_copy['time'] = signal['time'].isoformat()
                    signals_for_json.append(signal_copy)
                
                json.dump(signals_for_json, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Результаты тестирования сохранены: real_test_results_{timestamp}.json")
        
        return current_signals
    
    def run_real_analysis(self):
        """Запуск полного анализа на реальных данных"""
        print("🚀 АНАЛИЗ НА РЕАЛЬНЫХ ДАННЫХ")
        print("=" * 50)
        
        # 1. Сбор реальных данных
        self.collect_real_data()
        
        if not self.minimums_database:
            print("❌ Не удалось собрать реальные данные")
            return
        
        # 2. Обучение модели
        result = self.train_real_model()
        if not result:
            return
        
        model, feature_names, feature_importance = result
        
        # 3. Тестирование на текущих данных
        current_signals = self.test_on_current_data(model, feature_names)
        
        print(f"\n✅ АНАЛИЗ ЗАВЕРШЕН!")
        print(f"🎯 Модель обучена на {len(self.minimums_database)} реальных минимумах")
        print(f"📊 Протестировано на {len(current_signals)} текущих сигналах")

if __name__ == "__main__":
    trainer = RealTimeMinimumTrainer()
    trainer.run_real_analysis()
