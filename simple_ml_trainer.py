#!/usr/bin/env python3
"""
ПРОСТОЙ И ОПТИМАЛЬНЫЙ ML ПОДХОД
Сборщик обучающих данных для LONG позиций
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import json

logger = logging.getLogger(__name__)

class SimpleLongTrainer:
    """Простой тренер для LONG позиций"""
    
    def __init__(self):
        self.features_data = []  # Все признаки
        self.labels_data = []    # Все метки (прибыльно/нет)
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Параметры для определения прибыльности
        self.future_hours = 24      # Смотрим на 24 часа вперед
        self.min_profit = 0.02      # Минимум 2% прибыли
        self.max_loss = -0.05       # Максимум -5% потерь (стоп-лосс)
        
    def collect_training_data(self, symbols: List[str]) -> bool:
        """Сбор обучающих данных"""
        try:
            print(f"🔍 Собираю обучающие данные для {len(symbols)} символов...")
            
            for symbol in symbols:
                print(f"📊 Анализирую {symbol}...")
                
                # Получаем данные с 01.01.2025
                df = self._get_historical_data(symbol)
                if df is None or len(df) < 100:
                    print(f"❌ Недостаточно данных для {symbol}")
                    continue
                
                # Подготавливаем признаки
                df = self._prepare_features(df)
                if df is None:
                    continue
                
                # Собираем обучающие примеры
                examples_count = self._extract_training_examples(df, symbol)
                print(f"✅ {symbol}: собрано {examples_count} примеров")
            
            total_examples = len(self.features_data)
            positive_examples = sum(self.labels_data)
            negative_examples = total_examples - positive_examples
            
            print(f"📈 ИТОГО собрано {total_examples} примеров:")
            print(f"   ✅ Прибыльных: {positive_examples} ({positive_examples/total_examples*100:.1f}%)")
            print(f"   ❌ Убыточных: {negative_examples} ({negative_examples/total_examples*100:.1f}%)")
            
            return total_examples > 0
            
        except Exception as e:
            print(f"❌ Ошибка сбора данных: {e}")
            return False

    def _extract_training_examples(self, df: pd.DataFrame, symbol: str) -> int:
        """Извлечение обучающих примеров из данных"""
        try:
            examples_count = 0
            
            # Проходим по каждой свече (кроме последних future_hours)
            for i in range(len(df) - self.future_hours):
                current_row = df.iloc[i]
                current_price = current_row['close']
                
                # Извлекаем 9 групп признаков (27 значений)
                features = self._extract_features_from_row(current_row)
                if features is None:
                    continue
                
                # Смотрим на будущее: прибыльна ли LONG позиция?
                future_slice = df.iloc[i:i + self.future_hours]
                is_profitable = self._is_long_profitable(current_price, future_slice)
                
                # Добавляем пример в обучающую выборку
                self.features_data.append(features)
                self.labels_data.append(1 if is_profitable else 0)
                examples_count += 1
                
                # Логируем каждый 1000-й пример
                if examples_count % 1000 == 0:
                    print(f"📊 {symbol}: обработано {examples_count} примеров")
            
            return examples_count
            
        except Exception as e:
            print(f"❌ Ошибка извлечения примеров: {e}")
            return 0

    def _is_long_profitable(self, entry_price: float, future_data: pd.DataFrame) -> bool:
        """Определяет, прибыльна ли LONG позиция"""
        try:
            max_price = future_data['high'].max()
            min_price = future_data['low'].min()
            
            # Рассчитываем максимальную прибыль и максимальный убыток
            max_profit = (max_price - entry_price) / entry_price
            max_loss = (min_price - entry_price) / entry_price
            
            # Логика прибыльности:
            # 1. Если достигли стоп-лосс раньше прибыли - убыточно
            # 2. Если достигли прибыль раньше стоп-лосса - прибыльно
            
            # Ищем первую свечу, где достигли либо стоп-лосс, либо прибыль
            for _, row in future_data.iterrows():
                high_profit = (row['high'] - entry_price) / entry_price
                low_loss = (row['low'] - entry_price) / entry_price
                
                # Сначала проверяем стоп-лосс (может быть внутри свечи раньше)
                if low_loss <= self.max_loss:
                    return False  # Убыточно - сработал стоп-лосс
                
                # Потом проверяем прибыль
                if high_profit >= self.min_profit:
                    return True   # Прибыльно - достигли цель
            
            # Если за период не достигли ни стопа, ни цели
            final_price = future_data['close'].iloc[-1]
            final_return = (final_price - entry_price) / entry_price
            
            return final_return >= self.min_profit
            
        except Exception as e:
            print(f"❌ Ошибка расчета прибыльности: {e}")
            return False

    def _extract_features_from_row(self, row) -> Optional[List[float]]:
        """Извлечение 27 признаков из строки данных"""
        try:
            features = []
            
            # 1. Velocities (4)
            features.extend([
                row['price_velocity'],
                row['ema20_velocity'],
                row['ema50_velocity'],
                row['ema100_velocity']
            ])
            
            # 2. Accelerations (4)
            features.extend([
                row['price_acceleration'],
                row['ema20_acceleration'],
                row['ema50_acceleration'],
                row['ema100_acceleration']
            ])
            
            # 3. Velocity ratios (3)
            features.extend([
                row['price_to_ema20_velocity_ratio'],
                row['price_to_ema50_velocity_ratio'],
                row['price_to_ema100_velocity_ratio']
            ])
            
            # 4. Distances (3)
            features.extend([
                row['price_to_ema20_distance'],
                row['price_to_ema50_distance'],
                row['price_to_ema100_distance']
            ])
            
            # 5. Distance changes (3)
            features.extend([
                row['price_to_ema20_distance_change'],
                row['price_to_ema50_distance_change'],
                row['price_to_ema100_distance_change']
            ])
            
            # 6. Angles (3)
            features.extend([
                row['ema20_angle'],
                row['ema50_angle'],
                row['ema100_angle']
            ])
            
            # 7. Angle changes (3)
            features.extend([
                row['ema20_angle_change'],
                row['ema50_angle_change'],
                row['ema100_angle_change']
            ])
            
            # 8. EMA relationships (3)
            features.extend([
                row['ema20_to_ema50'],
                row['ema20_to_ema100'],
                row['ema50_to_ema100']
            ])
            
            # 9. Synchronizations (3)
            features.extend([
                row['price_ema20_sync'],
                row['price_ema50_sync'],
                row['price_ema100_sync']
            ])
            
            # Проверяем на NaN и бесконечность
            features = [0.0 if np.isnan(f) or np.isinf(f) else f for f in features]
            
            return features
            
        except Exception as e:
            print(f"❌ Ошибка извлечения признаков: {e}")
            return None

    def train_model(self) -> bool:
        """Обучение ML модели"""
        try:
            if len(self.features_data) == 0:
                print("❌ Нет данных для обучения")
                return False
            
            print(f"🤖 Начинаю обучение на {len(self.features_data)} примерах...")
            
            # Подготавливаем данные
            X = np.array(self.features_data)
            y = np.array(self.labels_data)
            
            # Разделяем на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Нормализуем признаки
            print("📊 Нормализую признаки...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Обучаем модель
            print("🧠 Обучаю Random Forest...")
            self.model.fit(X_train_scaled, y_train)
            
            # Оцениваем качество
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"📈 Точность на обучающей выборке: {train_score:.3f}")
            print(f"📊 Точность на тестовой выборке: {test_score:.3f}")
            
            # Детальный отчет
            y_pred = self.model.predict(X_test_scaled)
            print("\n📋 Подробный отчет:")
            print(classification_report(y_test, y_pred, target_names=['Убыточно', 'Прибыльно']))
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка обучения: {e}")
            return False

    def save_model(self, model_name: str = "simple_long_model") -> bool:
        """Сохранение обученной модели"""
        try:
            # Сохраняем модель и скалер
            joblib.dump(self.model, f'{model_name}.pkl')
            joblib.dump(self.scaler, f'{model_name}_scaler.pkl')
            
            # Сохраняем параметры
            params = {
                'future_hours': self.future_hours,
                'min_profit': self.min_profit,
                'max_loss': self.max_loss,
                'feature_count': 27
            }
            
            with open(f'{model_name}_params.json', 'w') as f:
                json.dump(params, f, indent=2)
            
            print(f"💾 Модель сохранена: {model_name}.pkl")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False

    def predict_long_probability(self, features: List[float]) -> float:
        """Предсказание вероятности прибыльной LONG позиции"""
        try:
            if len(features) != 27:
                return 0.0
            
            # Нормализуем признаки
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Получаем вероятность прибыльности
            probability = self.model.predict_proba(features_scaled)[0][1]
            
            return probability
            
        except Exception as e:
            print(f"❌ Ошибка предсказания: {e}")
            return 0.0

    def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Получение исторических данных с 01.01.2025"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Устанавливаем дату начала: 01.01.2025
            from datetime import datetime
            start_date = datetime(2025, 1, 1, 0, 0, 0)
            since = int(start_date.timestamp() * 1000)
            
            # Получаем данные порциями
            all_ohlcv = []
            current_since = since
            max_per_request = 1000
            
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=max_per_request)
                if not ohlcv or len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 3600000
                
                if current_since >= exchange.milliseconds():
                    break
                
                if len(all_ohlcv) > 20000:
                    break
            
            if not all_ohlcv:
                return None
            
            # Создаем DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения данных {symbol}: {e}")
            return None

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Подготовка всех 27 признаков"""
        try:
            # Рассчитываем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # 1. Velocity (скорость изменения)
            df['price_velocity'] = df['close'].pct_change()
            df['ema20_velocity'] = df['ema_20'].pct_change()
            df['ema50_velocity'] = df['ema_50'].pct_change()
            df['ema100_velocity'] = df['ema_100'].pct_change()
            
            # 2. Acceleration (ускорение)
            df['price_acceleration'] = df['price_velocity'].pct_change()
            df['ema20_acceleration'] = df['ema20_velocity'].pct_change()
            df['ema50_acceleration'] = df['ema50_velocity'].pct_change()
            df['ema100_acceleration'] = df['ema100_velocity'].pct_change()
            
            # 3. Velocity ratios
            df['price_to_ema20_velocity_ratio'] = df['price_velocity'] / (df['ema20_velocity'] + 1e-8)
            df['price_to_ema50_velocity_ratio'] = df['price_velocity'] / (df['ema50_velocity'] + 1e-8)
            df['price_to_ema100_velocity_ratio'] = df['price_velocity'] / (df['ema100_velocity'] + 1e-8)
            
            # 4. Distance to EMAs
            df['price_to_ema20_distance'] = (df['close'] - df['ema_20']) / df['close']
            df['price_to_ema50_distance'] = (df['close'] - df['ema_50']) / df['close']
            df['price_to_ema100_distance'] = (df['close'] - df['ema_100']) / df['close']
            
            # 5. Distance change
            df['price_to_ema20_distance_change'] = df['price_to_ema20_distance'].diff()
            df['price_to_ema50_distance_change'] = df['price_to_ema50_distance'].diff()
            df['price_to_ema100_distance_change'] = df['price_to_ema100_distance'].diff()
            
            # 6. EMA angles (наклон)
            df['ema20_angle'] = np.arctan(df['ema20_velocity']) * 180 / np.pi
            df['ema50_angle'] = np.arctan(df['ema50_velocity']) * 180 / np.pi
            df['ema100_angle'] = np.arctan(df['ema100_velocity']) * 180 / np.pi
            
            # 7. Angle change
            df['ema20_angle_change'] = df['ema20_angle'].diff()
            df['ema50_angle_change'] = df['ema50_angle'].diff()
            df['ema100_angle_change'] = df['ema100_angle'].diff()
            
            # 8. EMA relationships
            df['ema20_to_ema50'] = df['ema_20'] / (df['ema_50'] + 1e-8)
            df['ema20_to_ema100'] = df['ema_20'] / (df['ema_100'] + 1e-8)
            df['ema50_to_ema100'] = df['ema_50'] / (df['ema_100'] + 1e-8)
            
            # 9. Price-EMA synchronization
            window_size = 20
            df['price_ema20_sync'] = df['close'].rolling(window_size).corr(df['ema_20'])
            df['price_ema50_sync'] = df['close'].rolling(window_size).corr(df['ema_50'])
            df['price_ema100_sync'] = df['close'].rolling(window_size).corr(df['ema_100'])
            
            # Очищаем данные
            df = df.dropna()
            df = df.replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка подготовки признаков: {e}")
            return None

if __name__ == "__main__":
    # Пример использования
    trainer = SimpleLongTrainer()
    
    print("🎯 ПРОСТОЙ ML ТРЕНЕР ДЛЯ LONG ПОЗИЦИЙ")
    print("=====================================")
    
    # Список монет для обучения
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT']
    
    # Собираем данные
    print("📊 ШАГ 1: Сбор обучающих данных...")
    if trainer.collect_training_data(symbols):
        
        # Обучаем модель
        print("\n🤖 ШАГ 2: Обучение модели...")
        if trainer.train_model():
            
            # Сохраняем модель
            print("\n💾 ШАГ 3: Сохранение модели...")
            trainer.save_model("simple_long_model")
            
            print("\n✅ ГОТОВО! Модель обучена и сохранена.")
            print("Теперь можно использовать для предсказаний LONG позиций.")
        else:
            print("❌ Ошибка обучения модели")
    else:
        print("❌ Не удалось собрать данные для обучения")
