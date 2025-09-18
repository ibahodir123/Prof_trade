#!/usr/bin/env python3
"""
Нейронная сеть для предсказания стреляющих монет
Использует LSTM + Dense слои для анализа временных рядов
"""
import pandas as pd
import numpy as np
import json
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

# ML библиотеки
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ShootingStarPredictor:
    """Нейронная сеть для предсказания стреляющих монет"""
    
    def __init__(self, sequence_length: int = 24, prediction_horizon: int = 24):
        """
        Args:
            sequence_length: Количество часов истории для анализа (24 часа)
            prediction_horizon: На сколько часов вперед предсказываем (24 часа)
        """
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        self.feature_columns = []
        self.is_trained = False
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготавливает признаки для нейронной сети"""
        if df.empty:
            return df
            
        try:
            # Базовые признаки
            features = pd.DataFrame(index=df.index)
            
            # Ценовые признаки
            features['price'] = df['close']
            features['price_change_1h'] = df['close'].pct_change(1)
            features['price_change_4h'] = df['close'].pct_change(4)
            features['price_change_12h'] = df['close'].pct_change(12)
            features['price_change_24h'] = df['close'].pct_change(24)
            
            # EMA отношения
            features['price_to_ema20'] = df['close'] / df['ema_20']
            features['price_to_ema50'] = df['close'] / df['ema_50']
            features['price_to_ema100'] = df['close'] / df['ema_100']
            features['ema20_to_ema50'] = df['ema_20'] / df['ema_50']
            features['ema50_to_ema100'] = df['ema_50'] / df['ema_100']
            
            # RSI и его изменения
            features['rsi'] = df['rsi']
            features['rsi_change'] = df['rsi'].diff()
            
            # Bollinger Bands
            features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            
            # MACD
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_histogram'] = df['macd_histogram']
            
            # Объемные признаки
            features['volume'] = df['volume']
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ratio'] = df['volume_ratio']
            
            # Волатильность
            features['volatility'] = df['volatility']
            features['high_low_ratio'] = df['high'] / df['low']
            features['open_close_ratio'] = df['open'] / df['close']
            
            # Технические паттерны
            features['hammer'] = ((df['close'] - df['low']) / (df['high'] - df['low']) > 0.6).astype(int)
            features['doji'] = (abs(df['open'] - df['close']) / (df['high'] - df['low']) < 0.1).astype(int)
            features['green_candle'] = (df['close'] > df['open']).astype(int)
            
            # Momentum индикаторы
            features['price_momentum'] = df['close'].rolling(12).apply(lambda x: (x[-1] - x[0]) / x[0])
            features['volume_momentum'] = df['volume'].rolling(12).apply(lambda x: (x[-1] - x[0]) / x[0])
            
            # Временные признаки
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки признаков: {e}")
            return pd.DataFrame()
    
    def create_sequences(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Создает последовательности для LSTM"""
        try:
            X, y = [], []
            
            for i in range(self.sequence_length, len(features) - self.prediction_horizon + 1):
                # Входная последовательность
                X.append(features.iloc[i-self.sequence_length:i].values)
                
                # Целевая переменная
                y.append(targets.iloc[i + self.prediction_horizon - 1])
            
            return np.array(X), np.array(y)
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания последовательностей: {e}")
            return np.array([]), np.array([])
    
    def build_model(self, input_shape: Tuple[int, int], num_classes: int) -> Sequential:
        """Строит архитектуру нейронной сети"""
        model = Sequential([
            # LSTM слои для анализа временных рядов
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(64, return_sequences=True),
            BatchNormalization(),
            Dropout(0.2),
            
            LSTM(32, return_sequences=False),
            BatchNormalization(),
            Dropout(0.2),
            
            # Dense слои для финального предсказания
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(32, activation='relu'),
            Dropout(0.2),
            
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, data: Dict[str, pd.DataFrame], test_size: float = 0.2):
        """Обучает нейронную сеть"""
        logger.info("🚀 Начинаю обучение нейронной сети...")
        
        all_features = []
        all_targets = []
        
        # Собираем данные со всех монет
        for symbol, df in data.items():
            if df.empty or 'is_shooting_star' not in df.columns:
                continue
                
            logger.info(f"📊 Обрабатываю {symbol}...")
            
            # Подготавливаем признаки
            features = self.prepare_features(df)
            if features.empty:
                continue
            
            # Целевая переменная - категория роста
            targets = df['growth_category'].loc[features.index]
            
            # Создаем последовательности
            X_seq, y_seq = self.create_sequences(features, targets)
            
            if len(X_seq) > 0:
                all_features.append(X_seq)
                all_targets.append(y_seq)
        
        if not all_features:
            logger.error("❌ Нет данных для обучения")
            return False
        
        # Объединяем все данные
        X = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_targets, axis=0)
        
        logger.info(f"📊 Размер обучающих данных: {X.shape}")
        logger.info(f"📈 Распределение классов: {np.bincount(y.astype(int))}")
        
        # Кодируем целевые переменные
        y_encoded = to_categorical(y, num_classes=5)
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y
        )
        
        # Нормализуем признаки
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        
        # Строим модель
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]), 5)
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Обучаем модель
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Оценка модели
        y_pred = self.model.predict(X_test_scaled)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        logger.info(f"✅ Точность модели: {accuracy:.4f}")
        
        # Отчет по классификации
        class_names = ['Без роста', 'Малый рост', 'Средний рост', 'Большой рост', 'Огромный рост']
        report = classification_report(y_test_classes, y_pred_classes, target_names=class_names)
        logger.info(f"📊 Отчет по классификации:\n{report}")
        
        self.is_trained = True
        
        # Сохраняем модель
        self.save_model()
        
        return True
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """Предсказывает вероятность стреляющих монет"""
        if not self.is_trained or self.model is None:
            logger.error("❌ Модель не обучена")
            return {}
        
        try:
            # Подготавливаем признаки
            features = self.prepare_features(df)
            if features.empty or len(features) < self.sequence_length:
                return {}
            
            # Берем последнюю последовательность
            last_sequence = features.iloc[-self.sequence_length:].values
            last_sequence = last_sequence.reshape(1, self.sequence_length, -1)
            
            # Нормализуем
            last_sequence_reshaped = last_sequence.reshape(-1, last_sequence.shape[-1])
            last_sequence_scaled = self.scaler.transform(last_sequence_reshaped)
            last_sequence_scaled = last_sequence_scaled.reshape(last_sequence.shape)
            
            # Предсказание
            predictions = self.model.predict(last_sequence_scaled)
            probabilities = predictions[0]
            
            # Интерпретация результатов
            class_names = ['Без роста', 'Малый рост', 'Средний рост', 'Большой рост', 'Огромный рост']
            
            result = {
                'probabilities': dict(zip(class_names, probabilities)),
                'predicted_class': class_names[np.argmax(probabilities)],
                'confidence': float(np.max(probabilities)),
                'shooting_star_probability': float(probabilities[2:].sum()),  # Средний+ рост
                'high_growth_probability': float(probabilities[3:].sum()),   # Большой+ рост
                'explosive_growth_probability': float(probabilities[4])      # Огромный рост
            }
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Ошибка предсказания: {e}")
            return {}
    
    def save_model(self, filename: str = "shooting_star_model"):
        """Сохраняет модель и скейлер"""
        try:
            # Сохраняем модель
            self.model.save(f"{filename}.h5")
            
            # Сохраняем скейлер
            with open(f"{filename}_scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # Сохраняем метаданные
            metadata = {
                'sequence_length': self.sequence_length,
                'prediction_horizon': self.prediction_horizon,
                'feature_columns': self.feature_columns,
                'trained_at': datetime.now().isoformat()
            }
            
            with open(f"{filename}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"💾 Модель сохранена: {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения модели: {e}")
    
    def load_model(self, filename: str = "shooting_star_model"):
        """Загружает модель и скейлер"""
        try:
            # Загружаем модель
            self.model = tf.keras.models.load_model(f"{filename}.h5")
            
            # Загружаем скейлер
            with open(f"{filename}_scaler.pkl", 'rb') as f:
                self.scaler = pickle.load(f)
            
            # Загружаем метаданные
            with open(f"{filename}_metadata.json", 'r') as f:
                metadata = json.load(f)
            
            self.sequence_length = metadata['sequence_length']
            self.prediction_horizon = metadata['prediction_horizon']
            self.feature_columns = metadata.get('features', metadata.get('feature_columns', []))
            
            self.is_trained = True
            logger.info(f"📂 Модель загружена: {filename}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
            return False

def main():
    """Основная функция для тестирования"""
    from data_collector import HistoricalDataCollector
    
    print("🧠 ТЕСТИРОВАНИЕ НЕЙРОННОЙ СЕТИ ДЛЯ ПРЕДСКАЗАНИЯ СТРЕЛЯЮЩИХ МОНЕТ")
    print("=" * 70)
    
    # Загружаем данные (если есть)
    collector = HistoricalDataCollector()
    data = collector.load_data("historical_data.json")  # Предполагаем, что данные уже собраны
    
    if not data:
        print("❌ Данные не найдены. Сначала запустите data_collector.py")
        return
    
    # Создаем и обучаем модель
    predictor = ShootingStarPredictor()
    
    # Обучаем на части данных
    training_data = dict(list(data.items())[:20])  # Первые 20 монет для обучения
    
    success = predictor.train(training_data)
    
    if success:
        print("✅ Модель успешно обучена!")
        
        # Тестируем на одной монете
        test_symbol = list(data.keys())[0]
        test_df = data[test_symbol]
        
        prediction = predictor.predict(test_df)
        
        if prediction:
            print(f"\n🎯 ПРЕДСКАЗАНИЕ ДЛЯ {test_symbol}:")
            print(f"   - Предсказанный класс: {prediction['predicted_class']}")
            print(f"   - Уверенность: {prediction['confidence']:.2%}")
            print(f"   - Вероятность стреляющей монеты: {prediction['shooting_star_probability']:.2%}")
            print(f"   - Вероятность высокого роста: {prediction['high_growth_probability']:.2%}")
            print(f"   - Вероятность взрывного роста: {prediction['explosive_growth_probability']:.2%}")
    
    else:
        print("❌ Не удалось обучить модель")

if __name__ == "__main__":
    main()

