"""
Система обучения на исторических данных с 1 января 2025
Собирает данные и обучает модели для каждого типа тренда
"""

import pandas as pd
import numpy as np
import ccxt
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from advanced_trend_analyzer import TrendAnalyzer
import logging
from datetime import datetime, timedelta
import os
import json

logger = logging.getLogger(__name__)

class HistoricalTrendTrainer:
    """Тренер моделей на исторических данных"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'rateLimit': 1200,
        })
        
        self.analyzer = TrendAnalyzer()
        self.start_date = datetime(2025, 1, 1)
        self.models = {}
        
    def collect_historical_data(self, symbols: list, days: int = 30) -> dict:
        """Сбор исторических данных для обучения"""
        
        logger.info(f"📊 Начинаю сбор данных с {self.start_date} для {len(symbols)} символов...")
        
        all_data = {}
        
        for symbol in symbols:
            try:
                logger.info(f"📈 Собираю данные для {symbol}...")
                
                # Получаем данные с 1 января 2025
                since = int(self.start_date.timestamp() * 1000)
                limit = days * 24  # Часовые данные
                
                ohlcv = self.exchange.fetch_ohlcv(symbol, '1h', since=since, limit=limit)
                
                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    all_data[symbol] = df
                    logger.info(f"✅ {symbol}: {len(df)} записей")
                else:
                    logger.warning(f"⚠️ Нет данных для {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ Ошибка сбора данных {symbol}: {e}")
                continue
        
        logger.info(f"🎯 Собрано данных для {len(all_data)} символов")
        return all_data
    
    def prepare_training_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Подготовка признаков для обучения"""
        
        features = pd.DataFrame(index=df.index)
        
        # Технические индикаторы
        features['rsi'] = self._calculate_rsi(df['close'])
        features['ema_20'] = df['close'].ewm(span=20).mean()
        features['ema_50'] = df['close'].ewm(span=50).mean()
        features['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Волатильность
        features['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # Объем
        features['volume_ma'] = df['volume'].rolling(20).mean()
        features['volume_ratio'] = df['volume'] / features['volume_ma']
        
        # Ценовые паттерны
        features['price_change_1h'] = df['close'].pct_change(1)
        features['price_change_4h'] = df['close'].pct_change(4)
        features['price_change_24h'] = df['close'].pct_change(24)
        
        # Максимумы и минимумы
        features['high_20'] = df['high'].rolling(20).max()
        features['low_20'] = df['low'].rolling(20).min()
        
        # Относительные позиции
        features['price_position'] = (df['close'] - features['low_20']) / (features['high_20'] - features['low_20'])
        
        # Трендовые индикаторы
        features['trend_strength'] = (features['ema_20'] - features['ema_50']) / features['ema_50']
        
        # Удаляем NaN
        features = features.dropna()
        
        return features
    
    def create_training_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Создание меток для обучения на основе анализа трендов"""
        
        labels = pd.DataFrame(index=df.index)
        
        # Анализируем каждую точку данных
        for i in range(100, len(df)):  # Начинаем с 100-й свечи для стабильности индикаторов
            try:
                # Берем данные до текущей точки
                historical_data = df.iloc[:i+1]
                
                # Анализируем тренд
                analysis = self.analyzer.analyze_trend(historical_data)
                
                if 'error' not in analysis:
                    # Создаем метки на основе анализа
                    labels.loc[df.index[i], 'trend'] = analysis['trend_direction']
                    labels.loc[df.index[i], 'phase'] = analysis['current_phase']
                    labels.loc[df.index[i], 'signal'] = analysis['signal']['type']
                    
                    # Дополнительные метки
                    if analysis['signal']['type'] in ['LONG', 'SHORT']:
                        labels.loc[df.index[i], 'is_entry_point'] = 1
                        labels.loc[df.index[i], 'confidence'] = analysis['signal'].get('confidence', 0.5)
                    else:
                        labels.loc[df.index[i], 'is_entry_point'] = 0
                        labels.loc[df.index[i], 'confidence'] = 0
                        
            except Exception as e:
                logger.debug(f"Ошибка анализа для индекса {i}: {e}")
                continue
        
        return labels.dropna()
    
    def train_trend_models(self, historical_data: dict) -> dict:
        """Обучение моделей для разных типов трендов"""
        
        logger.info("🤖 Начинаю обучение моделей...")
        
        all_features = []
        all_labels = []
        
        # Собираем данные со всех символов
        for symbol, df in historical_data.items():
            try:
                logger.info(f"📊 Обрабатываю {symbol}...")
                
                # Подготавливаем признаки
                features = self.prepare_training_features(df)
                
                # Создаем метки
                labels = self.create_training_labels(df)
                
                # Синхронизируем индексы
                common_index = features.index.intersection(labels.index)
                features = features.loc[common_index]
                labels = labels.loc[common_index]
                
                if len(features) > 50:  # Минимум данных для обучения
                    all_features.append(features)
                    all_labels.append(labels)
                    logger.info(f"✅ {symbol}: {len(features)} образцов")
                
            except Exception as e:
                logger.error(f"❌ Ошибка обработки {symbol}: {e}")
                continue
        
        if not all_features:
            logger.error("❌ Нет данных для обучения!")
            return {}
        
        # Объединяем все данные
        combined_features = pd.concat(all_features, ignore_index=True)
        combined_labels = pd.concat(all_labels, ignore_index=True)
        
        logger.info(f"📊 Общий размер данных: {len(combined_features)} образцов")
        
        # Обучаем модели для разных задач
        models = {}
        
        # 1. Модель определения тренда
        if 'trend' in combined_labels.columns:
            X = combined_features
            y = combined_labels['trend']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            trend_model = RandomForestClassifier(n_estimators=100, random_state=42)
            trend_model.fit(X_train, y_train)
            
            models['trend_detector'] = trend_model
            
            # Оценка качества
            y_pred = trend_model.predict(X_test)
            logger.info(f"📊 Модель определения тренда:")
            logger.info(classification_report(y_test, y_pred))
        
        # 2. Модель определения фазы
        if 'phase' in combined_labels.columns:
            X = combined_features
            y = combined_labels['phase']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            phase_model = RandomForestClassifier(n_estimators=100, random_state=42)
            phase_model.fit(X_train, y_train)
            
            models['phase_detector'] = phase_model
            
            # Оценка качества
            y_pred = phase_model.predict(X_test)
            logger.info(f"📊 Модель определения фазы:")
            logger.info(classification_report(y_test, y_pred))
        
        # 3. Модель точек входа
        if 'is_entry_point' in combined_labels.columns:
            X = combined_features
            y = combined_labels['is_entry_point']
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            entry_model = RandomForestClassifier(n_estimators=100, random_state=42)
            entry_model.fit(X_train, y_train)
            
            models['entry_detector'] = entry_model
            
            # Оценка качества
            y_pred = entry_model.predict(X_test)
            logger.info(f"📊 Модель определения точек входа:")
            logger.info(classification_report(y_test, y_pred))
        
        # Сохраняем модели
        self._save_models(models)
        
        return models
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _save_models(self, models: dict):
        """Сохранение обученных моделей"""
        
        os.makedirs('models', exist_ok=True)
        
        for name, model in models.items():
            filename = f'models/{name}.pkl'
            joblib.dump(model, filename)
            logger.info(f"💾 Сохранена модель: {filename}")
    
    def load_models(self) -> dict:
        """Загрузка обученных моделей"""
        
        models = {}
        model_files = {
            'trend_detector': 'models/trend_detector.pkl',
            'phase_detector': 'models/phase_detector.pkl', 
            'entry_detector': 'models/entry_detector.pkl'
        }
        
        for name, filename in model_files.items():
            try:
                if os.path.exists(filename):
                    models[name] = joblib.load(filename)
                    logger.info(f"📂 Загружена модель: {name}")
                else:
                    logger.warning(f"⚠️ Модель не найдена: {filename}")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки модели {name}: {e}")
        
        return models
    
    def predict_with_models(self, features: pd.DataFrame, models: dict) -> dict:
        """Предсказание с использованием обученных моделей"""
        
        predictions = {}
        
        if 'trend_detector' in models:
            predictions['trend'] = models['trend_detector'].predict(features.iloc[-1:].values)[0]
            predictions['trend_proba'] = models['trend_detector'].predict_proba(features.iloc[-1:].values)[0]
        
        if 'phase_detector' in models:
            predictions['phase'] = models['phase_detector'].predict(features.iloc[-1:].values)[0]
            predictions['phase_proba'] = models['phase_detector'].predict_proba(features.iloc[-1:].values)[0]
        
        if 'entry_detector' in models:
            predictions['entry_probability'] = models['entry_detector'].predict_proba(features.iloc[-1:].values)[0][1]
        
        return predictions

def train_models_on_historical_data():
    """Функция для обучения моделей на исторических данных"""
    
    # Список символов для обучения
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
        'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
    ]
    
    trainer = HistoricalTrendTrainer()
    
    # Собираем данные
    historical_data = trainer.collect_historical_data(symbols, days=30)
    
    if not historical_data:
        logger.error("❌ Не удалось собрать данные для обучения")
        return
    
    # Обучаем модели
    models = trainer.train_trend_models(historical_data)
    
    if models:
        logger.info(f"🎯 Обучено {len(models)} моделей!")
        return models
    else:
        logger.error("❌ Не удалось обучить модели")
        return None

if __name__ == "__main__":
    train_models_on_historical_data()




