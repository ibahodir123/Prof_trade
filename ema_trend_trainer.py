"""
EMA Trend Trainer - Тренер моделей на основе EMA паттернов
Обучает ML модели на исторических данных с EMA индикаторами
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import ccxt

logger = logging.getLogger(__name__)

class EMATrendTrainer:
    """Тренер ML моделей на основе EMA паттернов"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        self.logger = logger
        
    def get_historical_data(self, symbol: str, timeframe: str = '1h', limit: int = 1000) -> List[List]:
        """Получение исторических данных с Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            self.logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return []
    
    def calculate_ema_features(self, prices: List[float]) -> Dict:
        """Вычисление EMA признаков"""
        if len(prices) < 100:
            return {}
            
        # EMA периоды
        ema20 = self._calculate_ema(prices, 20)
        ema50 = self._calculate_ema(prices, 50)
        ema100 = self._calculate_ema(prices, 100)
        
        features = {}
        
        # Текущие значения EMA
        features['ema20'] = ema20[-1] if not pd.isna(ema20[-1]) else 0
        features['ema50'] = ema50[-1] if not pd.isna(ema50[-1]) else 0
        features['ema100'] = ema100[-1] if not pd.isna(ema100[-1]) else 0
        
        # Отношения EMA
        if features['ema50'] > 0:
            features['ema20_ema50_ratio'] = features['ema20'] / features['ema50']
        else:
            features['ema20_ema50_ratio'] = 1
            
        if features['ema100'] > 0:
            features['ema50_ema100_ratio'] = features['ema50'] / features['ema100']
        else:
            features['ema50_ema100_ratio'] = 1
        
        # Расстояния от цены до EMA
        current_price = prices[-1]
        if features['ema20'] > 0:
            features['price_ema20_distance'] = (current_price - features['ema20']) / features['ema20']
        else:
            features['price_ema20_distance'] = 0
            
        if features['ema50'] > 0:
            features['price_ema50_distance'] = (current_price - features['ema50']) / features['ema50']
        else:
            features['price_ema50_distance'] = 0
        
        # Скорость изменения EMA (за последние 5 периодов)
        if len(ema20) >= 6:
            ema20_change = (ema20[-1] - ema20[-6]) / ema20[-6] if ema20[-6] > 0 else 0
            features['ema20_speed'] = ema20_change
        else:
            features['ema20_speed'] = 0
            
        if len(ema50) >= 6:
            ema50_change = (ema50[-1] - ema50[-6]) / ema50[-6] if ema50[-6] > 0 else 0
            features['ema50_speed'] = ema50_change
        else:
            features['ema50_speed'] = 0
        
        return features
    
    def _calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Вычисление EMA"""
        if len(prices) < period:
            return [np.nan] * len(prices)
            
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # Первое значение - простое среднее
        ema_values.append(sum(prices[:period]) / period)
        
        # Остальные значения
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
            
        return [np.nan] * (period - 1) + ema_values
    
    def create_training_data(self, symbols: List[str], lookback_days: int = 30) -> pd.DataFrame:
        """Создание обучающих данных"""
        training_data = []
        
        for symbol in symbols[:10]:  # Ограничиваем для тестирования
            try:
                self.logger.info(f"Обработка {symbol}...")
                
                # Получаем данные
                ohlcv = self.get_historical_data(symbol, '1h', lookback_days * 24)
                if len(ohlcv) < 100:
                    continue
                
                prices = [float(candle[4]) for candle in ohlcv]
                
                # Создаем окна данных для обучения
                window_size = 100
                for i in range(window_size, len(prices) - 10):
                    window_prices = prices[i-window_size:i+1]
                    
                    # Вычисляем признаки
                    features = self.calculate_ema_features(window_prices)
                    if not features:
                        continue
                    
                    # Определяем целевую переменную (будущее движение цены)
                    future_price = prices[i+5]  # Цена через 5 часов
                    current_price = prices[i]
                    price_change = (future_price - current_price) / current_price
                    
                    # Классификация
                    if price_change > 0.02:  # Рост > 2%
                        target = 'LONG'
                    elif price_change < -0.02:  # Падение > 2%
                        target = 'SHORT'
                    else:
                        target = 'HOLD'
                    
                    # Добавляем в данные
                    row = features.copy()
                    row['symbol'] = symbol
                    row['target'] = target
                    row['price_change'] = price_change
                    training_data.append(row)
                    
            except Exception as e:
                self.logger.error(f"Ошибка обработки {symbol}: {e}")
                continue
        
        if not training_data:
            self.logger.warning("Не удалось создать обучающие данные")
            return pd.DataFrame()
        
        df = pd.DataFrame(training_data)
        self.logger.info(f"Создано {len(df)} записей для обучения")
        return df
    
    def train_model(self, training_data: pd.DataFrame) -> Dict:
        """Обучение модели на EMA данных"""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import classification_report
            
            # Подготавливаем данные
            feature_columns = [col for col in training_data.columns 
                             if col not in ['symbol', 'target', 'price_change']]
            
            X = training_data[feature_columns].fillna(0)
            y = training_data['target']
            
            # Разделяем на train/test
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Обучаем модель
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            
            model.fit(X_train, y_train)
            
            # Оценка качества
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            return {
                'model': model,
                'feature_columns': feature_columns,
                'accuracy': report['accuracy'],
                'classification_report': report,
                'training_samples': len(X_train),
                'test_samples': len(X_test)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обучения модели: {e}")
            return {'error': str(e)}
    
    def save_model(self, model_data: Dict, filename: str = 'ema_model.pkl'):
        """Сохранение обученной модели"""
        try:
            import joblib
            
            save_data = {
                'model': model_data['model'],
                'feature_columns': model_data['feature_columns'],
                'training_date': datetime.now().isoformat(),
                'accuracy': model_data['accuracy']
            }
            
            joblib.dump(save_data, filename)
            self.logger.info(f"Модель сохранена в {filename}")
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения модели: {e}")
    
    def load_model(self, filename: str = 'ema_model.pkl') -> Dict:
        """Загрузка обученной модели"""
        try:
            import joblib
            
            model_data = joblib.load(filename)
            self.logger.info(f"Модель загружена из {filename}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели: {e}")
            return {'error': str(e)}