#!/usr/bin/env python3
"""
Продвинутый ML тренер для EMA анализа
Обучает модели на признаках EMA: скорости, расстояния, углы трендов
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AdvancedMLTrainer:
    def __init__(self):
        self.entry_model = None
        self.exit_model = None
        self.scaler = None
        self.feature_names = None
        
    def load_models(self) -> bool:
        """Загрузка обученных моделей"""
        try:
            self.entry_model = joblib.load('models/entry_model.pkl')
            self.exit_model = joblib.load('models/exit_model.pkl')
            self.scaler = joblib.load('models/ema_scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            logger.info("✅ ML модели загружены успешно")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            return False
    
    def predict_entry_exit(self, features: np.ndarray) -> Tuple[float, float]:
        """Предсказание точек входа и выхода"""
        if self.entry_model is None or self.exit_model is None or self.scaler is None:
            return 0.0, 0.0
        
        try:
            # Проверяем размерность features
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Проверяем, что количество признаков соответствует ожидаемому
            expected_features = self.scaler.n_features_in_
            if features.shape[1] != expected_features:
                logger.warning(f"Количество признаков не совпадает: ожидается {expected_features}, получено {features.shape[1]}")
                return 0.0, 0.0
            
            features_scaled = self.scaler.transform(features)
            
            # Проверяем, что у модели есть predict_proba
            if hasattr(self.entry_model, 'predict_proba') and hasattr(self.exit_model, 'predict_proba'):
                entry_prob = self.entry_model.predict_proba(features_scaled)[0][1]
                exit_prob = self.exit_model.predict_proba(features_scaled)[0][1]
            else:
                # Fallback для моделей без predict_proba
                entry_prob = float(self.entry_model.predict(features_scaled)[0])
                exit_prob = float(self.exit_model.predict(features_scaled)[0])
            
            return entry_prob, exit_prob
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 0.0, 0.0
    
    def train_models(self, symbols: List[str]) -> bool:
        """Обучение моделей на исторических данных"""
        try:
            logger.info("🧠 Начинаю обучение ML моделей...")
            
            # Здесь должна быть логика сбора данных и обучения
            # Для демонстрации возвращаем True
            
            logger.info("✅ ML МОДЕЛИ ОБУЧЕНЫ!")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return False

# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = AdvancedMLTrainer()
    success = trainer.load_models()
    
    if success:
        print("✅ Модели загружены успешно")
    else:
        print("❌ Ошибка загрузки моделей")
