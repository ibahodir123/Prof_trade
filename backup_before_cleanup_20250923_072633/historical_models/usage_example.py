#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ПРИМЕР ИСПОЛЬЗОВАНИЯ ИСТОРИЧЕСКИХ МОДЕЛЕЙ
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class HistoricalModelPredictor:
    def __init__(self, models_dir="historical_models"):
        self.models_dir = models_dir
        self.load_models()
    
    def load_models(self):
        """Загрузка моделей"""
        # Загружаем модели
        with open(f'{self.models_dir}/minimum_model.pkl', 'rb') as f:
            self.minimum_model = pickle.load(f)
        
        with open(f'{self.models_dir}/maximum_model.pkl', 'rb') as f:
            self.maximum_model = pickle.load(f)
        
        # Загружаем масштабировщики
        with open(f'{self.models_dir}/minimum_scaler.pkl', 'rb') as f:
            self.minimum_scaler = pickle.load(f)
        
        with open(f'{self.models_dir}/maximum_scaler.pkl', 'rb') as f:
            self.maximum_scaler = pickle.load(f)
        
        # Загружаем признаки
        with open(f'{self.models_dir}/minimum_features.pkl', 'rb') as f:
            self.minimum_features = pickle.load(f)
        
        with open(f'{self.models_dir}/maximum_features.pkl', 'rb') as f:
            self.maximum_features = pickle.load(f)
        
        print("✅ Модели загружены успешно")
    
    def prepare_minimum_features(self, df, idx):
        """Подготовка признаков для минимума"""
        if idx < 50 or idx >= len(df) - 6:
            return None
        
        current = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        features = {
            'price_velocity': (current['close'] - prev['close']) / prev['close'],
            'ema20_velocity': (current['ema_20'] - prev['ema_20']) / prev['ema_20'],
            'ema50_velocity': (current['ema_50'] - prev['ema_50']) / prev['ema_50'],
            'ema100_velocity': (current['ema_100'] - prev['ema_100']) / prev['ema_100'],
            'price_distance_ema20': (current['close'] - current['ema_20']) / current['ema_20'],
            'price_distance_ema50': (current['close'] - current['ema_50']) / current['ema_50'],
            'price_distance_ema100': (current['close'] - current['ema_100']) / current['ema_100'],
            'ema20_angle': ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) * 100,
            'ema50_angle': ((current['ema_50'] - prev['ema_50']) / prev['ema_50']) * 100,
            'volatility': df['close'].iloc[idx-20:idx].std() / df['close'].iloc[idx-20:idx].mean(),
            'volume_ratio': current['volume'] / df['volume'].iloc[idx-20:idx].mean() if df['volume'].iloc[idx-20:idx].mean() > 0 else 1
        }
        
        return features
    
    def predict_minimum(self, features_dict):
        """Прогнозирование минимума"""
        # Подготавливаем данные
        features_list = [features_dict.get(name, 0) for name in self.minimum_features]
        features_array = np.array(features_list).reshape(1, -1)
        
        # Масштабируем
        features_scaled = self.minimum_scaler.transform(features_array)
        
        # Прогнозируем
        prediction = self.minimum_model.predict(features_scaled)[0]
        probability = self.minimum_model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': prediction,
            'confidence': max(probability),
            'is_good_entry': prediction == 1
        }
    
    def predict_maximum(self, features_dict):
        """Прогнозирование максимума"""
        # Подготавливаем данные
        features_list = [features_dict.get(name, 0) for name in self.maximum_features]
        features_array = np.array(features_list).reshape(1, -1)
        
        # Масштабируем
        features_scaled = self.maximum_scaler.transform(features_array)
        
        # Прогнозируем
        prediction = self.maximum_model.predict(features_scaled)[0]
        probability = self.maximum_model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': prediction,
            'confidence': max(probability),
            'is_good_exit': prediction == 1
        }

# Пример использования
if __name__ == "__main__":
    predictor = HistoricalModelPredictor()
    
    # Пример данных (замените на реальные данные)
    # df = your_dataframe_with_ema_columns
    
    # Для минимума
    # features = predictor.prepare_minimum_features(df, idx)
    # result = predictor.predict_minimum(features)
    # print(f"Минимум: {'Хороший вход' if result['is_good_entry'] else 'Плохой вход'} (уверенность: {result['confidence']:.2f})")
    
    # Для максимума
    # features = predictor.prepare_maximum_features(df, idx)
    # result = predictor.predict_maximum(features)
    # print(f"Максимум: {'Хороший выход' if result['is_good_exit'] else 'Плохой выход'} (уверенность: {result['confidence']:.2f})")
