#!/usr/bin/env python3
"""
УМНЫЙ ПРЕДИКТОР ДВИЖЕНИЙ
Использует обученную ML модель для предсказания прибыльности
"""

import pickle
import numpy as np

class SmartPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.load_model()
    
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            with open('smart_predictor_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            print("✅ Модель загружена")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def predict_movement(self, features_dict):
        """
        Предсказание типа движения
        
        features_dict должен содержать 9 групп признаков:
        {
            'velocities': {'price': X, 'ema20': Y, ...},
            'accelerations': {...},
            'velocity_ratios': {...},
            'distances': {...},
            'distance_changes': {...},
            'angles': {...},
            'angle_changes': {...},
            'ema_relationships': {...},
            'synchronizations': {...}
        }
        
        Returns: (predicted_class, probabilities)
        """
        if self.model is None:
            return None, None
        
        # Преобразуем в вектор признаков
        feature_vector = self._dict_to_vector(features_dict)
        
        # Предсказание
        prediction = self.model.predict([feature_vector])[0]
        probabilities = self.model.predict_proba([feature_vector])[0]
        
        class_names = ['Малое (1-3%)', 'Среднее (3-7%)', 'Крупное (7%+)']
        
        return class_names[prediction], probabilities
    
    def _dict_to_vector(self, features_dict):
        """Преобразование словаря признаков в вектор"""
        vector = []
        
        # Скорости
        vel = features_dict.get('velocities', {})
        vector.extend([
            vel.get('price', 0), vel.get('ema20', 0),
            vel.get('ema50', 0), vel.get('ema100', 0)
        ])
        
        # Ускорения
        acc = features_dict.get('accelerations', {})
        vector.extend([
            acc.get('price', 0), acc.get('ema20', 0),
            acc.get('ema50', 0), acc.get('ema100', 0)
        ])
        
        # Соотношения скоростей
        ratio = features_dict.get('velocity_ratios', {})
        vector.extend([
            ratio.get('price_ema20', 0), ratio.get('price_ema50', 0),
            ratio.get('price_ema100', 0)
        ])
        
        # Расстояния
        dist = features_dict.get('distances', {})
        vector.extend([
            dist.get('price_ema20', 0), dist.get('price_ema50', 0),
            dist.get('price_ema100', 0)
        ])
        
        # Изменения расстояний
        dist_ch = features_dict.get('distance_changes', {})
        vector.extend([
            dist_ch.get('price_ema20', 0), dist_ch.get('price_ema50', 0),
            dist_ch.get('price_ema100', 0)
        ])
        
        # Углы
        angles = features_dict.get('angles', {})
        vector.extend([
            angles.get('ema20', 0), angles.get('ema50', 0),
            angles.get('ema100', 0)
        ])
        
        # Изменения углов
        angle_ch = features_dict.get('angle_changes', {})
        vector.extend([
            angle_ch.get('ema20', 0), angle_ch.get('ema50', 0),
            angle_ch.get('ema100', 0)
        ])
        
        # Взаимоотношения EMA
        rel = features_dict.get('ema_relationships', {})
        vector.extend([
            rel.get('ema20_ema50', 0), rel.get('ema20_ema100', 0),
            rel.get('ema50_ema100', 0)
        ])
        
        # Синхронизации
        sync = features_dict.get('synchronizations', {})
        vector.extend([
            sync.get('price_ema20', 0), sync.get('price_ema50', 0),
            sync.get('price_ema100', 0)
        ])
        
        return vector

# Пример использования
if __name__ == "__main__":
    predictor = SmartPredictor()
    
    # Тестовые данные (паттерн для среднего движения)
    test_features = {
        'velocities': {'price': -0.010411, 'ema20': 0.0001, 'ema50': 0.0001, 'ema100': 0.0001},
        'angles': {'ema20': -0.1, 'ema50': -0.05, 'ema100': 0},
        'synchronizations': {'price_ema20': 0.518, 'price_ema50': 0.5, 'price_ema100': 0.5}
    }
    
    prediction, probabilities = predictor.predict_movement(test_features)
    
    if prediction:
        print(f"🎯 Предсказание: {prediction}")
        print("📊 Вероятности:")
        for i, prob in enumerate(probabilities):
            class_name = ['Малое (1-3%)', 'Среднее (3-7%)', 'Крупное (7%+)'][i]
            print(f"   {class_name}: {prob:.1%}")
