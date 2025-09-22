#!/usr/bin/env python3
"""
УМНЫЙ ML ТРЕНЕР
Обучает модель на основе найденных паттернов из 809 движений
"""

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

class SmartMLTrainer:
    def __init__(self):
        self.movements = []
        self.model = None
        self.feature_names = []
        
    def load_movements(self):
        """Загрузка движений"""
        try:
            with open('movements_database.json', 'r') as f:
                self.movements = json.load(f)
            print(f"📊 Загружено {len(self.movements)} движений")
            return True
        except Exception as e:
            print(f"❌ Ошибка загрузки: {e}")
            return False
    
    def prepare_training_data(self):
        """Подготовка данных для обучения"""
        print("\n🔄 ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        print("=" * 35)
        
        features = []
        labels = []
        
        for movement in self.movements:
            # Извлекаем 9 ключевых признаков из MIN точки
            min_f = movement['min_features']
            
            feature_vector = []
            
            # 1. Скорости (4 признака)
            if 'velocities' in min_f:
                feature_vector.extend([
                    min_f['velocities'].get('price', 0),
                    min_f['velocities'].get('ema20', 0),
                    min_f['velocities'].get('ema50', 0),
                    min_f['velocities'].get('ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0, 0])
            
            # 2. Ускорения (4 признака)
            if 'accelerations' in min_f:
                feature_vector.extend([
                    min_f['accelerations'].get('price', 0),
                    min_f['accelerations'].get('ema20', 0),
                    min_f['accelerations'].get('ema50', 0),
                    min_f['accelerations'].get('ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0, 0])
            
            # 3. Соотношения скоростей (3 признака)
            if 'velocity_ratios' in min_f:
                feature_vector.extend([
                    min_f['velocity_ratios'].get('price_ema20', 0),
                    min_f['velocity_ratios'].get('price_ema50', 0),
                    min_f['velocity_ratios'].get('price_ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # 4. Расстояния до EMA (3 признака)
            if 'distances' in min_f:
                feature_vector.extend([
                    min_f['distances'].get('price_ema20', 0),
                    min_f['distances'].get('price_ema50', 0),
                    min_f['distances'].get('price_ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # 5. Изменения расстояний (3 признака)
            if 'distance_changes' in min_f:
                feature_vector.extend([
                    min_f['distance_changes'].get('price_ema20', 0),
                    min_f['distance_changes'].get('price_ema50', 0),
                    min_f['distance_changes'].get('price_ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # 6. Углы EMA (3 признака)
            if 'angles' in min_f:
                feature_vector.extend([
                    min_f['angles'].get('ema20', 0),
                    min_f['angles'].get('ema50', 0),
                    min_f['angles'].get('ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # 7. Изменения углов (3 признака)
            if 'angle_changes' in min_f:
                feature_vector.extend([
                    min_f['angle_changes'].get('ema20', 0),
                    min_f['angle_changes'].get('ema50', 0),
                    min_f['angle_changes'].get('ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # 8. Взаимоотношения EMA (3 признака)
            if 'ema_relationships' in min_f:
                feature_vector.extend([
                    min_f['ema_relationships'].get('ema20_ema50', 0),
                    min_f['ema_relationships'].get('ema20_ema100', 0),
                    min_f['ema_relationships'].get('ema50_ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            # 9. Синхронизации (3 признака)
            if 'synchronizations' in min_f:
                feature_vector.extend([
                    min_f['synchronizations'].get('price_ema20', 0),
                    min_f['synchronizations'].get('price_ema50', 0),
                    min_f['synchronizations'].get('price_ema100', 0)
                ])
            else:
                feature_vector.extend([0, 0, 0])
            
            features.append(feature_vector)
            
            # Создаем метки классов на основе прибыльности
            profit = movement['movement_percent']
            if profit >= 7:
                labels.append(2)  # Крупное движение
            elif profit >= 3:
                labels.append(1)  # Среднее движение
            else:
                labels.append(0)  # Малое движение
        
        # Создаем названия признаков
        self.feature_names = [
            'vel_price', 'vel_ema20', 'vel_ema50', 'vel_ema100',
            'acc_price', 'acc_ema20', 'acc_ema50', 'acc_ema100',
            'ratio_p_e20', 'ratio_p_e50', 'ratio_p_e100',
            'dist_p_e20', 'dist_p_e50', 'dist_p_e100',
            'dist_ch_e20', 'dist_ch_e50', 'dist_ch_e100',
            'angle_e20', 'angle_e50', 'angle_e100',
            'angle_ch_e20', 'angle_ch_e50', 'angle_ch_e100',
            'rel_e20_e50', 'rel_e20_e100', 'rel_e50_e100',
            'sync_p_e20', 'sync_p_e50', 'sync_p_e100'
        ]
        
        print(f"✅ Подготовлено {len(features)} образцов с {len(feature_vector)} признаками")
        print(f"📊 Классы: 0=малые, 1=средние, 2=крупные")
        
        # Статистика по классам
        unique, counts = np.unique(labels, return_counts=True)
        for cls, count in zip(unique, counts):
            cls_name = ['Малые', 'Средние', 'Крупные'][cls]
            print(f"   {cls_name}: {count} образцов")
        
        return np.array(features), np.array(labels)
    
    def train_model(self, X, y):
        """Обучение модели"""
        print("\n🤖 ОБУЧЕНИЕ ML МОДЕЛИ")
        print("=" * 25)
        
        # Разделяем на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📚 Обучающая выборка: {len(X_train)} образцов")
        print(f"🧪 Тестовая выборка: {len(X_test)} образцов")
        
        # Создаем и обучаем модель
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        
        print("🔄 Обучение модели...")
        self.model.fit(X_train, y_train)
        
        # Оценка качества
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"✅ Точность на обучении: {train_score:.3f}")
        print(f"✅ Точность на тесте: {test_score:.3f}")
        
        # Предсказания
        y_pred = self.model.predict(X_test)
        
        print("\n📊 ДЕТАЛЬНЫЙ ОТЧЕТ:")
        print(classification_report(y_test, y_pred, 
                                  target_names=['Малые', 'Средние', 'Крупные']))
        
        # Важность признаков
        self._show_feature_importance()
        
        return True
    
    def _show_feature_importance(self):
        """Показать важность признаков"""
        if self.model is None:
            return
        
        print("\n🎯 ТОП-10 ВАЖНЫХ ПРИЗНАКОВ:")
        print("-" * 30)
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        for i in range(min(10, len(importances))):
            idx = indices[i]
            print(f"{i+1:2d}. {self.feature_names[idx]:<15}: {importances[idx]:.3f}")
    
    def save_model(self):
        """Сохранение модели"""
        try:
            # Сохраняем модель
            with open('smart_predictor_model.pkl', 'wb') as f:
                pickle.dump(self.model, f)
            
            # Сохраняем названия признаков
            with open('feature_names.pkl', 'wb') as f:
                pickle.dump(self.feature_names, f)
            
            print("\n💾 МОДЕЛЬ СОХРАНЕНА:")
            print("   • smart_predictor_model.pkl")
            print("   • feature_names.pkl")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def create_predictor_script(self):
        """Создание скрипта для предсказаний"""
        predictor_code = '''#!/usr/bin/env python3
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
'''
        
        with open('smart_predictor.py', 'w', encoding='utf-8') as f:
            f.write(predictor_code)
        
        print("📁 Создан файл: smart_predictor.py")

if __name__ == "__main__":
    trainer = SmartMLTrainer()
    
    print("🤖 УМНЫЙ ML ТРЕНЕР")
    print("=" * 20)
    
    if trainer.load_movements():
        # Подготавливаем данные
        X, y = trainer.prepare_training_data()
        
        # Обучаем модель
        if trainer.train_model(X, y):
            
            # Сохраняем модель
            if trainer.save_model():
                
                # Создаем предиктор
                trainer.create_predictor_script()
                
                print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
                print("📁 Файлы созданы:")
                print("   • smart_predictor_model.pkl - обученная модель")
                print("   • feature_names.pkl - названия признаков")
                print("   • smart_predictor.py - скрипт для предсказаний")
                print("\n🧪 Тестируй: python smart_predictor.py")
