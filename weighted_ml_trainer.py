#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ML Тренер с взвешенной оценкой критериев для определения минимумов
Определяет удельный вес каждого из 4 критериев
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

class WeightedMLTrainer:
    def __init__(self):
        self.minimums_data = []
        self.model = None
        self.feature_names = ['price_velocity', 'ema20_velocity', 'ema20_angle', 'distance_to_ema20']
        self.feature_weights = {}
        
    def load_minimums_data(self, filename: str) -> bool:
        """Загрузка данных о минимумах"""
        try:
            print(f"📊 Загружаю данные из {filename}")
            
            with open(filename, 'r', encoding='utf-8') as f:
                self.minimums_data = json.load(f)
            
            print(f"✅ Загружено {len(self.minimums_data)} минимумов")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных: {e}")
            return False
    
    def analyze_feature_importance(self):
        """Анализ важности каждого критерия"""
        print("\\n🔍 АНАЛИЗ ВАЖНОСТИ КРИТЕРИЕВ")
        print("=" * 40)
        
        if not self.minimums_data:
            print("❌ Нет данных для анализа!")
            return
        
        # Разделяем на прибыльные и убыточные минимумы
        profitable = [m for m in self.minimums_data if m['is_profitable']]
        unprofitable = [m for m in self.minimums_data if not m['is_profitable']]
        
        print(f"📈 Прибыльных минимумов: {len(profitable)}")
        print(f"📉 Убыточных минимумов: {len(unprofitable)}")
        
        if len(profitable) == 0 or len(unprofitable) == 0:
            print("❌ Недостаточно данных для анализа важности!")
            return
        
        # Анализируем каждый критерий
        feature_scores = {}
        
        for feature in self.feature_names:
            # Значения критерия для прибыльных минимумов
            profitable_values = [m['criteria'][feature] for m in profitable]
            unprofitable_values = [m['criteria'][feature] for m in unprofitable]
            
            # Статистика
            prof_mean = np.mean(profitable_values)
            prof_std = np.std(profitable_values)
            unprof_mean = np.mean(unprofitable_values)
            unprof_std = np.std(unprofitable_values)
            
            # Расчет разделительной способности (чем больше разница средних, тем важнее критерий)
            separation_power = abs(prof_mean - unprof_mean) / (prof_std + unprof_std + 0.001)
            
            feature_scores[feature] = {
                'separation_power': separation_power,
                'profitable_mean': prof_mean,
                'profitable_std': prof_std,
                'unprofitable_mean': unprof_mean,
                'unprofitable_std': unprof_std
            }
            
            print(f"\\n📊 {feature}:")
            print(f"   Прибыльные: {prof_mean:.3f} ± {prof_std:.3f}")
            print(f"   Убыточные: {unprof_mean:.3f} ± {unprof_std:.3f}")
            print(f"   Разделительная сила: {separation_power:.3f}")
        
        # Нормализуем веса
        total_power = sum([scores['separation_power'] for scores in feature_scores.values()])
        
        print(f"\\n⚖️ УДЕЛЬНЫЕ ВЕСА КРИТЕРИЕВ:")
        for feature in self.feature_names:
            weight = feature_scores[feature]['separation_power'] / total_power
            self.feature_weights[feature] = weight
            print(f"   {feature}: {weight:.1%}")
        
        return feature_scores
    
    def prepare_training_data(self):
        """Подготовка данных для обучения"""
        if not self.minimums_data:
            return None, None
        
        X = []
        y = []
        
        for minimum in self.minimums_data:
            # Извлекаем признаки
            features = [minimum['criteria'][name] for name in self.feature_names]
            
            # Проверяем на NaN/Inf
            if any(np.isnan(f) or np.isinf(f) for f in features):
                continue
            
            X.append(features)
            
            # Метка класса (1 = прибыльный минимум, 0 = убыточный)
            y.append(1 if minimum['is_profitable'] else 0)
        
        return np.array(X), np.array(y)
    
    def train_model(self):
        """Обучение ML модели"""
        print("\\n🧠 ОБУЧЕНИЕ ML МОДЕЛИ")
        print("=" * 30)
        
        X, y = self.prepare_training_data()
        if X is None or len(X) == 0:
            print("❌ Нет данных для обучения!")
            return False
        
        print(f"📊 Подготовлено {len(X)} образцов")
        print(f"📈 Из них прибыльных: {sum(y)}")
        print(f"📉 Убыточных: {len(y) - sum(y)}")
        
        # Проверяем баланс классов
        if sum(y) == 0 or sum(y) == len(y):
            print("❌ Все примеры одного класса! Нельзя обучить модель.")
            return False
        
        # Разделяем на train/test
        if len(X) < 10:
            print("⚠️ Мало данных, используем все для обучения")
            X_train, X_test, y_train, y_test = X, X, y, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Обучаем модель
        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        
        self.model.fit(X_train, y_train)
        
        # Оценка модели
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"✅ Точность на обучении: {train_acc:.3f}")
        print(f"✅ Точность на тесте: {test_acc:.3f}")
        
        # Важность признаков от модели
        model_importances = self.model.feature_importances_
        print(f"\\n🏆 ВАЖНОСТЬ ПРИЗНАКОВ (по модели):")
        for i, feature in enumerate(self.feature_names):
            print(f"   {feature}: {model_importances[i]:.3f}")
        
        # Матрица ошибок
        if len(X_test) > 0:
            cm = confusion_matrix(y_test, test_pred)
            print(f"\\n📊 Матрица ошибок:")
            print(f"   Predicted:  [0, 1]")
            print(f"   Actual 0:   {cm[0]}")
            print(f"   Actual 1:   {cm[1]}")
        
        return True
    
    def calculate_minimum_probability(self, criteria_values: dict) -> dict:
        """Расчет вероятности минимума по критериям"""
        if not self.model:
            return {'error': 'Модель не обучена'}
        
        try:
            # Подготавливаем вектор признаков
            feature_vector = []
            for name in self.feature_names:
                value = criteria_values.get(name, 0.0)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Предсказание модели
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            # Взвешенная оценка (если есть веса)
            weighted_score = 0.0
            if self.feature_weights:
                for i, feature in enumerate(self.feature_names):
                    weight = self.feature_weights[feature]
                    # Нормализуем значение критерия для взвешивания
                    normalized_value = max(0, min(1, (abs(feature_vector[0][i]) / 10)))  # Приблизительная нормализация
                    weighted_score += weight * normalized_value
            
            result = {
                'is_minimum': bool(prediction),
                'probability': float(probabilities[1]),  # Вероятность класса 1 (прибыльный минимум)
                'weighted_score': weighted_score,
                'criteria_analysis': {}
            }
            
            # Детальный анализ критериев
            for i, feature in enumerate(self.feature_names):
                result['criteria_analysis'][feature] = {
                    'value': float(feature_vector[0][i]),
                    'weight': self.feature_weights.get(feature, 0.25),
                    'contribution': self.feature_weights.get(feature, 0.25) * abs(feature_vector[0][i])
                }
            
            return result
            
        except Exception as e:
            return {'error': f'Ошибка предсказания: {e}'}
    
    def save_model(self, model_filename: str = "minimum_detector_model.pkl"):
        """Сохранение обученной модели"""
        if not self.model:
            print("❌ Модель не обучена!")
            return False
        
        try:
            # Сохраняем модель
            with open(model_filename, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Сохраняем веса и метаданные
            metadata = {
                'feature_names': self.feature_names,
                'feature_weights': self.feature_weights,
                'training_date': datetime.now().isoformat(),
                'total_minimums': len(self.minimums_data)
            }
            
            metadata_filename = model_filename.replace('.pkl', '_metadata.json')
            with open(metadata_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Модель сохранена: {model_filename}")
            print(f"💾 Метаданные: {metadata_filename}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False
    
    def test_model_on_examples(self):
        """Тестирование модели на примерах"""
        if not self.model or not self.minimums_data:
            return
        
        print("\\n🧪 ТЕСТИРОВАНИЕ НА ПРИМЕРАХ")
        print("=" * 35)
        
        # Берем несколько примеров
        examples = self.minimums_data[:5]
        
        for i, example in enumerate(examples):
            print(f"\\n📊 Пример {i+1}: {example['symbol']} {example['time'][:10]}")
            print(f"   Реальный результат: {'✅ Прибыльный' if example['is_profitable'] else '❌ Убыточный'}")
            print(f"   Прибыль через 24ч: {example['future_profit_24h']:.2f}%")
            
            # Предсказание модели
            prediction = self.calculate_minimum_probability(example['criteria'])
            
            if 'error' not in prediction:
                print(f"   🤖 Предсказание модели: {'✅ Минимум' if prediction['is_minimum'] else '❌ Не минимум'}")
                print(f"   📊 Вероятность минимума: {prediction['probability']:.1%}")
                print(f"   ⚖️ Взвешенная оценка: {prediction['weighted_score']:.3f}")
            else:
                print(f"   ❌ {prediction['error']}")

if __name__ == "__main__":
    trainer = WeightedMLTrainer()
    
    print("🧠 ВЗВЕШЕННЫЙ ML ТРЕНЕР ДЛЯ МИНИМУМОВ")
    print("⚖️ Определяет важность каждого критерия")
    print("=" * 45)
    
    # Загружаем данные
    if trainer.load_minimums_data("minimums_202501.json"):
        # Анализируем важность критериев
        trainer.analyze_feature_importance()
        
        # Обучаем модель
        if trainer.train_model():
            # Сохраняем модель
            trainer.save_model()
            
            # Тестируем на примерах
            trainer.test_model_on_examples()
        else:
            print("❌ Не удалось обучить модель!")
    else:
        print("❌ Сначала запустите simple_min_detector.py для сбора данных!")


