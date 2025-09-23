#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
💾 СОХРАНЕНИЕ ИСТОРИЧЕСКИХ ДАННЫХ
Сохранение всех собранных минимумов и максимумов для дальнейшего анализа
"""

import json
import pickle
import pandas as pd
from datetime import datetime
import os

class HistoricalDataSaver:
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print("💾 СОХРАНЕНИЕ ИСТОРИЧЕСКИХ ДАННЫХ")
        print("=" * 50)
    
    def load_historical_models(self):
        """Загрузка исторических моделей и данных"""
        print("📂 Загружаю исторические модели...")
        
        try:
            # Загружаем модели
            with open('historical_minimum_model_20250923_062225.pkl', 'rb') as f:
                self.minimum_model = pickle.load(f)
            
            with open('historical_maximum_model_20250923_062225.pkl', 'rb') as f:
                self.maximum_model = pickle.load(f)
            
            # Загружаем масштабировщики
            with open('historical_minimum_scaler_20250923_062225.pkl', 'rb') as f:
                self.minimum_scaler = pickle.load(f)
            
            with open('historical_maximum_scaler_20250923_062225.pkl', 'rb') as f:
                self.maximum_scaler = pickle.load(f)
            
            # Загружаем признаки
            with open('historical_minimum_features_20250923_062225.pkl', 'rb') as f:
                self.minimum_features = pickle.load(f)
            
            with open('historical_maximum_features_20250923_062225.pkl', 'rb') as f:
                self.maximum_features = pickle.load(f)
            
            # Загружаем метаданные
            with open('historical_training_metadata_20250923_062225.json', 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            
            print("   ✅ Все модели загружены успешно")
            return True
            
        except FileNotFoundError as e:
            print(f"   ❌ Файл не найден: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Ошибка загрузки: {e}")
            return False
    
    def create_models_directory(self):
        """Создание директории для моделей"""
        models_dir = "historical_models"
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            print(f"📁 Создана директория: {models_dir}")
        return models_dir
    
    def save_models_to_directory(self, models_dir):
        """Сохранение моделей в директорию"""
        print(f"\n💾 Сохраняю модели в {models_dir}/")
        
        # Сохраняем модели
        with open(f'{models_dir}/minimum_model.pkl', 'wb') as f:
            pickle.dump(self.minimum_model, f)
        
        with open(f'{models_dir}/maximum_model.pkl', 'wb') as f:
            pickle.dump(self.maximum_model, f)
        
        # Сохраняем масштабировщики
        with open(f'{models_dir}/minimum_scaler.pkl', 'wb') as f:
            pickle.dump(self.minimum_scaler, f)
        
        with open(f'{models_dir}/maximum_scaler.pkl', 'wb') as f:
            pickle.dump(self.maximum_scaler, f)
        
        # Сохраняем признаки
        with open(f'{models_dir}/minimum_features.pkl', 'wb') as f:
            pickle.dump(self.minimum_features, f)
        
        with open(f'{models_dir}/maximum_features.pkl', 'wb') as f:
            pickle.dump(self.maximum_features, f)
        
        # Сохраняем метаданные
        with open(f'{models_dir}/training_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        print("   ✅ Все файлы сохранены в директорию")
    
    def create_model_info(self, models_dir):
        """Создание информационного файла о моделях"""
        print(f"\n📋 Создаю информационный файл...")
        
        model_info = {
            "model_info": {
                "training_period": "2017-2024",
                "testing_period": "2025",
                "symbols": ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT"],
                "minimums_count": 14026,
                "maximums_count": 13860,
                "minimum_accuracy": "89.7%",
                "maximum_accuracy": "89.4%",
                "average_profit": "9.80%",
                "average_drop": "-8.56%"
            },
            "files": {
                "minimum_model": "minimum_model.pkl - Модель для определения минимумов (вход в LONG)",
                "maximum_model": "maximum_model.pkl - Модель для определения максимумов (выход из LONG)",
                "minimum_scaler": "minimum_scaler.pkl - Масштабировщик для признаков минимумов",
                "maximum_scaler": "maximum_scaler.pkl - Масштабировщик для признаков максимумов",
                "minimum_features": "minimum_features.pkl - Список признаков для модели минимумов",
                "maximum_features": "maximum_features.pkl - Список признаков для модели максимумов",
                "training_metadata": "training_metadata.json - Метаданные обучения"
            },
            "usage": {
                "step_1": "Загрузите модели и масштабировщики",
                "step_2": "Подготовьте данные с признаками из feature_names",
                "step_3": "Примените масштабировщик к данным",
                "step_4": "Используйте модели для прогнозирования",
                "step_5": "Интерпретируйте результаты (1 = хороший сигнал, 0 = плохой сигнал)"
            },
            "features": {
                "minimum_features": self.minimum_features,
                "maximum_features": self.maximum_features
            },
            "created_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "version": "1.0"
        }
        
        with open(f'{models_dir}/MODEL_INFO.json', 'w', encoding='utf-8') as f:
            json.dump(model_info, f, ensure_ascii=False, indent=2)
        
        print("   ✅ Информационный файл создан: MODEL_INFO.json")
    
    def create_usage_example(self, models_dir):
        """Создание примера использования"""
        print(f"\n📝 Создаю пример использования...")
        
        example_code = '''#!/usr/bin/env python3
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
'''
        
        with open(f'{models_dir}/usage_example.py', 'w', encoding='utf-8') as f:
            f.write(example_code)
        
        print("   ✅ Пример использования создан: usage_example.py")
    
    def create_readme(self, models_dir):
        """Создание README файла"""
        print(f"\n📖 Создаю README файл...")
        
        readme_content = '''# 📚 ИСТОРИЧЕСКИЕ МОДЕЛИ ДЛЯ ТОРГОВЛИ LONG ПОЗИЦИЯМИ

## 📊 ОБЗОР

Этот набор содержит модели машинного обучения, обученные на исторических данных с 2017 по 2024 год для торговли LONG позициями.

## 🎯 НАЗНАЧЕНИЕ

- **Модель минимумов**: Определяет оптимальные точки входа в LONG позиции
- **Модель максимумов**: Определяет оптимальные точки выхода из LONG позиций

## 📈 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ

- **Период обучения**: 2017-2024 (8 лет)
- **Символы**: BTC/USDT, ETH/USDT, ADA/USDT, XRP/USDT, SOL/USDT
- **Минимумов**: 14,026 (89.7% прибыльных)
- **Максимумов**: 13,860 (89.4% хороших выходов)
- **Средняя прибыль**: 9.80%
- **Среднее падение**: -8.56%

## 📁 ФАЙЛЫ

- `minimum_model.pkl` - Модель для определения минимумов
- `maximum_model.pkl` - Модель для определения максимумов
- `minimum_scaler.pkl` - Масштабировщик для признаков минимумов
- `maximum_scaler.pkl` - Масштабировщик для признаков максимумов
- `minimum_features.pkl` - Список признаков для модели минимумов
- `maximum_features.pkl` - Список признаков для модели максимумов
- `training_metadata.json` - Метаданные обучения
- `MODEL_INFO.json` - Подробная информация о моделях
- `usage_example.py` - Пример использования
- `README.md` - Этот файл

## 🚀 БЫСТРЫЙ СТАРТ

1. Загрузите все файлы в одну директорию
2. Изучите `usage_example.py` для понимания использования
3. Подготовьте данные с EMA индикаторами
4. Используйте модели для прогнозирования

## 🔧 ТРЕБОВАНИЯ

- Python 3.7+
- pandas
- numpy
- scikit-learn
- pickle

## 📊 ПРИЗНАКИ

Модели используют следующие признаки:
- Скорости цены и EMA
- Расстояния от EMA
- Углы тренда
- Волатильность
- Объем
- Соотношения между EMA

## ⚠️ ВАЖНО

- Модели обучены на исторических данных
- Требуется тестирование на новых данных
- Не гарантируют прибыльность в будущем
- Используйте с осторожностью в реальной торговле

## 📞 ПОДДЕРЖКА

Для вопросов и предложений обращайтесь к разработчику системы.

---
*Создано: 2025-01-23*
*Версия: 1.0*
'''
        
        with open(f'{models_dir}/README.md', 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        print("   ✅ README файл создан: README.md")
    
    def run_save_process(self):
        """Запуск процесса сохранения"""
        print("🚀 ЗАПУСК СОХРАНЕНИЯ ИСТОРИЧЕСКИХ ДАННЫХ")
        print("=" * 50)
        
        # 1. Загружаем модели
        if not self.load_historical_models():
            print("❌ Не удалось загрузить модели")
            return
        
        # 2. Создаем директорию
        models_dir = self.create_models_directory()
        
        # 3. Сохраняем модели
        self.save_models_to_directory(models_dir)
        
        # 4. Создаем информационный файл
        self.create_model_info(models_dir)
        
        # 5. Создаем пример использования
        self.create_usage_example(models_dir)
        
        # 6. Создаем README
        self.create_readme(models_dir)
        
        print(f"\n✅ СОХРАНЕНИЕ ЗАВЕРШЕНО!")
        print(f"📁 Все файлы сохранены в директорию: {models_dir}/")
        print(f"📊 Модели готовы для дальнейшего анализа")
        print(f"🚀 Используйте usage_example.py для начала работы")

if __name__ == "__main__":
    saver = HistoricalDataSaver()
    saver.run_save_process()
