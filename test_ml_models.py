#!/usr/bin/env python3
"""
Тест ML моделей для проверки их работы
"""

import logging
import numpy as np
import joblib
import json
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ml_models():
    """Тестирование загруженных ML моделей"""
    try:
        logger.info("🧪 Тестирую ML модели...")
        
        # Проверяем существование файлов
        model_files = [
            'models/entry_model.pkl',
            'models/exit_model.pkl', 
            'models/ema_scaler.pkl',
            'models/feature_names.pkl'
        ]
        
        for file_path in model_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Файл модели не найден: {file_path}")
                return False
            else:
                logger.info(f"✅ Файл найден: {file_path}")
        
        # Загружаем модели
        entry_model = joblib.load('models/entry_model.pkl')
        exit_model = joblib.load('models/exit_model.pkl')
        scaler = joblib.load('models/ema_scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        logger.info(f"✅ Модели загружены успешно")
        logger.info(f"📊 Количество признаков: {scaler.n_features_in_}")
        logger.info(f"📊 Имена признаков: {feature_names}")
        
        # Создаем тестовые данные (10 признаков)
        test_features = np.array([
            [50000.0, 49000.0, 48000.0,  # EMA 20, 50, 100
             0.02, 0.015,  # Скорости EMA
             -0.03, 0.02, -0.025,  # Отношения цены к EMA
             15.5, 2.0]  # Угол тренда, тип тренда
        ])
        
        logger.info(f"🧪 Тестовые признаки: {test_features[0]}")
        
        # Масштабируем признаки
        test_features_scaled = scaler.transform(test_features)
        
        # Получаем предсказания
        if hasattr(entry_model, 'predict_proba') and hasattr(exit_model, 'predict_proba'):
            entry_prob = entry_model.predict_proba(test_features_scaled)[0][1]
            exit_prob = exit_model.predict_proba(test_features_scaled)[0][1]
            
            logger.info(f"🎯 Вероятность входа: {entry_prob:.3f}")
            logger.info(f"🎯 Вероятность выхода: {exit_prob:.3f}")
            
            # Проверяем, что модели работают (не возвращают 0.00)
            if entry_prob > 0 and exit_prob > 0:
                logger.info("✅ Модели работают корректно!")
                
                # Определяем сигнал
                if entry_prob > 0.4:
                    logger.info("🟢 Сигнал LONG (вход)")
                elif exit_prob > 0.4:
                    logger.info("🔴 Сигнал SHORT (выход)")
                else:
                    logger.info("⚪️ Сигнал ОЖИДАНИЕ")
                
                return True
            else:
                logger.error("❌ Модели возвращают 0.00 - проблема!")
                return False
        else:
            logger.error("❌ Модели не поддерживают predict_proba")
            return False
            
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
        return False

def show_training_metadata():
    """Показать метаданные обучения"""
    try:
        metadata_file = 'models/training_metadata.json'
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info("📁 Метаданные обучения:")
            logger.info(f"   Дата: {metadata.get('training_date')}")
            logger.info(f"   Символы: {', '.join(metadata.get('symbols_used', []))}")
            logger.info(f"   Образцов: {metadata.get('samples_count')}")
            logger.info(f"   Качество входа: {metadata.get('entry_model_score', 0):.3f}")
            logger.info(f"   Качество выхода: {metadata.get('exit_model_score', 0):.3f}")
            logger.info(f"   Источник данных: {metadata.get('data_source')}")
            
            if metadata.get('data_source') == 'real_binance_historical':
                logger.info("✅ Модели обучены на РЕАЛЬНЫХ данных!")
            else:
                logger.warning("⚠️ Модели НЕ обучены на реальных данных")
        else:
            logger.warning("⚠️ Файл метаданных не найден")
            
    except Exception as e:
        logger.error(f"❌ Ошибка чтения метаданных: {e}")

if __name__ == "__main__":
    print("🧪 Тестирование ML моделей")
    print("=" * 50)
    
    show_training_metadata()
    print()
    
    success = test_ml_models()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Тест пройден успешно!")
        print("🤖 ML модели работают корректно")
    else:
        print("❌ Тест провален!")
        print("🔧 Требуется исправление моделей")



