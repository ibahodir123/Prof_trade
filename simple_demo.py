#!/usr/bin/env python3
"""
Простая демонстрация сохраненной модели
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

def load_simple_model():
    """Загружает простую модель"""
    print("🔄 Загружаю модель...")
    
    # Загружаем модель
    model = tf.keras.models.load_model("simple_shooting_star_model.h5")
    print("✅ Модель загружена!")
    
    # Загружаем скейлер
    with open("simple_shooting_star_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    print("✅ Скейлер загружен!")
    
    # Загружаем метаданные
    with open("simple_shooting_star_metadata.json", 'r') as f:
        metadata = json.load(f)
    print("✅ Метаданные загружены!")
    
    return model, scaler, metadata

def create_test_data():
    """Создает тестовые данные"""
    print("📊 Создаю тестовые данные...")
    
    # Создаем синтетические данные для тестирования
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.exponential(1000, n_samples),
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'rsi': np.random.uniform(20, 80, n_samples)
    }
    
    df = pd.DataFrame(data)
    return df

def prepare_features(df):
    """Подготавливает признаки"""
    features = pd.DataFrame()
    features['close'] = df['close']
    features['volume'] = df['volume']
    features['high'] = df['high']
    features['low'] = df['low']
    features['rsi'] = df['rsi']
    
    return features.fillna(0)

def create_sequences(features, sequence_length=12):
    """Создает последовательности для LSTM"""
    X = []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i].values)
    
    return np.array(X)

def main():
    """Основная функция"""
    print("🧠 ДЕМОНСТРАЦИЯ СОХРАНЕННОЙ МОДЕЛИ")
    print("=" * 50)
    
    # Проверяем файлы
    required_files = [
        "simple_shooting_star_model.h5",
        "simple_shooting_star_scaler.pkl",
        "simple_shooting_star_metadata.json"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print("❌ Отсутствуют файлы:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n🔧 Для создания модели запустите:")
        print("   python simple_train.py")
        return
    
    print("✅ Все файлы найдены!")
    
    try:
        # Загружаем модель
        model, scaler, metadata = load_simple_model()
        
        print(f"\n📋 Информация о модели:")
        print(f"   - Длина последовательности: {metadata['sequence_length']}")
        print(f"   - Горизонт предсказания: {metadata['prediction_horizon']}")
        print(f"   - Признаки: {metadata['features']}")
        print(f"   - Классы: {metadata['classes']}")
        print(f"   - Точность: {metadata['accuracy']:.1%}")
        
        # Создаем тестовые данные
        test_df = create_test_data()
        features = prepare_features(test_df)
        
        print(f"\n📊 Тестовые данные: {len(features)} образцов")
        
        # Создаем последовательности
        X = create_sequences(features, metadata['sequence_length'])
        print(f"📈 Создано {len(X)} последовательностей")
        
        # Нормализуем данные
        X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        
        # Делаем предсказания
        print("\n🎯 Делаю предсказания...")
        predictions = model.predict(X_scaled, verbose=0)
        
        # Анализируем результаты
        predicted_classes = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        print(f"\n📊 Результаты анализа {len(predictions)} последовательностей:")
        
        # Подсчитываем распределение классов
        class_counts = np.bincount(predicted_classes)
        class_names = metadata['classes']
        
        for i, count in enumerate(class_counts):
            if count > 0:
                print(f"   {class_names[i]}: {count} ({count/len(predictions):.1%})")
        
        # Показываем лучшие предсказания
        best_indices = np.argsort(confidences)[-5:]  # Топ-5 по уверенности
        
        print(f"\n🏆 ТОП-5 ПРЕДСКАЗАНИЙ (по уверенности):")
        for i, idx in enumerate(reversed(best_indices)):
            pred_class = predicted_classes[idx]
            confidence = confidences[idx]
            class_name = class_names[pred_class]
            
            print(f"   {i+1}. Класс: {class_name}")
            print(f"      Уверенность: {confidence:.1%}")
            print(f"      Все вероятности: {predictions[idx]}")
            print()
        
        # Анализируем стреляющие монеты
        shooting_star_indices = predicted_classes >= 3  # Классы 3 и 4 - высокий/взрывной рост
        shooting_star_count = np.sum(shooting_star_indices)
        
        print(f"🚀 СТРЕЛЯЮЩИЕ МОНЕТЫ:")
        print(f"   - Найдено: {shooting_star_count} ({shooting_star_count/len(predictions):.1%})")
        
        if shooting_star_count > 0:
            print(f"   - Средняя уверенность: {np.mean(confidences[shooting_star_indices]):.1%}")
            
            # Показываем детали стреляющих монет
            print(f"\n📈 ДЕТАЛИ СТРЕЛЯЮЩИХ МОНЕТ:")
            for i, idx in enumerate(np.where(shooting_star_indices)[0][:3]):  # Показываем первые 3
                pred_class = predicted_classes[idx]
                confidence = confidences[idx]
                class_name = class_names[pred_class]
                
                print(f"   {i+1}. {class_name} (уверенность: {confidence:.1%})")
        
        print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
        print("\n💡 Выводы:")
        print("   ✅ Модель успешно загружается из файлов")
        print("   ✅ Предсказания работают в реальном времени")
        print("   ✅ Система сохранения/загрузки функционирует")
        print("   ✅ Не нужно переобучать каждый раз")
        
        print("\n📋 Доступные команды:")
        print("   - python simple_demo.py (эта демонстрация)")
        print("   - python shooting_star_bot.py (запуск бота)")
        print("   - python simple_train.py (переобучение)")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

