#!/usr/bin/env python3
"""
Простое обучение модели на минимальных данных
"""
import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import pickle

def create_simple_model():
    """Создает простую модель LSTM"""
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(12, 5)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')  # 5 классов
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def prepare_simple_features(df):
    """Подготавливает простые признаки"""
    features = pd.DataFrame()
    
    # Основные признаки
    features['close'] = df['close']
    features['volume'] = df['volume']
    features['high'] = df['high']
    features['low'] = df['low']
    features['rsi'] = df.get('rsi', 50)  # Если RSI нет, используем 50
    
    return features.fillna(0)

def create_simple_sequences(features, targets, sequence_length=12):
    """Создает последовательности для LSTM"""
    X, y = [], []
    
    for i in range(sequence_length, len(features)):
        X.append(features[i-sequence_length:i].values)
        y.append(targets[i])
    
    return np.array(X), np.array(y)

def simple_train():
    """Простое обучение на синтетических данных"""
    print("🚀 ПРОСТОЕ ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 40)
    
    # Создаем синтетические данные
    print("📊 Создаю синтетические данные...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Генерируем данные
    data = {
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.exponential(1000, n_samples),
        'high': np.random.randn(n_samples).cumsum() + 102,
        'low': np.random.randn(n_samples).cumsum() + 98,
        'rsi': np.random.uniform(20, 80, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Создаем целевые переменные (классы роста)
    df['price_change'] = df['close'].pct_change(12).fillna(0)
    
    def categorize_growth(change):
        if change > 0.2:
            return 4  # Взрывной рост
        elif change > 0.1:
            return 3  # Высокий рост
        elif change > 0.05:
            return 2  # Умеренный рост
        elif change > 0:
            return 1  # Небольшой рост
        else:
            return 0  # Падение/боковик
    
    df['growth_category'] = df['price_change'].apply(categorize_growth)
    df['is_shooting_star'] = (df['growth_category'] >= 3).astype(int)
    
    print(f"✅ Создано {len(df)} образцов")
    print(f"📈 Распределение классов: {df['growth_category'].value_counts().sort_index().to_dict()}")
    print(f"🎯 Стреляющих моментов: {df['is_shooting_star'].sum()}")
    
    # Подготавливаем признаки
    print("\n🔧 Подготавливаю признаки...")
    features = prepare_simple_features(df)
    targets = df['growth_category'].values
    
    # Создаем последовательности
    print("📊 Создаю последовательности...")
    X, y = create_simple_sequences(features, targets)
    
    print(f"✅ Создано {len(X)} последовательностей")
    print(f"📊 Размер данных: {X.shape}")
    
    # Нормализация
    print("🔧 Нормализую данные...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Создаем модель
    print("🧠 Создаю модель...")
    model = create_simple_model()
    
    # Обучение
    print("🚀 Начинаю обучение...")
    history = model.fit(
        X_scaled, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Сохраняем модель
    print("💾 Сохраняю модель...")
    model.save("simple_shooting_star_model.h5")
    
    # Сохраняем скейлер
    with open("simple_shooting_star_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    
    # Сохраняем метаданные
    metadata = {
        "sequence_length": 12,
        "prediction_horizon": 12,
        "features": list(features.columns),
        "classes": ["fall", "small_growth", "medium_growth", "high_growth", "explosive_growth"],
        "training_samples": len(X),
        "accuracy": float(history.history['val_accuracy'][-1])
    }
    
    with open("simple_shooting_star_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("✅ Модель сохранена!")
    print(f"📊 Точность: {history.history['val_accuracy'][-1]:.2%}")
    
    # Тестируем предсказание
    print("\n🎯 Тестирую предсказание...")
    test_sample = X_scaled[-1:].reshape(1, 12, 5)
    prediction = model.predict(test_sample, verbose=0)
    
    predicted_class = np.argmax(prediction[0])
    confidence = np.max(prediction[0])
    
    print(f"   - Предсказанный класс: {predicted_class}")
    print(f"   - Уверенность: {confidence:.1%}")
    print(f"   - Все вероятности: {prediction[0]}")
    
    print("\n🎉 ГОТОВО! Простая модель обучена!")
    print("\n📁 Созданные файлы:")
    print("   - simple_shooting_star_model.h5")
    print("   - simple_shooting_star_scaler.pkl")
    print("   - simple_shooting_star_metadata.json")
    
    return True

def main():
    """Основная функция"""
    print("🧠 ПРОСТОЕ ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ СТРЕЛЯЮЩИХ МОНЕТ")
    print("=" * 60)
    
    success = simple_train()
    
    if success:
        print("\n✅ УСПЕШНО! Модель готова к использованию!")
        print("\n📋 Следующие шаги:")
        print("   - python demo_saved_model.py (демонстрация)")
        print("   - python shooting_star_bot.py (запуск бота)")
    else:
        print("\n❌ ОБУЧЕНИЕ НЕ УДАЛОСЬ")

if __name__ == "__main__":
    main()

