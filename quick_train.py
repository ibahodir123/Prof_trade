#!/usr/bin/env python3
"""
Быстрое обучение модели с оптимизацией памяти
"""
import os
import json
import numpy as np
import pandas as pd
from neural_network_predictor import ShootingStarPredictor
from data_collector import HistoricalDataCollector

def optimize_data_for_training(data, max_samples_per_coin=1000):
    """Оптимизирует данные для обучения, ограничивая количество образцов"""
    optimized_data = {}
    
    for symbol, df_data in data.items():
        # Конвертируем JSON данные в DataFrame
        if isinstance(df_data, str):
            df = pd.read_json(df_data, orient='records')
        else:
            df = pd.DataFrame(df_data)
        
        if df.empty or 'is_shooting_star' not in df.columns:
            continue
            
        # Ограничиваем количество образцов
        if len(df) > max_samples_per_coin:
            # Берем последние max_samples_per_coin записей
            df_optimized = df.tail(max_samples_per_coin).copy()
        else:
            df_optimized = df.copy()
        
        optimized_data[symbol] = df_optimized
        print(f"   {symbol}: {len(df_optimized)} образцов (было {len(df)})")
    
    return optimized_data

def quick_train():
    """Быстрое обучение с оптимизацией"""
    print("🚀 БЫСТРОЕ ОБУЧЕНИЕ МОДЕЛИ")
    print("=" * 40)
    
    # Проверяем наличие данных
    data_files = [f for f in os.listdir(".") if f.startswith("data_batch_")]
    
    if not data_files:
        print("❌ Данные не найдены!")
        print("Запустите: python train_shooting_star_model.py --quick")
        return False
    
    # Загружаем данные напрямую
    with open(data_files[0], 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not data:
        print("❌ Не удалось загрузить данные")
        return False
    
    print(f"📊 Загружено {len(data)} монет")
    
    # Оптимизируем данные
    print("\n🔧 Оптимизирую данные...")
    optimized_data = optimize_data_for_training(data, max_samples_per_coin=500)
    
    if not optimized_data:
        print("❌ Нет данных для обучения")
        return False
    
    print(f"✅ Оптимизировано {len(optimized_data)} монет")
    
    # Создаем и обучаем модель
    print("\n🧠 Обучаю модель...")
    predictor = ShootingStarPredictor(
        sequence_length=12,  # Уменьшаем длину последовательности
        prediction_horizon=12  # Уменьшаем горизонт предсказания
    )
    
    try:
        success = predictor.train(optimized_data)
        
        if success:
            print("✅ Модель обучена успешно!")
            
            # Сохраняем модель
            predictor.save_model("shooting_star_model_quick")
            
            print("\n💾 Модель сохранена:")
            print("   - shooting_star_model_quick.h5")
            print("   - shooting_star_model_quick_scaler.pkl")
            print("   - shooting_star_model_quick_metadata.json")
            
            # Тестируем на одной монете
            test_symbol = list(optimized_data.keys())[0]
            test_df = optimized_data[test_symbol]
            
            prediction = predictor.predict(test_df)
            
            if prediction:
                print(f"\n🎯 ТЕСТОВОЕ ПРЕДСКАЗАНИЕ ({test_symbol}):")
                print(f"   - Класс: {prediction['predicted_class']}")
                print(f"   - Уверенность: {prediction['confidence']:.1%}")
                print(f"   - Стреляющая монета: {prediction['shooting_star_probability']:.1%}")
                print(f"   - Высокий рост: {prediction['high_growth_probability']:.1%}")
            
            print("\n🎉 ГОТОВО! Модель можно использовать!")
            print("\n📋 Команды для использования:")
            print("   - python demo_saved_model.py (демонстрация)")
            print("   - python shooting_star_bot.py (запуск бота)")
            
            return True
        else:
            print("❌ Не удалось обучить модель")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка обучения: {e}")
        return False

def main():
    """Основная функция"""
    print("🧠 БЫСТРОЕ ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ СТРЕЛЯЮЩИХ МОНЕТ")
    print("=" * 60)
    
    success = quick_train()
    
    if success:
        print("\n✅ УСПЕШНО! Модель готова к использованию!")
    else:
        print("\n❌ ОБУЧЕНИЕ НЕ УДАЛОСЬ")

if __name__ == "__main__":
    main()

