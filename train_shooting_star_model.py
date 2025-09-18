#!/usr/bin/env python3
"""
Скрипт для полного обучения модели предсказания стреляющих монет
Запускает весь процесс: сбор данных -> обучение -> тестирование
"""
import os
import sys
import time
import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Основная функция обучения"""
    print("🚀 ПОЛНОЕ ОБУЧЕНИЕ МОДЕЛИ ПРЕДСКАЗАНИЯ СТРЕЛЯЮЩИХ МОНЕТ")
    print("=" * 70)
    print(f"📅 Время начала: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Шаг 1: Сбор данных
        print("\n📊 ЭТАП 1: СБОР ИСТОРИЧЕСКИХ ДАННЫХ")
        print("-" * 50)
        
        from data_collector import HistoricalDataCollector
        
        collector = HistoricalDataCollector()
        data = collector.collect_all_data(max_pairs=100)  # Собираем данные для 100 монет
        
        if not data:
            print("❌ Не удалось собрать данные")
            return False
        
        # Сохраняем данные
        collector.save_data(data, "historical_data_full.json")
        
        # Статистика
        total_records = sum(len(df) for df in data.values())
        shooting_stars = sum(
            df['is_shooting_star'].sum() 
            for df in data.values() 
            if 'is_shooting_star' in df.columns
        )
        
        print(f"✅ Данные собраны:")
        print(f"   - Монет: {len(data)}")
        print(f"   - Записей: {total_records:,}")
        print(f"   - Стреляющих моментов: {shooting_stars:,}")
        print(f"   - Процент стреляющих: {(shooting_stars/total_records*100):.2f}%")
        
        # Шаг 2: Обучение модели
        print("\n🧠 ЭТАП 2: ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
        print("-" * 50)
        
        from neural_network_predictor import ShootingStarPredictor
        
        predictor = ShootingStarPredictor()
        
        print("🚀 Начинаю обучение модели...")
        success = predictor.train(data)
        
        if not success:
            print("❌ Не удалось обучить модель")
            return False
        
        print("✅ Модель обучена успешно!")
        
        # Шаг 3: Тестирование
        print("\n🧪 ЭТАП 3: ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("-" * 50)
        
        # Тестируем на нескольких монетах
        test_coins = list(data.keys())[:5]
        
        for coin in test_coins:
            try:
                df = data[coin]
                prediction = predictor.predict(df)
                
                if prediction:
                    print(f"📊 {coin}:")
                    print(f"   - Класс: {prediction['predicted_class']}")
                    print(f"   - Вероятность стреляющей: {prediction['shooting_star_probability']:.1%}")
                    print(f"   - Уверенность: {prediction['confidence']:.1%}")
                else:
                    print(f"❌ {coin}: Не удалось сделать предсказание")
                    
            except Exception as e:
                print(f"❌ {coin}: Ошибка тестирования - {e}")
        
        # Шаг 4: Сохранение результатов
        print("\n💾 ЭТАП 4: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 50)
        
        # Сохраняем модель
        predictor.save_model("shooting_star_model_final")
        
        # Создаем отчет
        report = {
            'training_date': datetime.now().isoformat(),
            'training_duration_minutes': (time.time() - start_time) / 60,
            'coins_processed': len(data),
            'total_records': total_records,
            'shooting_stars_found': shooting_stars,
            'shooting_stars_percentage': shooting_stars / total_records * 100,
            'model_parameters': {
                'sequence_length': predictor.sequence_length,
                'prediction_horizon': predictor.prediction_horizon
            }
        }
        
        import json
        with open('training_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("✅ Результаты сохранены:")
        print("   - Модель: shooting_star_model_final.h5")
        print("   - Скейлер: shooting_star_model_final_scaler.pkl")
        print("   - Метаданные: shooting_star_model_final_metadata.json")
        print("   - Отчет: training_report.json")
        
        # Финальная статистика
        end_time = time.time()
        duration = end_time - start_time
        
        print("\n🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
        print("=" * 70)
        print(f"⏱️  Время выполнения: {duration/60:.1f} минут")
        print(f"📊 Монет обработано: {len(data)}")
        print(f"🧠 Модель обучена: ✅")
        print(f"🎯 Готово к предсказаниям: ✅")
        print("=" * 70)
        
        print("\n📋 СЛЕДУЮЩИЕ ШАГИ:")
        print("1. Запустите shooting_star_bot.py для использования бота")
        print("2. Используйте команду /shooting_stars для поиска стреляющих монет")
        print("3. Модель автоматически загрузится при запуске бота")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        print(f"\n❌ ОБУЧЕНИЕ ПРЕРВАНО: {e}")
        return False

def quick_test():
    """Быстрый тест с небольшим количеством данных"""
    print("🧪 БЫСТРЫЙ ТЕСТ МОДЕЛИ")
    print("=" * 40)
    
    try:
        from data_collector import HistoricalDataCollector
        from neural_network_predictor import ShootingStarPredictor
        
        # Собираем данные для 10 монет
        collector = HistoricalDataCollector()
        data = collector.collect_all_data(max_pairs=10)
        
        if data:
            # Обучаем модель
            predictor = ShootingStarPredictor()
            success = predictor.train(data)
            
            if success:
                print("✅ Быстрый тест прошел успешно!")
                
                # Тестируем на одной монете
                test_coin = list(data.keys())[0]
                prediction = predictor.predict(data[test_coin])
                
                if prediction:
                    print(f"🎯 Тестовое предсказание для {test_coin}:")
                    print(f"   - Класс: {prediction['predicted_class']}")
                    print(f"   - Вероятность стреляющей: {prediction['shooting_star_probability']:.1%}")
                
                return True
            else:
                print("❌ Не удалось обучить модель")
                return False
        else:
            print("❌ Не удалось собрать данные")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка быстрого теста: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        # Быстрый тест
        success = quick_test()
    else:
        # Полное обучение
        success = main()
    
    sys.exit(0 if success else 1)


