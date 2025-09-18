#!/usr/bin/env python3
"""
Демонстрация работы с сохраненной моделью
Показывает, как загрузить уже обученную модель и использовать её
"""
import os
from neural_network_predictor import ShootingStarPredictor
from ml_bot_binance import get_binance_data

def demo_saved_model():
    """Демонстрация работы с сохраненной моделью"""
    print("🧠 ДЕМОНСТРАЦИЯ СОХРАНЕННОЙ МОДЕЛИ")
    print("=" * 50)
    
    # Проверяем, есть ли сохраненная модель
    model_files = [
        "shooting_star_model.h5",
        "shooting_star_model_scaler.pkl", 
        "shooting_star_model_metadata.json",
        "simple_shooting_star_model.h5",
        "simple_shooting_star_scaler.pkl",
        "simple_shooting_star_metadata.json"
    ]
    
    # Проверяем наличие любой из моделей
    simple_model_exists = all(os.path.exists(f) for f in model_files[3:6])  # simple_* файлы
    full_model_exists = all(os.path.exists(f) for f in model_files[0:3])    # shooting_star_* файлы
    
    model_exists = simple_model_exists or full_model_exists
    
    if not model_exists:
        print("❌ Обученная модель не найдена!")
        print("\n📋 Доступные файлы:")
        for file in os.listdir("."):
            if "shooting_star" in file:
                print(f"   - {file}")
        
        print("\n🔧 Для создания модели запустите:")
        print("   python train_shooting_star_model.py --quick")
        return False
    
    print("✅ Найдена сохраненная модель!")
    print("📁 Файлы модели:")
    for file in model_files:
        if os.path.exists(file):
            size = os.path.getsize(file) / 1024 / 1024  # MB
            print(f"   - {file} ({size:.1f} MB)")
    
    # Загружаем модель
    print("\n🔄 Загружаю модель...")
    predictor = ShootingStarPredictor()
    
    # Пробуем загрузить простую модель
    if simple_model_exists and predictor.load_model("simple_shooting_star_model"):
        print("✅ Модель загружена успешно!")
        print(f"   - Длина последовательности: {predictor.sequence_length}")
        print(f"   - Горизонт предсказания: {predictor.prediction_horizon}")
        print(f"   - Модель обучена: {predictor.is_trained}")
    else:
        print("❌ Ошибка загрузки модели")
        return False
    
    # Тестируем на реальных данных
    print("\n🎯 ТЕСТИРУЮ НА РЕАЛЬНЫХ ДАННЫХ")
    print("-" * 40)
    
    test_coins = ["BTC/USDT", "ETH/USDT", "BNB/USDT"]
    
    for coin in test_coins:
        try:
            print(f"\n📊 Анализирую {coin}...")
            
            # Получаем данные
            df = get_binance_data(coin, timeframe='1h', limit=500)
            if df is None or df.empty:
                print(f"   ❌ Нет данных для {coin}")
                continue
            
            # Добавляем индикаторы (упрощенно)
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Volume
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=24).std()
            
            df = df.dropna()
            
            if len(df) < predictor.sequence_length:
                print(f"   ❌ Недостаточно данных для {coin}")
                continue
            
            # Делаем предсказание
            prediction = predictor.predict(df)
            
            if prediction:
                print(f"   🎯 Результат для {coin}:")
                print(f"      - Класс: {prediction['predicted_class']}")
                print(f"      - Уверенность: {prediction['confidence']:.1%}")
                print(f"      - Стреляющая монета: {prediction['shooting_star_probability']:.1%}")
                print(f"      - Высокий рост: {prediction['high_growth_probability']:.1%}")
                print(f"      - Взрывной рост: {prediction['explosive_growth_probability']:.1%}")
                
                # Выводим все вероятности
                print(f"      - Все вероятности:")
                for class_name, prob in prediction['probabilities'].items():
                    print(f"        {class_name}: {prob:.1%}")
            else:
                print(f"   ❌ Не удалось сделать предсказание для {coin}")
                
        except Exception as e:
            print(f"   ❌ Ошибка анализа {coin}: {e}")
    
    print("\n🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА!")
    print("\n💡 Выводы:")
    print("   ✅ Модель загружается из файлов")
    print("   ✅ Предсказания работают в реальном времени")
    print("   ✅ Не нужно переобучать каждый раз")
    
    return True

def check_model_status():
    """Проверяет статус модели"""
    print("🔍 ПРОВЕРКА СТАТУСА МОДЕЛИ")
    print("=" * 40)
    
    # Ищем все файлы модели
    model_files = []
    for file in os.listdir("."):
        if "shooting_star" in file and any(ext in file for ext in ['.h5', '.pkl', '.json']):
            model_files.append(file)
    
    if model_files:
        print("📁 Найденные файлы модели:")
        for file in sorted(model_files):
            if os.path.exists(file):
                size = os.path.getsize(file) / 1024  # KB
                print(f"   ✅ {file} ({size:.1f} KB)")
    else:
        print("❌ Файлы модели не найдены")
        print("\n🔧 Для создания модели:")
        print("   python train_shooting_star_model.py --quick")
    
    # Проверяем данные
    if os.path.exists("historical_data.json"):
        size = os.path.getsize("historical_data.json") / 1024 / 1024  # MB
        print(f"\n📊 Данные: historical_data.json ({size:.1f} MB)")
    else:
        print("\n📊 Данные: historical_data.json - не найдены")
        print("   🔧 Для сбора данных:")
        print("   python data_collector.py")

def main():
    """Основная функция"""
    print("🚀 ДЕМОНСТРАЦИЯ СОХРАНЕННОЙ МОДЕЛИ ПРЕДСКАЗАНИЯ СТРЕЛЯЮЩИХ МОНЕТ")
    print("=" * 70)
    
    # Проверяем статус
    check_model_status()
    
    print("\n" + "=" * 70)
    
    # Запускаем демонстрацию
    success = demo_saved_model()
    
    if success:
        print("\n🎯 МОДЕЛЬ ГОТОВА К ИСПОЛЬЗОВАНИЮ!")
        print("\n📋 Доступные команды:")
        print("   - python shooting_star_bot.py (запуск бота)")
        print("   - python demo_saved_model.py (эта демонстрация)")
        print("   - python train_shooting_star_model.py (переобучение)")
    else:
        print("\n⚠️ ТРЕБУЕТСЯ ОБУЧЕНИЕ МОДЕЛИ")
        print("\n🔧 Команды для обучения:")
        print("   - python train_shooting_star_model.py --quick (быстрый тест)")
        print("   - python train_shooting_star_model.py (полное обучение)")

if __name__ == "__main__":
    main()

