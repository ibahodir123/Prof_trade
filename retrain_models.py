#!/usr/bin/env python3
import sys
sys.path.append('.')
from advanced_ml_trainer import AdvancedMLTrainer

# Создаем тренер
trainer = AdvancedMLTrainer()

# Получаем популярные символы
popular_symbols = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LTC/USDT',
    'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'FIL/USDT'
]

print("🔄 Переобучаю ML модели...")
try:
    success = trainer.train_models(popular_symbols)
    if success:
        print("✅ Модели успешно переобучены!")
    else:
        print("❌ Ошибка переобучения")
except Exception as e:
    print(f"❌ Ошибка: {e}")
