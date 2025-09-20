#!/usr/bin/env python3
"""
Скрипт для переобучения ML моделей на реальных данных
Сохраняет модели с метаданными для последующего использования
"""

import logging
import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from advanced_ml_trainer import AdvancedMLTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('retrain_models.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def load_symbols_from_file():
    """Загрузить список символов из файла"""
    try:
        # Сначала пробуем загрузить из top_coins_list.txt
        if os.path.exists('top_coins_list.txt'):
            with open('top_coins_list.txt', 'r', encoding='utf-8', errors='ignore') as f:
                symbols = []
                for line in f:
                    line = line.strip()
                    # Пропускаем комментарии и пустые строки
                    if line and not line.startswith('#'):
                        symbols.append(line)
                logger.info(f"📁 Загружено {len(symbols)} символов из top_coins_list.txt")
                return symbols
        
        # Если файла нет, используем базовый список
        logger.warning("⚠️ Файл top_coins_list.txt не найден, используем базовый список")
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'LTC/USDT',
            'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'FIL/USDT'
        ]
        
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки символов: {e}")
        return None

def main():
    """Основная функция переобучения моделей"""
    try:
        logger.info("🚀 Начинаю переобучение ML моделей на реальных данных...")
        
        # Загружаем список символов из файла
        symbols = load_symbols_from_file()
        if not symbols:
            logger.error("❌ Не удалось загрузить список символов")
            return False
        
        logger.info(f"📊 Будет использовано {len(symbols)} символов для обучения")
        
        # Создаем тренер
        trainer = AdvancedMLTrainer()
        
        # Проверяем существующие модели
        if trainer.load_models():
            logger.info("📁 Найдены существующие модели")
            if trainer.is_model_trained_on_real_data():
                logger.info("✅ Существующие модели уже обучены на реальных данных")
            else:
                logger.warning("⚠️ Существующие модели НЕ обучены на реальных данных")
        
        # Переобучаем модели
        logger.info("🧠 Начинаю переобучение...")
        success = trainer.train_models(symbols)
        
        if success:
            logger.info("✅ Модели успешно переобучены на реальных данных!")
            logger.info("📁 Модели сохранены с метаданными в папке models/")
            
            # Показываем информацию о метаданных
            metadata = trainer.load_training_metadata()
            if metadata:
                logger.info("📊 Информация о переобучении:")
                logger.info(f"   Дата: {metadata.get('training_date')}")
                logger.info(f"   Символы: {', '.join(metadata.get('symbols_used', []))}")
                logger.info(f"   Образцов: {metadata.get('samples_count')}")
                logger.info(f"   Качество входа: {metadata.get('entry_model_score', 0):.3f}")
                logger.info(f"   Качество выхода: {metadata.get('exit_model_score', 0):.3f}")
                logger.info(f"   Источник данных: {metadata.get('data_source')}")
        else:
            logger.error("❌ Ошибка переобучения моделей")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Критическая ошибка: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Переобучение завершено успешно!")
        print("📁 Модели сохранены в папке models/ с метаданными")
        print("🤖 Теперь можно запускать ботов с новыми моделями")
    else:
        print("\n❌ Переобучение завершилось с ошибками")
        print("📋 Проверьте логи в файле retrain_models.log")
        sys.exit(1)