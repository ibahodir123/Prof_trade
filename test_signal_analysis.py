#!/usr/bin/env python3
"""
Тест анализа сигналов для одной монеты
"""

import logging
import asyncio
import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from auto_signals_bot import AutoSignalsBot

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_signal_analysis():
    """Тестирование анализа сигналов"""
    try:
        logger.info("🧪 Тестирую анализ сигналов...")
        
        # Создаем бота
        bot = AutoSignalsBot()
        
        # Инициализируем компоненты
        if not bot.load_models():
            logger.error("❌ Не удалось загрузить модели")
            return False
        
        # Инициализируем Binance
        bot.initialize_binance()
        
        # Тестируем анализ одной монеты
        test_symbol = 'BTC/USDT'
        logger.info(f"📊 Анализирую {test_symbol}...")
        
        # Получаем данные
        try:
            ohlcv = bot.binance.fetch_ohlcv(test_symbol, '1h', limit=100)
            logger.info(f"✅ Получены данные для {test_symbol}: {len(ohlcv)} свечей")
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных для {test_symbol}: {e}")
            return False
        
        # Анализируем сигнал
        try:
            signal = await bot.analyze_coin_signal(test_symbol, ohlcv)
            if signal:
                logger.info(f"🎯 Результат анализа {test_symbol}:")
                logger.info(f"   Тип сигнала: {signal.get('signal_type', 'НЕТ')}")
                logger.info(f"   Вероятность: {signal.get('probability', 0):.1f}%")
                logger.info(f"   Цена входа: ${signal.get('entry_price', 0):.8f}")
                logger.info(f"   ML вход: {signal.get('ml_entry_prob', 0):.3f}")
                logger.info(f"   ML выход: {signal.get('ml_exit_prob', 0):.3f}")
                
                if signal.get('signal_type') != '⚪ ОЖИДАНИЕ':
                    logger.info("✅ Сигнал сгенерирован успешно!")
                    return True
                else:
                    logger.info("⚪️ Сигнал ОЖИДАНИЕ (нормально)")
                    return True
            else:
                logger.error("❌ Сигнал не сгенерирован")
                return False
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа сигнала: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка тестирования: {e}")
        return False

async def main():
    """Основная функция"""
    print("🧪 Тестирование анализа сигналов")
    print("=" * 50)
    
    success = await test_signal_analysis()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Тест пройден успешно!")
        print("🤖 Анализ сигналов работает корректно")
        print("📊 ML модели генерируют предсказания")
    else:
        print("❌ Тест провален!")
        print("🔧 Требуется дополнительная отладка")

if __name__ == "__main__":
    asyncio.run(main())
