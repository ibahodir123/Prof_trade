#!/usr/bin/env python3
"""
Тестовый скрипт для проверки узбекских переводов
"""
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_bot_binance import translate_to_uzbek, analyze_coin_signal_advanced_ema, bot_state

def test_uzbek_translations():
    """Тестирование узбекских переводов"""
    print("🇺🇿 Тестирование узбекских переводов")
    print("=" * 50)
    
    # Устанавливаем узбекский язык
    bot_state.language = "uz"
    
    # Тестовые тексты
    test_texts = [
        "📈 Сигнал для BTC/USDT",
        "🟢 LONG - Сильный",
        "⚪ ОЖИДАНИЕ - Слабый",
        "💰 Цена входа: $50000.00",
        "🎯 Take Profit: $52000.00",
        "🛡️ Stop Loss: $48000.00",
        "📊 RSI: 65.5",
        "🤖 ML статус: Активна",
        "💡 Что означает ОЖИДАНИЕ:",
        "❌ НЕ входить в позицию сейчас",
        "⏳ Ждать лучшего момента для входа",
        "📊 Мониторить цену и технические показатели",
        "🎯 Дождаться более благоприятных условий",
        "🔙 Назад",
        "Анализ монеты",
        "Поиск монет",
        "Статус системы"
    ]
    
    print("Русский → Узбекский:")
    print("-" * 30)
    
    for text in test_texts:
        translated = translate_to_uzbek(text, "uz")
        print(f"🇷🇺 {text}")
        print(f"🇺🇿 {translated}")
        print()
    
    print("✅ Тестирование переводов завершено!")

def test_coin_analysis():
    """Тестирование анализа монеты"""
    print("\n📊 Тестирование анализа монеты")
    print("=" * 50)
    
    # Устанавливаем русский язык для анализа
    bot_state.language = "ru"
    
    # Тестируем анализ BTC
    print("🔍 Анализирую BTC/USDT...")
    try:
        signal_data = analyze_coin_signal_advanced_ema("BTC/USDT")
        
        if signal_data:
            print(f"✅ Анализ завершен!")
            print(f"📊 Символ: {signal_data['symbol']}")
            print(f"🎯 Сигнал: {signal_data['signal_type']}")
            print(f"💪 Сила: {signal_data['strength_text']}")
            print(f"💰 Цена: ${signal_data['entry_price']:.8f}")
            
            # Теперь переводим результат на узбекский
            bot_state.language = "uz"
            
            message = f"""
📈 **Сигнал для {signal_data['symbol']}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            if "LONG" in signal_data['signal_type']:
                message += f"""
🎯 **Take Profit:** ${signal_data['take_profit']:.8f}
🛡️ **Stop Loss:** ${signal_data['stop_loss']:.8f}
                """
            elif "ОЖИДАНИЕ" in signal_data['signal_type']:
                message += f"""

💡 **Что означает ОЖИДАНИЕ:**
• ❌ **НЕ входить** в позицию сейчас
• ⏳ **Ждать** лучшего момента для входа
• 📊 **Мониторить** цену и технические показатели
• 🎯 **Дождаться** более благоприятных условий
                """
            
            translated_message = translate_to_uzbek(message)
            
            print("\n🇺🇿 УЗБЕКСКИЙ ПЕРЕВОД:")
            print("-" * 30)
            print(translated_message)
        else:
            print("❌ Ошибка анализа")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    print("🤖 Тестирование узбекских переводов в боте")
    print("=" * 60)
    
    # Тестируем переводы
    test_uzbek_translations()
    
    # Тестируем анализ монеты
    test_coin_analysis()
    
    print("\n🎉 Все тесты завершены!")
    print("💡 Для полного тестирования с Telegram удалите bot_config_local.json")






