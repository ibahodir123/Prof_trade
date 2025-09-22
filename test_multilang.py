#!/usr/bin/env python3
"""
Тестовый скрипт для проверки многоязычности бота
"""
import sys
import os

# Добавляем текущую директорию в путь для импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_bot_binance import translate_text, bot_state

def test_multilanguage():
    """Тестирование многоязычности"""
    print("🌍 Тестирование многоязычности бота")
    print("=" * 60)
    
    # Тестовые тексты
    test_texts = [
        "📈 Сигнал для BTC/USDT",
        "🟢 LONG - Сильный сигнал",
        "⚪ ОЖИДАНИЕ - Слабый сигнал",
        "💰 Цена входа: $50000.00",
        "🎯 Take Profit: $52000.00",
        "🛡️ Stop Loss: $48000.00",
        "📊 RSI: 65.5",
        "🤖 ML статус: Активна",
        "📊 Статус системы",
        "🪙 Выбор монет",
        "📈 Последние сигналы",
        "🔍 Анализ монеты",
        "🔙 Назад"
    ]
    
    languages = [
        ("ru", "🇷🇺 Русский"),
        ("uz", "🇺🇿 O'zbekcha"),
        ("en", "🇬🇧 English")
    ]
    
    for lang_code, lang_name in languages:
        print(f"\n{lang_name}:")
        print("-" * 30)
        
        for text in test_texts:
            translated = translate_text(text, lang_code)
            print(f"📝 {text}")
            print(f"   → {translated}")
            print()
    
    print("✅ Тестирование многоязычности завершено!")
    print("\n🎯 Доступные языки:")
    print("   🇷🇺 Русский - основной язык")
    print("   🇺🇿 O'zbekcha - для узбекских пользователей")
    print("   🇬🇧 English - для международных пользователей")

if __name__ == "__main__":
    test_multilanguage()






