#!/usr/bin/env python3
"""
Тест кнопки Контакты в главном меню
"""

import sys
import os

# Добавляем текущую директорию в путь
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_contacts_integration():
    """Тестируем интеграцию кнопки Контакты"""
    
    print("🧪 Тест кнопки Контакты")
    print("=" * 50)
    
    # Проверяем, что функция handle_contacts_menu существует
    try:
        from ml_bot_binance import handle_contacts_menu
        print("✅ Функция handle_contacts_menu импортирована успешно")
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        return False
    
    # Проверяем, что функция back_to_main_menu содержит кнопку Контакты
    try:
        from ml_bot_binance import back_to_main_menu
        import inspect
        
        # Читаем исходный код функции
        source = inspect.getsource(back_to_main_menu)
        
        if "📞 Контакты" in source:
            print("✅ Кнопка '📞 Контакты' найдена в back_to_main_menu")
        else:
            print("❌ Кнопка '📞 Контакты' НЕ найдена в back_to_main_menu")
            return False
            
        if "menu_contacts" in source:
            print("✅ callback_data 'menu_contacts' найден")
        else:
            print("❌ callback_data 'menu_contacts' НЕ найден")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка проверки back_to_main_menu: {e}")
        return False
    
    # Проверяем, что в button_callback есть обработчик
    try:
        from ml_bot_binance import button_callback
        source = inspect.getsource(button_callback)
        
        if "menu_contacts" in source and "handle_contacts_menu" in source:
            print("✅ Обработчик menu_contacts найден в button_callback")
        else:
            print("❌ Обработчик menu_contacts НЕ найден в button_callback")
            return False
            
    except Exception as e:
        print(f"❌ Ошибка проверки button_callback: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ Все тесты пройдены! Кнопка Контакты готова к использованию!")
    print("\n📱 Теперь пользователи смогут:")
    print("• Нажать кнопку '📞 Контакты' в главном меню")
    print("• Увидеть ваши контактные данные")
    print("• Связаться с вами для вопросов и сотрудничества")
    
    return True

if __name__ == "__main__":
    success = test_contacts_integration()
    if not success:
        sys.exit(1)
