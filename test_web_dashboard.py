#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧪 Тестирование веб-дашборда торгового бота
Скрипт для проверки работы веб-интерфейса перед развертыванием
"""

import subprocess
import sys
import os
import time
import requests
import json
from pathlib import Path

def print_header():
    print("🌐 ТЕСТИРОВАНИЕ ВЕБ-ДАШБОРДА ТОРГОВОГО БОТА")
    print("=" * 50)

def check_requirements():
    """Проверка зависимостей"""
    print("📦 Проверка зависимостей...")

    try:
        import flask
        print("✅ Flask установлен")
    except ImportError:
        print("❌ Flask не установлен")
        return False

    try:
        import ccxt
        print("✅ CCXT установлен")
    except ImportError:
        print("❌ CCXT не установлен")
        return False

    try:
        import plotly
        print("✅ Plotly установлен")
    except ImportError:
        print("❌ Plotly не установлен")
        return False

    try:
        import psutil
        print("✅ psutil установлен")
    except ImportError:
        print("❌ psutil не установлен")
        return False

    return True

def check_config():
    """Проверка конфигурации"""
    print("⚙️ Проверка конфигурации...")

    config_file = Path("bot_config.json")
    if not config_file.exists():
        print("❌ Файл bot_config.json не найден")
        return False

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # Проверка API ключей
        api_key = config.get('binance_api', {}).get('api_key', '')
        if not api_key or api_key == 'YOUR_BINANCE_API_KEY':
            print("⚠️ API ключ Binance не настроен")

        # Проверка настроек веб-дашборда
        web_config = config.get('web_dashboard', {})
        if not web_config:
            print("⚠️ Настройки веб-дашборда не найдены")
        else:
            print("✅ Настройки веб-дашборда найдены")

        return True

    except Exception as e:
        print(f"❌ Ошибка чтения конфигурации: {e}")
        return False

def test_api_endpoints():
    """Тестирование API endpoints"""
    print("🔗 Тестирование API endpoints...")

    # Импорт здесь, чтобы избежать ошибок импорта
    from web_trading_dashboard import WebTradingDashboard

    try:
        dashboard = WebTradingDashboard()

        # Тестирование создания приложения
        app = dashboard.app
        print("✅ Flask приложение создано")

        # Тестирование маршрутов
        with app.test_client() as client:
            # Тест главной страницы
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Главная страница доступна")
            else:
                print(f"❌ Главная страница недоступна: {response.status_code}")

            # Тест API статуса бота
            response = client.get('/api/bot/status')
            if response.status_code == 200:
                print("✅ API статуса бота работает")
            else:
                print(f"❌ API статуса бота не работает: {response.status_code}")

            # Тест API данных рынка
            response = client.get('/api/market/data?symbol=BTC/USDT&limit=10')
            if response.status_code == 200:
                print("✅ API данных рынка работает")
            else:
                print(f"❌ API данных рынка не работает: {response.status_code}")

        return True

    except Exception as e:
        print(f"❌ Ошибка тестирования API: {e}")
        return False

def start_test_server():
    """Запуск тестового сервера"""
    print("🚀 Запуск тестового сервера...")

    try:
        # Запуск в отдельном процессе
        process = subprocess.Popen([
            sys.executable, 'web_trading_dashboard.py',
            '--port', '5001',
            '--debug'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("✅ Тестовый сервер запущен на порту 5001")
        return process

    except Exception as e:
        print(f"❌ Ошибка запуска сервера: {e}")
        return None

def test_server_connection():
    """Тестирование подключения к серверу"""
    print("🌐 Тестирование подключения к серверу...")

    max_attempts = 10
    for attempt in range(max_attempts):
        try:
            response = requests.get('http://localhost:5001', timeout=5)
            if response.status_code == 200:
                print("✅ Сервер отвечает")
                return True
            else:
                print(f"⚠️ Сервер отвечает с кодом {response.status_code}")
                return False

        except requests.exceptions.ConnectionError:
            print(f"⏳ Попытка подключения {attempt + 1}/{max_attempts}...")
            time.sleep(2)

    print("❌ Не удалось подключиться к серверу")
    return False

def check_file_structure():
    """Проверка структуры файлов"""
    print("📁 Проверка структуры файлов...")

    required_files = [
        'web_trading_dashboard.py',
        'bot_config.json',
        'requirements_web.txt',
        'deploy_vultr.sh',
        'WEB_DASHBOARD_README.md'
    ]

    required_dirs = [
        'templates',
        'static'
    ]

    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Отсутствуют файлы: {', '.join(missing_files)}")
        return False

    for dir_name in required_dirs:
        if not Path(dir_name).exists():
            print(f"⚠️ Директория {dir_name} не найдена")
        else:
            print(f"✅ Директория {dir_name} найдена")

    print("✅ Структура файлов корректна")
    return True

def run_diagnostics():
    """Запуск диагностики"""
    print("🔍 Диагностика системы...")

    # Проверка Python версии
    python_version = sys.version.split()[0]
    print(f"🐍 Python версия: {python_version}")

    # Проверка свободного места
    try:
        import shutil
        total, used, free = shutil.disk_usage("/")
        free_gb = free // (1024**3)
        print(f"💾 Свободное место: {free_gb} GB")
    except:
        pass

    # Проверка памяти
    try:
        import psutil
        memory = psutil.virtual_memory()
        memory_gb = memory.total // (1024**3)
        print(f"🖥️ Оперативная память: {memory_gb} GB")
    except:
        pass

def main():
    """Основная функция тестирования"""
    print_header()

    print("\n🔍 ЗАПУСК ТЕСТИРОВАНИЯ\n")

    # Проверка требований
    if not check_requirements():
        print("\n❌ Тестирование прервано: не выполнены требования")
        return False

    # Проверка конфигурации
    if not check_config():
        print("\n⚠️ Проблемы с конфигурацией, но продолжаем тестирование")

    # Проверка структуры файлов
    if not check_file_structure():
        print("\n❌ Тестирование прервано: проблемы со структурой файлов")
        return False

    # Диагностика системы
    run_diagnostics()

    # Тестирование API
    if not test_api_endpoints():
        print("\n⚠️ Проблемы с API, но продолжаем тестирование")

    print("\n🚀 ПУСК ТЕСТОВОГО СЕРВЕРА\n")

    # Запуск тестового сервера
    server_process = start_test_server()

    if not server_process:
        print("\n❌ Не удалось запустить тестовый сервер")
        return False

    # Тестирование подключения
    if not test_server_connection():
        print("\n❌ Не удалось подключиться к тестовому серверу")
        server_process.terminate()
        return False

    print("\n✅ ТЕСТИРОВАНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print("\n📋 СЛЕДУЮЩИЕ ШАГИ:")
    print("1. Откройте браузер")
    print("2. Перейдите на http://localhost:5001")
    print("3. Проверьте работу интерфейса")
    print("4. Для остановки сервера: Ctrl+C")

    try:
        # Ожидание завершения
        server_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Остановка тестового сервера...")
        server_process.terminate()
        server_process.wait()
        print("✅ Сервер остановлен")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
