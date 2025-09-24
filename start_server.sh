#!/bin/bash

# 🚀 Простой запуск веб-дашборда на сервере
# Выполните после загрузки файлов на сервер

echo "🚀 Запуск веб-дашборда торгового бота..."

# Проверка наличия файлов
if [ ! -f "web_trading_dashboard.py" ]; then
    echo "❌ Файл web_trading_dashboard.py не найден!"
    echo "📤 Сначала загрузите файлы: scp -r /путь/к/файлам user@your-vultr-ip:/root/trading-bot/"
    exit 1
fi

if [ ! -f "requirements_web.txt" ]; then
    echo "❌ Файл requirements_web.txt не найден!"
    exit 1
fi

# Создание виртуального окружения
echo "🌐 Создание виртуального окружения..."
python3 -m venv trading_env
source trading_env/bin/activate

# Установка зависимостей
echo "📦 Установка зависимостей..."
pip install -r requirements_web.txt

# Настройка конфигурации (опционально)
if [ -f "bot_config.json.example" ]; then
    echo "⚙️ Настройка конфигурации..."
    cp bot_config.json.example bot_config.json
    echo "📝 Отредактируйте bot_config.json для настройки API ключей"
fi

# Запуск сервера
echo "🚀 Запуск Flask сервера..."
echo "🌐 Доступно на: http://$(hostname -I | awk '{print $1}'):5000"
python web_trading_dashboard.py --host 0.0.0.0 --port 5000
