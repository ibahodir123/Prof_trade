#!/bin/bash

# 🚀 Скрипт развертывания торгового бота на сервер Vultr
# Выполните эти команды на вашем сервере Vultr

echo "🚀 Начинаем развертывание торгового бота на Vultr..."

# 1. Обновление системы
echo "📦 Обновление системы..."
sudo apt update && sudo apt upgrade -y

# 2. Установка Python и pip
echo "🐍 Установка Python 3..."
sudo apt install python3 python3-pip python3-venv -y

# 3. Установка Git
echo "📥 Установка Git..."
sudo apt install git -y

# 4. Создание директории проекта
echo "📁 Создание директории проекта..."
mkdir -p /root/trading-bot
cd /root/trading-bot

# 4b. Загрузка файлов (выполните сначала: scp -r /путь/к/файлам user@your-vultr-ip:/root/trading-bot/)

# 5. Создание виртуального окружения
echo "🌐 Создание виртуального окружения..."
python3 -m venv trading_env
source trading_env/bin/activate

# 6. Установка зависимостей
echo "📦 Установка зависимостей..."
pip install -r requirements.txt
pip install -r requirements_web.txt

# 7. Настройка конфигурации
echo "⚙️ Настройка конфигурации..."
cp bot_config.json.example bot_config.json
nano bot_config.json  # Отредактируйте настройки API ключей

# 8. Создание systemd сервиса
echo "🔧 Создание сервиса для автозапуска..."
sudo tee /etc/systemd/system/trading-bot.service > /dev/null <<EOF
[Unit]
Description=Trading Bot Web Dashboard
After=network.target

[Service]
User=root
WorkingDirectory=/root/trading-bot
Environment=PATH=/root/trading-bot/trading_env/bin
ExecStart=/root/trading-bot/trading_env/bin/python web_trading_dashboard.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# 9. Настройка firewall
echo "🔒 Настройка firewall..."
sudo ufw allow 5000
sudo ufw allow ssh
sudo ufw --force enable

# 10. Запуск сервиса
echo "🚀 Запуск торгового бота..."
sudo systemctl daemon-reload
sudo systemctl start trading-bot
sudo systemctl enable trading-bot

# 11. Проверка статуса
echo "📊 Проверка статуса..."
sudo systemctl status trading-bot --no-pager

echo "✅ Развертывание завершено!"
echo "🌐 Ваш торговый бот доступен по адресу: http://ВАШ_IP_СЕРВЕРА:5000"
echo "🔑 Для остановки: sudo systemctl stop trading-bot"
echo "🔄 Для перезапуска: sudo systemctl restart trading-bot"
echo "📋 Логи: sudo journalctl -u trading-bot -f"
