#!/bin/bash

# 🚀 Настройка сервера Vultr для автоматического развертывания
# Выполните на вашем сервере Vultr

echo "🚀 Настройка сервера для автоматического развертывания..."

# 1. Обновление системы
echo "📦 Обновление системы..."
apt update && apt upgrade -y

# 2. Установка необходимых пакетов
echo "📥 Установка пакетов..."
apt install -y python3 python3-pip python3-venv curl wget git ufw rsync

# 3. Настройка firewall
echo "🔒 Настройка firewall..."
ufw allow 5000
ufw allow ssh
ufw --force enable

# 4. Создание SSH ключа для GitHub
echo "🔑 Генерация SSH ключа..."
ssh-keygen -t rsa -b 4096 -C "trading-bot-deploy" -f ~/.ssh/id_rsa -N ""

echo "📋 Ваш публичный SSH ключ (добавьте в GitHub):"
echo "================================================================="
cat ~/.ssh/id_rsa.pub
echo "================================================================="

# 5. Создание директории проекта
echo "📁 Создание директории проекта..."
mkdir -p /root/trading-bot
cd /root/trading-bot

# 6. Настройка Git для автоматического развертывания
echo "🔧 Настройка Git..."
git config --global user.name "GitHub Actions"
git config --global user.email "action@github.com"

# 7. Клонирование репозитория (замените на ваш)
echo "🔗 Клонирование репозитория..."
git clone https://github.com/ВАШ_USERNAME/ВАШ_REPO.git .
# Или загрузите файлы вручную: scp -r /путь/к/файлам root@ВАШ_IP:/root/trading-bot/

# 8. Создание виртуального окружения
echo "🌐 Создание виртуального окружения..."
python3 -m venv trading_env

# 9. Установка зависимостей
echo "📦 Установка зависимостей..."
source trading_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_web.txt

# 10. Установка systemd сервиса
echo "⚙️ Установка systemd сервиса..."
cp trading-bot.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable trading-bot

echo "✅ Настройка сервера завершена!"
echo ""
echo "🌐 Следующие шаги:"
echo "1. Добавьте SSH ключ выше в GitHub (Settings > SSH Keys)"
echo "2. Настройте secrets в GitHub репозитории:"
echo "   - SERVER_IP: ВАШ_IP_СЕРВЕРА"
echo "   - SERVER_SSH_KEY: содержимое ~/.ssh/id_rsa"
echo "3. Сделайте push в main ветку для автоматического развертывания"
echo ""
echo "🔍 Проверка: ssh root@ВАШ_IP_СЕРВЕРА"
