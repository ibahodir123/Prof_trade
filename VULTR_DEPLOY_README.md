# 🚀 Развертывание веб-дашборда на сервер Vultr

## 📋 Быстрый старт

### Вариант 1: Простое развертывание (рекомендуется)

#### Шаг 1: Подключение к серверу
```bash
ssh root@ВАШ_IP_СЕРВЕРА
```

#### Шаг 2: Загрузка файлов на сервер
```bash
# На вашем локальном компьютере выполните:
scp -r D:\Best_trade\* root@ВАШ_IP_СЕРВЕРА:/root/trading-bot/
```

#### Шаг 3: Запуск на сервере
```bash
cd /root/trading-bot
chmod +x start_server.sh
./start_server.sh
```

### Вариант 2: Полное развертывание с автозапуском

#### Шаг 1: Выполните на сервере
```bash
# Подключитесь к серверу
ssh root@ВАШ_IP_СЕРВЕРА

# Обновление системы
apt update && apt upgrade -y

# Установка необходимых пакетов
apt install python3 python3-pip python3-venv curl wget git -y

# Создание директории проекта
mkdir -p /root/trading-bot
cd /root/trading-bot
```

#### Шаг 2: Загрузка файлов
```bash
# На локальном компьютере:
scp -r D:\Best_trade\* root@ВАШ_IP_СЕРВЕРА:/root/trading-bot/
```

#### Шаг 3: Установка зависимостей и запуск
```bash
# На сервере:
cd /root/trading-bot

# Создание виртуального окружения
python3 -m venv trading_env
source trading_env/bin/activate

# Установка зависимостей
pip install -r requirements_web.txt

# Запуск сервера
python web_trading_dashboard.py --host 0.0.0.0 --port 5000
```

## ⚙️ Настройка автозапуска (systemd)

### Шаг 1: Копирование сервис файла
```bash
# На сервере:
cp /root/trading-bot/trading-bot.service /etc/systemd/system/
```

### Шаг 2: Настройка firewall
```bash
# Разрешаем доступ к порту 5000
ufw allow 5000
ufw allow ssh
ufw --force enable
```

### Шаг 3: Запуск сервиса
```bash
# Перезагрузка systemd
systemctl daemon-reload

# Запуск сервиса
systemctl start trading-bot

# Добавление в автозапуск
systemctl enable trading-bot

# Проверка статуса
systemctl status trading-bot
```

## 🌐 Доступ к веб-дашборду

После запуска откройте браузер и перейдите:

**http://ВАШ_IP_СЕРВЕРА:5000**

## 🔧 Управление сервисом

```bash
# Статус сервиса
systemctl status trading-bot

# Остановка
systemctl stop trading-bot

# Запуск
systemctl start trading-bot

# Перезапуск
systemctl restart trading-bot

# Просмотр логов
journalctl -u trading-bot -f
```

## 📝 Настройка конфигурации

### API ключи Binance
Отредактируйте файл `/root/trading-bot/bot_config.json`:

```json
{
  "binance_api": {
    "api_key": "ВАШ_API_KEY",
    "secret_key": "ВАШ_SECRET_KEY"
  }
}
```

### Сетевые настройки
```json
{
  "web_dashboard": {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": false
  }
}
```

## 🛠️ Устранение проблем

### Порт уже используется
```bash
# Проверьте, что использует порт 5000
netstat -tlnp | grep :5000

# Остановите процесс
kill -9 PID_ПРОЦЕССА

# Или измените порт в web_trading_dashboard.py
```

### Проблемы с firewall
```bash
# Проверить статус firewall
ufw status

# Разрешить порт
ufw allow 5000

# Перезапустить firewall
ufw reload
```

### Проблемы с зависимостями
```bash
# Переустановка в виртуальном окружении
source trading_env/bin/activate
pip install --upgrade pip
pip install -r requirements_web.txt
```

## 📊 Мониторинг

### Системные ресурсы
```bash
# CPU и память
htop

# Дисковое пространство
df -h

# Сетевые соединения
netstat -tlnp
```

### Логи приложения
```bash
# В реальном времени
journalctl -u trading-bot -f

# Последние 100 строк
journalctl -u trading-bot -n 100
```

## 🔒 Безопасность

### Рекомендации
1. **Используйте HTTPS** для продакшена (nginx + certbot)
2. **Настройте firewall** (ufw)
3. **Создайте отдельного пользователя** для запуска сервиса
4. **Регулярно обновляйте** систему и зависимости

### Пример настройки nginx (опционально)
```bash
# Установка nginx
apt install nginx -y

# Настройка reverse proxy для HTTPS
# (конфигурация в /etc/nginx/sites-available/trading-bot)
```

---

**🎉 Удачи! Ваш веб-дашборд торгового бота готов к работе на сервере Vultr!**

*Для дополнительной помощи обратитесь к документации или создайте issue в репозитории*
