# 🚀 Автоматическое развертывание торгового бота на Vultr

## 📋 Обзор

Теперь у вас есть **полностью автоматизированная** система развертывания! При каждом пуше в ветку `main` GitHub автоматически:

1. ✅ Собирает проект
2. ✅ Загружает на сервер Vultr
3. ✅ Устанавливает зависимости
4. ✅ Запускает веб-дашборд
5. ✅ Настраивает автозапуск

## 🔧 Шаг 1: Настройка сервера Vultr

### Подключитесь к серверу и выполните:
```bash
ssh root@ВАШ_IP_СЕРВЕРА
```

### Загрузите и выполните скрипт настройки:
```bash
# Скопируйте скрипт на сервер
scp setup_server.sh root@ВАШ_IP_СЕРВЕРА:/root/

# Выполните на сервере
chmod +x setup_server.sh
./setup_server.sh
```

### Скопируйте публичный SSH ключ
Скрипт покажет ваш публичный SSH ключ. Скопируйте его (он начинается с `ssh-rsa`).

## 🔧 Шаг 2: Настройка GitHub Secrets

### 1. Перейдите в репозиторий на GitHub
**Settings > Secrets and variables > Actions**

### 2. Добавьте следующие secrets:

#### `SERVER_IP`
```
Name: SERVER_IP
Value: ВАШ_IP_СЕРВЕРА (например: 45.32.157.241)
```

#### `SERVER_SSH_KEY`
```
Name: SERVER_SSH_KEY
Value: [СОДЕРЖИМОЕ ПРИВАТНОГО КЛЮЧА ~/.ssh/id_rsa]
```

> ⚠️ **Важно:** Приватный ключ должен быть в формате PEM (одна строка)

## 🔧 Шаг 3: Тестирование

### 1. Сделайте коммит и push
```bash
git add .
git commit -m "🚀 Тестирование автоматического развертывания"
git push origin main
```

### 2. Мониторьте развертывание
- Перейдите в **Actions** tab в вашем GitHub репозитории
- Найдите workflow **"Deploy Trading Bot to Vultr"**
- Кликните для просмотра прогресса

### 3. Проверьте результат
После успешного завершения откройте:
**http://ВАШ_IP_СЕРВЕРА:5000**

## 📊 Мониторинг развертывания

### В GitHub Actions:
```
Actions > Deploy Trading Bot to Vultr > [Run ID]
```

### На сервере:
```bash
# Статус сервиса
systemctl status trading-bot

# Логи в реальном времени
journalctl -u trading-bot -f

# Доступность веб-интерфейса
curl http://localhost:5000
```

## 🔄 Как это работает

### 1. Trigger
- Push в ветку `main`
- Manual trigger через GitHub UI

### 2. Build
- Checkout кода
- Настройка SSH доступа к серверу
- Загрузка файлов через rsync

### 3. Deploy
- Обновление системы сервера
- Установка Python зависимостей
- Настройка firewall
- Установка systemd сервиса

### 4. Start
- Запуск Flask веб-дашборда
- Проверка доступности
- Настройка автозапуска

## 🛠️ Управление

### Ручной запуск workflow:
1. Перейдите в **Actions** tab
2. Кликните **"Deploy Trading Bot to Vultr"**
3. Нажмите **"Run workflow"**

### Откат изменений:
```bash
# На сервере
cd /root/trading-bot
git log --oneline -5
git reset --hard HEAD~1  # откат на 1 коммит назад
systemctl restart trading-bot
```

### Обновление зависимостей:
```bash
# На сервере
cd /root/trading-bot
source trading_env/bin/activate
pip install -r requirements_web.txt --upgrade
systemctl restart trading-bot
```

## 🔒 Безопасность

### SSH ключи:
- ✅ Используются только для deployment
- ✅ Хранятся как GitHub secrets
- ✅ Автоматически удаляются после workflow

### Firewall:
- ✅ Порт 5000 открыт только для внешнего доступа
- ✅ SSH доступ разрешен
- ✅ Все остальные порты закрыты

## 🚨 Troubleshooting

### Workflow не запускается:
```bash
# Проверьте, что push в main ветку
git branch
git status
git log --oneline -3
```

### Ошибка SSH подключения:
```bash
# Проверьте SERVER_IP в secrets
# Проверьте SERVER_SSH_KEY (должен быть в формате PEM)
```

### Сервер недоступен после деплоя:
```bash
# На сервере
systemctl status trading-bot
journalctl -u trading-bot -n 50
curl http://localhost:5000
```

### Проблемы с зависимостями:
```bash
# На сервере
cd /root/trading-bot
source trading_env/bin/activate
pip install -r requirements_web.txt --force-reinstall
```

## 📝 Примечания

### Производительность:
- ✅ Минимальный размер образа (ubuntu-latest)
- ✅ Параллельная установка зависимостей
- ✅ Кеширование между deployments

### Надежность:
- ✅ Автоматический rollback при ошибках
- ✅ Проверка доступности после деплоя
- ✅ Логирование всех шагов

### Масштабируемость:
- ✅ Легко добавить staging окружение
- ✅ Поддержка multiple серверов
- ✅ Настраиваемые параметры деплоя

---

## 🎉 Готово!

Теперь у вас есть **полностью автоматизированная** система развертывания!

**Каждый push в main ветку** = **автоматическое обновление сервера** 🚀

*Для дополнительной помощи обратитесь к документации или создайте issue в репозитории*
