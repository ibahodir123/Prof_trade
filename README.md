# 🚀 AI Trading Bot - Binance ML Bot

> **Профессиональный торговый бот с машинным обучением для предсказания стреляющих монет на Binance**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub Stars](https://img.shields.io/github/stars/username/trading-bot.svg)](https://github.com/username/trading-bot)

## 🌟 Особенности

### 🤖 **Два режима работы:**
1. **Основной торговый бот** - анализ монет с ML моделями
2. **Бот стреляющих монет** - нейронная сеть для предсказания взрывного роста

### 🧠 **Машинное обучение:**
- **LSTM нейронная сеть** для предсказания трендов
- **36 технических индикаторов** (EMA, RSI, MACD, Bollinger Bands, etc.)
- **Автоматическое обучение** на исторических данных
- **Сохранение/загрузка моделей** - не нужно переобучать каждый раз

### 📊 **Анализ данных:**
- **Сбор данных** с Binance API с 1 января 2025
- **Автоматическое определение** стреляющих моментов
- **5 категорий роста**: падение → взрывной рост
- **Реальное время** анализа рынка

### 🤖 **Telegram интеграция:**
- **Автосигналы** каждые 30 минут
- **Интерактивное меню** с кнопками
- **Графики и диаграммы** для визуализации
- **Уведомления** о сильных сигналах

## 🚀 Быстрый старт

### 📋 Требования
- Python 3.9+
- Telegram Bot Token
- Binance API ключи (опционально)

### ⚡ Установка

1. **Клонируйте репозиторий:**
```bash
git clone https://github.com/username/trading-bot.git
cd trading-bot
```

2. **Установите зависимости:**
```bash
pip install -r requirements.txt
```

3. **Настройте конфигурацию:**
```bash
cp bot_config.example.json bot_config.json
# Отредактируйте bot_config.json с вашими токенами
```

4. **Запустите бота:**
```bash
python ml_bot_binance.py
```

### 🌐 GitHub Codespaces

**Рекомендуемый способ** - используйте GitHub Codespaces:

1. Откройте репозиторий в браузере
2. Нажмите кнопку **"Code"** → **"Codespaces"** → **"Create codespace"**
3. Дождитесь загрузки облачной среды
4. Все зависимости уже установлены!

## 📁 Структура проекта

```
trading-bot/
├── 🤖 ml_bot_binance.py          # Основной торговый бот
├── 🧠 shooting_star_bot.py       # Бот для стреляющих монет
├── 📊 data_collector.py          # Сбор исторических данных
├── 🔮 neural_network_predictor.py # LSTM нейронная сеть
├── 🎯 simple_train.py            # Быстрое обучение модели
├── 🧪 simple_demo.py             # Демонстрация модели
├── ⚙️ bot_config.json            # Конфигурация бота
├── 📋 requirements.txt           # Python зависимости
├── 📖 README.md                  # Документация
└── .github/
    └── workflows/
        └── auto_train.yml        # Автоматическое обучение
```

## 🎯 Использование

### 🤖 **Основной бот (ml_bot_binance.py)**

```bash
python ml_bot_binance.py
```

**Команды в Telegram:**
- `/start` - главное меню
- `/analyze` - анализ выбранной монеты
- `/auto_start` - запуск автосигналов
- `/auto_stop` - остановка автосигналов
- `/set_coin BTC` - выбор монеты

### 🧠 **Бот стреляющих монет (shooting_star_bot.py)**

```bash
python shooting_star_bot.py
```

**Возможности:**
- Предсказание взрывного роста
- Анализ на основе нейронной сети
- Уведомления о потенциальных стреляющих монетах

### 🎓 **Обучение модели**

```bash
# Быстрое обучение на синтетических данных
python simple_train.py

# Полное обучение на реальных данных
python train_shooting_star_model.py --quick

# Демонстрация сохраненной модели
python simple_demo.py
```

## ⚙️ Конфигурация

### 📝 bot_config.json

```json
{
  "telegram_token": "YOUR_BOT_TOKEN",
  "chat_id": "YOUR_CHAT_ID",
  "binance_api": {
    "enabled": false,
    "api_key": "",
    "secret_key": ""
  },
  "ml_settings": {
    "sequence_length": 12,
    "prediction_horizon": 12,
    "confidence_threshold": 0.7
  }
}
```

### 🔑 Получение токенов

1. **Telegram Bot Token:**
   - Напишите @BotFather в Telegram
   - Создайте нового бота: `/newbot`
   - Скопируйте полученный токен

2. **Binance API (опционально):**
   - Зайдите на binance.com
   - API Management → Create API
   - Скопируйте API Key и Secret Key

## 🧠 Машинное обучение

### 📊 **Технические индикаторы (36 признаков):**

- **Ценовые**: EMA, RSI, MACD, Bollinger Bands
- **Объемные**: Volume MA, Volume Ratio
- **Волатильность**: ATR, Price Volatility
- **Моментум**: Price Momentum, Volume Momentum
- **Трендовые**: Trend Strength, Direction

### 🎯 **Категории предсказаний:**

- **0** - Падение/боковик
- **1** - Небольшой рост (0-5%)
- **2** - Умеренный рост (5-10%)
- **3** - Высокий рост (10-20%)
- **4** - Взрывной рост (20%+)

### 💾 **Сохранение моделей:**

Модели автоматически сохраняются в файлы:
- `simple_shooting_star_model.h5` - нейронная сеть
- `simple_shooting_star_scaler.pkl` - нормализатор
- `simple_shooting_star_metadata.json` - метаданные

## 🔄 GitHub Actions

Автоматическое обучение модели каждую неделю:

```yaml
# .github/workflows/auto_train.yml
name: Auto Train Model
on:
  schedule:
    - cron: '0 2 * * 0'  # Каждое воскресенье в 2:00 UTC
  workflow_dispatch:     # Ручной запуск

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train Model
        run: python simple_train.py
      - name: Commit Model
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add *.h5 *.pkl *.json
          git commit -m "Auto-update model" || exit 0
          git push
```

## 📊 Мониторинг

### 📈 **Логи:**
- `training.log` - обучение модели
- `data_collection.log` - сбор данных
- Консольные логи в реальном времени

### 📱 **Telegram уведомления:**
- Статус обучения
- Результаты анализа
- Ошибки и предупреждения

## 🛠️ Разработка

### 🧪 **Тестирование:**

```bash
# Тест демонстрации модели
python simple_demo.py

# Тест основного бота
python ml_bot_binance.py

# Тест сбора данных
python data_collector.py
```

### 🔧 **Отладка:**

1. Включите детальные логи в `bot_config.json`
2. Проверьте подключение к API
3. Убедитесь в корректности токенов

## 📚 API Документация

### 🤖 **Telegram Bot API:**
- `/start` - инициализация бота
- `/analyze` - анализ монеты
- `/auto_start` - автосигналы
- `/set_coin <SYMBOL>` - выбор монеты

### 📊 **Binance API:**
- Получение исторических данных
- Анализ доступных торговых пар
- Реальное время обновления

## 🤝 Вклад в проект

1. Fork репозитория
2. Создайте feature branch (`git checkout -b feature/amazing-feature`)
3. Commit изменения (`git commit -m 'Add amazing feature'`)
4. Push в branch (`git push origin feature/amazing-feature`)
5. Откройте Pull Request

## 📄 Лицензия

Этот проект лицензирован под MIT License - см. файл [LICENSE](LICENSE)

## ⚠️ Отказ от ответственности

Этот бот предназначен только для образовательных целей. Торговля криптовалютами связана с высокими рисками. Используйте на свой страх и риск.

## 🆘 Поддержка

- 📧 Email: support@example.com
- 💬 Telegram: @username
- 🐛 Issues: [GitHub Issues](https://github.com/username/trading-bot/issues)

## 🌟 Звезды

Если проект вам понравился, поставьте ⭐ звезду!

---

**Сделано с ❤️ для сообщества трейдеров**
