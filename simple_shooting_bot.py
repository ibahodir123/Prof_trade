#!/usr/bin/env python3
"""
Упрощенный бот для предсказания стреляющих монет
Без TensorFlow, использует простые алгоритмы
"""
import asyncio
import json
import logging
import os
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
import ccxt
import pandas as pd
import numpy as np

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
def load_config():
    """Загружает конфигурацию"""
    try:
        with open('bot_config.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return None

config = load_config()

class SimpleShootingStarPredictor:
    """Простой предиктор стреляющих монет без ML"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True
            }
        })
    
    def get_binance_data(self, symbol, timeframe='1h', limit=100):
        """Получает данные с Binance"""
        try:
            # Синхронизация времени
            server_time = self.exchange.fetch_time()
            local_time = self.exchange.milliseconds()
            time_diff = server_time - local_time
            
            if abs(time_diff) > 1000:
                self.exchange.options['timeDifference'] = time_diff
            
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Вычисляет технические индикаторы"""
        if df is None or df.empty:
            return None
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price change
        df['price_change_1h'] = df['close'].pct_change(1)
        df['price_change_4h'] = df['close'].pct_change(4)
        df['price_change_24h'] = df['close'].pct_change(24)
        
        return df.dropna()
    
    def predict_shooting_star(self, df):
        """Предсказывает стреляющие монеты"""
        if df is None or len(df) < 50:
            return None
        
        latest = df.iloc[-1]
        
        # Критерии для стреляющей монеты
        criteria = {
            'rsi_oversold': latest['rsi'] < 30,  # RSI перепродан
            'price_above_ema20': latest['close'] > latest['ema_20'],  # Цена выше EMA20
            'volume_surge': latest['volume_ratio'] > 2.0,  # Объем в 2 раза больше среднего
            'bb_squeeze': (latest['bb_upper'] - latest['bb_lower']) / latest['close'] < 0.05,  # Сжатие Боллинджера
            'recent_dip': latest['price_change_24h'] < -0.05,  # Недавнее падение 5%+
            'momentum_building': latest['price_change_1h'] > 0 and latest['price_change_4h'] > 0  # Нарастающий импульс
        }
        
        # Подсчет критериев
        score = sum(criteria.values())
        
        # Определение вероятности
        if score >= 5:
            probability = 0.9
            category = "🚀 ВЗРЫВНОЙ РОСТ"
        elif score >= 4:
            probability = 0.7
            category = "📈 ВЫСОКИЙ РОСТ"
        elif score >= 3:
            probability = 0.5
            category = "📊 УМЕРЕННЫЙ РОСТ"
        else:
            probability = 0.2
            category = "📉 СЛАБЫЙ СИГНАЛ"
        
        return {
            'probability': probability,
            'category': category,
            'score': score,
            'criteria': criteria,
            'rsi': latest['rsi'],
            'volume_ratio': latest['volume_ratio'],
            'price_change_24h': latest['price_change_24h'],
            'bb_width': (latest['bb_upper'] - latest['bb_lower']) / latest['close']
        }
    
    def analyze_coin(self, symbol):
        """Анализирует монету"""
        try:
            # Получаем данные
            df = self.get_binance_data(symbol)
            if df is None:
                return None
            
            # Вычисляем индикаторы
            df = self.calculate_indicators(df)
            if df is None:
                return None
            
            # Делаем предсказание
            prediction = self.predict_shooting_star(df)
            if prediction is None:
                return None
            
            return {
                'symbol': symbol,
                'current_price': df['close'].iloc[-1],
                'prediction': prediction,
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None

# Глобальная переменная для предиктора
predictor = SimpleShootingStarPredictor()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start"""
    keyboard = [
        [InlineKeyboardButton("🔍 Анализ монеты", callback_data="analyze_coin")],
        [InlineKeyboardButton("🚀 Поиск стреляющих монет", callback_data="find_shooting_stars")],
        [InlineKeyboardButton("📊 Топ сигналов", callback_data="top_signals")],
        [InlineKeyboardButton("ℹ️ Помощь", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "🎯 **БОТ ПРЕДСКАЗАНИЯ СТРЕЛЯЮЩИХ МОНЕТ**\n\n"
        "Я помогаю найти монеты с потенциалом взрывного роста!\n\n"
        "Выберите действие:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка нажатий кнопок"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "analyze_coin":
        await query.edit_message_text(
            "🔍 **АНАЛИЗ МОНЕТЫ**\n\n"
            "Отправьте символ монеты (например: BTC, ETH, BNB)\n\n"
            "Или используйте команду:\n"
            "/analyze BTC"
        )
    
    elif query.data == "find_shooting_stars":
        await find_shooting_stars(update, context)
    
    elif query.data == "top_signals":
        await top_signals(update, context)
    
    elif query.data == "help":
        await help_command(update, context)
    
    elif query.data == "back_to_main":
        await start(update, context)

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /analyze"""
    if not context.args:
        await update.message.reply_text(
            "❌ Укажите символ монеты!\n\n"
            "Пример: /analyze BTC"
        )
        return
    
    symbol = context.args[0].upper()
    if not symbol.endswith('USDT'):
        symbol += '/USDT'
    
    await update.message.reply_text(f"🔍 Анализирую {symbol}...")
    
    # Анализируем монету
    result = predictor.analyze_coin(symbol)
    
    if result is None:
        await update.message.reply_text(f"❌ Не удалось проанализировать {symbol}")
        return
    
    prediction = result['prediction']
    
    # Формируем сообщение
    message = f"""
🎯 **АНАЛИЗ {symbol}**

💰 **Цена:** ${result['current_price']:.8f}
📊 **Данных:** {result['data_points']} свечей

🚀 **ПРЕДСКАЗАНИЕ:**
{prediction['category']}
🎯 **Вероятность:** {prediction['probability']:.1%}
⭐ **Оценка:** {prediction['score']}/6

📈 **ДЕТАЛИ:**
• RSI: {prediction['rsi']:.1f}
• Объем: {prediction['volume_ratio']:.1f}x среднего
• Изменение 24ч: {prediction['price_change_24h']:.1%}
• Ширина BB: {prediction['bb_width']:.1%}

🔍 **КРИТЕРИИ:**
{'✅' if prediction['criteria']['rsi_oversold'] else '❌'} RSI перепродан (<30)
{'✅' if prediction['criteria']['price_above_ema20'] else '❌'} Цена выше EMA20
{'✅' if prediction['criteria']['volume_surge'] else '❌'} Всплеск объема (2x+)
{'✅' if prediction['criteria']['bb_squeeze'] else '❌'} Сжатие Боллинджера
{'✅' if prediction['criteria']['recent_dip'] else '❌'} Недавнее падение 5%+
{'✅' if prediction['criteria']['momentum_building'] else '❌'} Нарастающий импульс
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        message,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def find_shooting_stars(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Поиск стреляющих монет"""
    if update.callback_query:
        await update.callback_query.edit_message_text("🚀 Ищу стреляющие монеты...")
        message_func = update.callback_query.edit_message_text
    else:
        await update.message.reply_text("🚀 Ищу стреляющие монеты...")
        message_func = update.message.reply_text
    
    try:
        # Получаем список популярных монет
        popular_coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC']
        
        results = []
        for coin in popular_coins:
            symbol = f"{coin}/USDT"
            result = predictor.analyze_coin(symbol)
            if result and result['prediction']['probability'] > 0.3:
                results.append(result)
        
        if not results:
            await message_func("😔 Стреляющие монеты не найдены в данный момент")
            return
        
        # Сортируем по вероятности
        results.sort(key=lambda x: x['prediction']['probability'], reverse=True)
        
        # Формируем сообщение
        message = "🚀 **НАЙДЕННЫЕ СТРЕЛЯЮЩИЕ МОНЕТЫ:**\n\n"
        
        for i, result in enumerate(results[:5], 1):
            pred = result['prediction']
            message += f"{i}. **{result['symbol']}**\n"
            message += f"   💰 ${result['current_price']:.8f}\n"
            message += f"   {pred['category']}\n"
            message += f"   🎯 {pred['probability']:.1%} ({pred['score']}/6)\n\n"
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await message_func(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"Ошибка поиска стреляющих монет: {e}")
        await message_func("❌ Ошибка при поиске стреляющих монет")

async def top_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Топ сигналов"""
    if update.callback_query:
        await update.callback_query.edit_message_text("📊 Анализирую топ сигналы...")
        message_func = update.callback_query.edit_message_text
    else:
        await update.message.reply_text("📊 Анализирую топ сигналы...")
        message_func = update.message.reply_text
    
    try:
        # Получаем список популярных монет
        popular_coins = ['BTC', 'ETH', 'BNB', 'ADA', 'SOL', 'XRP', 'DOT', 'DOGE', 'AVAX', 'MATIC', 
                        'LINK', 'UNI', 'LTC', 'BCH', 'ATOM', 'NEAR', 'FTM', 'ALGO', 'VET', 'ICP']
        
        results = []
        for coin in popular_coins:
            symbol = f"{coin}/USDT"
            result = predictor.analyze_coin(symbol)
            if result:
                results.append(result)
        
        if not results:
            await message_func("❌ Не удалось получить данные")
            return
        
        # Сортируем по вероятности
        results.sort(key=lambda x: x['prediction']['probability'], reverse=True)
        
        # Формируем сообщение
        message = "📊 **ТОП СИГНАЛОВ:**\n\n"
        
        for i, result in enumerate(results[:10], 1):
            pred = result['prediction']
            emoji = "🚀" if pred['probability'] > 0.7 else "📈" if pred['probability'] > 0.5 else "📊"
            
            message += f"{emoji} {i}. **{result['symbol']}**\n"
            message += f"   💰 ${result['current_price']:.8f}\n"
            message += f"   🎯 {pred['probability']:.1%} - {pred['category']}\n\n"
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await message_func(
            message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    except Exception as e:
        logger.error(f"Ошибка получения топ сигналов: {e}")
        await message_func("❌ Ошибка при получении топ сигналов")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда помощи"""
    help_text = """
ℹ️ **ПОМОЩЬ**

🎯 **ОСНОВНЫЕ ФУНКЦИИ:**
• 🔍 Анализ монеты - детальный анализ конкретной монеты
• 🚀 Поиск стреляющих монет - автоматический поиск монет с потенциалом роста
• 📊 Топ сигналов - рейтинг всех проанализированных монет

📋 **КОМАНДЫ:**
• /start - главное меню
• /analyze BTC - анализ конкретной монеты
• /help - эта справка

🔍 **КРИТЕРИИ АНАЛИЗА:**
• RSI перепродан (<30)
• Цена выше EMA20
• Всплеск объема (2x+ среднего)
• Сжатие полос Боллинджера
• Недавнее падение 5%+
• Нарастающий импульс

⚠️ **ВАЖНО:**
Это инструмент для анализа, не инвестиционный совет!
"""
    
    keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    if update.callback_query:
        await update.callback_query.edit_message_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
    else:
        await update.message.reply_text(
            help_text,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ошибок"""
    logger.error(f"Ошибка: {context.error}")
    
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Произошла ошибка. Попробуйте позже."
        )

def main():
    """Основная функция"""
    if not config:
        logger.error("Не удалось загрузить конфигурацию")
        return
    
    # Создаем приложение
    application = Application.builder().token(config["telegram_token"]).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    # Обработчик ошибок
    application.add_error_handler(error_handler)
    
    # Запускаем бота
    logger.info("🚀 Запускаю бота для предсказания стреляющих монет...")
    application.run_polling()

if __name__ == "__main__":
    main()
