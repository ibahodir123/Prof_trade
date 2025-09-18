#!/usr/bin/env python3
"""
Интегрированный бот с системой предсказания стреляющих монет
Объединяет существующий ML бот с нейронной сетью для предсказания
"""
import asyncio
import logging
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Импорты из существующего бота
from ml_bot_binance import (
    load_config, get_binance_data, prepare_ml_features,
    calculate_dynamic_percentages, create_advanced_trading_chart
)

# Импорты для нейронной сети
from neural_network_predictor import ShootingStarPredictor
from data_collector import HistoricalDataCollector

# Telegram bot imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler, 
    ContextTypes, MessageHandler, filters
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShootingStarBot:
    """Бот с предсказанием стреляющих монет"""
    
    def __init__(self):
        self.config = load_config()
        self.current_coin = "BTC/USDT"
        self.predictor = ShootingStarPredictor()
        self.data_collector = HistoricalDataCollector()
        self.shooting_stars_cache = {}
        self.last_update = None
        
        # Загружаем модель если она существует
        self.load_predictor_model()
    
    def load_predictor_model(self):
        """Загружает обученную модель предсказания"""
        try:
            if self.predictor.load_model():
                logger.info("✅ Модель предсказания стреляющих монет загружена")
            else:
                logger.warning("⚠️ Модель предсказания не найдена. Запустите обучение.")
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки модели: {e}")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда /start"""
        keyboard = [
            [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")],
            [InlineKeyboardButton("🪙 Выбор монет", callback_data="menu_coins")],
            [InlineKeyboardButton("📈 Последние сигналы", callback_data="menu_signals")],
            [InlineKeyboardButton("🔍 Анализ монеты", callback_data="menu_analyze")],
            [InlineKeyboardButton("🚀 Стреляющие монеты", callback_data="menu_shooting_stars")],
            [InlineKeyboardButton("🧠 Обучение модели", callback_data="menu_training")],
            [InlineKeyboardButton("🤖 Авто сигналы", callback_data="menu_auto")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""🤖 **Shooting Star Trading Bot**

🪙 **Текущая монета:** {self.current_coin}
🧠 **Модель предсказания:** {'✅ Загружена' if self.predictor.is_trained else '❌ Не обучена'}

**Доступные команды:**
• `/analyze` - Анализ текущей монеты
• `/shooting_stars` - Найти стреляющие монеты
• `/train_model` - Обновить модель
• `/collect_data` - Собрать новые данные

Выберите действие из меню ниже:"""
        
        await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
    
    async def shooting_stars_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда поиска стреляющих монет"""
        if not self.predictor.is_trained:
            await update.message.reply_text(
                "❌ **Модель не обучена**\n\n"
                "Сначала запустите обучение:\n"
                "`/train_model` - обновить модель\n"
                "`/collect_data` - собрать данные",
                parse_mode='Markdown'
            )
            return
        
        await update.message.reply_text("🔍 Ищу стреляющие монеты...")
        
        try:
            # Получаем список монет для анализа
            coins = self.get_popular_coins()[:20]  # Анализируем топ-20
            
            shooting_stars = []
            
            for coin in coins:
                try:
                    # Получаем данные
                    df = get_binance_data(coin, timeframe='1h', limit=500)
                    if df is None or df.empty:
                        continue
                    
                    # Добавляем индикаторы
                    df = self.add_technical_indicators(df)
                    
                    # Предсказание
                    prediction = self.predictor.predict(df)
                    
                    if prediction and prediction['shooting_star_probability'] > 0.3:
                        shooting_stars.append({
                            'coin': coin,
                            'probability': prediction['shooting_star_probability'],
                            'class': prediction['predicted_class'],
                            'confidence': prediction['confidence'],
                            'high_growth_prob': prediction['high_growth_probability']
                        })
                        
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка анализа {coin}: {e}")
                    continue
            
            # Сортируем по вероятности
            shooting_stars.sort(key=lambda x: x['probability'], reverse=True)
            
            if shooting_stars:
                message = "🚀 **СТРЕЛЯЮЩИЕ МОНЕТЫ** (следующие 24 часа)\n\n"
                
                for i, star in enumerate(shooting_stars[:5], 1):
                    message += f"**{i}. {star['coin']}**\n"
                    message += f"   🎯 Вероятность роста: {star['probability']:.1%}\n"
                    message += f"   📈 Класс: {star['class']}\n"
                    message += f"   🔥 Высокий рост: {star['high_growth_prob']:.1%}\n"
                    message += f"   ✅ Уверенность: {star['confidence']:.1%}\n\n"
                
                message += "⚠️ **Предупреждение:** Это прогнозы ИИ. Торгуйте ответственно!"
                
                # Создаем кнопки для анализа
                keyboard = []
                for star in shooting_stars[:3]:
                    keyboard.append([InlineKeyboardButton(
                        f"📊 Анализ {star['coin']}", 
                        callback_data=f"analyze_shooting_{star['coin']}"
                    )])
                
                keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")])
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                await update.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
            else:
                await update.message.reply_text(
                    "😴 **Стреляющих монет не найдено**\n\n"
                    "В данный момент модель не видит монет с высокой вероятностью роста в ближайшие 24 часа.",
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"❌ Ошибка поиска стреляющих монет: {e}")
            await update.message.reply_text("❌ Ошибка при поиске стреляющих монет")
    
    async def train_model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда обучения модели"""
        await update.message.reply_text(
            "🧠 **ОБУЧЕНИЕ МОДЕЛИ**\n\n"
            "Для обучения модели нужно:\n"
            "1. 📊 Собрать исторические данные\n"
            "2. 🧠 Обучить нейронную сеть\n"
            "3. ✅ Протестировать модель\n\n"
            "⚠️ **Внимание:** Обучение может занять 30-60 минут!\n\n"
            "Начать обучение?",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("📊 Собрать данные", callback_data="collect_data")],
                [InlineKeyboardButton("🧠 Обучить модель", callback_data="train_neural_network")],
                [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
            ]),
            parse_mode='Markdown'
        )
    
    async def collect_data_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Команда сбора данных"""
        await update.message.reply_text(
            "📊 **СБОР ИСТОРИЧЕСКИХ ДАННЫХ**\n\n"
            "Собираю данные всех монет с 1 января 2025 года...\n"
            "Это может занять 10-20 минут.\n\n"
            "⏳ Пожалуйста, подождите..."
        )
        
        try:
            # Запускаем сбор данных в фоне
            asyncio.create_task(self.collect_data_background(update))
            
        except Exception as e:
            logger.error(f"❌ Ошибка сбора данных: {e}")
            await update.message.reply_text("❌ Ошибка при сборе данных")
    
    async def collect_data_background(self, update: Update):
        """Фоновый сбор данных"""
        try:
            # Собираем данные (ограничиваем для демо)
            data = self.data_collector.collect_all_data(max_pairs=30)
            
            if data:
                # Сохраняем данные
                self.data_collector.save_data(data, "historical_data.json")
                
                # Статистика
                total_records = sum(len(df) for df in data.values())
                shooting_stars = sum(
                    df['is_shooting_star'].sum() 
                    for df in data.values() 
                    if 'is_shooting_star' in df.columns
                )
                
                message = f"""✅ **ДАННЫЕ СОБРАНЫ УСПЕШНО!**

📊 **Статистика:**
• Монет обработано: {len(data)}
• Всего записей: {total_records:,}
• Стреляющих моментов: {shooting_stars:,}
• Процент стреляющих: {(shooting_stars/total_records*100):.2f}%

Теперь можно обучать модель!"""
                
                await update.message.reply_text(
                    message,
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🧠 Обучить модель", callback_data="train_neural_network")],
                        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
                    ]),
                    parse_mode='Markdown'
                )
            else:
                await update.message.reply_text("❌ Не удалось собрать данные")
                
        except Exception as e:
            logger.error(f"❌ Ошибка фонового сбора: {e}")
            await update.message.reply_text("❌ Ошибка при сборе данных")
    
    async def train_neural_network_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обучение нейронной сети"""
        query = update.callback_query
        await query.answer()
        
        await query.edit_message_text(
            "🧠 **ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ**\n\n"
            "Обучаю модель на собранных данных...\n"
            "Это может занять 30-60 минут.\n\n"
            "⏳ Пожалуйста, подождите..."
        )
        
        try:
            # Запускаем обучение в фоне
            asyncio.create_task(self.train_model_background(query))
            
        except Exception as e:
            logger.error(f"❌ Ошибка обучения: {e}")
            await query.edit_message_text("❌ Ошибка при обучении модели")
    
    async def train_model_background(self, query):
        """Фоновое обучение модели"""
        try:
            # Загружаем данные
            data = self.data_collector.load_data("historical_data.json")
            
            if not data:
                await query.edit_message_text("❌ Данные не найдены. Сначала соберите данные.")
                return
            
            # Обучаем модель
            success = self.predictor.train(data)
            
            if success:
                await query.edit_message_text(
                    "✅ **МОДЕЛЬ ОБУЧЕНА УСПЕШНО!**\n\n"
                    "Теперь можно искать стреляющие монеты!\n\n"
                    "Попробуйте команду `/shooting_stars`",
                    reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("🚀 Найти стреляющие", callback_data="menu_shooting_stars")],
                        [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
                    ]),
                    parse_mode='Markdown'
                )
            else:
                await query.edit_message_text("❌ Ошибка при обучении модели")
                
        except Exception as e:
            logger.error(f"❌ Ошибка фонового обучения: {e}")
            await query.edit_message_text("❌ Ошибка при обучении модели")
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет технические индикаторы к данным"""
        try:
            # EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # Volume
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=24).std()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"❌ Ошибка добавления индикаторов: {e}")
            return df
    
    def get_popular_coins(self) -> List[str]:
        """Возвращает список популярных монет"""
        return [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
            "XRP/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT", "LINK/USDT",
            "UNI/USDT", "LTC/USDT", "ATOM/USDT", "NEAR/USDT", "ALGO/USDT",
            "VET/USDT", "FIL/USDT", "TRX/USDT", "ETC/USDT", "XLM/USDT"
        ]
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработчик callback кнопок"""
        query = update.callback_query
        await query.answer()
        
        data = query.data
        
        if data == "menu_shooting_stars":
            await self.shooting_stars_command(query)
        elif data == "menu_training":
            await self.train_model_command(query)
        elif data == "collect_data":
            await self.collect_data_command(query)
        elif data == "train_neural_network":
            await self.train_neural_network_callback(update, context)
        elif data == "back_to_main":
            await self.start_command(query, context)
        # Добавьте другие обработчики по необходимости
    
    def run(self):
        """Запускает бота"""
        application = Application.builder().token(self.config['telegram_token']).build()
        
        # Добавляем обработчики
        application.add_handler(CommandHandler("start", self.start_command))
        application.add_handler(CommandHandler("shooting_stars", self.shooting_stars_command))
        application.add_handler(CommandHandler("train_model", self.train_model_command))
        application.add_handler(CommandHandler("collect_data", self.collect_data_command))
        application.add_handler(CallbackQueryHandler(self.handle_callback))
        
        logger.info("🚀 Запускаю Shooting Star Bot...")
        application.run_polling()

def main():
    """Основная функция"""
    bot = ShootingStarBot()
    bot.run()

if __name__ == "__main__":
    main()


