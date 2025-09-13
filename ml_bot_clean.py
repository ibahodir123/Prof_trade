#!/usr/bin/env python3
"""
Чистый Telegram бот для ML сигналов БЕЗ Mastra
Только стандартный python-telegram-bot
"""
import asyncio
import logging
from telegram import Bot, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import json
import os
import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import time

# Настройка matplotlib для работы без GUI
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
import io

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_config():
    """Загрузка конфигурации бота"""
    try:
        with open('bot_config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        logger.error(f"Ошибка загрузки конфигурации: {e}")
        return None

def calculate_dynamic_percentages(signal_strength, signal_type):
    """Расчет динамических процентов на основе силы сигнала"""
    
    if signal_strength > 0.9:
        # Очень сильный сигнал
        profit_pct = 0.05  # +5%
        loss_pct = 0.03    # -3%
        strength_text = "🔥 Очень сильный"
    elif signal_strength > 0.8:
        # Сильный сигнал
        profit_pct = 0.04  # +4%
        loss_pct = 0.025   # -2.5%
        strength_text = "💪 Сильный"
    elif signal_strength > 0.7:
        # Средний сигнал
        profit_pct = 0.03  # +3%
        loss_pct = 0.02    # -2%
        strength_text = "⚡ Средний"
    else:
        # Слабый сигнал
        profit_pct = 0.02  # +2%
        loss_pct = 0.015   # -1.5%
        strength_text = "🌱 Слабый"
    
    return profit_pct, loss_pct, strength_text

def analyze_coin_signal(symbol):
    """Анализ монеты и генерация сигнала"""
    try:
        # Инициализация биржи
        exchange = ccxt.binance({
            'apiKey': '',
            'secret': '',
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Получение данных
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Расчет индикаторов
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Расчет RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Загрузка ML моделей
        try:
            scaler = joblib.load('models/scaler.pkl')
            min_detector = joblib.load('models/minimum_detector.pkl')
            max_detector = joblib.load('models/maximum_detector.pkl')
            
            # Подготовка данных для ML
            features = df[['close', 'ema_20', 'ema_50', 'ema_100', 'rsi']].fillna(0)
            features_scaled = scaler.transform(features)
            
            # Предсказания
            min_prob = min_detector.predict_proba(features_scaled[-1:])[0][1]
            max_prob = max_detector.predict_proba(features_scaled[-1:])[0][1]
            
            # Определение сигнала
            if max_prob > 0.7:
                signal_type = "⚪ ОЖИДАНИЕ"
                strength_text = f"Возможное падение {max_prob*100:.1f}%"
                profit_pct, loss_pct, _ = calculate_dynamic_percentages(max_prob, "SHORT")
                entry_price = df['close'].iloc[-1]
                take_profit = entry_price * (1 - profit_pct)
                stop_loss = entry_price * (1 + loss_pct)
                ml_status = "Активна"
                
            elif min_prob > 0.7:
                signal_type = "🟢 LONG"
                strength_text = f"Рост {min_prob*100:.1f}%"
                profit_pct, loss_pct, _ = calculate_dynamic_percentages(min_prob, "LONG")
                entry_price = df['close'].iloc[-1]
                take_profit = entry_price * (1 + profit_pct)
                stop_loss = entry_price * (1 - loss_pct)
                ml_status = "Активна"
                
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                strength_text = "Нет четкого сигнала"
                entry_price = df['close'].iloc[-1]
                take_profit = None
                stop_loss = None
                ml_status = "Активна"
                
        except Exception as e:
            # Fallback к простому анализу
            latest_close = df['close'].iloc[-1]
            ema_20_latest = df['ema_20'].iloc[-1]
            ema_50_latest = df['ema_50'].iloc[-1]
            
            if latest_close > ema_20_latest > ema_50_latest:
                signal_type = "🟢 LONG"
                strength_text = "Простой анализ: восходящий тренд"
                profit_pct, loss_pct, _ = calculate_dynamic_percentages(0.6, "LONG")
                entry_price = latest_close
                take_profit = entry_price * (1 + profit_pct)
                stop_loss = entry_price * (1 - loss_pct)
                ml_status = "Fallback"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                strength_text = "Простой анализ: неопределенность"
                entry_price = latest_close
                take_profit = None
                stop_loss = None
                ml_status = "Fallback"
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'strength_text': strength_text,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'rsi': df['rsi'].iloc[-1],
            'ml_status': ml_status,
            'df': df
        }
        
    except Exception as e:
        logger.error(f"Ошибка анализа {symbol}: {e}")
        return None

def create_advanced_trading_chart(symbol, df, signal_data):
    """Создание продвинутого графика в стиле TradingView"""
    try:
        # Настройка стиля TradingView
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
        
        # Основной график
        ax1.set_facecolor('#1e1e1e')
        ax1.grid(True, alpha=0.3, color='#333333')
        
        # Последние 100 свечей для лучшей видимости
        recent_df = df.tail(100)
        x_pos = range(len(recent_df))
        
        # Свечи
        for i, (idx, row) in enumerate(recent_df.iterrows()):
            color = '#26a69a' if row['close'] >= row['open'] else '#ef5350'
            # Тело свечи
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            ax1.bar(i, body_height, bottom=body_bottom, width=0.6, color=color, alpha=0.8)
            # Тени
            ax1.plot([i, i], [row['low'], row['high']], color=color, linewidth=1)
        
        # EMA линии
        ax1.plot(x_pos, recent_df['ema_20'], color='#ffeb3b', linewidth=2, label='EMA 20')
        ax1.plot(x_pos, recent_df['ema_50'], color='#ff9800', linewidth=2, label='EMA 50')
        ax1.plot(x_pos, recent_df['ema_100'], color='#e91e63', linewidth=2, label='EMA 100')
        
        # Точки входа и тейк-профита
        current_price = signal_data['entry_price']
        current_idx = len(recent_df) - 1
        
        if signal_data['signal_type'] == "🟢 LONG":
            # Точка входа
            ax1.scatter(current_idx, current_price, color='#4caf50', s=100, marker='^', 
                       label='Вход LONG', zorder=5)
            
            if signal_data['take_profit']:
                # Take Profit зона
                tp_price = signal_data['take_profit']
                ax1.axhline(y=tp_price, color='#4caf50', linestyle='--', alpha=0.7, label=f'TP: ${tp_price:.4f}')
                ax1.fill_between(x_pos, tp_price, tp_price * 1.001, alpha=0.3, color='#4caf50')
                
                # Stop Loss зона
                if signal_data['stop_loss']:
                    sl_price = signal_data['stop_loss']
                    ax1.axhline(y=sl_price, color='#f44336', linestyle='--', alpha=0.7, label=f'SL: ${sl_price:.4f}')
                    ax1.fill_between(x_pos, sl_price * 0.999, sl_price, alpha=0.3, color='#f44336')
        
        # RSI график
        ax2.set_facecolor('#1e1e1e')
        ax2.grid(True, alpha=0.3, color='#333333')
        ax2.plot(x_pos, recent_df['rsi'], color='#9c27b0', linewidth=2)
        ax2.axhline(y=70, color='#f44336', linestyle='--', alpha=0.7, label='Перекупленность')
        ax2.axhline(y=30, color='#4caf50', linestyle='--', alpha=0.7, label='Перепроданность')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel('RSI', color='white')
        
        # Настройки осей
        ax1.set_title(f'{symbol} - {signal_data["signal_type"]} {signal_data["strength_text"]}', 
                     color='white', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Цена ($)', color='white')
        ax1.legend(loc='upper left', framealpha=0.8)
        ax2.legend(loc='upper right', framealpha=0.8)
        
        # Информационная панель
        info_text = f"""📊 Анализ: {signal_data['ml_status']}
💰 Текущая цена: ${current_price:.4f}
📈 RSI: {signal_data['rsi']:.1f}"""
        
        if signal_data['signal_type'] == "🟢 LONG":
            info_text += f"""
🎯 Take Profit: ${signal_data['take_profit']:.4f}
🛡️ Stop Loss: ${signal_data['stop_loss']:.4f}"""
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#2d2d2d', alpha=0.8),
                verticalalignment='top', fontsize=9, color='white')
        
        plt.tight_layout()
        
        # Сохранение в буфер
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#1e1e1e', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        return buffer
        
    except Exception as e:
        logger.error(f"Ошибка создания продвинутого графика для {symbol}: {e}")
        return None

# Глобальные переменные
current_coin = "BTC/USDT"
auto_signals_enabled = False

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start с красивым меню"""
    global current_coin
    
    keyboard = [
        [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")],
        [InlineKeyboardButton("🪙 Выбор монет", callback_data="menu_coins")],
        [InlineKeyboardButton("📈 Последние сигналы", callback_data="menu_signals")],
        [InlineKeyboardButton("🔍 Анализ монеты", callback_data="menu_analyze")],
        [InlineKeyboardButton("🔍 Поиск монет", callback_data="menu_search")],
        [InlineKeyboardButton("🤖 Авто сигналы", callback_data="menu_auto")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""
🤖 **Trading Bot с автоматическими сигналами!**

🪙 **Текущая монета:** {current_coin}

**Выберите действие из меню ниже:**
    """
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок меню"""
    global current_coin
    
    query = update.callback_query
    await query.answer()
    
    print(f"🔘 Нажата кнопка: {query.data}")  # Отладка
    logger.info(f"🔘 Нажата кнопка: {query.data}")
    
    try:
        if query.data == "menu_status":
            await handle_status_menu(query, context)
        elif query.data == "menu_coins":
            await handle_coins_menu(query, context)
        elif query.data == "menu_signals":
            await handle_signals_menu(query, context)
        elif query.data == "menu_analyze":
            await handle_analyze_menu(query, context)
        elif query.data == "menu_search":
            await handle_search_menu(query, context)
        elif query.data == "menu_auto":
            await handle_auto_menu(query, context)
        elif query.data.startswith("select_"):
            await handle_coin_selection(query, context)
        elif query.data == "back_to_main":
            await back_to_main_menu(query, context)
            
    except Exception as e:
        print(f"❌ Ошибка в button_callback: {e}")
        logger.error(f"❌ Ошибка в button_callback: {e}")
        try:
            await query.edit_message_text(f"❌ Ошибка: {str(e)}")
        except:
            pass

async def handle_status_menu(query, context):
    """Обработка кнопки Статус системы"""
    try:
        status_message = f"""
📊 **Статус системы**

🪙 **Текущая монета:** {current_coin}
🤖 **Авто сигналы:** {'✅ Включены' if auto_signals_enabled else '❌ Выключены'}
⏰ **Время:** {datetime.now().strftime('%H:%M:%S')}

**Доступные команды:**
/start - Главное меню
/status - Статус системы
/coins - Список монет
/signals - Сигналы для {current_coin}
/analyze - Анализ {current_coin}
/search - Поиск монет
        """
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(status_message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения статуса: {str(e)}")

async def handle_coins_menu(query, context):
    """Обработка кнопки Выбор монет"""
    try:
        popular_coins = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT",
            "XRP/USDT", "DOT/USDT", "DOGE/USDT", "AVAX/USDT", "MATIC/USDT"
        ]
        
        keyboard = []
        for i in range(0, len(popular_coins), 2):
            row = []
            for j in range(2):
                if i + j < len(popular_coins):
                    coin = popular_coins[i + j]
                    row.append(InlineKeyboardButton(coin, callback_data=f"select_{coin}"))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = "🪙 **Выберите монету для анализа:**"
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения списка монет: {str(e)}")

async def handle_signals_menu(query, context):
    """Обработка кнопки Последние сигналы"""
    try:
        signal_data = analyze_coin_signal(current_coin)
        if not signal_data:
            await query.edit_message_text(f"❌ Ошибка анализа {current_coin}")
            return
        
        # Создание графика
        chart_buffer = create_advanced_trading_chart(current_coin, signal_data['df'], signal_data)
        
        if chart_buffer:
            # Отправка графика с подписью
            message = f"""
📈 **Сигнал для {current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.4f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            if signal_data['signal_type'] == "🟢 LONG":
                message += f"""
🎯 **Take Profit:** ${signal_data['take_profit']:.4f}
🛡️ **Stop Loss:** ${signal_data['stop_loss']:.4f}
                """
            
            keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_photo(
                photo=chart_buffer,
                caption=message,
                reply_markup=reply_markup
            )
            await query.edit_message_text("📈 График отправлен!")
        else:
            # Fallback без графика
            message = f"""
📈 **Сигнал для {current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.4f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.edit_message_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения сигналов: {str(e)}")

async def handle_analyze_menu(query, context):
    """Обработка кнопки Анализ монеты"""
    await handle_signals_menu(query, context)  # Пока используем ту же логику

async def handle_search_menu(query, context):
    """Обработка кнопки Поиск монет"""
    try:
        message = """
🔍 **Поиск монет**

Для поиска монет используйте команду:
/search <название>

Примеры:
/search BTC
/search ETH
/search BNB
        """
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка поиска: {str(e)}")

async def handle_auto_menu(query, context):
    """Обработка кнопки Авто сигналы"""
    try:
        status = "✅ Включены" if auto_signals_enabled else "❌ Выключены"
        action = "Остановить" if auto_signals_enabled else "Запустить"
        callback = "auto_stop" if auto_signals_enabled else "auto_start"
        
        message = f"""
🤖 **Автоматические сигналы**

**Статус:** {status}

Автоматические сигналы отправляются каждые 30 минут с лучшими сигналами.
        """
        
        keyboard = [
            [InlineKeyboardButton(f"🔄 {action}", callback_data=callback)],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка управления авто сигналами: {str(e)}")

async def handle_coin_selection(query, context):
    """Обработка выбора монеты"""
    global current_coin
    coin = query.data.replace("select_", "")
    current_coin = coin
    
    await query.edit_message_text(f"✅ Выбрана монета: {coin}")
    
    # Автоматически показываем анализ
    await asyncio.sleep(1)
    await handle_signals_menu(query, context)

async def back_to_main_menu(query, context):
    """Возврат в главное меню"""
    global current_coin
    
    keyboard = [
        [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")],
        [InlineKeyboardButton("🪙 Выбор монет", callback_data="menu_coins")],
        [InlineKeyboardButton("📈 Последние сигналы", callback_data="menu_signals")],
        [InlineKeyboardButton("🔍 Анализ монеты", callback_data="menu_analyze")],
        [InlineKeyboardButton("🔍 Поиск монет", callback_data="menu_search")],
        [InlineKeyboardButton("🤖 Авто сигналы", callback_data="menu_auto")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""
🤖 **Trading Bot с автоматическими сигналами!**

🪙 **Текущая монета:** {current_coin}

**Выберите действие из меню ниже:**
    """
    
    await query.edit_message_text(welcome_message, reply_markup=reply_markup)

def main():
    """Основная функция"""
    print("🤖 Запуск ЧИСТОГО ML Telegram Bot (без Mastra)")
    
    # Загружаем конфигурацию
    config = load_config()
    if not config:
        print("❌ Не удалось загрузить конфигурацию")
        return
    
    # Создаем приложение
    application = Application.builder().token(config["telegram_token"]).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    print("✅ Бот настроен успешно")
    print("🚀 Запускаю бота...")
    
    # Запускаем бота
    application.run_polling()

if __name__ == "__main__":
    main()
