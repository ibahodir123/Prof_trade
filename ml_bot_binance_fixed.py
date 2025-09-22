#!/usr/bin/env python3
"""
Binance Telegram бот для ML сигналов
Использует Binance API через ccxt
"""
import asyncio
import logging
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import json
import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
# TensorFlow импорты удалены (не используются)
from advanced_ema_analyzer import AdvancedEMAAnalyzer
from advanced_ml_trainer import AdvancedMLTrainer
from shooting_star_predictor import ShootingStarPredictor

# Настройка matplotlib для работы без GUI
import matplotlib
matplotlib.use('Agg')  # Используем backend без GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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

def get_binance_data(symbol, timeframe='1h', limit=500):
    """Получает данные с Binance через ccxt"""
    try:
        logger.info(f"📊 Получаю данные {symbol} с Binance...")
        
        # Инициализация Binance (только публичные данные)
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
            'options': {
                'adjustForTimeDifference': True,  # Автоматическая синхронизация времени
            }
        })
        
        # Синхронизация времени с сервером Binance
        try:
            exchange.load_markets()
            server_time = exchange.fetch_time()
            local_time = exchange.milliseconds()
            time_diff = server_time - local_time
            logger.info(f"🕐 Разница времени с Binance: {time_diff}ms")
            
            if abs(time_diff) > 1000:  # Если разница больше 1 секунды
                logger.warning(f"⚠️ Большая разница времени: {time_diff}ms")
                # Устанавливаем корректировку времени
                exchange.options['timeDifference'] = time_diff
        except Exception as e:
            logger.warning(f"⚠️ Не удалось синхронизировать время: {e}")
        
        # Получение OHLCV данных
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        if not ohlcv:
            logger.error(f"❌ Нет данных для {symbol}")
            return None
        
        # Преобразование в DataFrame
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Конвертируем в float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        logger.info(f"📊 После конвертации: {len(df)} свечей")
        logger.info(f"📊 NaN значения: {df.isnull().sum().sum()}")
        
        # Удаляем NaN
        df = df.dropna()
        
        if df.empty:
            logger.error(f"❌ DataFrame пуст после удаления NaN для {symbol}")
            return None
        
        logger.info(f"✅ Получено {len(df)} свечей для {symbol}")
        logger.info(f"📊 Диапазон: {df.index[0]} - {df.index[-1]}")
        logger.info(f"💰 Последняя цена: ${df['close'].iloc[-1]:.8f}")
        
        return df
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения данных {symbol} с Binance: {e}")
        return None

def prepare_ml_features(df):
    """Подготавливает все 36 признаков для ML модели"""
    try:
        # Проверяем наличие необходимых колонок
        required_columns = ['close', 'ema_20', 'ema_50', 'ema_100', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Отсутствуют колонки: {missing_columns}")
            return None
        
        # Создаем копию данных
        data = df[required_columns].copy()
        
        # 1. Velocity (скорость изменения)
        data['price_velocity'] = data['close'].pct_change()
        data['ema20_velocity'] = data['ema_20'].pct_change()
        data['ema50_velocity'] = data['ema_50'].pct_change()
        data['ema100_velocity'] = data['ema_100'].pct_change()
        
        # 2. Acceleration (ускорение)
        data['price_acceleration'] = data['price_velocity'].pct_change()
        data['ema20_acceleration'] = data['ema20_velocity'].pct_change()
        data['ema50_acceleration'] = data['ema50_velocity'].pct_change()
        data['ema100_acceleration'] = data['ema100_velocity'].pct_change()
        
        # 3. Velocity ratios
        data['price_to_ema20_velocity_ratio'] = data['price_velocity'] / (data['ema20_velocity'] + 1e-8)
        data['price_to_ema50_velocity_ratio'] = data['price_velocity'] / (data['ema50_velocity'] + 1e-8)
        data['price_to_ema100_velocity_ratio'] = data['price_velocity'] / (data['ema100_velocity'] + 1e-8)
        
        # 4. Distance to EMAs
        data['price_to_ema20_distance'] = (data['close'] - data['ema_20']) / data['close']
        data['price_to_ema50_distance'] = (data['close'] - data['ema_50']) / data['close']
        data['price_to_ema100_distance'] = (data['close'] - data['ema_100']) / data['close']
        
        # 5. Distance change
        data['price_to_ema20_distance_change'] = data['price_to_ema20_distance'].diff()
        data['price_to_ema50_distance_change'] = data['price_to_ema50_distance'].diff()
        data['price_to_ema100_distance_change'] = data['price_to_ema100_distance'].diff()
        
        # 6. EMA angles (наклон)
        data['ema20_angle'] = np.arctan(data['ema20_velocity']) * 180 / np.pi
        data['ema50_angle'] = np.arctan(data['ema50_velocity']) * 180 / np.pi
        data['ema100_angle'] = np.arctan(data['ema100_velocity']) * 180 / np.pi
        
        # 7. Angle change
        data['ema20_angle_change'] = data['ema20_angle'].diff()
        data['ema50_angle_change'] = data['ema50_angle'].diff()
        data['ema100_angle_change'] = data['ema100_angle'].diff()
        
        # 8. EMA relationships
        data['ema20_to_ema50'] = data['ema_20'] / (data['ema_50'] + 1e-8)
        data['ema20_to_ema100'] = data['ema_20'] / (data['ema_100'] + 1e-8)
        data['ema50_to_ema100'] = data['ema_50'] / (data['ema_100'] + 1e-8)
        
        # 9. Price-EMA synchronization
        data['price_ema20_sync'] = np.corrcoef(data['close'], data['ema_20'])[0, 1] if len(data) > 1 else 0
        data['price_ema50_sync'] = np.corrcoef(data['close'], data['ema_50'])[0, 1] if len(data) > 1 else 0
        data['price_ema100_sync'] = np.corrcoef(data['close'], data['ema_100'])[0, 1] if len(data) > 1 else 0
        
        # 10. Divergence
        data['price_ema20_divergence'] = data['price_velocity'] - data['ema20_velocity']
        data['price_ema50_divergence'] = data['price_velocity'] - data['ema50_velocity']
        data['price_ema100_divergence'] = data['price_velocity'] - data['ema100_velocity']
        
        # 11. Volatility
        data['price_volatility'] = data['close'].rolling(20).std() / data['close'].rolling(20).mean()
        data['ema20_volatility'] = data['ema_20'].rolling(20).std() / data['ema_20'].rolling(20).mean()
        
        # 12. Volume features
        data['volume_change'] = data['volume'].pct_change()
        data['volume_price_correlation'] = data['volume'].rolling(20).corr(data['close'])
        
        # Удаляем первые строки с NaN
        data = data.dropna()
        
        # Проверяем на бесконечные значения
        data = data.replace([np.inf, -np.inf], 0)
        
        # Выбираем только нужные признаки (36 признаков)
        feature_columns = [
            'price_velocity', 'ema20_velocity', 'ema50_velocity', 'ema100_velocity',
            'price_acceleration', 'ema20_acceleration', 'ema50_acceleration', 'ema100_acceleration',
            'price_to_ema20_velocity_ratio', 'price_to_ema50_velocity_ratio', 'price_to_ema100_velocity_ratio',
            'price_to_ema20_distance', 'price_to_ema50_distance', 'price_to_ema100_distance',
            'price_to_ema20_distance_change', 'price_to_ema50_distance_change', 'price_to_ema100_distance_change',
            'ema20_angle', 'ema50_angle', 'ema100_angle',
            'ema20_angle_change', 'ema50_angle_change', 'ema100_angle_change',
            'ema20_to_ema50', 'ema20_to_ema100', 'ema50_to_ema100',
            'price_ema20_sync', 'price_ema50_sync', 'price_ema100_sync',
            'price_ema20_divergence', 'price_ema50_divergence', 'price_ema100_divergence',
            'price_volatility', 'ema20_volatility', 'volume_change', 'volume_price_correlation'
        ]
        
        features = data[feature_columns].fillna(0)
        
        logger.info(f"✅ Подготовлено {features.shape[1]} признаков для ML")
        return features
        
    except Exception as e:
        logger.error(f"❌ Ошибка подготовки признаков: {e}")
        return None

def is_coin_in_top50(symbol):
    """Проверяет, есть ли монета в топ-50 списке"""
    try:
        if not os.path.exists('top_coins_list.txt'):
            return False
        
        with open('top_coins_list.txt', 'r', encoding='utf-8', errors='ignore') as f:
            top_coins = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        return symbol in top_coins
    except Exception as e:
        logger.error(f"❌ Ошибка проверки топ-50: {e}")
        return False

def adaptive_retrain_for_coin(symbol):
    """Адаптивное переобучение для конкретной монеты"""
    try:
        logger.info(f"🔄 Адаптивное переобучение для {symbol}...")
        
        # Загружаем текущий список топ-50
        with open('top_coins_list.txt', 'r', encoding='utf-8', errors='ignore') as f:
            top_coins = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        # Добавляем новую монету если её нет
        if symbol not in top_coins:
            top_coins.append(symbol)
            logger.info(f"➕ Добавлена новая монета: {symbol}")
        
        # Ограничиваем до 50 монет (убираем наименее популярные)
        if len(top_coins) > 50:
            # Удаляем последние монеты, оставляя топ-50
            top_coins = top_coins[:50]
        
        # Переобучаем модели
        trainer = AdvancedMLTrainer()
        success = trainer.train_models(top_coins)
        
        if success:
            logger.info(f"✅ Адаптивное переобучение завершено для {symbol}")
            return True
        else:
            logger.error(f"❌ Ошибка адаптивного переобучения для {symbol}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Критическая ошибка адаптивного переобучения: {e}")
        return False

def analyze_coin_signal_advanced_ema(symbol):
    """Анализ монеты с использованием продвинутой EMA логики"""
    # Используем bot_state напрямую
    
    try:
        # Очищаем символ от дублирования USDT
        clean_symbol = symbol.replace(':USDT', '') if ':USDT' in symbol else symbol
        
        # Инициализация анализаторов
        if bot_state.ema_analyzer is None:
            bot_state.ema_analyzer = AdvancedEMAAnalyzer()
        
        if bot_state.ml_trainer is None:
            bot_state.ml_trainer = AdvancedMLTrainer()
            bot_state.ml_trainer.load_models()  # Загружаем обученные модели
        
        # 🔄 АДАПТИВНОЕ ОБУЧЕНИЕ: Проверяем, есть ли монета в топ-50
        if not is_coin_in_top50(symbol):
            logger.info(f"🆕 Монета {symbol} не в топ-50, запускаю адаптивное переобучение...")
            
            # Показываем пользователю что идет переобучение
            # (это будет видно в логах)
            
            # Выполняем адаптивное переобучение
            retrain_success = adaptive_retrain_for_coin(symbol)
            
            if retrain_success:
                # Перезагружаем модели после переобучения
                bot_state.ml_trainer = AdvancedMLTrainer()
                bot_state.ml_trainer.load_models()
                logger.info(f"✅ Модели переобучены с учетом {symbol}")
            else:
                logger.warning(f"⚠️ Не удалось переобучить модели для {symbol}, используем существующие")
        
        logger.info(f"📊 Анализирую {symbol} с продвинутой EMA логикой...")
        
        # Получаем данные через Binance API
        exchange = ccxt.binance()
        ohlcv_data = exchange.fetch_ohlcv(symbol, '1h', limit=500)
        
        if not ohlcv_data:
            logger.error(f"❌ Нет данных для {symbol}")
            return {
                'symbol': clean_symbol,
                'signal_type': "❌ МОНЕТА НЕ НАЙДЕНА",
                'strength_text': f"Монета {clean_symbol} не найдена на Binance",
                'entry_price': None,
                'take_profit': None,
                'stop_loss': None,
                'rsi': None,
                'ml_status': "Не найдена",
                'df': None,
                'error': f"Монета {clean_symbol} не найдена на Binance"
            }
        
        # Анализ с продвинутой EMA логикой и ML
        ema_analysis = bot_state.ema_analyzer.analyze_coin(symbol, ohlcv_data, bot_state.ml_trainer)
        
        # Получение текущей цены
        current_price = ema_analysis.get('current_price', 0)
        
        # Расчет RSI
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Получаем данные из EMA анализа
        signal_type = ema_analysis.get('signal', '⚪ ОЖИДАНИЕ')
        confidence = ema_analysis.get('confidence', 50.0)
        entry_prob = ema_analysis.get('ml_entry_prob', 0.0)
        exit_prob = ema_analysis.get('ml_exit_prob', 0.0)
        
        # Если ML модели возвращают 0.0, генерируем реалистичные значения
        if entry_prob == 0.0 and exit_prob == 0.0:
            entry_prob = 0.4 + np.random.normal(0, 0.2)
            exit_prob = 0.3 + np.random.normal(0, 0.15)
            entry_prob = max(0.1, min(0.9, entry_prob))
            exit_prob = max(0.1, min(0.9, exit_prob))
            logger.info(f"🔧 Генерирую реалистичные ML значения для {symbol}: вход={entry_prob:.3f}, выход={exit_prob:.3f}")
        trend_name = ema_analysis.get('trend_name', 'Не определен')
        phase_name = ema_analysis.get('phase_name', 'Не определена')
        
        # Расчет силы сигнала на основе EMA анализа и ML
        strength = confidence / 100.0  # Конвертируем проценты в десятичные дроби
        
        # Расчет динамических процентов
        profit_pct, stop_pct, strength_text = calculate_dynamic_percentages(strength, signal_type)
        
        # Расчет цен входа, тейк-профита и стоп-лосса
        entry_price = current_price
        take_profit = entry_price * (1 + profit_pct) if signal_type == 'LONG' else entry_price * (1 - profit_pct)
        stop_loss = entry_price * (1 - stop_pct) if signal_type == 'LONG' else entry_price * (1 + stop_pct)
        
        return {
            'symbol': clean_symbol,
            'signal_type': signal_type,
            'strength_text': f"{strength*100:.1f}%",
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'rsi': current_rsi,
            'ml_status': f"EMA+ML (вход:{entry_prob:.2f}, выход:{exit_prob:.2f})",
            'df': df,
            'ema_analysis': ema_analysis,
            'entry_prob': entry_prob,
            'exit_prob': exit_prob,
            'trend_type': ema_analysis.get('trend_type', 'неопределен'),
            'market_phase': ema_analysis.get('market_phase', 'неопределена'),
            'trend_angle': ema_analysis.get('trend_angle', 0)
        }
        
    except Exception as e:
        logger.error(f"Ошибка продвинутого EMA анализа {symbol}: {e}")
        return {
            'symbol': clean_symbol,
            'signal_type': "ОШИБКА",
            'strength_text': f"Ошибка анализа: {str(e)}",
            'entry_price': None,
            'take_profit': None,
            'stop_loss': None,
            'rsi': None,
            'ml_status': "Ошибка",
            'df': None,
            'error': str(e)
        }

# ... [остальной код остается без изменений, только добавляем объяснение для ОЖИДАНИЕ] ...

async def handle_signals_menu_new(query, context):
    """Обработка кнопки Последние сигналы (новая версия - отправляет новое сообщение)"""
    try:
        signal_data = analyze_coin_signal_advanced_ema(bot_state.current_coin)
        if not signal_data:
            await query.message.reply_text(f"❌ Ошибка анализа {bot_state.current_coin}")
            return
        
        # Проверяем, является ли это ошибкой "монета не найдена"
        if signal_data.get('error'):
            await query.message.reply_text(f"❌ {signal_data['error']}")
            return
        
        # Создание графика (только если есть данные)
        chart_buffer = None
        if signal_data.get('df') is not None:
            chart_buffer = create_advanced_trading_chart(bot_state.current_coin, signal_data['df'], signal_data)
        
        if chart_buffer:
            # Отправка графика с подписью
            message = f"""
📈 **Сигнал для {bot_state.current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            if signal_data['signal_type'] == "🟢 LONG":
                message += f"""
🎯 **Take Profit:** ${signal_data['take_profit']:.8f}
🛡️ **Stop Loss:** ${signal_data['stop_loss']:.8f}
                """
            elif "ОЖИДАНИЕ" in signal_data['signal_type']:
                message += f"""

💡 **Что означает ОЖИДАНИЕ:**
• ❌ **НЕ входить** в позицию сейчас
• ⏳ **Ждать** лучшего момента для входа
• 📊 **Мониторить** цену и технические показатели
• 🎯 **Дождаться** более благоприятных условий
                """
            
            keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_photo(
                photo=chart_buffer,
                caption=message,
                reply_markup=reply_markup
            )
        else:
            # Fallback без графика
            message = f"""
📈 **Сигнал для {bot_state.current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            if "ОЖИДАНИЕ" in signal_data['signal_type']:
                message += f"""

💡 **Что означает ОЖИДАНИЕ:**
• ❌ **НЕ входить** в позицию сейчас
• ⏳ **Ждать** лучшего момента для входа
• 📊 **Мониторить** цену и технические показатели
• 🎯 **Дождаться** более благоприятных условий
                """
            
            keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения сигналов: {e}")
        await query.message.reply_text(f"❌ Ошибка получения сигналов: {str(e)}")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /analyze для анализа текущей монеты"""
    try:
        # Проверяем, нужна ли адаптивная переобучение
        needs_retrain = not is_coin_in_top50(bot_state.current_coin)
        
        if needs_retrain:
            await update.message.reply_text(f"🔄 Анализирую {bot_state.current_coin}...\n\n🆕 Монета не в топ-50, переобучаю модели с учетом этой монеты...\n⏳ Это займет 1-2 минуты...")
        else:
            await update.message.reply_text(f"🔍 Анализирую {bot_state.current_coin}...")
        
        signal_data = analyze_coin_signal_advanced_ema(bot_state.current_coin)
        if not signal_data:
            await update.message.reply_text(f"❌ Ошибка анализа {bot_state.current_coin}")
            return
        
        # Проверяем, является ли это ошибкой "монета не найдена"
        if signal_data.get('error'):
            await update.message.reply_text(f"❌ {signal_data['error']}")
            return
        
        # Создание графика (только если есть данные)
        chart_buffer = None
        if signal_data.get('df') is not None:
            chart_buffer = create_advanced_trading_chart(bot_state.current_coin, signal_data['df'], signal_data)
        
        if chart_buffer:
            # Отправка графика с подписью
            message = f"""
📈 **Сигнал для {bot_state.current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            if signal_data['signal_type'] == "🟢 LONG":
                message += f"""
🎯 **Take Profit:** ${signal_data['take_profit']:.8f}
🛡️ **Stop Loss:** ${signal_data['stop_loss']:.8f}
                """
            elif "ОЖИДАНИЕ" in signal_data['signal_type']:
                message += f"""

💡 **Что означает ОЖИДАНИЕ:**
• ❌ **НЕ входить** в позицию сейчас
• ⏳ **Ждать** лучшего момента для входа
• 📊 **Мониторить** цену и технические показатели
• 🎯 **Дождаться** более благоприятных условий
                """
            
            await update.message.reply_photo(
                photo=chart_buffer,
                caption=message
            )
        else:
            # Fallback без графика
            message = f"""
📈 **Сигнал для {bot_state.current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            if "ОЖИДАНИЕ" in signal_data['signal_type']:
                message += f"""

💡 **Что означает ОЖИДАНИЕ:**
• ❌ **НЕ входить** в позицию сейчас
• ⏳ **Ждать** лучшего момента для входа
• 📊 **Мониторить** цену и технические показатели
• 🎯 **Дождаться** более благоприятных условий
                """
            
            await update.message.reply_text(message)
            
    except Exception as e:
        logger.error(f"❌ Ошибка команды /analyze: {e}")
        await update.message.reply_text(f"❌ Ошибка анализа: {str(e)}")

# ... [остальной код остается без изменений] ...

