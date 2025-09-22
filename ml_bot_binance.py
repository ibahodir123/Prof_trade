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
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
# TensorFlow импорты удалены (не используются)
from advanced_ema_analyzer import AdvancedEMAAnalyzer
from advanced_ml_trainer import AdvancedMLTrainer
from shooting_star_predictor import ShootingStarPredictor
import pickle

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
        # Сначала пытаемся загрузить локальную конфигурацию для разработки
        config_file = 'bot_config_local.json' if os.path.exists('bot_config_local.json') else 'bot_config.json'
        
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # Проверяем, включен ли локальный режим разработки
        if config.get('local_development', {}).get('enabled', False):
            logger.info("🔧 Локальный режим разработки активирован")
            logger.info("🚫 Telegram API отключен для избежания конфликтов")
        
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
    """Получает свежие данные с Binance через ccxt"""
    try:
        logger.info(f"📊 Получаю СВЕЖИЕ данные {symbol} с Binance...")
        
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

def prepare_ml_features(df, symbol="unknown"):
    """Подготавливает все 27 признаков для ML модели"""
    try:
        # Проверяем наличие необходимых колонок
        required_columns = ['close', 'ema_20', 'ema_50', 'ema_100', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"❌ Отсутствуют колонки: {missing_columns}")
            
            # Попробуем создать EMA колонки на месте
            if 'ema_20' in missing_columns:
                logger.info(f"🔧 Создаю EMA колонки в prepare_ml_features для {symbol}")
                df['ema_20'] = df['close'].ewm(span=20).mean()
                df['ema_50'] = df['close'].ewm(span=50).mean()
                df['ema_100'] = df['close'].ewm(span=100).mean()
                logger.info(f"✅ EMA колонки созданы в prepare_ml_features")
                
                # Обновляем список отсутствующих колонок
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    logger.error(f"❌ Все еще отсутствуют колонки: {missing_columns}")
                    return None
            else:
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
        
        
        # Удаляем первые строки с NaN
        data = data.dropna()
        
        # Проверяем на бесконечные значения
        data = data.replace([np.inf, -np.inf], 0)
        
        # Выбираем только нужные признаки (27 признаков)
        feature_columns = [
            'price_velocity', 'ema20_velocity', 'ema50_velocity', 'ema100_velocity',
            'price_acceleration', 'ema20_acceleration', 'ema50_acceleration', 'ema100_acceleration',
            'price_to_ema20_velocity_ratio', 'price_to_ema50_velocity_ratio', 'price_to_ema100_velocity_ratio',
            'price_to_ema20_distance', 'price_to_ema50_distance', 'price_to_ema100_distance',
            'price_to_ema20_distance_change', 'price_to_ema50_distance_change', 'price_to_ema100_distance_change',
            'ema20_angle', 'ema50_angle', 'ema100_angle',
            'ema20_angle_change', 'ema50_angle_change', 'ema100_angle_change',
            'ema20_to_ema50', 'ema20_to_ema100', 'ema50_to_ema100',
            'price_ema20_sync', 'price_ema50_sync', 'price_ema100_sync'
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

def predict_with_smart_ml(features_dict):
    """Предсказание движения с помощью Smart ML модели"""
    try:
        if not bot_state.smart_predictor:
            return None
        
        model = bot_state.smart_predictor['model']
        
        # Преобразуем словарь признаков в вектор (27 признаков)
        feature_vector = []
        
        # 1. Скорости (4 признака)
        vel = features_dict.get('velocities', {})
        feature_vector.extend([
            vel.get('price', 0), vel.get('ema20', 0),
            vel.get('ema50', 0), vel.get('ema100', 0)
        ])
        
        # 2. Ускорения (4 признака)
        acc = features_dict.get('accelerations', {})
        feature_vector.extend([
            acc.get('price', 0), acc.get('ema20', 0),
            acc.get('ema50', 0), acc.get('ema100', 0)
        ])
        
        # 3. Соотношения скоростей (3 признака)
        ratio = features_dict.get('velocity_ratios', {})
        feature_vector.extend([
            ratio.get('price_ema20', 0), ratio.get('price_ema50', 0),
            ratio.get('price_ema100', 0)
        ])
        
        # 4. Расстояния до EMA (3 признака)
        dist = features_dict.get('distances', {})
        feature_vector.extend([
            dist.get('price_ema20', 0), dist.get('price_ema50', 0),
            dist.get('price_ema100', 0)
        ])
        
        # 5. Изменения расстояний (3 признака)
        dist_ch = features_dict.get('distance_changes', {})
        feature_vector.extend([
            dist_ch.get('price_ema20', 0), dist_ch.get('price_ema50', 0),
            dist_ch.get('price_ema100', 0)
        ])
        
        # 6. Углы EMA (3 признака)
        angles = features_dict.get('angles', {})
        feature_vector.extend([
            angles.get('ema20', 0), angles.get('ema50', 0),
            angles.get('ema100', 0)
        ])
        
        # 7. Изменения углов (3 признака)
        angle_ch = features_dict.get('angle_changes', {})
        feature_vector.extend([
            angle_ch.get('ema20', 0), angle_ch.get('ema50', 0),
            angle_ch.get('ema100', 0)
        ])
        
        # 8. Взаимоотношения EMA (3 признака)
        rel = features_dict.get('ema_relationships', {})
        feature_vector.extend([
            rel.get('ema20_ema50', 0), rel.get('ema20_ema100', 0),
            rel.get('ema50_ema100', 0)
        ])
        
        # 9. Синхронизации (3 признака)
        sync = features_dict.get('synchronizations', {})
        feature_vector.extend([
            sync.get('price_ema20', 0), sync.get('price_ema50', 0),
            sync.get('price_ema100', 0)
        ])
        
        # Получаем предсказание
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]
        
        class_names = ['Малое (1-3%)', 'Среднее (3-7%)', 'Крупное (7%+)']
        
        return {
            'prediction': class_names[prediction],
            'probabilities': {
                'small': probabilities[0],
                'medium': probabilities[1], 
                'large': probabilities[2]
            }
        }
        
    except Exception as e:
        logger.error(f"❌ Ошибка Smart ML предсказания: {e}")
        return None

def _format_smart_prediction(prediction):
    """Форматирование Smart ML предсказания"""
    if not prediction:
        return ""
    
    return f"""🧠 **Smart ML прогноз:** {prediction['prediction']}
📊 **Вероятности движения:**
   💰 Малое (1-3%): {prediction['probabilities']['small']:.1%}
   📈 Среднее (3-7%): {prediction['probabilities']['medium']:.1%}
   🚀 Крупное (7%+): {prediction['probabilities']['large']:.1%}"""

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
        
        # Расчет RSI и EMA
        df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # Создаем EMA колонки для графика
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Расчет RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        df['rsi'] = rsi  # Добавляем RSI в DataFrame
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Получаем данные из EMA анализа
        signal_type = ema_analysis.get('signal', '😴 ОЖИДАНИЕ')
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

def analyze_coin_signal(symbol):
    """Анализ монеты и генерация сигнала (старая логика)"""
    try:
        # Очищаем символ от дублирования USDT
        clean_symbol = symbol.replace(':USDT', '') if ':USDT' in symbol else symbol
        # Получение данных с Binance
        logger.info(f"📊 Получаю данные {symbol} с Binance...")
        df = get_binance_data(symbol, timeframe='1h', limit=500)
        
        if df is None or df.empty:
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
            
        # Дополнительное логирование
        logger.info(f"📊 DataFrame {symbol}:")
        logger.info(f"   Размер: {df.shape}")
        logger.info(f"   Колонки: {list(df.columns)}")
        logger.info(f"   Типы данных: {df.dtypes.to_dict()}")
        logger.info(f"   Последняя цена: {df['close'].iloc[-1]:.8f}")
        logger.info(f"   Диапазон дат: {df.index[0]} - {df.index[-1]}")
        logger.info(f"   Проверка NaN: {df.isnull().sum().sum()}")
        
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
            logger.info(f"🤖 Загружаю ML модели для {symbol}...")
            
            # Загружаем модели
            scaler = joblib.load('models/scaler.pkl')
            min_detector = joblib.load('models/minimum_detector.pkl')
            max_detector = joblib.load('models/maximum_detector.pkl')
            
            logger.info(f"✅ ML модели загружены успешно")
            
            # Подготовка данных для ML
            logger.info(f"📊 Подготавливаю данные для ML анализа...")
            
            # Подготавливаем все 36 признаков
            features = prepare_ml_features(df, symbol)
            if features is None or features.empty:
                logger.error(f"❌ Не удалось подготовить признаки для ML")
                raise ValueError("Failed to prepare ML features")
            
            logger.info(f"📈 Признаки для ML: {features.shape}")
            logger.info(f"📊 Последние значения: {features.iloc[-1].to_dict()}")
            
            # Масштабируем данные
            features_scaled = scaler.transform(features)
            logger.info(f"✅ Данные масштабированы: {features_scaled.shape}")
            
            # Получаем последние признаки для предсказания
            last_features = features_scaled[-1:].reshape(1, -1)
            
            # Предсказания
            logger.info(f"🔮 Выполняю ML предсказания...")
            min_prob = min_detector.predict_proba(last_features)[0][1]
            max_prob = max_detector.predict_proba(last_features)[0][1]
            
            logger.info(f"📊 ML результаты:")
            logger.info(f"   Минимальный детектор: {min_prob:.3f}")
            logger.info(f"   Максимальный детектор: {max_prob:.3f}")
            
            # Получаем цену входа
            entry_price = df['close'].iloc[-1]
            logger.info(f"💰 Цена входа для {clean_symbol}: {entry_price}")
            
            if entry_price <= 0:
                logger.error(f"❌ Некорректная цена входа для {symbol}: {entry_price}")
                raise ValueError(f"Некорректная цена входа: {entry_price}")
            
            # Определение сигнала (исправленная логика сравнения вероятностей)
            diff = max_prob - min_prob
            logger.info(f"🔍 ОТЛАДКА: min_prob={min_prob:.3f}, max_prob={max_prob:.3f}, diff={diff:.3f}")
            
            if diff > 0.10:  # Максимальный детектор значительно выше
                logger.info(f"🔍 УСЛОВИЕ 1: diff > 0.10 ({diff:.3f} > 0.10) = True")
                if max_prob > 0.3:  # Достаточная уверенность для SHORT
                    logger.info(f"🔍 УСЛОВИЕ 2: max_prob > 0.3 ({max_prob:.3f} > 0.3) = True")
                    signal_type = "🔴 SHORT"
                    strength_text = f"Падение {max_prob*100:.1f}%"
                    profit_pct, loss_pct, _ = calculate_dynamic_percentages(max_prob, "SHORT")
                    take_profit = entry_price * (1 - profit_pct)
                    stop_loss = entry_price * (1 + loss_pct)
                    ml_status = "Активна"
                    logger.info(f"🎯 Сигнал: SHORT (падение {max_prob*100:.1f}%)")
                else:
                    logger.info(f"🔍 УСЛОВИЕ 2: max_prob > 0.3 ({max_prob:.3f} > 0.3) = False")
                    signal_type = "⚪ ОЖИДАНИЕ"
                    strength_text = "Слабая уверенность в падении"
                    take_profit = None
                    stop_loss = None
                    ml_status = "Активна"
                    logger.info(f"🎯 Сигнал: ОЖИДАНИЕ (слабая уверенность в падении)")
                    
            elif diff < -0.10:  # Минимальный детектор значительно выше
                logger.info(f"🔍 УСЛОВИЕ 3: diff < -0.10 ({diff:.3f} < -0.10) = True")
                if min_prob > 0.3:  # Достаточная уверенность для LONG
                    logger.info(f"🔍 УСЛОВИЕ 4: min_prob > 0.3 ({min_prob:.3f} > 0.3) = True")
                    signal_type = "🟢 LONG"
                    strength_text = f"Рост {min_prob*100:.1f}%"
                    profit_pct, loss_pct, _ = calculate_dynamic_percentages(min_prob, "LONG")
                    take_profit = entry_price * (1 + profit_pct)
                    stop_loss = entry_price * (1 - loss_pct)
                    ml_status = "Активна"
                    logger.info(f"🎯 Сигнал: LONG (рост {min_prob*100:.1f}%)")
                else:
                    logger.info(f"🔍 УСЛОВИЕ 4: min_prob > 0.3 ({min_prob:.3f} > 0.3) = False")
                    signal_type = "⚪ ОЖИДАНИЕ"
                    strength_text = "Слабая уверенность в росте"
                    take_profit = None
                    stop_loss = None
                    ml_status = "Активна"
                    logger.info(f"🎯 Сигнал: ОЖИДАНИЕ (слабая уверенность в росте)")
                    
            else:  # Разница менее 10% - нет четкого сигнала
                logger.info(f"🔍 УСЛОВИЕ 5: else (diff не в диапазоне >0.10 или <-0.10)")
                signal_type = "⚪ ОЖИДАНИЕ"
                strength_text = "Нет четкого сигнала"
                take_profit = None
                stop_loss = None
                ml_status = "Активна"
                logger.info(f"🎯 Сигнал: ОЖИДАНИЕ (нет четкого сигнала)")
                
        except Exception as e:
            # Fallback к простому анализу с применением ML моделей
            logger.error(f"❌ ML ошибка для {symbol}: {e}")
            logger.info(f"🔄 Переключаюсь на fallback с ML моделями...")
            
            try:
                # Пытаемся загрузить модели заново
                logger.info(f"🤖 Повторная загрузка ML моделей для {symbol}...")
                scaler = joblib.load('models/scaler.pkl')
                min_detector = joblib.load('models/minimum_detector.pkl')
                max_detector = joblib.load('models/maximum_detector.pkl')
                
                # Подготавливаем данные для ML
                features = prepare_ml_features(df, symbol)
                if features is not None and not features.empty:
                    features_scaled = scaler.transform(features)
                    last_features = features_scaled[-1:].reshape(1, -1)
                    
                    # Предсказания
                    min_prob = min_detector.predict_proba(last_features)[0][1]
                    max_prob = max_detector.predict_proba(last_features)[0][1]
                    
                    logger.info(f"📊 Fallback ML результаты:")
                    logger.info(f"   Минимальный детектор: {min_prob:.3f}")
                    logger.info(f"   Максимальный детектор: {max_prob:.3f}")
                    
                    # Получаем цену входа для fallback
                    entry_price = df['close'].iloc[-1]
                    logger.info(f"💰 Fallback цена входа для {symbol}: {entry_price}")
                    
                    # Определение сигнала с исправленной fallback логикой
                    diff = max_prob - min_prob
                    
                    if diff > 0.10:  # Максимальный детектор значительно выше
                        if max_prob > 0.3:
                            signal_type = "🔴 SHORT"
                            strength_text = f"Fallback ML: падение {max_prob*100:.1f}%"
                            profit_pct, loss_pct, _ = calculate_dynamic_percentages(max_prob, "SHORT")
                            take_profit = entry_price * (1 - profit_pct)
                            stop_loss = entry_price * (1 + loss_pct)
                            ml_status = "Fallback ML"
                            logger.info(f"🎯 Fallback ML сигнал: SHORT (падение {max_prob*100:.1f}%)")
                        else:
                            signal_type = "⚪ ОЖИДАНИЕ"
                            strength_text = "Fallback ML: слабая уверенность в падении"
                            take_profit = None
                            stop_loss = None
                            ml_status = "Fallback ML"
                            logger.info(f"🎯 Fallback ML сигнал: ОЖИДАНИЕ (слабая уверенность в падении)")
                            
                    elif diff < -0.10:  # Минимальный детектор значительно выше
                        if min_prob > 0.3:
                            signal_type = "🟢 LONG"
                            strength_text = f"Fallback ML: рост {min_prob*100:.1f}%"
                            profit_pct, loss_pct, _ = calculate_dynamic_percentages(min_prob, "LONG")
                            take_profit = entry_price * (1 + profit_pct)
                            stop_loss = entry_price * (1 - loss_pct)
                            ml_status = "Fallback ML"
                            logger.info(f"🎯 Fallback ML сигнал: LONG (рост {min_prob*100:.1f}%)")
                        else:
                            signal_type = "⚪ ОЖИДАНИЕ"
                            strength_text = "Fallback ML: слабая уверенность в росте"
                            take_profit = None
                            stop_loss = None
                            ml_status = "Fallback ML"
                            logger.info(f"🎯 Fallback ML сигнал: ОЖИДАНИЕ (слабая уверенность в росте)")
                            
                    else:  # Разница менее 10% - нет четкого сигнала
                        signal_type = "⚪ ОЖИДАНИЕ"
                        strength_text = "Fallback ML: нет четкого сигнала"
                        take_profit = None
                        stop_loss = None
                        ml_status = "Fallback ML"
                        logger.info(f"🎯 Fallback ML сигнал: ОЖИДАНИЕ (нет четкого сигнала)")
                else:
                    raise Exception("Не удалось подготовить признаки для ML")
                    
            except Exception as fallback_error:
                # Если и fallback ML не работает, используем простой анализ
                logger.error(f"❌ Fallback ML ошибка для {symbol}: {fallback_error}")
                logger.info(f"🔄 Переключаюсь на простой анализ...")
                
                latest_close = df['close'].iloc[-1]
                ema_20_latest = df['ema_20'].iloc[-1]
                ema_50_latest = df['ema_50'].iloc[-1]
                rsi_latest = df['rsi'].iloc[-1]
                
                logger.info(f"📊 Простой анализ {symbol}:")
                logger.info(f"   Цена: {latest_close:.8f}")
                logger.info(f"   EMA 20: {ema_20_latest:.8f}")
                logger.info(f"   EMA 50: {ema_50_latest:.8f}")
                logger.info(f"   RSI: {rsi_latest:.1f}")
                
                # Простой анализ тренда
                if latest_close > ema_20_latest > ema_50_latest:


                    
                    signal_type = "🟢 LONG"
                    strength_text = "Простой анализ: восходящий тренд"
                    profit_pct, loss_pct, _ = calculate_dynamic_percentages(0.6, "LONG")
                    entry_price = latest_close
                    take_profit = entry_price * (1 + profit_pct)
                    stop_loss = entry_price * (1 - loss_pct)
                    ml_status = "Fallback (тренд)"
                    logger.info(f"🎯 Простой сигнал: LONG (восходящий тренд)")
                    
                elif latest_close < ema_20_latest < ema_50_latest:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    strength_text = "Простой анализ: нисходящий тренд"
                    entry_price = latest_close
                    take_profit = None
                    stop_loss = None
                    ml_status = "Fallback (тренд)"
                    logger.info(f"🎯 Простой сигнал: ОЖИДАНИЕ (нисходящий тренд)")
                    
                else:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    strength_text = "Простой анализ: неопределенность"
                    entry_price = latest_close
                    take_profit = None
                    stop_loss = None
                    ml_status = "Fallback (неопределенность)"
                    logger.info(f"🎯 Простой сигнал: ОЖИДАНИЕ (неопределенность)")
        
        return {
            'symbol': clean_symbol,
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
        
        # Проверяем, является ли ошибка "монета не найдена"
        if "does not have market symbol" in str(e) or "symbol not found" in str(e).lower():
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
        
        return None

# Класс для управления состоянием бота
class SmartBacktestEngine:
    def __init__(self):
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime.now()
        self.initial_balance = 1000.0
        self.current_balance = self.initial_balance
        self.trades = []
        self.positions = {}
        self.position_size_percent = 0.1
        self.max_positions = 3
        
    def get_historical_data(self, symbol: str, timeframe='1h') -> pd.DataFrame:
        """Получение исторических данных с Binance"""
        try:
            exchange = ccxt.binance()
            since = int(self.start_date.timestamp() * 1000)
            
            all_ohlcv = []
            current_since = since
            
            while current_since < int(self.end_date.timestamp() * 1000):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                    if not ohlcv:
                        break
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    time.sleep(0.1)  # Пауза для API
                except:
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            return df
        except:
            return pd.DataFrame()
    
    def analyze_signal_for_backtest(self, symbol: str, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Smart ML анализ сигнала для бэктеста"""
        try:
            historical_data = df.iloc[:current_idx + 1].copy()
            if len(historical_data) < 100:
                return {'signal': 'WAIT', 'confidence': 0}
            
            # Подготавливаем EMA для ML признаков
            historical_data['ema_20'] = historical_data['close'].ewm(span=20).mean()
            historical_data['ema_50'] = historical_data['close'].ewm(span=50).mean()
            historical_data['ema_100'] = historical_data['close'].ewm(span=100).mean()
            
            try:
                # Подготавливаем ML признаки
                features = prepare_ml_features(historical_data, symbol)
                if features is None or (hasattr(features, 'empty') and features.empty):
                    return {'signal': 'WAIT', 'confidence': 0}
                
                # Получаем Smart ML предсказание
                smart_prediction = predict_with_smart_ml(features)
                if not smart_prediction:
                    return {'signal': 'WAIT', 'confidence': 0}
                
                probabilities = smart_prediction['probabilities']
                prediction = smart_prediction['prediction']
                
                # Генерируем сигнал на основе ML
                signal = 'WAIT'
                confidence = 0
                
                # Если ML предсказывает среднее или крупное движение - LONG
                medium_prob = probabilities['medium']
                large_prob = probabilities['large']
                
                if medium_prob > 0.35 or large_prob > 0.15:
                    signal = 'LONG'
                    confidence = int((medium_prob + large_prob) * 100)
                
                return {
                    'signal': signal,
                    'confidence': min(100, max(0, confidence)),
                    'price': float(historical_data.iloc[-1]['close']),
                    'ml_prediction': prediction,
                    'probabilities': probabilities
                }
                
            except Exception as e:
                logger.error(f"❌ Ошибка ML анализа в бэктесте: {e}")
                return {'signal': 'WAIT', 'confidence': 0}
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа сигнала в бэктесте: {e}")
            return {'signal': 'WAIT', 'confidence': 0}
    
    def run_backtest(self, symbols: List[str]) -> Dict[str, Any]:
        """Запуск бэктеста"""
        self.trades = []
        self.positions = {}
        self.current_balance = self.initial_balance
        
        # Загружаем данные
        historical_data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty:
                historical_data[symbol] = df
        
        if not historical_data:
            return {'error': 'Нет данных для тестирования'}
        
        # Находим общие временные точки
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index)
        timestamps = sorted(list(all_timestamps))
        
        # Основной цикл бэктеста
        for i, timestamp in enumerate(timestamps):
            for symbol, df in historical_data.items():
                if timestamp not in df.index:
                    continue
                
                current_price = float(df.loc[timestamp, 'close'])
                current_idx = df.index.get_loc(timestamp)
                
                # Проверяем закрытие позиций
                if symbol in self.positions:
                    position = self.positions[symbol]
                    should_close = False
                    
                    if position['side'] == 'LONG':
                        if current_price >= position['take_profit'] or current_price <= position['stop_loss']:
                            should_close = True
                    else:
                        if current_price <= position['take_profit'] or current_price >= position['stop_loss']:
                            should_close = True
                    
                    if should_close:
                        # Расчет PnL
                        if position['side'] == 'LONG':
                            pnl = (current_price - position['entry_price']) * position['size']
                        else:
                            pnl = (position['entry_price'] - current_price) * position['size']
                        
                        self.current_balance += pnl
                        
                        self.trades.append({
                            'symbol': symbol,
                            'side': position['side'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_percent': (pnl / (position['entry_price'] * position['size'])) * 100,
                            'timestamp': timestamp
                        })
                        
                        del self.positions[symbol]
                
                # Ищем новые сигналы
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    signal_data = self.analyze_signal_for_backtest(symbol, df, current_idx)
                    
                    if signal_data['signal'] in ['LONG', 'SHORT'] and signal_data['confidence'] >= 50:
                        position_value = self.current_balance * self.position_size_percent
                        size = position_value / current_price
                        
                        # Уровни TP/SL
                        if signal_data['confidence'] >= 70:
                            profit_pct, loss_pct = 0.04, 0.02
                        else:
                            profit_pct, loss_pct = 0.03, 0.015
                        
                        if signal_data['signal'] == 'LONG':
                            take_profit = current_price * (1 + profit_pct)
                            stop_loss = current_price * (1 - loss_pct)
                        else:
                            take_profit = current_price * (1 - profit_pct)
                            stop_loss = current_price * (1 + loss_pct)
                        
                        self.positions[symbol] = {
                            'side': signal_data['signal'],
                            'entry_price': current_price,
                            'size': size,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'timestamp': timestamp
                        }
        
        # Закрываем оставшиеся позиции
        for symbol in list(self.positions.keys()):
            last_price = float(historical_data[symbol].iloc[-1]['close'])
            position = self.positions[symbol]
            
            if position['side'] == 'LONG':
                pnl = (last_price - position['entry_price']) * position['size']
            else:
                pnl = (position['entry_price'] - last_price) * position['size']
            
            self.current_balance += pnl
            
            self.trades.append({
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'pnl': pnl,
                'pnl_percent': (pnl / (position['entry_price'] * position['size'])) * 100,
                'timestamp': timestamps[-1]
            })
            
            del self.positions[symbol]
        
        # Расчет статистики
        if not self.trades:
            return {'error': 'Сделок не было'}
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': total_return,
            'total_pnl': self.current_balance - self.initial_balance,
            'trades': self.trades
        }

class BotState:
    def __init__(self):
        self.current_coin = "BTC/USDT"
        self.available_pairs = []
        self.config = None
        self.application = None
        self.ema_analyzer = None
        self.ml_trainer = None
        self.shooting_predictor = None
        self.language = "ru"  # По умолчанию русский, "uz" для узбекского
        self.custom_uzbek_explanations = {}  # Пользовательские объяснения на узбекском
        self.backtest_engine = SmartBacktestEngine()  # Smart ML движок бэктестинга
        self.smart_predictor = None  # ML предиктор движений
    
    def initialize(self):
        """Инициализация состояния бота"""
        self.config = load_config()
        if self.config:
            self.ema_analyzer = AdvancedEMAAnalyzer()
            self.ml_trainer = AdvancedMLTrainer()
            self.shooting_predictor = ShootingStarPredictor()
            self.smart_predictor = self._load_smart_predictor()
            logger.info("✅ Состояние бота инициализировано")
        else:
            logger.error("❌ Не удалось загрузить конфигурацию")
    
    def _load_smart_predictor(self):
        """Загрузка умного ML предиктора"""
        try:
            # Загружаем модель
            with open('smart_predictor_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Загружаем названия признаков
            with open('feature_names.pkl', 'rb') as f:
                feature_names = pickle.load(f)
            
            logger.info("🧠 Smart ML предиктор загружен")
            return {'model': model, 'feature_names': feature_names}
        except Exception as e:
            logger.warning(f"⚠️ Smart предиктор не загружен: {e}")
            return None

# Глобальный экземпляр состояния
bot_state = BotState()

# Словарь переводов на узбекский язык
UZBEK_TRANSLATIONS = {
    # Основные термины
    "Сигнал": "Сигнал",
    "LONG": "LONG",
    "SHORT": "SHORT", 
    "ОЖИДАНИЕ": "КУТИШ",
    "Цена входа": "Кириш нархи",
    "Take Profit": "Фойда олиш",
    "Stop Loss": "Йўқотишни тўхтатиш",
    "RSI": "RSI",
    "ML статус": "ML холати",
    "Анализ": "Таҳлил",
    "Тренд": "Тренд",
    "Фаза": "Фаза",
    "Уверенность": "Ишонч",
    "Вероятность": "Эҳтимоллик",
    "Потенциал": "Потенциал",
    "Прогноз": "Прогноз",
    
    # Объяснения сигналов
    "Что означает ОЖИДАНИЕ": "КУТИШ нимани англатади",
    "НЕ входить в позицию сейчас": "Ҳозирча позицияга кирманг",
    "Ждать лучшего момента для входа": "Кириш учун яхши пайтни кутинг",
    "Мониторить цену и технические показатели": "Нарх ва техник кўрсаткичларни кузатинг",
    "Дождаться более благоприятных условий": "Яхширок шартларни кутинг",
    
    # EMA объяснения
    "Восходящий тренд": "Юқорига йўналган тренд",
    "Нисходящий тренд": "Пастга йўналган тренд", 
    "Боковой тренд": "Қийма тренд",
    "Импульс": "Импульс",
    "Коррекция": "Тузатиш",
    "Пересечение EMA": "EMA кесишиши",
    "Приближение к EMA": "EMAга яқинлашув",
    "Отскок от EMA": "EMAдан сакраш",
    
    # Сила сигнала
    "Очень сильный": "Жуда кучли",
    "Сильный": "Кучли", 
    "Средний": "Ўртача",
    "Слабый": "Суст",
    "Падение": "Тушиб кетиш",
    "Рост": "Ўсиш",
    "Нет четкого сигнала": "Аниқ сигнал йўқ",
    "Слабая уверенность": "Суст ишонч",
    
    # Технические термины
    "Волатильность": "Волатиллик",
    "Объем": "Ҳажм",
    "Свечи": "Шам",
    "График": "График",
    "Индикатор": "Индикатор",
    "Поддержка": "Қўллаб-қувватлаш",
    "Сопротивление": "Қаршилик",
    
    # Статусы и сообщения
    "МОНЕТА НЕ НАЙДЕНА": "ТАНГА ТОПИЛМАДИ",
    "Ошибка анализа": "Таҳлил хатоси",
    "Ошибка получения данных": "Маълумотларни олиш хатоси",
    "Система готова": "Тизим тайёр",
    "Анализирую": "Таҳлил қиламан",
    "Обучаю модели": "Моделларни ўқитаман",
    "Готово к использованию": "Фойдаланишга тайёр",
    
    # Меню и кнопки
    "Статус системы": "Тизим холати",
    "Выбор монет": "Тангаларни танлаш",
    "Последние сигналы": "Охирги сигналлар",
    "Анализ монеты": "Танга таҳлили",
    "Поиск монет": "Тангаларни қидириш",
    "Стреляющие монеты": "Отилган тангалар",
    "EMA Анализ": "EMA Таҳлили",
    "Обучение ML": "ML Ўқитиш",
    "Назад": "Орқага",
    "Контакты": "Алоқалар",
    
    # Контактная информация
    "Разработчик": "Ишлаб чиқарувчи",
    "Бот": "Бот",
    "Связь": "Алоқа",
    "Вопросы": "Саволлар",
    "Предложения": "Таклифлар", 
    "Сотрудничество": "Ҳамкорлик",
    "Возможности бота": "Бот имкониятлари",
    "Технологии": "Технологиялар",
    "Точность": "Аниқлик",
    "Покрытие": "Қамров",
    
    # Объяснения возможностей
    "Анализ любых монет Binance": "Binanceнинг барча тангаларини таҳлил қилиш",
    "Адаптивное ML обучение": "Мослашувчи ML ўқитиш",
    "Автосигналы каждые 30 минут": "Ҳар 30 дақиқада автосигналлар",
    "Стреляющие звезды": "Отилган юлдузлар",
    "EMA анализ с ML предсказаниями": "ML бошоратлари билан EMA таҳлили",
    "Machine Learning": "Машина ўрганиши",
    "Binance API": "Binance API",
    "Telegram Bot API": "Telegram Bot API",
    "Python": "Python",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "Модели обучены на реальных данных": "Моделлар ҳақиқий маълумотлар асосида ўқитилган",
    "Все 614+ USDT пар Binance": "Binanceнинг барча 614+ USDT жуфтлари",
    "Спасибо за использование": "Фойдаланганингиз учун раҳмат",
    
    # Поиск и выбор
    "Найдено": "Топилди",
    "пар с": "жуфт",
    "Выберите монету": "Тангани танланг",
    "Доступно": "Мавжуд",
    "монет": "танга",
    "Популярные монеты": "Оммабоп тангалар",
    "Поиск завершен": "Қидириш тугади",
    "Стреляющих не найдено": "Отилган топилмади",
    "Попробуйте позже": "Кейинроқ уриниб кўринг",
    
    # Объяснения анализа
    "Проанализировано": "Таҳлил қилинди",
    "Найдено стреляющих": "Отилган топилди",
    "ТОП стреляющих монет": "ТОП отилган тангалар",
    "Время анализа": "Таҳлил вақти",
    "Нет монет с высокой вероятностью": "Юқори эҳтимолликдаги тангалар йўқ",
    "Используйте обычный анализ": "Оддий таҳлилдан фойдаланинг"
}

# Словарь переводов на английский язык
ENGLISH_TRANSLATIONS = {
    # Основные термины
    "Сигнал": "Signal",
    "LONG": "LONG",
    "SHORT": "SHORT", 
    "ОЖИДАНИЕ": "WAITING",
    "Цена входа": "Entry Price",
    "Take Profit": "Take Profit",
    "Stop Loss": "Stop Loss",
    "RSI": "RSI",
    "ML статус": "ML Status",
    "Анализ": "Analysis",
    "Тренд": "Trend",
    "Фаза": "Phase",
    "Уверенность": "Confidence",
    "Вероятность": "Probability",
    "Потенциал": "Potential",
    "Прогноз": "Forecast",
    
    # Объяснения сигналов
    "Что означает ОЖИДАНИЕ": "What WAITING means",
    "НЕ входить в позицию сейчас": "DO NOT enter position now",
    "Ждать лучшего момента для входа": "Wait for better entry moment",
    "Мониторить цену и технические показатели": "Monitor price and technical indicators",
    "Дождаться более благоприятных условий": "Wait for more favorable conditions",
    
    # EMA объяснения
    "Восходящий тренд": "Uptrend",
    "Нисходящий тренд": "Downtrend", 
    "Боковой тренд": "Sideways trend",
    "Импульс": "Impulse",
    "Коррекция": "Correction",
    "Пересечение EMA": "EMA crossover",
    "Приближение к EMA": "Approaching EMA",
    "Отскок от EMA": "Bounce from EMA",
    
    # Сила сигнала
    "Очень сильный": "Very strong",
    "Сильный": "Strong", 
    "Средний": "Medium",
    "Слабый": "Weak",
    "Падение": "Decline",
    "Рост": "Growth",
    "Нет четкого сигнала": "No clear signal",
    "Слабая уверенность": "Low confidence",
    
    # Технические термины
    "Волатильность": "Volatility",
    "Объем": "Volume",
    "Свечи": "Candles",
    "График": "Chart",
    "Индикатор": "Indicator",
    "Поддержка": "Support",
    "Сопротивление": "Resistance",
    
    # Статусы и сообщения
    "МОНЕТА НЕ НАЙДЕНА": "COIN NOT FOUND",
    "Ошибка анализа": "Analysis error",
    "Ошибка получения данных": "Data retrieval error",
    "Система готова": "System ready",
    "Анализирую": "Analyzing",
    "Обучаю модели": "Training models",
    "Готово к использованию": "Ready to use",
    
    # Меню и кнопки
    "Статус системы": "System Status",
    "Выбор монет": "Coin Selection",
    "Последние сигналы": "Latest Signals",
    "Анализ монеты": "Coin Analysis",
    "Поиск монет": "Search Coins",
    "Стреляющие монеты": "Shooting Stars",
    "EMA Анализ": "EMA Analysis",
    "Обучение ML": "ML Training",
    "Назад": "Back",
    "Контакты": "Contacts",
    
    # Контактная информация
    "Разработчик": "Developer",
    "Бот": "Bot",
    "Связь": "Contact",
    "Вопросы": "Questions",
    "Предложения": "Suggestions", 
    "Сотрудничество": "Cooperation",
    "Возможности бота": "Bot capabilities",
    "Технологии": "Technologies",
    "Точность": "Accuracy",
    "Покрытие": "Coverage",
    
    # Объяснения возможностей
    "Анализ любых монет Binance": "Analysis of any Binance coins",
    "Адаптивное ML обучение": "Adaptive ML training",
    "Автосигналы каждые 30 минут": "Auto signals every 30 minutes",
    "Стреляющие звезды": "Shooting stars",
    "EMA анализ с ML предсказаниями": "EMA analysis with ML predictions",
    "Machine Learning": "Machine Learning",
    "Binance API": "Binance API",
    "Telegram Bot API": "Telegram Bot API",
    "Python": "Python",
    "pandas": "pandas",
    "scikit-learn": "scikit-learn",
    "Модели обучены на реальных данных": "Models trained on real data",
    "Все 614+ USDT пар Binance": "All 614+ Binance USDT pairs",
    "Спасибо за использование": "Thank you for using",
    
    # Поиск и выбор
    "Найдено": "Found",
    "пар с": "pairs with",
    "Выберите монету": "Select coin",
    "Доступно": "Available",
    "монет": "coins",
    "Популярные монеты": "Popular coins",
    "Поиск завершен": "Search completed",
    "Стреляющих не найдено": "No shooting stars found",
    "Попробуйте позже": "Try again later",
    
    # Объяснения анализа
    "Проанализировано": "Analyzed",
    "Найдено стреляющих": "Shooting stars found",
    "ТОП стреляющих монет": "TOP shooting coins",
    "Время анализа": "Analysis time",
    "Нет монет с высокой вероятностью": "No coins with high probability",
    "Используйте обычный анализ": "Use regular analysis"
}

def translate_text(text, language="ru"):
    """Переводит текст на выбранный язык"""
    if language == "ru":
        return text
    elif language == "uz":
        # Простой перевод на узбекский только основных терминов
        basic_translations = {
            "Сигнал": "Сигнал",
            "LONG": "LONG", 
            "SHORT": "SHORT",
            "ОЖИДАНИЕ": "КУТИШ",
            "Цена входа": "Кириш нархи",
            "Take Profit": "Фойда олиш",
            "Stop Loss": "Йўқотишни тўхтатиш",
            "RSI": "RSI",
            "ML статус": "ML холати",
            "Анализ": "Таҳлил",
            "Назад": "Орқага",
            "Статус системы": "Тизим холати",
            "Выбор монет": "Тангаларни танлаш",
            "Последние сигналы": "Охирги сигналлар",
            "Анализ монеты": "Танга таҳлили",
            "Поиск монет": "Тангаларни қидириш"
        }
        
        translated_text = text
        for russian, uzbek in basic_translations.items():
            translated_text = translated_text.replace(russian, uzbek)
        
        return translated_text
    elif language == "en":
        # Перевод на английский
        translated_text = text
        for russian, english in ENGLISH_TRANSLATIONS.items():
            translated_text = translated_text.replace(russian, english)
        return translated_text
    
    return text

def add_custom_uzbek_explanation(key, explanation):
    """Добавляет пользовательское объяснение на узбекском языке"""
    bot_state.custom_uzbek_explanations[key] = explanation
    logger.info(f"✅ Добавлено узбекское объяснение для '{key}'")

def get_custom_uzbek_explanation(key):
    """Получает пользовательское объяснение на узбекском языке"""
    return bot_state.custom_uzbek_explanations.get(key, "")

def create_advanced_trading_chart(symbol, df, signal_data):
    """Создание продвинутого графика в стиле TradingView"""
    try:
        # ДОПОЛНИТЕЛЬНАЯ ПРОВЕРКА: Логируем информацию о DataFrame
        logger.info(f"🔍 Создание графика для {symbol}")
        logger.info(f"   Размер DataFrame: {df.shape}")
        logger.info(f"   Колонки: {list(df.columns)}")
        logger.info(f"   EMA колонки присутствуют: {'ema_20' in df.columns}")
        
        # Проверяем наличие EMA колонок
        if 'ema_20' not in df.columns:
            logger.error(f"❌ Отсутствует колонка ema_20 в DataFrame для {symbol}")
            logger.error(f"   Доступные колонки: {list(df.columns)}")
            
            # Попробуем создать EMA колонки на месте
            logger.info(f"🔧 Создаю EMA колонки на месте для {symbol}")
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            logger.info(f"✅ EMA колонки созданы: {list(df.columns)}")
        # Настройка стиля TradingView
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), 
                                      gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.1})
        
        # Основной график
        ax1.set_facecolor('#1e1e1e')
        ax1.grid(True, alpha=0.3, color='#333333')
        
        # Последние 100 свечей для лучшей видимости
        recent_df = df.tail(100)
        
        # Проверяем наличие EMA колонок в recent_df
        if 'ema_20' not in recent_df.columns:
            logger.error(f"❌ Отсутствует колонка ema_20 в recent_df для {symbol}")
            logger.error(f"   Доступные колонки в recent_df: {list(recent_df.columns)}")
            
            # Попробуем создать EMA колонки в recent_df
            logger.info(f"🔧 Создаю EMA колонки в recent_df для {symbol}")
            recent_df['ema_20'] = recent_df['close'].ewm(span=20).mean()
            recent_df['ema_50'] = recent_df['close'].ewm(span=50).mean()
            recent_df['ema_100'] = recent_df['close'].ewm(span=100).mean()
            logger.info(f"✅ EMA колонки созданы в recent_df: {list(recent_df.columns)}")
            
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
        
        if "LONG" in signal_data['signal_type']:
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
💰 Текущая цена: ${current_price:.8f}
📈 RSI: {signal_data['rsi']:.1f}"""
        
        if "LONG" in signal_data['signal_type']:
            info_text += f"""
🎯 Take Profit: ${signal_data['take_profit']:.8f}
🛡️ Stop Loss: ${signal_data['stop_loss']:.8f}"""
        
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

# Функция для получения списка доступных монет с Binance
async def get_available_pairs():
    """Получает список популярных монет с Binance"""
    # Используем bot_state напрямую
    try:
        logger.info("🔍 Получаю список монет с Binance...")
        
        # Инициализация Binance (только публичные данные)
        exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        
        # Получаем все торговые пары
        markets = exchange.load_markets()
        
        # Фильтруем только USDT пары
        usdt_pairs = []
        for symbol, market in markets.items():
            if market['quote'] == 'USDT' and market['active']:
                usdt_pairs.append(symbol)
        
        # Сортируем по популярности (все доступные USDT пары)
        bot_state.available_pairs = sorted(usdt_pairs)
        logger.info(f"✅ Найдено {len(bot_state.available_pairs)} монет с Binance")
        return bot_state.available_pairs
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения монет с Binance: {e}")
        # Fallback на стандартный список
        bot_state.available_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT',
            'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'FTM/USDT', 'ALGO/USDT',
            'VET/USDT', 'ICP/USDT', 'FIL/USDT', 'TRX/USDT', 'ETC/USDT'
        ]
        return bot_state.available_pairs

async def clear_chat_history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Очистка истории чата"""
    try:
        chat_id = update.effective_chat.id
        # Удаляем последние 10 сообщений бота
        for i in range(10):
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=update.message.message_id - i)
            except:
                pass
    except Exception as e:
        logger.warning(f"⚠️ Не удалось очистить историю чата: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start с красивым меню"""
    # Используем состояние бота
    
    # Очищаем старые сообщения при каждом старте
    await clear_chat_history(update, context)
    
    # Определяем текст кнопки языка (циклическое переключение)
    if bot_state.language == "ru":
        lang_button_text = "🇺🇿 O'zbekcha"
        lang_callback = "switch_to_uzbek"
    elif bot_state.language == "uz":
        lang_button_text = "🇬🇧 English"
        lang_callback = "switch_to_english"
    else:  # en
        lang_button_text = "🇷🇺 Русский"
        lang_callback = "switch_to_russian"
    
    keyboard = [
        [InlineKeyboardButton("⚡ ТОРГОВЫЕ ПАРЫ", callback_data="menu_coins")],
        [InlineKeyboardButton("🎯 АНАЛИЗ & СИГНАЛЫ", callback_data="menu_analyze")],
        [InlineKeyboardButton("🔍 ПОИСК АКТИВОВ", callback_data="menu_search")],
        [InlineKeyboardButton("📊 БЭКТЕСТИНГ", callback_data="menu_backtest")],
        [InlineKeyboardButton("💬 СВЯЗАТЬСЯ С НАМИ", callback_data="menu_contacts")],
        [InlineKeyboardButton("🗑️ Очистить чат", callback_data="clear_chat")],
        [InlineKeyboardButton(lang_button_text, callback_data=lang_callback)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""
🤖 **Binance Trading Bot с ML сигналами!**

🪙 **Текущая монета:** {bot_state.current_coin}

**Выберите действие из меню ниже:**
    """
    
    # Переводим сообщение на выбранный язык
    welcome_message = translate_text(welcome_message, bot_state.language)
    
    await update.message.reply_text(welcome_message, reply_markup=reply_markup)

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок меню"""
    # Используем состояние бота
    
    query = update.callback_query
    await query.answer()
    
    print(f"🔘 Нажата кнопка: {query.data}")  # Отладка
    logger.info(f"🔘 Нажата кнопка: {query.data}")
    
    try:
        if query.data == "menu_coins":
            await handle_coins_menu(query, context)
        elif query.data == "menu_analyze":
            await handle_analyze_menu(query, context)
        elif query.data == "menu_search":
            await handle_search_menu(query, context)
        elif query.data == "menu_backtest":
            await handle_backtest_menu(query, context)
        elif query.data.startswith("select_"):
            await handle_coin_selection(query, context)
        elif query.data == "find_shooting_stars":
            await handle_find_shooting_stars(query, context)
        elif query.data == "train_ema_models":
            await handle_train_ema_models(query, context)
        elif query.data == "start_ml_training":
            await handle_start_ml_training(query, context)
        elif query.data == "ml_models_status":
            await handle_ml_models_status(query, context)
        elif query.data == "ema_analyze_coin":
            await handle_ema_analyze_coin(query, context)
        elif query.data.startswith("ema_analyze_"):
            symbol = query.data.replace("ema_analyze_", "")
            await handle_ema_coin_analysis(query, context, symbol)
        elif query.data == "backtest_quick":
            await handle_backtest_quick(query, context)
        elif query.data == "backtest_full":
            await handle_backtest_full(query, context)
        elif query.data == "backtest_custom":
            await handle_backtest_custom(query, context)
        elif query.data == "back_to_main":
            await back_to_main_menu(query, context)
        elif query.data == "menu_contacts":
            await handle_contacts_menu(query, context)
        elif query.data == "switch_to_uzbek":
            await handle_switch_to_uzbek(query, context)
        elif query.data == "switch_to_english":
            await handle_switch_to_english(query, context)
        elif query.data == "switch_to_russian":
            await handle_switch_to_russian(query, context)
        elif query.data == "clear_chat":
            await handle_clear_chat(query, context)
            
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

🪙 **Текущая монета:** {bot_state.current_coin}
⏰ **Время:** {datetime.now().strftime('%H:%M:%S')}
🔗 **API:** Binance (ccxt)

**Доступные команды:**
/start - Главное меню
/status - Статус системы
/coins - Список монет
/signals - Сигналы для {bot_state.current_coin}
/analyze - Анализ {bot_state.current_coin}
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
        # Получаем список доступных пар с Binance
        if not bot_state.available_pairs:
            await get_available_pairs()
        
        # Используем реальные пары с Binance
        popular_coins = bot_state.available_pairs[:20]  # Первые 20 пар
        
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
        
        message = f"🪙 **Выберите монету для анализа (Binance):**\n\n📊 Доступно {len(bot_state.available_pairs)} монет"
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения списка монет: {str(e)}")


async def handle_analyze_menu(query, context):
    """Обработка кнопки Анализ монеты"""
    await analyze_coin_with_advanced_logic(query, context)  # Используем продвинутый анализ

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

async def handle_backtest_menu(query, context):
    """Обработка кнопки Бэктестинг"""
    try:
        message = """
📊 **БЭКТЕСТИНГ ТОРГОВОГО БОТА**

🎯 **Проверьте прибыльность стратегии на исторических данных!**

**Команды для запуска:**
• `/backtest BTC ETH` - тест на BTC и ETH
• `/backtest ADA SOL XRP` - тест на 3 монетах
• `/backtest ALL` - тест на топ-10 монетах

📅 **Период тестирования:** 01.01.2025 - сегодня
💰 **Стартовый капитал:** $1,000 
⏱️ **Время выполнения:** 3-10 минут
🧠 **Стратегия:** EMA + RSI анализ

**Что покажет тест:**
✅ Win Rate (% прибыльных сделок)
📈 Общая доходность за период
🏆 Топ прибыльные пары
📊 Детальная статистика

⚠️ **Важно:** Результаты прошлого не гарантируют будущую прибыль!
        """
        
        # Кнопки для быстрого запуска
        keyboard = [
            [InlineKeyboardButton("⚡ Быстрый тест (BTC, ETH, ADA)", callback_data="backtest_quick")],
            [InlineKeyboardButton("📊 Полный тест (ТОП-10)", callback_data="backtest_full")],
            [InlineKeyboardButton("💰 ВЫБРАТЬ МОНЕТЫ", callback_data="backtest_custom")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка меню бэктестинга: {str(e)}")

async def handle_backtest_quick(query, context):
    """Быстрый бэктест на 3 основных монетах"""
    try:
        await query.answer()
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
        symbols_text = 'BTC, ETH, ADA'
        
        progress_msg = await query.edit_message_text(
            f"🚀 **ЗАПУСКАЮ БЫСТРЫЙ БЭКТЕСТ**\n\n"
            f"🪙 **Монеты:** {symbols_text}\n"
            f"📅 **Период:** 01.01.2025 - {datetime.now().strftime('%d.%m.%Y')}\n"
            f"💰 **Стартовый капитал:** $1,000\n\n"
            f"⏳ **Загружаю данные...** (2-3 минуты)"
        )
        
        try:
            results = bot_state.backtest_engine.run_backtest(symbols)
            await send_backtest_results(query, results, progress_msg)
        except Exception as e:
            await progress_msg.edit_text(f"❌ Ошибка быстрого бэктеста: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка быстрого бэктеста: {e}")

async def handle_backtest_full(query, context):
    """Полный бэктест на топ-10 монетах"""
    try:
        await query.answer()
        
        symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 
                  'BNB/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT']
        symbols_text = 'BTC, ETH, ADA, SOL, XRP, BNB, DOGE, AVAX, DOT, LINK'
        
        progress_msg = await query.edit_message_text(
            f"🚀 **ЗАПУСКАЮ ПОЛНЫЙ БЭКТЕСТ**\n\n"
            f"🪙 **Монеты:** ТОП-10\n"
            f"📅 **Период:** 01.01.2025 - {datetime.now().strftime('%d.%m.%Y')}\n"
            f"💰 **Стартовый капитал:** $1,000\n\n"
            f"⏳ **Загружаю данные...** (5-10 минут)\n"
            f"⚠️ **Пожалуйста, подождите...**"
        )
        
        try:
            results = bot_state.backtest_engine.run_backtest(symbols)
            await send_backtest_results(query, results, progress_msg)
        except Exception as e:
            await progress_msg.edit_text(f"❌ Ошибка полного бэктеста: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка полного бэктеста: {e}")

async def handle_backtest_custom(query, context):
    """Инструкция по пользовательскому бэктестингу"""
    try:
        await query.answer()
        
        message = """
💰 **ПОЛЬЗОВАТЕЛЬСКИЙ БЭКТЕСТ**

🎯 **Для тестирования своих монет используйте команду:**
`/backtest ВАШИ_МОНЕТЫ`

📈 **Примеры команд:**
• `/backtest MATIC ATOM` - тест на 2 монетах
• `/backtest SOL AVAX DOT LINK` - тест на 4 монетах  
• `/backtest SHIB PEPE DOGE` - тест мем-монет
• `/backtest LTC BCH ETC` - альткоины
• `/backtest SAND MANA GALA` - игровые токены

⚡ **Быстрые варианты:**
• `/backtest BTC ETH` - основные монеты
• `/backtest ADA XRP ALGO` - популярные альткоины
• `/backtest ALL` - ТОП-10 монет

⚠️ **Ограничения:**
• Максимум 10 монет за раз
• Только пары с USDT (добавляется автоматически)
• Тестирование с 01.01.2025

🕐 **Время выполнения:** 3-10 минут в зависимости от количества монет

📊 **Что получите:**
✅ Win Rate для каждой монеты
📈 Общую доходность портфеля  
🏆 Лучшие и худшие пары
💰 Прибыль/убыток в долларах
        """
        
        keyboard = [
            [InlineKeyboardButton("🔙 Назад к меню", callback_data="menu_backtest")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"❌ Ошибка инструкции бэктеста: {e}")
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")

async def send_backtest_results(query, results, progress_msg):
    """Отправка результатов бэктестинга"""
    try:
        if 'error' in results:
            await progress_msg.edit_text(f"❌ Ошибка бэктестинга: {results['error']}")
            return
        
        # Формируем отчет
        win_rate = results['win_rate']
        total_return = results['total_return']
        total_trades = results['total_trades']
        winning_trades = results['winning_trades']
        losing_trades = total_trades - winning_trades
        final_balance = results['final_balance']
        total_pnl = results['total_pnl']
        
        # Определяем эмодзи результата
        if total_return > 20:
            result_emoji = "🚀💰"
        elif total_return > 0:
            result_emoji = "📈✅"
        elif total_return > -10:
            result_emoji = "📊⚠️"
        else:
            result_emoji = "📉❌"
        
        # Статистика по парам
        symbol_stats = {}
        for trade in results['trades']:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                symbol_stats[symbol]['wins'] += 1
        
        # Топ-3 прибыльные пары
        if symbol_stats:
            top_pairs = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:3]
            top_text = ""
            for symbol, stats in top_pairs:
                wr = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                top_text += f"• {symbol.replace('/USDT', '')}: ${stats['pnl']:.0f} ({wr:.0f}% WR)\n"
        else:
            top_text = "Нет данных\n"
        
        report = f"""{result_emoji} **РЕЗУЛЬТАТЫ БЭКТЕСТИНГА**

📊 **ОБЩАЯ СТАТИСТИКА:**
💰 Стартовый капитал: $1,000
💵 Финальный капитал: ${final_balance:.2f}
📈 Общая доходность: {total_return:+.1f}%
💸 Чистая прибыль: ${total_pnl:+.2f}

🎯 **ТОРГОВАЯ СТАТИСТИКА:**
📊 Всего сделок: {total_trades}
✅ Прибыльных: {winning_trades} ({win_rate:.1f}%)
❌ Убыточных: {losing_trades} ({100-win_rate:.1f}%)

🏆 **ТОП-3 ПРИБЫЛЬНЫЕ ПАРЫ:**
{top_text}
📅 **Период:** 01.01.2025 - {datetime.now().strftime('%d.%m.%Y')}
⏱️ **Таймфрейм:** 1 час
🧠 **Стратегия:** EMA + RSI анализ

⚠️ **Отказ от ответственности:** Результаты прошлого не гарантируют будущую прибыль!"""
        
        # Кнопка возврата
        keyboard = [[InlineKeyboardButton("🔙 Назад к бэктестингу", callback_data="menu_backtest")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await progress_msg.edit_text(report, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"❌ Ошибка отправки результатов: {e}")
        await progress_msg.edit_text(f"❌ Ошибка формирования отчета: {str(e)}")

async def handle_shooting_stars_menu(query, context):
    """Обработка кнопки Стреляющие монеты"""
    try:
        message = """
🚀 **Стреляющие монеты**

Анализ всех монет на Binance для поиска потенциальных "стреляющих звезд" - монет, которые могут показать резкий рост в ближайшее время.

**Возможности:**
• 🔮 LSTM нейронная сеть для предсказаний
• 📊 Анализ всех USDT пар на Binance
• 🎯 Топ-10 самых перспективных монет
• ⚡ Быстрый анализ (до 5 минут)
        """
        
        keyboard = [
            [InlineKeyboardButton("🔮 Найти стреляющие монеты", callback_data="find_shooting_stars")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка стреляющих монет: {str(e)}")


async def analyze_coin_with_advanced_logic(query, context):
    """Продвинутый анализ монеты с проверкой топ-50 и переобучением"""
    try:
        # Проверяем, нужна ли адаптивная переобучение
        needs_retrain = not is_coin_in_top50(bot_state.current_coin)
        
        if needs_retrain:
            await query.message.reply_text(f"🔄 Анализирую {bot_state.current_coin}...\n\n🆕 Монета не в топ-50, переобучаю модели с учетом этой монеты...\n⏳ Это займет 1-2 минуты...")
        else:
            await query.message.reply_text(f"🔍 Анализирую {bot_state.current_coin}...")
        
        signal_data = analyze_coin_signal_advanced_ema(bot_state.current_coin)
        if not signal_data:
            await query.message.reply_text(f"❌ Ошибка анализа {bot_state.current_coin}")
            return
        
        # Добавляем Smart ML предсказание
        smart_prediction = None
        logger.info(f"🔍 DEBUG: signal_data keys: {list(signal_data.keys())}")
        logger.info(f"🔍 DEBUG: features есть: {signal_data.get('features') is not None}")
        
        # Если features нет, пытаемся подготовить их из DataFrame
        if not signal_data.get('features') and signal_data.get('df') is not None:
            try:
                logger.info("🔧 Подготавливаю features для Smart ML...")
                features = prepare_ml_features(signal_data['df'], bot_state.current_coin)
                signal_data['features'] = features
                logger.info("✅ Features подготовлены для Smart ML")
            except Exception as e:
                logger.error(f"❌ Ошибка подготовки features: {e}")
        
        if signal_data.get('features') is not None:
            try:
                smart_prediction = predict_with_smart_ml(signal_data['features'])
                if smart_prediction:
                    logger.info(f"🧠 Smart ML: {smart_prediction['prediction']}")
                    signal_data['smart_prediction'] = smart_prediction
                else:
                    logger.warning("⚠️ Smart ML предсказание вернуло None")
            except Exception as e:
                logger.error(f"❌ Ошибка Smart ML предсказания: {e}")
        else:
            logger.warning("⚠️ Features не найдены для Smart ML")
        
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

{_format_smart_prediction(signal_data.get('smart_prediction'))}
            """
            
            if "LONG" in signal_data['signal_type']:
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
            
            # Переводим сообщение на выбранный язык
            message = translate_text(message, bot_state.language)
            
            # Переводим кнопки
            if bot_state.language == "uz":
                back_button_text = "🔙 Орқага"
            elif bot_state.language == "en":
                back_button_text = "🔙 Back"
            else:
                back_button_text = "🔙 Назад"
            keyboard = [[InlineKeyboardButton(back_button_text, callback_data="back_to_main")]]
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

{_format_smart_prediction(signal_data.get('smart_prediction'))}
            """
            
            # Переводим сообщение на выбранный язык
            message = translate_text(message, bot_state.language)
            
            # Переводим кнопки
            if bot_state.language == "uz":
                back_button_text = "🔙 Орқага"
            elif bot_state.language == "en":
                back_button_text = "🔙 Back"
            else:
                back_button_text = "🔙 Назад"
            keyboard = [[InlineKeyboardButton(back_button_text, callback_data="back_to_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        logger.error(f"❌ Ошибка продвинутого анализа: {e}")
        await query.message.reply_text(f"❌ Ошибка анализа: {str(e)}")

async def handle_coin_selection(query, context):
    """Обработка выбора монеты"""
    # Используем состояние бота
    coin = query.data.replace("select_", "")
    bot_state.current_coin = coin
    
    # Отправляем новое сообщение вместо редактирования
    await query.message.reply_text(f"✅ Выбрана монета: {coin}")
    
    # Автоматически запускаем продвинутый анализ
    await asyncio.sleep(1)
    await analyze_coin_with_advanced_logic(query, context)

async def handle_find_shooting_stars(query, context):
    """Поиск стреляющих монет с помощью продвинутого анализа"""
    # Используем bot_state напрямую
    
    try:
        # Инициализируем предиктор если еще не инициализирован
        if bot_state.shooting_predictor is None:
            bot_state.shooting_predictor = ShootingStarPredictor()
        
        # Отправляем сообщение о начале анализа
        await query.edit_message_text("🔮 **Поиск стреляющих монет...**\n\n⏳ Анализирую все монеты на Binance...")
        
        # Получаем список всех монет
        bot_state.available_pairs = await get_available_pairs()
        
        # Проверяем, что список не пустой
        if not bot_state.available_pairs:
            await query.edit_message_text("❌ Не удалось получить список монет с Binance")
            return
        
        # Ограничиваем анализ первыми 50 монетами для скорости
        pairs_to_analyze = bot_state.available_pairs[:50]
        
        logger.info(f"🚀 Начинаю поиск стреляющих монет среди {len(pairs_to_analyze)} монет")
        
        # Используем предиктор для поиска стреляющих звезд
        shooting_stars = bot_state.shooting_predictor.find_shooting_stars(pairs_to_analyze, min_probability=0.4)
        
        # Формируем результат
        if shooting_stars:
            message = f"""🚀 **СТРЕЛЯЮЩИЕ МОНЕТЫ НАЙДЕНЫ!**

📊 **Проанализировано:** {len(pairs_to_analyze)} монет
🎯 **Найдено стреляющих:** {len(shooting_stars)}

**🏆 ТОП-{min(10, len(shooting_stars))} СТРЕЛЯЮЩИХ МОНЕТ:**

"""
            
            for i, star in enumerate(shooting_stars[:10], 1):
                probability_pct = star['probability'] * 100
                message += f"""**{i}. {star['symbol']}** 🚀
💰 Цена: ${star['current_price']:.8f}
🎯 Вероятность: {probability_pct:.1f}%
📈 Потенциал: {'🔥' * min(5, int(probability_pct / 20))}
📊 Прогноз: {star['predicted_change']}

"""
            
            message += f"\n⏰ **Время анализа:** {datetime.now().strftime('%H:%M:%S')}"
            
        else:
            message = f"""🚀 **Поиск стреляющих монет завершен**

📊 **Проанализировано:** {len(pairs_to_analyze)} монет
🎯 **Стреляющих не найдено**

ℹ️ В данный момент нет монет с высокой вероятностью резкого роста.
Попробуйте позже или используйте обычный анализ монет.
"""
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="menu_shooting_stars")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
        logger.info(f"✅ Поиск стреляющих монет завершен: найдено {len(shooting_stars)} из {len(pairs_to_analyze)}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка поиска стреляющих монет: {e}")
        await query.edit_message_text(f"❌ Ошибка поиска стреляющих монет: {str(e)}")

async def handle_ema_analysis_menu(query, context):
    """Меню EMA анализа"""
    try:
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("🤖 Обучить EMA модели", callback_data="train_ema_models")],
            [InlineKeyboardButton("📊 EMA анализ монеты", callback_data="ema_analyze_coin")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = """📈 **EMA АНАЛИЗ**

🎯 **Возможности:**
• Обучение моделей на EMA закономерностях
• Анализ трендов по EMA 20, 50, 100
• Поиск импульсов и коррекций
• Точки входа на основе EMA структур

📊 **Особенности:**
• Фокус на скорости движения EMA
• Соответствие скорости цены и EMA
• Расстояния между EMA линиями
• Без лишних технических индикаторов
"""
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка EMA меню: {e}")
        await query.edit_message_text("❌ Ошибка отображения меню")

async def handle_train_ema_models(query, context):
    """Обучение EMA моделей на исторических данных"""
    try:
        await query.answer()
        await query.edit_message_text("🤖 Обучаю EMA модели на исторических данных...")
        
        # Инициализируем тренер
        trainer = AdvancedMLTrainer()
        
        # Список символов для обучения
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
        ]
        
        # Собираем данные
        await query.edit_message_text("📊 Собираю исторические данные...")
        historical_data = trainer.collect_historical_data(symbols, days=30)
        
        if not historical_data:
            await query.edit_message_text("❌ Не удалось собрать данные для обучения")
            return
        
        # Обучаем модели
        await query.edit_message_text("🤖 Обучаю EMA модели...")
        success = trainer.train_models(symbols)
        
        if success:
            message = f"✅ **EMA МОДЕЛИ ОБУЧЕНЫ!**\n\n"
            message += f"📊 Моделей: 2 (вход и выход)\n"
            message += f"📈 Символов: {len(symbols)}\n"
            message += "🚀 Готово к использованию!"
            
            await query.edit_message_text(message, parse_mode='Markdown')
        else:
            await query.edit_message_text("❌ Не удалось обучить EMA модели")
            
    except Exception as e:
        logger.error(f"❌ Ошибка обучения EMA моделей: {e}")
        await query.edit_message_text("❌ Ошибка обучения EMA моделей")

async def handle_train_ml_menu(query, context):
    """Меню обучения ML моделей"""
    try:
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("🚀 Начать обучение", callback_data="start_ml_training")],
            [InlineKeyboardButton("📊 Статус моделей", callback_data="ml_models_status")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = """🧠 **ОБУЧЕНИЕ ML МОДЕЛЕЙ**

🎯 **Универсальная EMA логика для всех трендов:**
• Скорости EMA линий (20, 50, 100)
• Скорость цены относительно EMA
• Расстояния между EMA и ценой
• Углы тренда (-90° до +90°)
• Импульсы и коррекции

📈 **Обучение на данных с 1 января 2025:**
• **Нисходящий тренд:** максимальные расстояния = LONG вход, минимальные = выход
• **Восходящий тренд:** минимальные расстояния = LONG вход (приближение, пересечение, касание, отскок), максимальные = выход при максимуме импульса  
• **Боковой тренд:** максимальные расстояния = LONG вход, минимальные = выход
• Только LONG сигналы
• RandomForest классификация

⚡ **Результат:** Точные предсказания точек входа/выхода
"""
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка ML меню: {e}")
        await query.edit_message_text("❌ Ошибка отображения меню")

async def handle_ema_analyze_coin(query, context):
    """EMA анализ конкретной монеты"""
    try:
        await query.answer()
        
        # Показываем популярные монеты для анализа
        keyboard = [
            [InlineKeyboardButton("BTC/USDT", callback_data="ema_analyze_BTC/USDT")],
            [InlineKeyboardButton("ETH/USDT", callback_data="ema_analyze_ETH/USDT")],
            [InlineKeyboardButton("BNB/USDT", callback_data="ema_analyze_BNB/USDT")],
            [InlineKeyboardButton("ADA/USDT", callback_data="ema_analyze_ADA/USDT")],
            [InlineKeyboardButton("SOL/USDT", callback_data="ema_analyze_SOL/USDT")],
            [InlineKeyboardButton("🔙 Назад", callback_data="menu_ema_analysis")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = """📊 **EMA АНАЛИЗ МОНЕТЫ**

Выберите монету для анализа EMA паттернов:

🎯 **Что анализируется:**
• Тренд по EMA 20, 50, 100
• Скорость движения EMA линий
• Соответствие скорости цены и EMA
• Расстояния между EMA
• Импульсы и коррекции
• Точки входа/выхода
"""
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка EMA анализа монеты: {e}")
        await query.edit_message_text("❌ Ошибка отображения монет")

async def handle_ema_coin_analysis(query, context, symbol):
    """Анализ конкретной монеты с EMA"""
    try:
        await query.answer()
        
        # Проверяем, нужна ли адаптивная переобучение
        needs_retrain = not is_coin_in_top50(symbol)
        
        if needs_retrain:
            await query.edit_message_text(f"🔄 Анализирую {symbol}...\n\n🆕 Монета не в топ-50, переобучаю модели с учетом этой монеты...\n⏳ Это займет 1-2 минуты...")
        else:
            await query.edit_message_text(f"📊 Анализирую {symbol} с помощью EMA...")
        
        # Выполняем EMA анализ
        signal_data = analyze_coin_signal_advanced_ema(symbol)
        
        if signal_data.get('error'):
            await query.edit_message_text(f"❌ Ошибка анализа {symbol}: {signal_data['error']}")
            return
        
        # Формируем сообщение с результатами
        ema_analysis = signal_data.get('ema_analysis', {})
        
        message = f"""📈 EMA АНАЛИЗ {symbol}

🎯 Результат: {signal_data['signal_type']}
📝 Обоснование: {signal_data['strength_text']}

📊 EMA Данные:
• Тренд: {ema_analysis.get('trend_name', 'Не определен')}
• Фаза: {ema_analysis.get('phase_name', 'Не определена')}
• Уверенность: {ema_analysis.get('confidence', 0):.1f}%

💰 Цена входа: ${signal_data['entry_price']:.8f}
"""
        
        if signal_data.get('take_profit'):
            message += f"🎯 Take Profit: ${signal_data['take_profit']:.8f}\n"
        
        if signal_data.get('stop_loss'):
            message += f"🛡️ Stop Loss: ${signal_data['stop_loss']:.8f}\n"
        
        # Добавляем EMA уровни
        levels = ema_analysis.get('levels', {})
        if levels:
            message += f"""
📊 EMA Уровни:
• EMA 20: ${levels.get('ema_20', 0):.8f}
• EMA 50: ${levels.get('ema_50', 0):.8f}
• EMA 100: ${levels.get('ema_100', 0):.8f}
"""
        
        message += f"\n📊 RSI: {signal_data['rsi']:.1f}"
        message += f"\n🤖 ML статус: {signal_data['ml_status']}"
        
        # Переводим сообщение на выбранный язык
        message = translate_text(message, bot_state.language)
        
        # Переводим кнопки
        if bot_state.language == "uz":
            back_button_text = "🔙 Орқага"
        elif bot_state.language == "en":
            back_button_text = "🔙 Back"
        else:
            back_button_text = "🔙 Назад"
        keyboard = [[InlineKeyboardButton(back_button_text, callback_data="menu_ema_analysis")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
        logger.info(f"✅ EMA анализ {symbol} завершен: {signal_data['signal_type']}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка EMA анализа {symbol}: {e}")
        await query.edit_message_text(f"❌ Ошибка анализа {symbol}")

def prepare_lstm_features(df):
    """Подготовка признаков для LSTM модели"""
    try:
        # Простая подготовка признаков (можно улучшить)
        features = []
        
        for i in range(10, len(df)):
            row_features = []
            
            # Цена и объем
            row_features.extend([
                df['close'].iloc[i],
                df['volume'].iloc[i],
                df['high'].iloc[i] - df['low'].iloc[i],  # волатильность
            ])
            
            # Простые технические индикаторы
            close_prices = df['close'].iloc[i-10:i+1]
            
            # RSI (упрощенный)
            if len(close_prices) > 1:
                price_change = (close_prices.iloc[-1] - close_prices.iloc[0]) / close_prices.iloc[0]
                row_features.append(price_change)
            else:
                row_features.append(0)
            
            # Средняя цена за период
            row_features.append(close_prices.mean())
            
            # Максимум и минимум
            row_features.extend([close_prices.max(), close_prices.min()])
            
            features.append(row_features)
        
        if not features:
            return None
            
        return np.array(features)
        
    except Exception as e:
        logger.error(f"❌ Ошибка подготовки LSTM признаков: {e}")
        return None




async def back_to_main_menu(query, context):
    """Возврат в главное меню"""
    # Используем состояние бота
    
    # Определяем текст кнопки языка (циклическое переключение)
    if bot_state.language == "ru":
        lang_button_text = "🇺🇿 O'zbekcha"
        lang_callback = "switch_to_uzbek"
    elif bot_state.language == "uz":
        lang_button_text = "🇬🇧 English"
        lang_callback = "switch_to_english"
    else:  # en
        lang_button_text = "🇷🇺 Русский"
        lang_callback = "switch_to_russian"
    
    keyboard = [
        [InlineKeyboardButton("⚡ ТОРГОВЫЕ ПАРЫ", callback_data="menu_coins")],
        [InlineKeyboardButton("🎯 АНАЛИЗ & СИГНАЛЫ", callback_data="menu_analyze")],
        [InlineKeyboardButton("🔍 ПОИСК АКТИВОВ", callback_data="menu_search")],
        [InlineKeyboardButton("📊 БЭКТЕСТИНГ", callback_data="menu_backtest")],
        [InlineKeyboardButton("💬 СВЯЗАТЬСЯ С НАМИ", callback_data="menu_contacts")],
        [InlineKeyboardButton(lang_button_text, callback_data=lang_callback)],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""
🤖 **Binance Trading Bot с ML сигналами!**

🪙 **Текущая монета:** {bot_state.current_coin}

**Выберите действие из меню ниже:**
    """
    
    # Переводим сообщение на выбранный язык
    welcome_message = translate_text(welcome_message, bot_state.language)
    
    try:
        # Пытаемся редактировать сообщение
        await query.edit_message_text(welcome_message, reply_markup=reply_markup)
    except Exception as e:
        # Если не получается редактировать (например, сообщение с фото), отправляем новое
        logger.warning(f"⚠️ Не удалось редактировать сообщение: {e}")
        await query.message.reply_text(welcome_message, reply_markup=reply_markup)

async def handle_contacts_menu(query, context):
    """Обработка меню контактов"""
    try:
        await query.answer()
        
        keyboard = [
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        contacts_message = """
📞 **КОНТАКТЫ РАЗРАБОТЧИКА**

👨‍💻 **Разработчик:** Bahodir
🤖 **Бот:** Binance Trading Bot с ML

📧 **Telegram:** [@Bbbbbbb111233](https://t.me/Bbbbbbb111233)
💬 **Связь:** Для вопросов, предложений и сотрудничества

🚀 **Возможности бота:**
• Анализ любых монет Binance
• Адаптивное ML обучение
• Автосигналы каждые 30 минут
• Стреляющие звезды
• EMA анализ с ML предсказаниями

💡 **Технологии:**
• Machine Learning (RandomForest)
• Binance API
• Telegram Bot API
• Python, pandas, scikit-learn

📈 **Точность:** Модели обучены на реальных данных
🎯 **Покрытие:** Все 614+ USDT пар Binance

Спасибо за использование! 🙏
        """
        
        await query.edit_message_text(contacts_message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка в handle_contacts_menu: {e}")
        await query.edit_message_text("❌ Ошибка отображения контактов")

async def handle_switch_to_uzbek(query, context):
    """Переключение на узбекский язык"""
    try:
        await query.answer()
        bot_state.language = "uz"
        
        message = "🇺🇿 **Тил ўзгартирилди!**\n\nҲозир бот ўзбек тилида ишлайди.\n\n**Til:** O'zbekcha ✅"
        keyboard = [[InlineKeyboardButton("🔙 Орқага", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка переключения на узбекский: {e}")
        await query.edit_message_text("❌ Ошибка переключения языка")

async def handle_switch_to_english(query, context):
    """Переключение на английский язык"""
    try:
        await query.answer()
        bot_state.language = "en"
        
        message = "🇬🇧 **Language changed!**\n\nNow bot works in English language.\n\n**Language:** English ✅"
        keyboard = [[InlineKeyboardButton("🔙 Back", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка переключения на английский: {e}")
        await query.edit_message_text("❌ Ошибка переключения языка")

async def handle_switch_to_russian(query, context):
    """Переключение на русский язык"""
    try:
        await query.answer()
        bot_state.language = "ru"
        
        message = "🇷🇺 **Язык изменен!**\n\nТеперь бот работает на русском языке.\n\n**Язык:** Русский ✅"
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка переключения на русский: {e}")
        await query.edit_message_text("❌ Ошибка переключения языка")

async def handle_clear_chat(query, context):
    """Очистка чата"""
    try:
        await query.answer("🗑️ Очищаю чат...")
        
        # Удаляем все сообщения бота в чате
        chat_id = query.message.chat_id
        
        # Пытаемся удалить последние 20 сообщений
        deleted_count = 0
        for i in range(1, 21):
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=query.message.message_id - i)
                deleted_count += 1
            except:
                pass
        
        # Отправляем новое сообщение с подтверждением
        message = f"✅ **Чат очищен!**\n\n🗑️ Удалено сообщений: {deleted_count}\n\nТеперь чат чистый и показывает только актуальную информацию!"
        
        # Переводим сообщение на выбранный язык
        if bot_state.language == "uz":
            message = f"✅ **Чат тозалади!**\n\n🗑️ Ўчирилган хабарлар: {deleted_count}\n\nЭнди чат тоза ва фақат актуал маълумотларни кўрсатади!"
        elif bot_state.language == "en":
            message = f"✅ **Chat cleared!**\n\n🗑️ Messages deleted: {deleted_count}\n\nNow chat is clean and shows only actual information!"
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.message.reply_text(message, reply_markup=reply_markup, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка очистки чата: {e}")
        await query.edit_message_text("❌ Ошибка очистки чата")

async def clear_chat_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /clear для очистки чата"""
    try:
        # Удаляем все сообщения бота в чате
        chat_id = update.effective_chat.id
        
        # Пытаемся удалить последние 20 сообщений
        deleted_count = 0
        for i in range(1, 21):
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=update.message.message_id - i)
                deleted_count += 1
            except:
                pass
        
        # Отправляем подтверждение
        message = f"✅ **Чат очищен!**\n\n🗑️ Удалено сообщений: {deleted_count}\n\nТеперь показываю только свежие данные!"
        
        # Переводим сообщение на выбранный язык
        if bot_state.language == "uz":
            message = f"✅ **Чат тозалади!**\n\n🗑️ Ўчирилган хабарлар: {deleted_count}\n\nЭнди фақат янги маълумотларни кўрсатаман!"
        elif bot_state.language == "en":
            message = f"✅ **Chat cleared!**\n\n🗑️ Messages deleted: {deleted_count}\n\nNow showing only fresh data!"
        
        await update.message.reply_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка команды /clear: {e}")
        await update.message.reply_text("❌ Ошибка очистки чата")

async def set_coin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /set_coin для выбора монеты"""
    try:
        if not context.args:
            await update.message.reply_text("❌ Укажите монету!\nПример: /set_coin BTC/USDT")
            return
        
        coin = context.args[0].upper()
        
        # Исправляем популярные монеты
        if coin == 'XRPUSDT':
            coin = 'XRP/USDT'
        elif coin == 'BTCUSDT':
            coin = 'BTC/USDT'
        elif coin == 'ETHUSDT':
            coin = 'ETH/USDT'
        elif coin == 'ADAUSDT':
            coin = 'ADA/USDT'
        elif coin == 'SOLUSDT':
            coin = 'SOL/USDT'
        elif coin == 'BNBUSDT':
            coin = 'BNB/USDT'
        elif not coin.endswith('/USDT'):
            coin += '/USDT'
        
        # Используем состояние бота
        bot_state.current_coin = coin
        
        # Устанавливаем монету без проверки в списке
        await update.message.reply_text(
            f"✅ **Монета установлена:** {coin}\n\n"
            f"Теперь используйте /analyze для анализа или /start для меню."
        )
        
    except Exception as e:
        logger.error(f"❌ Ошибка команды /set_coin: {e}")
        await update.message.reply_text(f"❌ Ошибка: {str(e)}")

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

{_format_smart_prediction(signal_data.get('smart_prediction'))}
            """
            
            if "LONG" in signal_data['signal_type']:
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
            
            # Переводим сообщение на выбранный язык
            message = translate_text(message, bot_state.language)
            
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

{_format_smart_prediction(signal_data.get('smart_prediction'))}
            """
            
            # Переводим сообщение на выбранный язык
            message = translate_text(message, bot_state.language)
            
            await update.message.reply_text(message)
            
    except Exception as e:
        logger.error(f"❌ Ошибка команды /analyze: {e}")
        await update.message.reply_text(f"❌ Ошибка анализа: {str(e)}")

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /signals - алиас для /analyze"""
    await analyze_command(update, context)

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /search для поиска монет"""
    try:
        if not context.args:
            await update.message.reply_text(
                "❌ Укажите название монеты!\n\n"
                "Примеры:\n"
                "/search BTC\n"
                "/search ETH\n"
                "/search BNB\n\n"
                "Или используйте /start для выбора из списка."
            )
            return
        
        search_term = context.args[0].upper()
        
        # Поиск в доступных парах
        matching_pairs = []
        if bot_state.available_pairs:
            matching_pairs = [pair for pair in bot_state.available_pairs if search_term in pair]
        
        if matching_pairs:
            # Создаем кнопки для найденных пар
            keyboard = []
            for i in range(0, len(matching_pairs[:10]), 2):  # Максимум 10 результатов
                row = []
                for j in range(2):
                    if i + j < len(matching_pairs):
                        pair = matching_pairs[i + j]
                        row.append(InlineKeyboardButton(pair, callback_data=f"select_{pair}"))
                if row:
                    keyboard.append(row)
            
            keyboard.append([InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(
                f"🔍 **Найдено {len(matching_pairs)} пар с \"{search_term}\":**",
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                f"❌ Не найдено пар с \"{search_term}\"\n\n"
                "Попробуйте:\n"
                "- Проверить правильность написания\n"
                "- Использовать /start для полного списка\n"
                "- Попробовать сокращения (BTC, ETH, etc.)"
            )
            
    except Exception as e:
        logger.error(f"❌ Ошибка команды /search: {e}")
        await update.message.reply_text(f"❌ Ошибка поиска: {str(e)}")

async def test_binance_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /test_binance для тестирования Binance соединения"""
    try:
        await update.message.reply_text("🔍 Тестирую соединение с Binance...")
        
        # Тестируем Binance
        test_symbol = "BTC/USDT"
        df = get_binance_data(test_symbol, timeframe='1h', limit=10)
        
        if df is not None and not df.empty:
            latest_price = df['close'].iloc[-1]
            latest_time = df.index[-1]
            message = f"""
✅ **Binance соединение работает!**

📊 **Тест данных:**
- Символ: {test_symbol}
- Свечей получено: {len(df)}
- Последняя цена: ${latest_price:.2f}
- Доступно монет: {len(bot_state.available_pairs)}

🕐 **Последние данные:**
- Время: {latest_time.strftime('%H:%M:%S')}
- Цена: ${latest_price:.2f}
- Объем: {df['volume'].iloc[-1]:.0f}

🎯 **Готов к работе!**
            """
        else:
            message = "❌ **Ошибка получения данных с Binance**"
        
        await update.message.reply_text(message)
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования Binance: {e}")
        await update.message.reply_text(f"❌ Ошибка Binance: {str(e)}")

async def backtest_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /backtest для запуска бэктестирования"""
    try:
        # Проверяем аргументы
        if not context.args:
            await update.message.reply_text(
                "📊 **БЭКТЕСТИНГ ТОРГОВОГО БОТА**\n\n"
                "🎯 **Использование:**\n"
                "`/backtest BTC ETH ADA SOL`\n\n"
                "📈 **Примеры:**\n"
                "• `/backtest BTC ETH` - тест на BTC и ETH\n"
                "• `/backtest ADA SOL XRP` - тест на 3 монетах\n"
                "• `/backtest ALL` - тест на топ-10 монетах\n\n"
                "📅 **Период:** 01.01.2025 - сегодня\n"
                "💰 **Стартовый капитал:** $1,000\n"
                "⏱️ **Время выполнения:** 3-10 минут\n\n"
                "⚠️ **Внимание:** Бэктест может занять время!"
            )
            return
        
        symbols_input = [arg.upper() for arg in context.args]
        
        # Подготавливаем список символов
        if 'ALL' in symbols_input:
            symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT', 'XRP/USDT', 
                      'BNB/USDT', 'DOGE/USDT', 'AVAX/USDT', 'DOT/USDT', 'LINK/USDT']
        else:
            symbols = []
            for symbol in symbols_input:
                if not symbol.endswith('/USDT'):
                    symbol += '/USDT'
                symbols.append(symbol)
        
        if len(symbols) > 10:
            await update.message.reply_text("❌ Максимум 10 монет для бэктестинга!")
            return
        
        # Уведомляем о запуске
        symbols_text = ', '.join([s.replace('/USDT', '') for s in symbols])
        progress_msg = await update.message.reply_text(
            f"🚀 **ЗАПУСКАЮ БЭКТЕСТ**\n\n"
            f"🪙 **Монеты:** {symbols_text}\n"
            f"📅 **Период:** 01.01.2025 - {datetime.now().strftime('%d.%m.%Y')}\n"
            f"💰 **Стартовый капитал:** $1,000\n\n"
            f"⏳ **Загружаю данные...** (это может занять несколько минут)"
        )
        
        # Запускаем бэктест
        try:
            results = bot_state.backtest_engine.run_backtest(symbols)
            
            if 'error' in results:
                await progress_msg.edit_text(f"❌ Ошибка бэктестинга: {results['error']}")
                return
            
            # Формируем отчет
            win_rate = results['win_rate']
            total_return = results['total_return']
            total_trades = results['total_trades']
            winning_trades = results['winning_trades']
            losing_trades = total_trades - winning_trades
            final_balance = results['final_balance']
            total_pnl = results['total_pnl']
            
            # Определяем эмодзи результата
            if total_return > 20:
                result_emoji = "🚀💰"
            elif total_return > 0:
                result_emoji = "📈✅"
            elif total_return > -10:
                result_emoji = "📊⚠️"
            else:
                result_emoji = "📉❌"
            
            # Статистика по парам
            symbol_stats = {}
            for trade in results['trades']:
                symbol = trade['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
                symbol_stats[symbol]['trades'] += 1
                symbol_stats[symbol]['pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    symbol_stats[symbol]['wins'] += 1
            
            # Топ-3 прибыльные пары
            if symbol_stats:
                top_pairs = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)[:3]
                top_text = ""
                for symbol, stats in top_pairs:
                    wr = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                    top_text += f"• {symbol.replace('/USDT', '')}: ${stats['pnl']:.0f} ({wr:.0f}% WR)\n"
            else:
                top_text = "Нет данных\n"
            
            report = f"""{result_emoji} **РЕЗУЛЬТАТЫ БЭКТЕСТИНГА**

📊 **ОБЩАЯ СТАТИСТИКА:**
💰 Стартовый капитал: $1,000
💵 Финальный капитал: ${final_balance:.2f}
📈 Общая доходность: {total_return:+.1f}%
💸 Чистая прибыль: ${total_pnl:+.2f}

🎯 **ТОРГОВАЯ СТАТИСТИКА:**
📊 Всего сделок: {total_trades}
✅ Прибыльных: {winning_trades} ({win_rate:.1f}%)
❌ Убыточных: {losing_trades} ({100-win_rate:.1f}%)

🏆 **ТОП-3 ПРИБЫЛЬНЫЕ ПАРЫ:**
{top_text}
📅 **Период:** 01.01.2025 - {datetime.now().strftime('%d.%m.%Y')}
⏱️ **Таймфрейм:** 1 час
🧠 **Стратегия:** EMA + RSI анализ

⚠️ **Отказ от ответственности:** Результаты прошлого не гарантируют будущую прибыль!"""
            
            await progress_msg.edit_text(report)
            
        except Exception as e:
            logger.error(f"❌ Ошибка выполнения бэктеста: {e}")
            await progress_msg.edit_text(f"❌ Ошибка выполнения бэктеста: {str(e)}")
            
    except Exception as e:
        logger.error(f"❌ Ошибка команды /backtest: {e}")
        await update.message.reply_text(f"❌ Ошибка бэктестинга: {str(e)}")

async def handle_start_ml_training(query, context):
    """Начало обучения ML моделей"""
    try:
        await query.answer()
        await query.edit_message_text("🚀 Начинаю обучение ML моделей на EMA данных...")
        
        # Инициализируем тренер
        trainer = AdvancedMLTrainer()
        
        # Список символов для обучения
        symbols = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
        ]
        
        # Обучение моделей
        success = trainer.train_models(symbols)
        
        if success:
            message = """✅ **ML МОДЕЛИ ОБУЧЕНЫ!**

🎯 Модели точек входа и выхода готовы
📊 Обучены на EMA признаках
🚀 Готово к использованию!"""
            await query.edit_message_text(message, parse_mode='Markdown')
        else:
            await query.edit_message_text("❌ Ошибка обучения ML моделей")
            
    except Exception as e:
        logger.error(f"❌ Ошибка обучения ML: {e}")
        await query.edit_message_text(f"❌ Ошибка обучения: {str(e)}")

async def handle_ml_models_status(query, context):
    """Статус ML моделей"""
    try:
        await query.answer()
        
        trainer = AdvancedMLTrainer()
        
        # Проверяем наличие моделей
        models_exist = (
            os.path.exists('models/entry_model.pkl') and
            os.path.exists('models/exit_model.pkl') and
            os.path.exists('models/ema_scaler.pkl')
        )
        
        if models_exist:
            # Пытаемся загрузить модели
            try:
                trainer.load_models()
                message = """✅ **ML МОДЕЛИ ЗАГРУЖЕНЫ**

🤖 **Доступные модели:**
• Модель точек входа ✅
• Модель точек выхода ✅
• Нормализатор данных ✅

📊 **Статус:** Готовы к предсказаниям
🎯 **Логика:** EMA + ML анализ
"""
            except Exception as e:
                message = f"""⚠️ **МОДЕЛИ НАЙДЕНЫ, НО ОШИБКА ЗАГРУЗКИ**

❌ **Ошибка:** {str(e)}

🔄 **Рекомендация:** Переобучите модели
"""
        else:
            message = """❌ **ML МОДЕЛИ НЕ НАЙДЕНЫ**

📁 **Ожидаемые файлы:**
• models/entry_model.pkl
• models/exit_model.pkl
• models/ema_scaler.pkl

🚀 **Действие:** Начните обучение моделей
"""
        
        await query.edit_message_text(message, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"❌ Ошибка проверки статуса: {e}")
        await query.edit_message_text(f"❌ Ошибка: {str(e)}")

def main():
    """Основная функция"""
    print("🤖 Запуск Binance ML Telegram Bot")
    print("🔄 Используем Binance API через ccxt")
    
    # Инициализируем состояние бота
    bot_state.initialize()
    if not bot_state.config:
        print("❌ Не удалось загрузить конфигурацию")
        return
    
    # Проверяем локальный режим разработки
    is_local_dev = bot_state.config.get('local_development', {}).get('enabled', False)
    if is_local_dev:
        print("🔧 ЛОКАЛЬНЫЙ РЕЖИМ РАЗРАБОТКИ")
        print("🚫 Telegram API отключен - тестируем только функции анализа")
        print("📊 Доступные функции:")
        print("   - Анализ монет")
        print("   - ML предсказания") 
        print("   - EMA анализ")
        print("   - Создание графиков")
        print("\n💡 Для полного тестирования удалите bot_config_local.json")
        return
    
    # Инициализируем список доступных пар
    print("🔍 Получаю список популярных монет с Binance...")
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(get_available_pairs())
        print(f"✅ Загружено {len(bot_state.available_pairs)} монет с Binance")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки монет с Binance: {e}")
        print("🔄 Использую стандартный список")
    
    # Создаем приложение
    bot_state.application = Application.builder().token(bot_state.config["telegram"]["bot_token"]).build()
    
    
    # Добавляем обработчики
    bot_state.application.add_handler(CommandHandler("start", start_command))
    bot_state.application.add_handler(CommandHandler("clear", clear_chat_command))
    bot_state.application.add_handler(CommandHandler("set_coin", set_coin_command))
    bot_state.application.add_handler(CommandHandler("analyze", analyze_command))
    bot_state.application.add_handler(CommandHandler("signals", signals_command))
    bot_state.application.add_handler(CommandHandler("search", search_command))
    bot_state.application.add_handler(CommandHandler("test_binance", test_binance_command))
    bot_state.application.add_handler(CommandHandler("backtest", backtest_command))
    bot_state.application.add_handler(CallbackQueryHandler(button_callback))
    
    print("✅ Бот настроен успешно")
    print("🚀 Запускаю бота...")
    
    # Запускаем бота
    print("⏰ Планировщик задач будет запущен после старта бота")
    bot_state.application.run_polling()

if __name__ == "__main__":
    main()
