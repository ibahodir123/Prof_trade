#!/usr/bin/env python3
"""
Binance Telegram бот для ML сигналов
Использует Binance API через ccxt
"""
import asyncio
import logging
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
import json
import ccxt
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import tensorflow as tf
from tensorflow.keras.models import load_model

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

def analyze_coin_signal(symbol):
    """Анализ монеты и генерация сигнала"""
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
            features = prepare_ml_features(df)
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
                features = prepare_ml_features(df)
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

# Глобальные переменные
current_coin = "BTC/USDT"
auto_signals_enabled = False
available_pairs = []
config = None
scheduler = None
application = None

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
💰 Текущая цена: ${current_price:.8f}
📈 RSI: {signal_data['rsi']:.1f}"""
        
        if signal_data['signal_type'] == "🟢 LONG":
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
    global available_pairs
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
        available_pairs = sorted(usdt_pairs)
        logger.info(f"✅ Найдено {len(available_pairs)} монет с Binance")
        return available_pairs
        
    except Exception as e:
        logger.error(f"❌ Ошибка получения монет с Binance: {e}")
        # Fallback на стандартный список
        available_pairs = [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
            'XRP/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'LINK/USDT',
            'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'FTM/USDT', 'ALGO/USDT',
            'VET/USDT', 'ICP/USDT', 'FIL/USDT', 'TRX/USDT', 'ETC/USDT'
        ]
        return available_pairs

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start с красивым меню"""
    global current_coin
    
    keyboard = [
        [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")],
        [InlineKeyboardButton("🪙 Выбор монет", callback_data="menu_coins")],
        [InlineKeyboardButton("📈 Последние сигналы", callback_data="menu_signals")],
        [InlineKeyboardButton("🔍 Анализ монеты", callback_data="menu_analyze")],
        [InlineKeyboardButton("🔍 Поиск монет", callback_data="menu_search")],
        [InlineKeyboardButton("🚀 Стреляющие монеты", callback_data="menu_shooting_stars")],
        [InlineKeyboardButton("🤖 Авто сигналы", callback_data="menu_auto")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = f"""
🤖 **Binance Trading Bot с ML сигналами!**

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
            await handle_signals_menu_new(query, context)
        elif query.data == "menu_analyze":
            await handle_analyze_menu(query, context)
        elif query.data == "menu_search":
            await handle_search_menu(query, context)
        elif query.data == "menu_auto":
            await handle_auto_menu(query, context)
        elif query.data == "menu_shooting_stars":
            await handle_shooting_stars_menu(query, context)
        elif query.data.startswith("select_"):
            await handle_coin_selection(query, context)
        elif query.data == "auto_start":
            await handle_auto_start(query, context)
        elif query.data == "auto_stop":
            await handle_auto_stop(query, context)
        elif query.data == "find_shooting_stars":
            await handle_find_shooting_stars(query, context)
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
🔗 **API:** Binance (ccxt)

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
        # Получаем список доступных пар с Binance
        if not available_pairs:
            await get_available_pairs()
        
        # Используем реальные пары с Binance
        popular_coins = available_pairs[:20]  # Первые 20 пар
        
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
        
        message = f"🪙 **Выберите монету для анализа (Binance):**\n\n📊 Доступно {len(available_pairs)} монет"
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения списка монет: {str(e)}")

async def handle_signals_menu_new(query, context):
    """Обработка кнопки Последние сигналы (новая версия - отправляет новое сообщение)"""
    try:
        signal_data = analyze_coin_signal(current_coin)
        if not signal_data:
            await query.message.reply_text(f"❌ Ошибка анализа {current_coin}")
            return
        
        # Проверяем, является ли это ошибкой "монета не найдена"
        if signal_data.get('error'):
            await query.message.reply_text(f"❌ {signal_data['error']}")
            return
        
        # Создание графика (только если есть данные)
        chart_buffer = None
        if signal_data.get('df') is not None:
            chart_buffer = create_advanced_trading_chart(current_coin, signal_data['df'], signal_data)
        
        if chart_buffer:
            # Отправка графика с подписью
            message = f"""
📈 **Сигнал для {current_coin}**

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
📈 **Сигнал для {current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
            keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await query.message.reply_text(message, reply_markup=reply_markup)
            
    except Exception as e:
        logger.error(f"❌ Ошибка получения сигналов: {e}")
        await query.message.reply_text(f"❌ Ошибка получения сигналов: {str(e)}")

async def handle_analyze_menu(query, context):
    """Обработка кнопки Анализ монеты"""
    await handle_signals_menu_new(query, context)  # Используем новую версию без ошибок редактирования

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
    
    # Отправляем новое сообщение вместо редактирования
    await query.message.reply_text(f"✅ Выбрана монета: {coin}")
    
    # Автоматически показываем анализ в новом сообщении
    await asyncio.sleep(1)
    await handle_signals_menu_new(query, context)

async def handle_find_shooting_stars(query, context):
    """Поиск стреляющих монет с помощью LSTM модели"""
    try:
        # Отправляем сообщение о начале анализа
        await query.edit_message_text("🔮 **Поиск стреляющих монет...**\n\n⏳ Анализирую все монеты на Binance...")
        
        # Получаем список всех монет
        available_pairs = get_available_pairs()
        
        # Ограничиваем анализ первыми 50 монетами для скорости
        pairs_to_analyze = available_pairs[:50]
        
        logger.info(f"🚀 Начинаю поиск стреляющих монет среди {len(pairs_to_analyze)} монет")
        
        shooting_stars = []
        analyzed_count = 0
        
        for symbol in pairs_to_analyze:
            try:
                # Получаем данные для анализа
                df = get_binance_data(symbol, '1h', 100)
                if df is None or len(df) < 50:
                    continue
                
                # Загружаем LSTM модель
                try:
                    model = load_model('simple_shooting_star_model.h5')
                    scaler = joblib.load('simple_shooting_star_scaler.pkl')
                    
                    # Подготавливаем данные для LSTM
                    features = prepare_lstm_features(df)
                    if features is None:
                        continue
                    
                    # Нормализуем данные
                    features_scaled = scaler.transform(features)
                    
                    # Делаем предсказание
                    prediction = model.predict(features_scaled[-1:].reshape(1, -1, features_scaled.shape[1]))
                    shooting_probability = prediction[0][0]
                    
                    # Если вероятность высокая, добавляем в список
                    if shooting_probability > 0.7:
                        current_price = df['close'].iloc[-1]
                        shooting_stars.append({
                            'symbol': symbol,
                            'probability': shooting_probability,
                            'price': current_price
                        })
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка LSTM анализа для {symbol}: {e}")
                    continue
                
                analyzed_count += 1
                
                # Обновляем прогресс каждые 10 монет
                if analyzed_count % 10 == 0:
                    progress_msg = f"🔮 **Поиск стреляющих монет...**\n\n📊 Проанализировано: {analyzed_count}/{len(pairs_to_analyze)}\n🎯 Найдено стреляющих: {len(shooting_stars)}"
                    await query.edit_message_text(progress_msg)
                
            except Exception as e:
                logger.error(f"❌ Ошибка анализа {symbol}: {e}")
                continue
        
        # Сортируем по вероятности
        shooting_stars.sort(key=lambda x: x['probability'], reverse=True)
        
        # Формируем результат
        if shooting_stars:
            message = f"""🚀 **СТРЕЛЯЮЩИЕ МОНЕТЫ НАЙДЕНЫ!**

📊 **Проанализировано:** {analyzed_count} монет
🎯 **Найдено стреляющих:** {len(shooting_stars)}

**🏆 ТОП-{min(10, len(shooting_stars))} СТРЕЛЯЮЩИХ МОНЕТ:**

"""
            
            for i, star in enumerate(shooting_stars[:10], 1):
                probability_pct = star['probability'] * 100
                message += f"""**{i}. {star['symbol']}** 🚀
💰 Цена: ${star['price']:.8f}
🎯 Вероятность: {probability_pct:.1f}%
📈 Потенциал: {'🔥' * min(5, int(probability_pct / 20))}

"""
            
            message += f"\n⏰ **Время анализа:** {datetime.now().strftime('%H:%M:%S')}"
            
        else:
            message = f"""🚀 **Поиск стреляющих монет завершен**

📊 **Проанализировано:** {analyzed_count} монет
🎯 **Стреляющих не найдено**

ℹ️ В данный момент нет монет с высокой вероятностью резкого роста.
Попробуйте позже или используйте обычный анализ монет.
"""
        
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="menu_shooting_stars")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
        logger.info(f"✅ Поиск стреляющих монет завершен: найдено {len(shooting_stars)} из {analyzed_count}")
        
    except Exception as e:
        logger.error(f"❌ Ошибка поиска стреляющих монет: {e}")
        await query.edit_message_text(f"❌ Ошибка поиска стреляющих монет: {str(e)}")

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

async def handle_auto_start(query, context):
    """Обработка запуска авто сигналов"""
    global auto_signals_enabled, scheduler
    
    try:
        auto_signals_enabled = True
        logger.info("🤖 Авто сигналы включены пользователем")
        
        # Добавляем задачу в планировщик (каждые 30 минут)
        if scheduler:
            scheduler.add_job(
                send_auto_signals,
                trigger=IntervalTrigger(minutes=30),
                id='auto_signals',
                replace_existing=True
            )
            logger.info("⏰ Планировщик автосигналов запущен (каждые 30 минут)")
        
        message = """
🤖 **Автоматические сигналы ЗАПУЩЕНЫ!**

✅ **Статус:** Включены
⏰ **Интервал:** Каждые 30 минут
📊 **Мониторинг:** Лучшие сигналы

Автоматические сигналы будут отправляться в этот чат с лучшими торговыми сигналами.
        """
        
        keyboard = [
            [InlineKeyboardButton("🛑 Остановить", callback_data="auto_stop")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"❌ Ошибка запуска авто сигналов: {e}")
        await query.edit_message_text(f"❌ Ошибка запуска авто сигналов: {str(e)}")

async def handle_auto_stop(query, context):
    """Обработка остановки авто сигналов"""
    global auto_signals_enabled, scheduler
    
    try:
        auto_signals_enabled = False
        logger.info("🤖 Авто сигналы остановлены пользователем")
        
        # Удаляем задачу из планировщика
        if scheduler:
            try:
                scheduler.remove_job('auto_signals')
                logger.info("⏰ Планировщик автосигналов остановлен")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось остановить планировщик: {e}")
        
        message = """
🤖 **Автоматические сигналы ОСТАНОВЛЕНЫ!**

❌ **Статус:** Выключены
⏰ **Интервал:** Неактивен
📊 **Мониторинг:** Приостановлен

Автоматические сигналы больше не будут отправляться.
        """
        
        keyboard = [
            [InlineKeyboardButton("🔄 Запустить", callback_data="auto_start")],
            [InlineKeyboardButton("🔙 Назад", callback_data="back_to_main")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
    except Exception as e:
        logger.error(f"❌ Ошибка остановки авто сигналов: {e}")
        await query.edit_message_text(f"❌ Ошибка остановки авто сигналов: {str(e)}")

async def send_auto_signals():
    """Отправка автоматических сигналов - анализирует все доступные монеты и выбирает топ-5"""
    global auto_signals_enabled, application, config, available_pairs
    
    if not auto_signals_enabled or not application or not config:
        return
    
    try:
        logger.info("🤖 Отправляю автосигналы...")
        
        # Получаем список всех доступных монет для анализа
        if not available_pairs:
            await get_available_pairs()
        
        # Анализируем ВСЕ доступные монеты (но с ограничением по времени)
        coins_to_check = available_pairs
        logger.info(f"📊 Анализирую {len(coins_to_check)} монет для автосигналов")
        
        all_signals = []
        analyzed_count = 0
        max_analysis_time = 300  # Максимум 5 минут на анализ
        start_time = datetime.now()
        
        for coin in coins_to_check:
            try:
                # Проверяем время выполнения
                elapsed_time = (datetime.now() - start_time).total_seconds()
                if elapsed_time > max_analysis_time:
                    logger.info(f"⏰ Время анализа истекло ({max_analysis_time}с), проанализировано {analyzed_count} монет")
                    break
                
                if analyzed_count % 10 == 0:  # Логируем каждые 10 монет
                    logger.info(f"🔍 Анализирую {coin}... ({analyzed_count}/{len(coins_to_check)})")
                
                signal_data = analyze_coin_signal(coin)
                if signal_data and signal_data.get('symbol'):
                    # Добавляем все сигналы (не только LONG)
                    all_signals.append((signal_data['symbol'], signal_data))
                
                analyzed_count += 1
                    
            except Exception as e:
                logger.warning(f"⚠️ Ошибка анализа {coin}: {e}")
                analyzed_count += 1
                continue
        
        if not all_signals:
            logger.info("ℹ️ Нет сигналов для отправки")
            return
        
        # Фильтруем LONG и SHORT сигналы и сортируем по силе
        trading_signals = []
        for coin, signal_data in all_signals:
            if signal_data.get('signal_type') in ['🟢 LONG', '🔴 SHORT']:
                # Извлекаем силу сигнала из strength_text
                strength_text = signal_data.get('strength_text', '')
                if 'Рост' in strength_text:
                    try:
                        # Извлекаем процент из текста "Рост 52.3%"
                        strength = float(strength_text.split('Рост ')[1].replace('%', '')) / 100
                        signal_data['signal_strength'] = strength
                    except:
                        signal_data['signal_strength'] = 0.5  # По умолчанию для LONG
                elif 'Падение' in strength_text:
                    try:
                        # Извлекаем процент из текста "Падение 45.2%"
                        strength = float(strength_text.split('Падение ')[1].replace('%', '')) / 100
                        signal_data['signal_strength'] = strength
                    except:
                        signal_data['signal_strength'] = 0.5  # По умолчанию для SHORT
                else:
                    signal_data['signal_strength'] = 0.5
                
                trading_signals.append((coin, signal_data))
        
        # Сортируем по силе сигнала
        trading_signals.sort(key=lambda x: x[1].get('signal_strength', 0), reverse=True)
        
        if trading_signals:
            # Берем топ-5 лучших сигналов
            top_signals = trading_signals[:5]
            
            # Создаем сообщение с топ-5 сигналами
            # Подсчитываем LONG и SHORT сигналы
            long_count = sum(1 for _, data in trading_signals if data.get('signal_type') == '🟢 LONG')
            short_count = sum(1 for _, data in trading_signals if data.get('signal_type') == '🔴 SHORT')
            
            message = f"""🤖 **АВТОМАТИЧЕСКИЕ СИГНАЛЫ**
⏰ **Время:** {datetime.now().strftime('%H:%M:%S')}
📊 **Проанализировано:** {analyzed_count} монет (из {len(available_pairs)})
🟢 **LONG сигналов:** {long_count}
🔴 **SHORT сигналов:** {short_count}

**🏆 ТОП-{len(top_signals)} ЛУЧШИХ СИГНАЛОВ:**

"""
            
            for i, (coin, signal_data) in enumerate(top_signals, 1):
                strength = signal_data.get('signal_strength', 0.7)
                signal_type = signal_data.get('signal_type', '⚪ ОЖИДАНИЕ')
                signal_emoji = "🟢" if "LONG" in signal_type else "🔴" if "SHORT" in signal_type else "⚪"
                signal_name = "LONG" if "LONG" in signal_type else "SHORT" if "SHORT" in signal_type else "ОЖИДАНИЕ"
                
                message += f"""**{i}. {coin}** {signal_emoji} {signal_name}
💰 Цена: ${signal_data['entry_price']:.8f}
📈 Сила: {strength*100:.1f}%
📊 RSI: {signal_data['rsi']:.1f}
🎯 TP: ${signal_data['take_profit']:.8f}
🛡️ SL: ${signal_data['stop_loss']:.8f}

"""
            
            # Отправляем сообщение
            await application.bot.send_message(
                chat_id=config['chat_id'],
                text=message
            )
            
            logger.info(f"✅ Автосигналы отправлены: топ-{len(top_signals)} из {len(trading_signals)} торговых сигналов")
        else:
            # Если нет торговых сигналов, показываем статистику
            message = f"""🤖 **АВТОМАТИЧЕСКИЕ СИГНАЛЫ**
⏰ **Время:** {datetime.now().strftime('%H:%M:%S')}
📊 **Проанализировано:** {analyzed_count} монет (из {len(available_pairs)})
🟢 **LONG сигналов:** 0
🔴 **SHORT сигналов:** 0

ℹ️ В данный момент нет сильных торговых сигналов.
Попробуйте позже или используйте /analyze для анализа конкретной монеты.
            """
            
            await application.bot.send_message(
                chat_id=config['chat_id'],
                text=message
            )
            
            logger.info("ℹ️ Нет LONG сигналов для отправки")
            
    except Exception as e:
        logger.error(f"❌ Ошибка отправки автосигналов: {e}")

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
🤖 **Binance Trading Bot с ML сигналами!**

🪙 **Текущая монета:** {current_coin}

**Выберите действие из меню ниже:**
    """
    
    try:
        # Пытаемся редактировать сообщение
        await query.edit_message_text(welcome_message, reply_markup=reply_markup)
    except Exception as e:
        # Если не получается редактировать (например, сообщение с фото), отправляем новое
        logger.warning(f"⚠️ Не удалось редактировать сообщение: {e}")
        await query.message.reply_text(welcome_message, reply_markup=reply_markup)

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
        
        global current_coin
        current_coin = coin
        
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
        await update.message.reply_text(f"🔍 Анализирую {current_coin}...")
        
        signal_data = analyze_coin_signal(current_coin)
        if not signal_data:
            await update.message.reply_text(f"❌ Ошибка анализа {current_coin}")
            return
        
        # Проверяем, является ли это ошибкой "монета не найдена"
        if signal_data.get('error'):
            await update.message.reply_text(f"❌ {signal_data['error']}")
            return
        
        # Создание графика (только если есть данные)
        chart_buffer = None
        if signal_data.get('df') is not None:
            chart_buffer = create_advanced_trading_chart(current_coin, signal_data['df'], signal_data)
        
        if chart_buffer:
            # Отправка графика с подписью
            message = f"""
📈 **Сигнал для {current_coin}**

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
            
            await update.message.reply_photo(
                photo=chart_buffer,
                caption=message
            )
        else:
            # Fallback без графика
            message = f"""
📈 **Сигнал для {current_coin}**

{signal_data['signal_type']} - {signal_data['strength_text']}

💰 **Цена входа:** ${signal_data['entry_price']:.8f}
📊 **RSI:** {signal_data['rsi']:.1f}
🤖 **ML статус:** {signal_data['ml_status']}
            """
            
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
        if available_pairs:
            matching_pairs = [pair for pair in available_pairs if search_term in pair]
        
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
- Доступно монет: {len(available_pairs)}

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

def main():
    """Основная функция"""
    global config, scheduler, application
    print("🤖 Запуск Binance ML Telegram Bot")
    print("🔄 Используем Binance API через ccxt")
    
    # Загружаем конфигурацию
    config = load_config()
    if not config:
        print("❌ Не удалось загрузить конфигурацию")
        return
    
    # Инициализируем список доступных пар
    print("🔍 Получаю список популярных монет с Binance...")
    try:
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(get_available_pairs())
        print(f"✅ Загружено {len(available_pairs)} монет с Binance")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки монет с Binance: {e}")
        print("🔄 Использую стандартный список")
    
    # Создаем приложение
    application = Application.builder().token(config["telegram_token"]).build()
    
    # Инициализируем планировщик после создания приложения
    scheduler = AsyncIOScheduler()
    print("⏰ Планировщик задач готов к запуску")
    
    # Функция для запуска планировщика после старта бота
    async def post_init(application):
        """Запуск планировщика после инициализации бота"""
        global scheduler
        try:
            scheduler.start()
            print("✅ Планировщик задач запущен успешно")
        except Exception as e:
            print(f"⚠️ Ошибка запуска планировщика: {e}")
    
    # Добавляем обработчик для запуска планировщика
    application.post_init = post_init
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("set_coin", set_coin_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("test_binance", test_binance_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    print("✅ Бот настроен успешно")
    print("🚀 Запускаю бота...")
    
    # Запускаем бота
    print("⏰ Планировщик задач будет запущен после старта бота")
    application.run_polling()

if __name__ == "__main__":
    main()
