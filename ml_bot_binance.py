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
import tensorflow as tf
from tensorflow.keras.models import load_model
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

def analyze_coin_signal_advanced_ema(symbol):
    """Анализ монеты с использованием продвинутой EMA логики"""
    global ema_analyzer, ml_trainer
    
    try:
        # Очищаем символ от дублирования USDT
        clean_symbol = symbol.replace(':USDT', '') if ':USDT' in symbol else symbol
        
        # Инициализация анализаторов
        if ema_analyzer is None:
            ema_analyzer = AdvancedEMAAnalyzer()
        
        if ml_trainer is None:
            ml_trainer = AdvancedMLTrainer()
            ml_trainer.load_models()  # Загружаем обученные модели
        
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
        
        # Анализ с продвинутой EMA логикой
        ema_analysis = ema_analyzer.analyze_coin(symbol, ohlcv_data)
        
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
        
        # Определение сигнала на основе EMA анализа
        signal_type = ema_analysis.get('signal', 'ОЖИДАНИЕ')
        
        # ML предсказание (если модели загружены)
        entry_prob = 0.0
        exit_prob = 0.0
        
        if ml_trainer.entry_model is not None and ml_trainer.exit_model is not None:
            # Подготавливаем признаки для ML
            df_processed = ema_analyzer.calculate_ema_features(df)
            feature_columns = [
                'ema20_speed', 'ema50_speed', 'ema100_speed',
                'price_speed_vs_ema20', 'price_speed_vs_ema50', 'price_speed_vs_ema100',
                'ema20_to_ema50', 'ema50_to_ema100', 'ema20_to_ema100',
                'price_to_ema20', 'price_to_ema50', 'price_to_ema100',
                'trend_angle', 'trend_type', 'market_phase'
            ]
            
            if all(col in df_processed.columns for col in feature_columns):
                features = df_processed[feature_columns].iloc[-1].values
                if len(features) > 0 and not np.isnan(features).any():
                    entry_prob, exit_prob = ml_trainer.predict_entry_exit(features)
        
        # Расчет силы сигнала на основе EMA анализа и ML
        strength = 0.3  # Базовая сила
        
        if signal_type == 'LONG':
            strength = 0.8 + (entry_prob * 0.2)  # EMA + ML
        elif signal_type == 'ТЕЙК ПРОФИТ':
            strength = 0.7 + (exit_prob * 0.3)   # EMA + ML
        elif entry_prob > 0.6:
            strength = 0.6 + (entry_prob * 0.3)  # Только ML
        elif exit_prob > 0.6:
            strength = 0.5 + (exit_prob * 0.3)   # Только ML
        
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
available_pairs = []
config = None
application = None

# Новые анализаторы
ema_analyzer = None
ml_trainer = None
shooting_predictor = None

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
        [InlineKeyboardButton("📈 EMA Анализ", callback_data="menu_ema_analysis")],
        [InlineKeyboardButton("🧠 Обучение ML", callback_data="menu_train_ml")],
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
        elif query.data == "menu_shooting_stars":
            await handle_shooting_stars_menu(query, context)
        elif query.data == "menu_ema_analysis":
            await handle_ema_analysis_menu(query, context)
        elif query.data == "menu_train_ml":
            await handle_train_ml_menu(query, context)
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
        signal_data = analyze_coin_signal_advanced_ema(current_coin)
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
    """Поиск стреляющих монет с помощью продвинутого анализа"""
    global shooting_predictor
    
    try:
        # Инициализируем предиктор если еще не инициализирован
        if shooting_predictor is None:
            shooting_predictor = ShootingStarPredictor()
        
        # Отправляем сообщение о начале анализа
        await query.edit_message_text("🔮 **Поиск стреляющих монет...**\n\n⏳ Анализирую все монеты на Binance...")
        
        # Получаем список всех монет
        available_pairs = await get_available_pairs()
        
        # Проверяем, что список не пустой
        if not available_pairs:
            await query.edit_message_text("❌ Не удалось получить список монет с Binance")
            return
        
        # Ограничиваем анализ первыми 50 монетами для скорости
        pairs_to_analyze = available_pairs[:50]
        
        logger.info(f"🚀 Начинаю поиск стреляющих монет среди {len(pairs_to_analyze)} монет")
        
        # Используем предиктор для поиска стреляющих звезд
        shooting_stars = shooting_predictor.find_shooting_stars(pairs_to_analyze, min_probability=0.4)
        
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
            [InlineKeyboardButton("🔙 Назад", callback_data="start")]
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
            [InlineKeyboardButton("🔙 Назад", callback_data="start")]
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
• Тренд: {ema_analysis.get('trend', 'Не определен')}
• Фаза: {ema_analysis.get('phase', 'Не определена')}
• Уверенность: {ema_analysis.get('confidence', 0)*100:.1f}%

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
        
        # Кнопка назад
        keyboard = [[InlineKeyboardButton("🔙 Назад", callback_data="ema_analyze_coin")]]
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
    global current_coin
    
    keyboard = [
        [InlineKeyboardButton("📊 Статус системы", callback_data="menu_status")],
        [InlineKeyboardButton("🪙 Выбор монет", callback_data="menu_coins")],
        [InlineKeyboardButton("📈 Последние сигналы", callback_data="menu_signals")],
        [InlineKeyboardButton("🔍 Анализ монеты", callback_data="menu_analyze")],
        [InlineKeyboardButton("🔍 Поиск монет", callback_data="menu_search")],
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
        
        signal_data = analyze_coin_signal_advanced_ema(current_coin)
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
    global config, application
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
