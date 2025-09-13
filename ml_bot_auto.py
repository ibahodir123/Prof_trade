#!/usr/bin/env python3
"""
Telegram бот для ML сигналов с автоматической отправкой каждые 30 минут
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
        profit_pct = 0.035 # +3.5%
        loss_pct = 0.025  # -2.5%
        strength_text = "💪 Сильный"
    else:
        # Средний сигнал
        profit_pct = 0.025 # +2.5%
        loss_pct = 0.02   # -2%
        strength_text = "⚡ Средний"
    
    return profit_pct, loss_pct, strength_text

def create_ml_features(df):
    """Создание признаков для ML модели"""
    import pandas as pd
    import numpy as np
    
    # Создаем копию данных
    df_features = df.copy()
    
    # Добавляем EMA если их нет
    if 'ema_20' not in df_features.columns:
        df_features['ema_20'] = df_features['close'].ewm(span=20).mean()
    if 'ema_50' not in df_features.columns:
        df_features['ema_50'] = df_features['close'].ewm(span=50).mean()
    if 'ema_100' not in df_features.columns:
        df_features['ema_100'] = df_features['close'].ewm(span=100).mean()
    
    # Создаем признаки
    df_features['price_ema20_ratio'] = df_features['close'] / df_features['ema_20']
    df_features['price_ema50_ratio'] = df_features['close'] / df_features['ema_50']
    df_features['price_ema100_ratio'] = df_features['close'] / df_features['ema_100']
    
    df_features['ema20_ema50_ratio'] = df_features['ema_20'] / df_features['ema_50']
    df_features['ema20_ema100_ratio'] = df_features['ema_20'] / df_features['ema_100']
    df_features['ema50_ema100_ratio'] = df_features['ema_50'] / df_features['ema_100']
    
    # Скорости изменения
    df_features['price_velocity'] = df_features['close'].pct_change()
    df_features['ema20_velocity'] = df_features['ema_20'].pct_change()
    df_features['ema50_velocity'] = df_features['ema_50'].pct_change()
    
    # Расстояния до EMA
    df_features['distance_to_ema20'] = (df_features['close'] - df_features['ema_20']) / df_features['ema_20']
    df_features['distance_to_ema50'] = (df_features['close'] - df_features['ema_50']) / df_features['ema_50']
    df_features['distance_to_ema100'] = (df_features['close'] - df_features['ema_100']) / df_features['ema_100']
    
    # Волатильность
    df_features['volatility'] = df_features['close'].rolling(20).std()
    
    # Корреляция с объемом
    df_features['volume_price_corr'] = df_features['volume'].rolling(20).corr(df_features['close'])
    
    # Заполняем NaN значения
    df_features = df_features.bfill().ffill()
    
    return df_features

def get_popular_coins():
    """Получаем список популярных монет для анализа"""
    return [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "SOL/USDT", "DOGE/USDT", "DOT/USDT", "AVAX/USDT", "MATIC/USDT",
        "LINK/USDT", "UNI/USDT", "LTC/USDT", "ATOM/USDT", "FIL/USDT",
        "TRX/USDT", "ETC/USDT", "XLM/USDT", "BCH/USDT", "ALGO/USDT"
    ]

def analyze_coin_signal(symbol, exchange):
    """Анализ одной монеты и получение сигнала"""
    try:
        # Получаем OHLCV данные
        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Добавляем EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Получаем текущую цену
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # Пытаемся загрузить ML модели
        try:
            minimum_detector = joblib.load('models/minimum_detector.pkl')
            maximum_detector = joblib.load('models/maximum_detector.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            # Создаем признаки для ML
            df_features = create_ml_features(df)
            
            # Берем последние данные для предсказания
            if len(df_features) > 0:
                features = df_features.iloc[-1:][feature_names].values
                features_scaled = scaler.transform(features)
                
                # Предсказания
                min_prob = minimum_detector.predict_proba(features_scaled)[0][1]
                max_prob = maximum_detector.predict_proba(features_scaled)[0][1]
                
                # Определяем сигнал с динамическими процентами
                if min_prob > 0.7:
                    signal_type = "🟢 ЛОНГ"
                    entry_price = current_price
                    
                    # Получаем динамические проценты
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(min_prob, "LONG")
                    
                    take_profit = current_price * (1 + profit_pct)
                    stop_loss = current_price * (1 - loss_pct)
                    probability = min_prob * 100
                    signal_strength = min_prob
                    
                elif max_prob > 0.7:
                    signal_type = "🔴 ШОРТ"
                    entry_price = current_price
                    
                    # Получаем динамические проценты
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(max_prob, "SHORT")
                    
                    take_profit = current_price * (1 - profit_pct)  # Для шорта
                    stop_loss = current_price * (1 + loss_pct)     # Для шорта
                    probability = max_prob * 100
                    signal_strength = max_prob
                    
                else:
                    return None  # Нет сильного сигнала
                    
                ml_status = "Активна"
                
            else:
                return None
                
        except Exception as e:
            # Fallback к простому анализу
            latest_close = df['close'].iloc[-1]
            ema_20_latest = df['ema_20'].iloc[-1]
            ema_50_latest = df['ema_50'].iloc[-1]
            
            if latest_close > ema_20_latest > ema_50_latest:
                signal_type = "🟢 ЛОНГ"
                entry_price = current_price
                take_profit = current_price * 1.035  # +3.5% для симуляции
                stop_loss = current_price * 0.975    # -2.5% для симуляции
                probability = 75.0
                signal_strength = 0.75
                strength_text = "💪 Сильный"
            elif latest_close < ema_20_latest < ema_50_latest:
                signal_type = "🔴 ШОРТ"
                entry_price = current_price
                take_profit = current_price * 0.965  # -3.5% для шорта
                stop_loss = current_price * 1.025    # +2.5% для шорта
                probability = 75.0
                signal_strength = 0.75
                strength_text = "💪 Сильный"
            else:
                return None  # Нет сильного сигнала
                
            ml_status = "Симуляция"
        
        # Рассчитываем потенциал
        price_change_pct = ((take_profit - entry_price) / entry_price) * 100
        
        return {
            'symbol': symbol,
            'signal_type': signal_type,
            'current_price': current_price,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'probability': probability,
            'signal_strength': signal_strength,
            'strength_text': strength_text,
            'price_change_pct': price_change_pct,
            'ml_status': ml_status,
            'chart_data': df  # Добавляем данные для графика
        }
        
    except Exception as e:
        logger.error(f"Ошибка анализа {symbol}: {e}")
        return None

def create_trading_chart(symbol, df, signal_data):
    """Создание графика в стиле TradingView с точками входа и тейк-профита"""
    try:
        # Настройка стиля графика в стиле TradingView
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(14, 10))
        fig.patch.set_facecolor('#131722')  # Темный фон TradingView
        
        # Создаем сетку для размещения элементов
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 4, 1], width_ratios=[1, 1, 1, 1])
        
        # Основной график
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.set_facecolor('#131722')
        
        # Берем последние 60 свечей для графика
        chart_data = df.tail(60)
        
        # Строим улучшенный свечной график
        for i, (timestamp, row) in enumerate(chart_data.iterrows()):
            # Определяем цвета как в TradingView
            if row['close'] >= row['open']:
                color = '#26a69a'  # Зеленый для роста
                alpha = 0.8
            else:
                color = '#ef5350'  # Красный для падения
                alpha = 0.8
            
            # Тени свечей
            ax_main.plot([i, i], [row['low'], row['high']], color=color, linewidth=1, alpha=alpha)
            # Тела свечей
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            ax_main.bar(i, body_height, bottom=body_bottom, width=0.8, color=color, alpha=alpha)
        
        # Добавляем EMA линии в стиле TradingView
        ax_main.plot(range(len(chart_data)), chart_data['ema_20'], color='#ffeb3b', linewidth=2, label='EMA 20', alpha=0.9)
        ax_main.plot(range(len(chart_data)), chart_data['ema_50'], color='#ff9800', linewidth=2, label='EMA 50', alpha=0.9)
        ax_main.plot(range(len(chart_data)), chart_data['ema_100'], color='#9c27b0', linewidth=2, label='EMA 100', alpha=0.9)
        
        # Получаем данные сигнала
        current_price = signal_data['current_price']
        entry_price = signal_data['entry_price']
        take_profit = signal_data['take_profit']
        stop_loss = signal_data['stop_loss']
        signal_type = signal_data['signal_type']
        
        # Определяем цветовую схему в зависимости от типа сигнала
        if "ЛОНГ" in signal_type:
            buy_color = '#26a69a'
            sell_color = '#ef5350'
            profit_zone_color = '#26a69a'
            loss_zone_color = '#ef5350'
        else:
            buy_color = '#ef5350'
            sell_color = '#26a69a'
            profit_zone_color = '#ef5350'
            loss_zone_color = '#26a69a'
        
        # Добавляем заштрихованные зоны как на TradingView
        chart_length = len(chart_data)
        
        # Зона тейк-профита (зеленая/красная зона выше/ниже цены)
        if "ЛОНГ" in signal_type:
            # Для лонга: зеленая зона выше текущей цены
            profit_zone = Rectangle((chart_length - 10, current_price), 10, take_profit - current_price, 
                                  facecolor=profit_zone_color, alpha=0.2, edgecolor=profit_zone_color, linewidth=2)
        else:
            # Для шорта: красная зона ниже текущей цены
            profit_zone = Rectangle((chart_length - 10, take_profit), 10, current_price - take_profit, 
                                  facecolor=profit_zone_color, alpha=0.2, edgecolor=profit_zone_color, linewidth=2)
        ax_main.add_patch(profit_zone)
        
        # Зона стоп-лосса (красная зона)
        if "ЛОНГ" in signal_type:
            # Для лонга: красная зона ниже текущей цены
            loss_zone = Rectangle((chart_length - 10, stop_loss), 10, current_price - stop_loss, 
                                facecolor=loss_zone_color, alpha=0.2, edgecolor=loss_zone_color, linewidth=2)
        else:
            # Для шорта: красная зона выше текущей цены
            loss_zone = Rectangle((chart_length - 10, current_price), 10, stop_loss - current_price, 
                                facecolor=loss_zone_color, alpha=0.2, edgecolor=loss_zone_color, linewidth=2)
        ax_main.add_patch(loss_zone)
        
        # Горизонтальные линии уровней как на TradingView
        ax_main.axhline(y=entry_price, color='#2196f3', linestyle='--', alpha=0.8, linewidth=2, label=f'Вход: ${entry_price:.4f}')
        ax_main.axhline(y=take_profit, color=profit_zone_color, linestyle='--', alpha=0.8, linewidth=2, label=f'Тейк-профит: ${take_profit:.4f}')
        ax_main.axhline(y=stop_loss, color=loss_zone_color, linestyle='--', alpha=0.8, linewidth=2, label=f'Стоп-лосс: ${stop_loss:.4f}')
        
        # Добавляем маркеры на текущую позицию
        current_pos = chart_length - 1
        ax_main.scatter([current_pos], [current_price], color='white', s=150, zorder=10, edgecolors='black', linewidth=2)
        ax_main.scatter([current_pos], [entry_price], color='#2196f3', s=120, zorder=10, edgecolors='white', linewidth=1)
        
        # Настройка основного графика
        ax_main.set_title(f'{symbol} - {signal_type}', fontsize=18, color='white', fontweight='bold', pad=20)
        ax_main.set_ylabel('Цена (USDT)', fontsize=14, color='white')
        ax_main.legend(loc='upper left', fontsize=11, framealpha=0.8)
        ax_main.grid(True, alpha=0.2, color='#2a2e39')
        ax_main.tick_params(colors='white')
        
        # Настройка осей X
        ax_main.set_xticks(range(0, len(chart_data), 10))
        ax_main.set_xticklabels([chart_data.index[i].strftime('%H:%M') for i in range(0, len(chart_data), 10)], 
                               rotation=45, color='white')
        
        # Создаем панель кнопок в стиле TradingView (в верхней части)
        ax_buttons = fig.add_subplot(gs[0, :])
        ax_buttons.set_facecolor('#131722')
        ax_buttons.axis('off')
        
        # Добавляем кнопки КУПИТЬ/ПРОДАТЬ с реальными ценами bid/ask
        button_width = 0.15
        button_height = 0.6
        
        # Рассчитываем реальные цены bid/ask (спред ~0.1%)
        bid_price = current_price * 0.9995  # Цена продажи (немного ниже)
        ask_price = current_price * 1.0005  # Цена покупки (немного выше)
        
        # Кнопка ПРОДАТЬ (красная) - показывает цену bid
        sell_rect = Rectangle((0.05, 0.2), button_width, button_height, 
                            facecolor='#ef5350', edgecolor='white', linewidth=2)
        ax_buttons.add_patch(sell_rect)
        ax_buttons.text(0.05 + button_width/2, 0.5, 'ПРОДАТЬ', ha='center', va='center', 
                       color='white', fontsize=14, fontweight='bold')
        ax_buttons.text(0.05 + button_width/2, 0.35, f'${bid_price:.4f}', ha='center', va='center', 
                       color='white', fontsize=12)
        
        # Кнопка КУПИТЬ (синяя) - показывает цену ask
        buy_rect = Rectangle((0.25, 0.2), button_width, button_height, 
                           facecolor='#26a69a', edgecolor='white', linewidth=2)
        ax_buttons.add_patch(buy_rect)
        ax_buttons.text(0.25 + button_width/2, 0.5, 'КУПИТЬ', ha='center', va='center', 
                       color='white', fontsize=14, fontweight='bold')
        ax_buttons.text(0.25 + button_width/2, 0.35, f'${ask_price:.4f}', ha='center', va='center', 
                       color='white', fontsize=12)
        
        # Добавляем информацию о текущей цене
        ax_buttons.text(0.5, 0.7, f'Текущая цена: ${current_price:.4f}', 
                       color='white', fontsize=16, fontweight='bold')
        ax_buttons.text(0.5, 0.5, f'Таймфрейм: 1H', 
                       color='#9e9e9e', fontsize=12)
        ax_buttons.text(0.5, 0.3, f'Вероятность: {signal_data["probability"]:.1f}%', 
                       color='#4caf50', fontsize=14, fontweight='bold')
        
        # Создаем информационную панель (в нижней части)
        ax_info = fig.add_subplot(gs[2, :])
        ax_info.set_facecolor('#131722')
        ax_info.axis('off')
        
        # Добавляем детальную информацию в зависимости от типа сигнала
        if "ОЖИДАНИЕ" in signal_type and "падение" in signal_data["strength_text"]:
            info_text = f"""
        📉 АНАЛИЗ:  Цена: ${entry_price:.4f}  |  Потенциальное падение: -{abs(signal_data["price_change_pct"]):.2f}%
        📊 Статус: {signal_data["strength_text"]}  |  ML статус: {signal_data["ml_status"]}
        """
        else:
            info_text = f"""
        🎯 ТОЧКИ ТОРГОВЛИ:  Вход: ${entry_price:.4f}  |  Тейк-профит: ${take_profit:.4f}  |  Стоп-лосс: ${stop_loss:.4f}
        📊 Потенциал: {signal_data["price_change_pct"]:+.2f}%  |  Сила сигнала: {signal_data["strength_text"]}  |  ML статус: {signal_data["ml_status"]}
        """
        ax_info.text(0.02, 0.5, info_text, color='white', fontsize=11, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#2a2e39', alpha=0.8))
        
        # Убираем отступы
        plt.tight_layout()
        
        # Сохраняем график в байты
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                   facecolor='#131722', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        return buffer
        
    except Exception as e:
        logger.error(f"Ошибка создания графика для {symbol}: {e}")
        return None

def create_advanced_trading_chart(symbol, df, signal_data):
    """Создание продвинутого графика в стиле TradingView с дополнительными индикаторами"""
    try:
        logger.info(f"Начинаю создание продвинутого графика для {symbol}")
        # Настройка стиля графика в стиле TradingView
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor('#131722')
        
        # Создаем сетку для размещения элементов (4 ряда)
        gs = fig.add_gridspec(4, 4, height_ratios=[1, 3, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # Основной график цены
        ax_main = fig.add_subplot(gs[1, :])
        ax_main.set_facecolor('#131722')
        
        # Берем последние 80 свечей для графика
        chart_data = df.tail(80)
        
        # Строим профессиональный свечной график
        for i, (timestamp, row) in enumerate(chart_data.iterrows()):
            # Определяем цвета как в TradingView
            if row['close'] >= row['open']:
                color = '#26a69a'  # Зеленый для роста
                alpha = 0.9
            else:
                color = '#ef5350'  # Красный для падения
                alpha = 0.9
            
            # Тени свечей (high-low)
            ax_main.plot([i, i], [row['low'], row['high']], color=color, linewidth=1.5, alpha=alpha)
            
            # Тела свечей (open-close)
            body_height = abs(row['close'] - row['open'])
            body_bottom = min(row['open'], row['close'])
            if body_height > 0:
                ax_main.bar(i, body_height, bottom=body_bottom, width=0.7, color=color, alpha=alpha)
            else:
                # Доджи (open == close)
                ax_main.plot([i-0.3, i+0.3], [row['close'], row['close']], color=color, linewidth=2, alpha=alpha)
        
        # Добавляем EMA линии с улучшенными цветами
        ax_main.plot(range(len(chart_data)), chart_data['ema_20'], color='#ffeb3b', linewidth=2.5, label='EMA 20', alpha=0.9)
        ax_main.plot(range(len(chart_data)), chart_data['ema_50'], color='#ff9800', linewidth=2.5, label='EMA 50', alpha=0.9)
        ax_main.plot(range(len(chart_data)), chart_data['ema_100'], color='#9c27b0', linewidth=2.5, label='EMA 100', alpha=0.9)
        
        # Получаем данные сигнала
        current_price = signal_data['current_price']
        entry_price = signal_data['entry_price']
        take_profit = signal_data['take_profit']
        stop_loss = signal_data['stop_loss']
        signal_type = signal_data['signal_type']
        
        # Определяем цветовую схему
        if "ЛОНГ" in signal_type:
            profit_zone_color = '#26a69a'
            loss_zone_color = '#ef5350'
            signal_color = '#26a69a'
        else:
            profit_zone_color = '#ef5350'
            loss_zone_color = '#26a69a'
            signal_color = '#ef5350'
        
        # Добавляем визуальные элементы в зависимости от типа сигнала
        chart_length = len(chart_data)
        zone_width = 15
        
        if "ОЖИДАНИЕ" in signal_type and "падение" in signal_data["strength_text"]:
            # Для сигналов ожидания с падением показываем только текущую цену
            ax_main.axhline(y=entry_price, color='#ff9800', linestyle='--', alpha=0.8, linewidth=2.5, label=f'Текущая цена: ${entry_price:.4f}')
        else:
            # Для обычных сигналов показываем все торговые уровни
            # Зона тейк-профита
            if "ЛОНГ" in signal_type:
                profit_zone = Rectangle((chart_length - zone_width, current_price), zone_width, take_profit - current_price, 
                                      facecolor=profit_zone_color, alpha=0.15, edgecolor=profit_zone_color, linewidth=2)
            else:
                profit_zone = Rectangle((chart_length - zone_width, take_profit), zone_width, current_price - take_profit, 
                                      facecolor=profit_zone_color, alpha=0.15, edgecolor=profit_zone_color, linewidth=2)
            ax_main.add_patch(profit_zone)
            
            # Зона стоп-лосса
            if "ЛОНГ" in signal_type:
                loss_zone = Rectangle((chart_length - zone_width, stop_loss), zone_width, current_price - stop_loss, 
                                    facecolor=loss_zone_color, alpha=0.15, edgecolor=loss_zone_color, linewidth=2)
            else:
                loss_zone = Rectangle((chart_length - zone_width, current_price), zone_width, stop_loss - current_price, 
                                    facecolor=loss_zone_color, alpha=0.15, edgecolor=loss_zone_color, linewidth=2)
            ax_main.add_patch(loss_zone)
            
            # Горизонтальные линии уровней
            ax_main.axhline(y=entry_price, color='#2196f3', linestyle='--', alpha=0.8, linewidth=2.5, label=f'Вход: ${entry_price:.4f}')
            ax_main.axhline(y=take_profit, color=profit_zone_color, linestyle='--', alpha=0.8, linewidth=2.5, label=f'Тейк-профит: ${take_profit:.4f}')
            ax_main.axhline(y=stop_loss, color=loss_zone_color, linestyle='--', alpha=0.8, linewidth=2.5, label=f'Стоп-лосс: ${stop_loss:.4f}')
        
        # Добавляем маркеры на текущую позицию
        current_pos = chart_length - 1
        ax_main.scatter([current_pos], [current_price], color='white', s=200, zorder=15, edgecolors='black', linewidth=3)
        ax_main.scatter([current_pos], [entry_price], color='#2196f3', s=150, zorder=15, edgecolors='white', linewidth=2)
        
        # Настройка основного графика
        ax_main.set_title(f'{symbol} - {signal_type}', fontsize=20, color='white', fontweight='bold', pad=25)
        ax_main.set_ylabel('Цена (USDT)', fontsize=16, color='white')
        ax_main.legend(loc='upper left', fontsize=12, framealpha=0.9)
        ax_main.grid(True, alpha=0.15, color='#2a2e39')
        ax_main.tick_params(colors='white', labelsize=12)
        
        # Настройка осей X
        ax_main.set_xticks(range(0, len(chart_data), 15))
        ax_main.set_xticklabels([chart_data.index[i].strftime('%H:%M') for i in range(0, len(chart_data), 15)], 
                               rotation=45, color='white')
        
        # Создаем панель кнопок в стиле TradingView
        ax_buttons = fig.add_subplot(gs[0, :])
        ax_buttons.set_facecolor('#131722')
        ax_buttons.axis('off')
        
        # Рассчитываем реальные цены bid/ask
        bid_price = current_price * 0.9995
        ask_price = current_price * 1.0005
        
        # Улучшенные кнопки КУПИТЬ/ПРОДАТЬ
        button_width = 0.12
        button_height = 0.7
        
        # Кнопка ПРОДАТЬ
        sell_rect = FancyBboxPatch((0.05, 0.15), button_width, button_height, 
                                  boxstyle="round,pad=0.02", facecolor='#ef5350', 
                                  edgecolor='white', linewidth=2)
        ax_buttons.add_patch(sell_rect)
        ax_buttons.text(0.05 + button_width/2, 0.55, 'ПРОДАТЬ', ha='center', va='center', 
                       color='white', fontsize=15, fontweight='bold')
        ax_buttons.text(0.05 + button_width/2, 0.35, f'${bid_price:.4f}', ha='center', va='center', 
                       color='white', fontsize=13)
        
        # Кнопка КУПИТЬ
        buy_rect = FancyBboxPatch((0.22, 0.15), button_width, button_height, 
                                 boxstyle="round,pad=0.02", facecolor='#26a69a', 
                                 edgecolor='white', linewidth=2)
        ax_buttons.add_patch(buy_rect)
        ax_buttons.text(0.22 + button_width/2, 0.55, 'КУПИТЬ', ha='center', va='center', 
                       color='white', fontsize=15, fontweight='bold')
        ax_buttons.text(0.22 + button_width/2, 0.35, f'${ask_price:.4f}', ha='center', va='center', 
                       color='white', fontsize=13)
        
        # Информационная панель
        ax_buttons.text(0.45, 0.75, f'Текущая цена: ${current_price:.4f}', 
                       color='white', fontsize=18, fontweight='bold')
        ax_buttons.text(0.45, 0.55, f'Таймфрейм: 1H', 
                       color='#9e9e9e', fontsize=14)
        ax_buttons.text(0.45, 0.35, f'Вероятность: {signal_data["probability"]:.1f}%', 
                       color='#4caf50', fontsize=16, fontweight='bold')
        
        # Создаем панель RSI
        ax_rsi = fig.add_subplot(gs[2, :])
        ax_rsi.set_facecolor('#131722')
        
        # Рассчитываем RSI
        def calculate_rsi(prices, period=14):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        
        rsi = calculate_rsi(chart_data['close'])
        
        # Рисуем RSI
        ax_rsi.plot(range(len(chart_data)), rsi, color='#ff9800', linewidth=2)
        ax_rsi.axhline(y=70, color='#ef5350', linestyle='--', alpha=0.7, label='Перекупленность')
        ax_rsi.axhline(y=30, color='#26a69a', linestyle='--', alpha=0.7, label='Перепроданность')
        ax_rsi.axhline(y=50, color='#9e9e9e', linestyle='-', alpha=0.5)
        
        ax_rsi.set_title('RSI (14)', fontsize=14, color='white', fontweight='bold')
        ax_rsi.set_ylabel('RSI', fontsize=12, color='white')
        ax_rsi.set_ylim(0, 100)
        ax_rsi.grid(True, alpha=0.15)
        ax_rsi.tick_params(colors='white')
        ax_rsi.legend(fontsize=10)
        
        # Создаем информационную панель
        ax_info = fig.add_subplot(gs[3, :])
        ax_info.set_facecolor('#131722')
        ax_info.axis('off')
        
        # Детальная информация в зависимости от типа сигнала
        if "ОЖИДАНИЕ" in signal_type and "падение" in signal_data["strength_text"]:
            info_text = f"""
        📉 АНАЛИЗ:  Цена: ${entry_price:.4f}  |  Потенциальное падение: -{abs(signal_data["price_change_pct"]):.2f}%
        📊 СТАТУС:  {signal_data["strength_text"]}  |  ML: {signal_data["ml_status"]}
        📈 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:  EMA20: ${chart_data['ema_20'].iloc[-1]:.4f}  |  EMA50: ${chart_data['ema_50'].iloc[-1]:.4f}  |  RSI: {rsi.iloc[-1]:.1f}
        """
        else:
            info_text = f"""
        🎯 ТОРГОВЫЕ УРОВНИ:  Вход: ${entry_price:.4f}  |  Тейк-профит: ${take_profit:.4f}  |  Стоп-лосс: ${stop_loss:.4f}
        📊 АНАЛИЗ:  Потенциал: {signal_data["price_change_pct"]:+.2f}%  |  Сила: {signal_data["strength_text"]}  |  ML: {signal_data["ml_status"]}
        📈 ТЕХНИЧЕСКИЕ ИНДИКАТОРЫ:  EMA20: ${chart_data['ema_20'].iloc[-1]:.4f}  |  EMA50: ${chart_data['ema_50'].iloc[-1]:.4f}  |  RSI: {rsi.iloc[-1]:.1f}
        """
        ax_info.text(0.02, 0.5, info_text, color='white', fontsize=12, 
                    bbox=dict(boxstyle="round,pad=0.4", facecolor='#2a2e39', alpha=0.9))
        
        plt.tight_layout()
        
        # Сохраняем график
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=200, bbox_inches='tight', 
                   facecolor='#131722', edgecolor='none')
        buffer.seek(0)
        plt.close()
        
        logger.info(f"Продвинутый график для {symbol} создан успешно")
        return buffer
        
    except Exception as e:
        logger.error(f"Ошибка создания продвинутого графика для {symbol}: {e}")
        return None

def test_chart_creation():
    """Тестовая функция для проверки создания графиков"""
    try:
        logger.info("Тестирую создание простого графика...")
        
        # Создаем простой тестовый график
        plt.figure(figsize=(10, 6))
        plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
        plt.title('Тестовый график')
        
        # Сохраняем в байты
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150)
        buffer.seek(0)
        plt.close()
        
        logger.info(f"Тестовый график создан успешно, размер: {len(buffer.getvalue())} байт")
        return buffer
        
    except Exception as e:
        logger.error(f"Ошибка создания тестового графика: {e}")
        return None

async def send_auto_signals(context: ContextTypes.DEFAULT_TYPE):
    """Автоматическая отправка лучших сигналов каждые 30 минут"""
    try:
        logger.info("🔍 Начинаю анализ всех монет...")
        
        exchange = ccxt.binance()
        popular_coins = get_popular_coins()
        
        # Анализируем все монеты
        signals = []
        for coin in popular_coins:
            signal = analyze_coin_signal(coin, exchange)
            if signal:
                signals.append(signal)
            time.sleep(0.1)  # Небольшая пауза между запросами
        
        if not signals:
            logger.info("❌ Сильных сигналов не найдено")
            return
        
        # Сортируем по силе сигнала (от большего к меньшему)
        signals.sort(key=lambda x: x['signal_strength'], reverse=True)
        
        # Берем топ-3 лучших сигнала
        top_signals = signals[:3]
        
        # Формируем сообщение
        message = f"""
🚨 **АВТОМАТИЧЕСКИЕ СИГНАЛЫ** 🚨
🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')}

📊 **Проанализировано монет:** {len(popular_coins)}
🎯 **Найдено сигналов:** {len(signals)}
🏆 **Топ-{len(top_signals)} лучших сигналов:**

"""
        
        for i, signal in enumerate(top_signals, 1):
            message += f"""
**{i}. {signal['symbol']}** {signal['signal_type']}
💰 **Цена:** ${signal['current_price']:,.4f}
📈 **Прогноз:** ${signal['take_profit']:,.4f} ({signal['price_change_pct']:+.1f}%)
🎯 **Вероятность:** {signal['probability']:.1f}% {signal['strength_text']}

🎯 **ТОЧКИ ТОРГОВЛИ:**
• **Вход:** ${signal['entry_price']:,.4f}
• **Тейк-профит:** ${signal['take_profit']:,.4f}
• **Стоп-лосс:** ${signal['stop_loss']:,.4f}

"""
        
        message += f"""
🤖 **ML статус:** {top_signals[0]['ml_status']}
⏰ **Следующий анализ через 30 минут**

💡 **Используйте /analyze для детального анализа конкретной монеты**
        """
        
        # Отправляем сообщение
        await context.bot.send_message(
            chat_id=context.job.data['chat_id'],
            text=message
        )
        
        logger.info(f"✅ Отправлено {len(top_signals)} автоматических сигналов")
        
    except Exception as e:
        logger.error(f"❌ Ошибка автоматических сигналов: {e}")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /start с красивым меню"""
    global current_coin
    
    # Создаем главное меню
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

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /status"""
    try:
        import ccxt
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(current_coin)
        current_price = ticker['last']
        
        # Проверяем статус автоматических сигналов
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        auto_status = "🟢 Активны" if jobs else "🔴 Остановлены"
        
        status_message = f"""
📊 **Статус бота:**

✅ Бот активен
🪙 Текущая монета: {current_coin}
📈 Цена {current_coin}: ${current_price:,.4f}
📡 Источник данных: Binance
🤖 ML модель: Активна
🎯 Динамические проценты: Включены
🕐 Автоматические сигналы: {auto_status}

Бот работает корректно! 🎉
        """
        await update.message.reply_text(status_message)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения статуса: {str(e)}")

async def auto_start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /auto_start - запуск автоматических сигналов"""
    try:
        # Останавливаем существующие задачи
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        for job in jobs:
            job.schedule_removal()
        
        # Запускаем новую задачу
        context.job_queue.run_repeating(
            send_auto_signals,
            interval=1800,  # 30 минут = 1800 секунд
            first=10,       # Первый запуск через 10 секунд
            name="auto_signals",
            data={'chat_id': update.effective_chat.id}
        )
        
        await update.message.reply_text("""
✅ **Автоматические сигналы запущены!**

🕐 **Интервал:** каждые 30 минут
📊 **Анализ:** 20 популярных монет
🏆 **Отправка:** топ-3 лучших сигнала
⏰ **Первый сигнал:** через 10 секунд

Используйте /auto_stop для остановки
        """)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка запуска автоматических сигналов: {str(e)}")

async def auto_stop_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /auto_stop - остановка автоматических сигналов"""
    try:
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        for job in jobs:
            job.schedule_removal()
        
        await update.message.reply_text("""
🛑 **Автоматические сигналы остановлены!**

Используйте /auto_start для повторного запуска
        """)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка остановки автоматических сигналов: {str(e)}")

async def coins_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /coins"""
    try:
        # Используем наш список популярных монет
        popular_coins = get_popular_coins()[:10]
        
        keyboard = []
        for coin in popular_coins:
            keyboard.append([InlineKeyboardButton(coin, callback_data=f"select_{coin}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = "🪙 **Выберите монету для анализа:**"
        await update.message.reply_text(message, reply_markup=reply_markup)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения списка монет: {str(e)}")

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /signals с динамическими процентами"""
    global current_coin
    
    try:
        # Получаем данные для текущей монеты
        import ccxt
        import pandas as pd
        import numpy as np
        import joblib
        from datetime import datetime
        
        exchange = ccxt.binance()
        
        # Получаем OHLCV данные
        ohlcv = exchange.fetch_ohlcv(current_coin, '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Добавляем EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Получаем текущую цену
        ticker = exchange.fetch_ticker(current_coin)
        current_price = ticker['last']
        
        # Пытаемся загрузить ML модели
        try:
            minimum_detector = joblib.load('models/minimum_detector.pkl')
            maximum_detector = joblib.load('models/maximum_detector.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            # Создаем признаки для ML
            df_features = create_ml_features(df)
            
            # Берем последние данные для предсказания
            if len(df_features) > 0:
                features = df_features.iloc[-1:][feature_names].values
                features_scaled = scaler.transform(features)
                
                # Предсказания
                min_prob = minimum_detector.predict_proba(features_scaled)[0][1]
                max_prob = maximum_detector.predict_proba(features_scaled)[0][1]
                
                # Определяем сигнал с динамическими процентами
                if min_prob > 0.7:
                    signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                    entry_price = current_price
                    
                    # Получаем динамические проценты
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(min_prob, "LONG")
                    
                    take_profit = current_price * (1 + profit_pct)
                    stop_loss = current_price * (1 - loss_pct)
                    probability = min_prob * 100
                    
                elif max_prob > 0.7:
                    # Отключаем ШОРТ сигналы, показываем только как ожидание с информацией о падении
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    
                    # Рассчитываем потенциальное падение
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(max_prob, "SHORT")
                    potential_fall_pct = profit_pct * 100  # Процент потенциального падения
                    
                    take_profit = current_price * 1.02  # Консервативный +2%
                    stop_loss = current_price * 0.98   # Консервативный -2%
                    probability = max_prob * 100
                    strength_text = f"⚪ Слабый (падение {potential_fall_pct:.1f}%)"
                    
                else:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    take_profit = current_price * 1.02  # Консервативный +2%
                    stop_loss = current_price * 0.98   # Консервативный -2%
                    probability = max(min_prob, max_prob) * 100
                    strength_text = "⚪ Слабый"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02
                stop_loss = current_price * 0.98
                probability = 50.0
                strength_text = "⚪ Слабый"
                
            ml_status = "Активна"
            
        except Exception as e:
            # Fallback к простому анализу
            latest_close = df['close'].iloc[-1]
            ema_20_latest = df['ema_20'].iloc[-1]
            ema_50_latest = df['ema_50'].iloc[-1]
            
            if latest_close > ema_20_latest > ema_50_latest:
                signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                entry_price = current_price
                take_profit = current_price * 1.035  # +3.5% для симуляции
                stop_loss = current_price * 0.975    # -2.5% для симуляции
                probability = 75.0
                strength_text = "💪 Сильный"
            elif latest_close < ema_20_latest < ema_50_latest:
                # Отключаем ШОРТ сигналы в fallback тоже, но показываем потенциальное падение
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02  # Консервативный +2%
                stop_loss = current_price * 0.98   # Консервативный -2%
                probability = 75.0
                strength_text = "⚪ Слабый (падение 3.5%)"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02
                stop_loss = current_price * 0.98
                probability = 50.0
                strength_text = "⚪ Слабый"
                
            ml_status = "Симуляция"
        
        # Рассчитываем потенциал
        price_change_pct = ((take_profit - entry_price) / entry_price) * 100
        
        # Создаем данные сигнала для графика
        signal_data = {
            'symbol': current_coin,
            'signal_type': signal_type,
            'current_price': current_price,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'probability': probability,
            'strength_text': strength_text,
            'price_change_pct': price_change_pct,
            'ml_status': ml_status
        }
        
        # Формируем сообщение в зависимости от типа сигнала
        if "ОЖИДАНИЕ" in signal_type and "падение" in strength_text:
            # Для сигналов ожидания с падением не показываем торговые уровни
            message = f"""
📊 **Последние сигналы для {current_coin}:**

🚨 {signal_type}

💰 **Цена:** ${current_price:,.4f}
📉 **Потенциальное падение:** -{abs(price_change_pct):.1f}%
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📉 Ожидание лучшего входа

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}

💡 **Используйте /analyze для детального анализа**
            """
        else:
            # Для обычных сигналов показываем полную информацию
            message = f"""
📊 **Последние сигналы для {current_coin}:**

🚨 {signal_type}

💰 **Цена:** ${current_price:,.4f}
📈 **Прогноз:** ${take_profit:,.4f} ({price_change_pct:+.1f}%)
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📈 Восходящий тренд

🎯 **ТОЧКИ ТОРГОВЛИ:**
• **Вход:** ${entry_price:,.4f}
• **Тейк-профит:** ${take_profit:,.4f}
• **Стоп-лосс:** ${stop_loss:,.4f}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}

💡 **Используйте /analyze для детального анализа**
            """
        
        # Создаем и отправляем график
        try:
            global use_advanced_chart
            logger.info(f"Создание графика для {current_coin}, продвинутый режим: {use_advanced_chart}")
            
            if use_advanced_chart:
                logger.info("Создаю продвинутый график...")
                chart_buffer = create_advanced_trading_chart(current_coin, df, signal_data)
                chart_caption = f"📈 Продвинутый график {current_coin} в стиле TradingView\n🎯 С кнопками КУПИТЬ/ПРОДАТЬ и зонами тейк-профита"
            else:
                logger.info("Создаю обычный график...")
                chart_buffer = create_trading_chart(current_coin, df, signal_data)
                chart_caption = f"📈 График {current_coin} с торговыми уровнями"
            
            if chart_buffer:
                logger.info("График создан успешно, отправляю...")
                # Отправляем график с текстом в caption (график сверху, текст снизу)
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=chart_buffer,
                    caption=message
                )
                logger.info("График с текстом отправлен успешно")
            else:
                logger.error("Не удалось создать график, отправляю только текст")
                await update.message.reply_text(message)
        except Exception as chart_error:
            logger.error(f"Ошибка создания графика: {chart_error}")
            import traceback
            logger.error(f"Подробности ошибки: {traceback.format_exc()}")
            await update.message.reply_text(message)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка получения сигналов: {str(e)}")

async def analyze_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /analyze с динамическими процентами"""
    global current_coin
    
    try:
        # Получаем данные для выбранной монеты
        import ccxt
        import pandas as pd
        import numpy as np
        import joblib
        from datetime import datetime
        
        exchange = ccxt.binance()
        
        # Получаем OHLCV данные
        ohlcv = exchange.fetch_ohlcv(current_coin, '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Добавляем EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Получаем текущую цену
        ticker = exchange.fetch_ticker(current_coin)
        current_price = ticker['last']
        
        # Пытаемся загрузить ML модели
        try:
            minimum_detector = joblib.load('models/minimum_detector.pkl')
            maximum_detector = joblib.load('models/maximum_detector.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            # Создаем признаки для ML
            df_features = create_ml_features(df)
            
            # Берем последние данные для предсказания
            if len(df_features) > 0:
                features = df_features.iloc[-1:][feature_names].values
                features_scaled = scaler.transform(features)
                
                # Предсказания
                min_prob = minimum_detector.predict_proba(features_scaled)[0][1]
                max_prob = maximum_detector.predict_proba(features_scaled)[0][1]
                
                # Определяем сигнал с динамическими процентами
                if min_prob > 0.7:
                    signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                    entry_price = current_price
                    
                    # Получаем динамические проценты
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(min_prob, "LONG")
                    
                    take_profit = current_price * (1 + profit_pct)
                    stop_loss = current_price * (1 - loss_pct)
                    probability = min_prob * 100
                    
                elif max_prob > 0.7:
                    # Отключаем ШОРТ сигналы, показываем только как ожидание с информацией о падении
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    
                    # Рассчитываем потенциальное падение
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(max_prob, "SHORT")
                    potential_fall_pct = profit_pct * 100  # Процент потенциального падения
                    
                    take_profit = current_price * 1.02  # Консервативный +2%
                    stop_loss = current_price * 0.98   # Консервативный -2%
                    probability = max_prob * 100
                    strength_text = f"⚪ Слабый (падение {potential_fall_pct:.1f}%)"
                    
                else:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    take_profit = current_price * 1.02  # Консервативный +2%
                    stop_loss = current_price * 0.98   # Консервативный -2%
                    probability = max(min_prob, max_prob) * 100
                    strength_text = "⚪ Слабый"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02
                stop_loss = current_price * 0.98
                probability = 50.0
                strength_text = "⚪ Слабый"
                
            ml_status = "Активна"
            
        except Exception as e:
            # Fallback к простому анализу
            latest_close = df['close'].iloc[-1]
            ema_20_latest = df['ema_20'].iloc[-1]
            ema_50_latest = df['ema_50'].iloc[-1]
            
            if latest_close > ema_20_latest > ema_50_latest:
                signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                entry_price = current_price
                take_profit = current_price * 1.035  # +3.5% для симуляции
                stop_loss = current_price * 0.975    # -2.5% для симуляции
                probability = 75.0
                strength_text = "💪 Сильный"
            elif latest_close < ema_20_latest < ema_50_latest:
                # Отключаем ШОРТ сигналы в fallback тоже, но показываем потенциальное падение
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02  # Консервативный +2%
                stop_loss = current_price * 0.98   # Консервативный -2%
                probability = 75.0
                strength_text = "⚪ Слабый (падение 3.5%)"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02
                stop_loss = current_price * 0.98
                probability = 50.0
                strength_text = "⚪ Слабый"
                
            ml_status = "Симуляция"
        
        # Рассчитываем потенциал
        price_change_pct = ((take_profit - entry_price) / entry_price) * 100
        
        # Создаем данные сигнала для графика
        signal_data = {
            'symbol': current_coin,
            'signal_type': signal_type,
            'current_price': current_price,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'probability': probability,
            'strength_text': strength_text,
            'price_change_pct': price_change_pct,
            'ml_status': ml_status
        }
        
        # Добавляем информацию о потенциальном падении для отключенных шорт-сигналов
        potential_fall_info = ""
        if "ОЖИДАНИЕ" in signal_type and "падение" in strength_text:
            if "max_prob" in locals() and max_prob > 0.7:
                profit_pct, loss_pct, _ = calculate_dynamic_percentages(max_prob, "SHORT")
                potential_fall_pct = profit_pct * 100
                potential_fall_info = f"\n📉 **Потенциальное падение:** -{potential_fall_pct:.1f}%"
            else:
                potential_fall_info = f"\n📉 **Потенциальное падение:** -3.5%"
        
        # Формируем сообщение в зависимости от типа сигнала
        if "ОЖИДАНИЕ" in signal_type and "падение" in strength_text:
            # Для сигналов ожидания с падением не показываем торговые уровни
            message = f"""
🚨 {signal_type}

💰 **{current_coin}:** ${current_price:,.4f}
📉 **Потенциальное падение:** -{abs(price_change_pct):.1f}%
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📉 Ожидание лучшего входа

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}
            """
        else:
            # Для обычных сигналов показываем полную информацию
            message = f"""
🚨 {signal_type}

💰 **{current_coin}:** ${current_price:,.4f}
📈 **Прогноз:** ${take_profit:,.4f} ({price_change_pct:+.1f}%){potential_fall_info}
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📈 Восходящий тренд

🎯 **ТОЧКИ ТОРГОВЛИ:**
• **Вход:** ${entry_price:,.4f}
• **Тейк-профит:** ${take_profit:,.4f}
• **Стоп-лосс:** ${stop_loss:,.4f}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}
            """
        
        # Создаем и отправляем график
        try:
            global use_advanced_chart
            logger.info(f"Создание графика для {current_coin}, продвинутый режим: {use_advanced_chart}")
            
            if use_advanced_chart:
                logger.info("Создаю продвинутый график...")
                chart_buffer = create_advanced_trading_chart(current_coin, df, signal_data)
                chart_caption = f"📈 Продвинутый график {current_coin} в стиле TradingView\n🎯 С кнопками КУПИТЬ/ПРОДАТЬ и зонами тейк-профита"
            else:
                logger.info("Создаю обычный график...")
                chart_buffer = create_trading_chart(current_coin, df, signal_data)
                chart_caption = f"📈 График {current_coin} с торговыми уровнями"
            
            if chart_buffer:
                logger.info("График создан успешно, отправляю...")
                # Отправляем график с текстом в caption (график сверху, текст снизу)
                await context.bot.send_photo(
                    chat_id=update.effective_chat.id,
                    photo=chart_buffer,
                    caption=message
                )
                logger.info("График с текстом отправлен успешно")
            else:
                logger.error("Не удалось создать график, отправляю только текст")
                await update.message.reply_text(message)
        except Exception as chart_error:
            logger.error(f"Ошибка создания графика: {chart_error}")
            import traceback
            logger.error(f"Подробности ошибки: {traceback.format_exc()}")
            await update.message.reply_text(message)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка анализа {current_coin}: {str(e)}")

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /search"""
    try:
        query = ' '.join(context.args) if context.args else 'BTC'
        
        # Используем наш список популярных монет для поиска
        popular_coins = get_popular_coins()
        results = [coin for coin in popular_coins if query.upper() in coin.upper()][:5]
        
        if results:
            message = f"🔍 **Результаты поиска '{query}':**\n\n"
            for coin in results:
                message += f"• {coin}\n"
        else:
            message = f"❌ Монеты с названием '{query}' не найдены"
        
        await update.message.reply_text(message)
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка поиска: {str(e)}")

async def set_coin_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /set_coin для быстрой смены монеты"""
    global current_coin
    
    if not context.args:
        await update.message.reply_text("❌ Укажите монету: /set_coin BTCUSDT")
        return
    
    coin = context.args[0].upper()
    
    # Проверяем формат монеты
    if not coin.endswith('USDT'):
        coin = f"{coin}USDT"
    
    # Конвертируем в формат с слешем для API
    coin_with_slash = coin.replace('USDT', '/USDT')
    
    try:
        import ccxt
        exchange = ccxt.binance()
        
        # Проверяем, существует ли монета
        ticker = exchange.fetch_ticker(coin_with_slash)
        current_price = ticker['last']
        
        # Обновляем текущую монету
        current_coin = coin_with_slash
        
        message = f"""
✅ **Монета изменена на: {coin_with_slash}**
💰 **Текущая цена:** ${current_price:,.4f}

Используйте /analyze для анализа
        """
        await update.message.reply_text(message)
        
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка установки монеты {coin}: {str(e)}")

async def toggle_chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /toggle_chart для переключения типа графиков"""
    global use_advanced_chart
    
    use_advanced_chart = not use_advanced_chart
    
    chart_type = "Продвинутые графики в стиле TradingView" if use_advanced_chart else "Обычные графики"
    
    message = f"""
🔄 **Тип графиков изменен!**

📊 **Текущий режим:** {chart_type}

{'✅ Включены:' if use_advanced_chart else '❌ Отключены:'}
• Кнопки КУПИТЬ/ПРОДАТЬ с ценами bid/ask
• Заштрихованные зоны тейк-профита и стоп-лосса
• Индикатор RSI
• Расширенная информационная панель
• Профессиональный дизайн в стиле TradingView

Используйте /analyze для просмотра графика
    """
    
    await update.message.reply_text(message)

async def test_chart_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Команда /test_chart для тестирования создания графиков"""
    try:
        await update.message.reply_text("🔄 Тестирую создание графика...")
        
        # Тестируем создание простого графика
        test_buffer = test_chart_creation()
        
        if test_buffer:
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=test_buffer,
                caption="✅ Тестовый график создан успешно!"
            )
        else:
            await update.message.reply_text("❌ Ошибка создания тестового графика")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка тестирования: {str(e)}")

# Функции-обработчики для меню
async def handle_status_menu(query, context):
    """Обработка кнопки Статус системы"""
    try:
        import ccxt
        exchange = ccxt.binance()
        ticker = exchange.fetch_ticker(current_coin)
        current_price = ticker['last']
        
        # Проверяем статус автоматических сигналов
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        auto_status = "🟢 Активны" if jobs else "🔴 Остановлены"
        
        status_message = f"""
📊 **Статус системы:**

✅ Бот активен
🪙 Текущая монета: {current_coin}
📈 Цена {current_coin}: ${current_price:,.4f}
📡 Источник данных: Binance
🤖 ML модель: Активна
🎯 Динамические проценты: Включены
🕐 Автоматические сигналы: {auto_status}

Бот работает корректно! 🎉
        """
        
        # Кнопка возврата в меню
        keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(status_message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения статуса: {str(e)}")

async def handle_coins_menu(query, context):
    """Обработка кнопки Выбор монет"""
    try:
        popular_coins = get_popular_coins()[:10]
        
        keyboard = []
        for coin in popular_coins:
            keyboard.append([InlineKeyboardButton(coin, callback_data=f"select_{coin}")])
        
        # Добавляем кнопку возврата
        keyboard.append([InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = "🪙 **Выберите монету для анализа:**"
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения списка монет: {str(e)}")

async def handle_signals_menu(query, context):
    """Обработка кнопки Последние сигналы"""
    try:
        # Получаем данные для текущей монеты
        import ccxt
        import pandas as pd
        import numpy as np
        import joblib
        from datetime import datetime
        
        exchange = ccxt.binance()
        
        # Получаем OHLCV данные
        ohlcv = exchange.fetch_ohlcv(current_coin, '1h', limit=200)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Добавляем EMA
        df['ema_20'] = df['close'].ewm(span=20).mean()
        df['ema_50'] = df['close'].ewm(span=50).mean()
        df['ema_100'] = df['close'].ewm(span=100).mean()
        
        # Получаем текущую цену
        ticker = exchange.fetch_ticker(current_coin)
        current_price = ticker['last']
        
        # Пытаемся загрузить ML модели
        try:
            minimum_detector = joblib.load('models/minimum_detector.pkl')
            maximum_detector = joblib.load('models/maximum_detector.pkl')
            scaler = joblib.load('models/scaler.pkl')
            feature_names = joblib.load('models/feature_names.pkl')
            
            # Создаем признаки для ML
            df_features = create_ml_features(df)
            
            # Берем последние данные для предсказания
            if len(df_features) > 0:
                features = df_features.iloc[-1:][feature_names].values
                features_scaled = scaler.transform(features)
                
                # Предсказания
                min_prob = minimum_detector.predict_proba(features_scaled)[0][1]
                max_prob = maximum_detector.predict_proba(features_scaled)[0][1]
                
                # Определяем сигнал с динамическими процентами
                if min_prob > 0.7:
                    signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                    entry_price = current_price
                    
                    # Получаем динамические проценты
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(min_prob, "LONG")
                    
                    take_profit = current_price * (1 + profit_pct)
                    stop_loss = current_price * (1 - loss_pct)
                    probability = min_prob * 100
                    
                elif max_prob > 0.7:
                    # Отключаем ШОРТ сигналы, показываем только как ожидание с информацией о падении
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    
                    # Рассчитываем потенциальное падение
                    profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(max_prob, "SHORT")
                    potential_fall_pct = profit_pct * 100  # Процент потенциального падения
                    
                    take_profit = current_price * 1.02  # Консервативный +2%
                    stop_loss = current_price * 0.98   # Консервативный -2%
                    probability = max_prob * 100
                    strength_text = f"⚪ Слабый (падение {potential_fall_pct:.1f}%)"
                    
                else:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    take_profit = current_price * 1.02  # Консервативный +2%
                    stop_loss = current_price * 0.98   # Консервативный -2%
                    probability = max(min_prob, max_prob) * 100
                    strength_text = "⚪ Слабый"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02
                stop_loss = current_price * 0.98
                probability = 50.0
                strength_text = "⚪ Слабый"
                
            ml_status = "Активна"
            
        except Exception as e:
            # Fallback к простому анализу
            latest_close = df['close'].iloc[-1]
            ema_20_latest = df['ema_20'].iloc[-1]
            ema_50_latest = df['ema_50'].iloc[-1]
            
            if latest_close > ema_20_latest > ema_50_latest:
                signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                entry_price = current_price
                take_profit = current_price * 1.035  # +3.5% для симуляции
                stop_loss = current_price * 0.975    # -2.5% для симуляции
                probability = 75.0
                strength_text = "💪 Сильный"
            elif latest_close < ema_20_latest < ema_50_latest:
                # Отключаем ШОРТ сигналы в fallback тоже, но показываем потенциальное падение
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02  # Консервативный +2%
                stop_loss = current_price * 0.98   # Консервативный -2%
                probability = 75.0
                strength_text = "⚪ Слабый (падение 3.5%)"
            else:
                signal_type = "⚪ ОЖИДАНИЕ"
                entry_price = current_price
                take_profit = current_price * 1.02
                stop_loss = current_price * 0.98
                probability = 50.0
                strength_text = "⚪ Слабый"
                
            ml_status = "Симуляция"
        
        # Рассчитываем потенциал
        price_change_pct = ((take_profit - entry_price) / entry_price) * 100
        
        # Создаем данные сигнала для графика
        signal_data = {
            'symbol': current_coin,
            'signal_type': signal_type,
            'current_price': current_price,
            'entry_price': entry_price,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'probability': probability,
            'strength_text': strength_text,
            'price_change_pct': price_change_pct,
            'ml_status': ml_status
        }
        
        message = f"""
📊 **Последние сигналы для {current_coin}:**

🚨 {signal_type}

💰 **Цена:** ${current_price:,.4f}
📈 **Прогноз:** ${take_profit:,.4f} ({price_change_pct:+.1f}%)
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📈 Восходящий тренд

🎯 **ТОЧКИ ТОРГОВЛИ:**
• **Вход:** ${entry_price:,.4f}
• **Тейк-профит:** ${take_profit:,.4f}
• **Стоп-лосс:** ${stop_loss:,.4f}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}
        """
        
        # Кнопка возврата в меню
        keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Создаем и отправляем график (продвинутый или обычный)
        try:
            global use_advanced_chart
            if use_advanced_chart:
                chart_buffer = create_advanced_trading_chart(current_coin, df, signal_data)
                chart_caption = f"📈 Продвинутый график {current_coin} в стиле TradingView\n🎯 С кнопками КУПИТЬ/ПРОДАТЬ и зонами тейк-профита"
            else:
                chart_buffer = create_trading_chart(current_coin, df, signal_data)
                chart_caption = f"📈 График {current_coin} с торговыми уровнями"
            
            if chart_buffer:
                # Отправляем график с текстом в caption (график сверху, текст снизу)
                await context.bot.send_photo(
                    chat_id=query.message.chat_id,
                    photo=chart_buffer,
                    caption=message
                )
            else:
                await query.edit_message_text(message, reply_markup=reply_markup)
        except Exception as chart_error:
            logger.error(f"Ошибка создания графика: {chart_error}")
            await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка получения сигналов: {str(e)}")

async def handle_analyze_menu(query, context):
    """Обработка кнопки Анализ монеты"""
    await handle_signals_menu(query, context)  # Пока используем ту же логику

async def handle_search_menu(query, context):
    """Обработка кнопки Поиск монет"""
    try:
        popular_coins = get_popular_coins()
        
        # Показываем популярные монеты для поиска
        message = "🔍 **Популярные монеты для поиска:**\n\n"
        for i, coin in enumerate(popular_coins[:10], 1):
            message += f"{i}. {coin}\n"
        
        message += "\n💡 **Используйте команду:** /set_coin <монета>"
        
        # Кнопка возврата в меню
        keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка поиска: {str(e)}")

async def handle_auto_menu(query, context):
    """Обработка кнопки Авто сигналы"""
    try:
        # Проверяем статус автоматических сигналов
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        auto_status = "🟢 Активны" if jobs else "🔴 Остановлены"
        
        message = f"""
🤖 **Автоматические сигналы:**

📊 **Статус:** {auto_status}
🕐 **Интервал:** каждые 30 минут
📈 **Анализ:** 20 популярных монет
🏆 **Отправка:** топ-3 лучших сигнала
🎯 **Динамические проценты:** Включены

**Выберите действие:**
        """
        
        # Кнопки управления
        keyboard = []
        if jobs:
            keyboard.append([InlineKeyboardButton("🛑 Остановить авто сигналы", callback_data="auto_stop")])
        else:
            keyboard.append([InlineKeyboardButton("▶️ Запустить авто сигналы", callback_data="auto_start")])
        
        keyboard.append([InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка управления авто сигналами: {str(e)}")

async def handle_auto_start(query, context):
    """Обработка запуска автоматических сигналов"""
    try:
        # Останавливаем существующие задачи
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        for job in jobs:
            job.schedule_removal()
        
        # Запускаем новую задачу
        context.job_queue.run_repeating(
            send_auto_signals,
            interval=1800,  # 30 минут = 1800 секунд
            first=10,       # Первый запуск через 10 секунд
            name="auto_signals",
            data={'chat_id': query.message.chat_id}
        )
        
        message = """
✅ **Автоматические сигналы запущены!**

🕐 **Интервал:** каждые 30 минут
📊 **Анализ:** 20 популярных монет
🏆 **Отправка:** топ-3 лучших сигнала
⏰ **Первый сигнал:** через 10 секунд
        """
        
        # Кнопка возврата в меню
        keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка запуска автоматических сигналов: {str(e)}")

async def handle_auto_stop(query, context):
    """Обработка остановки автоматических сигналов"""
    try:
        jobs = context.job_queue.get_jobs_by_name("auto_signals")
        for job in jobs:
            job.schedule_removal()
        
        message = """
🛑 **Автоматические сигналы остановлены!**

Используйте меню для повторного запуска
        """
        
        # Кнопка возврата в меню
        keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text(message, reply_markup=reply_markup)
        
    except Exception as e:
        await query.edit_message_text(f"❌ Ошибка остановки автоматических сигналов: {str(e)}")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик кнопок меню"""
    global current_coin
    
    query = update.callback_query
    await query.answer()
    
    logger.info(f"🔘 Нажата кнопка: {query.data}")
    
    if query.data.startswith("menu_"):
        # Обработка кнопок главного меню
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
        elif query.data == "menu_back":
            # Возврат в главное меню
            logger.info("🔙 Возврат в главное меню")
            try:
                await start_command_from_callback(query, context)
                logger.info("✅ Главное меню создано успешно")
            except Exception as e:
                logger.error(f"❌ Ошибка при создании главного меню: {e}")
    
    elif query.data.startswith("select_"):
        # Обработка выбора монеты с автоматическим анализом
        coin = query.data.replace("select_", "")
        current_coin = coin
        
        # Показываем сообщение о выборе монеты
        await query.edit_message_text(f"🔄 Анализирую {coin}...")
        
        # Автоматически выполняем анализ выбранной монеты
        try:
            # Получаем данные для выбранной монеты
            import ccxt
            import pandas as pd
            import numpy as np
            import joblib
            from datetime import datetime
            
            exchange = ccxt.binance()
            
            # Получаем OHLCV данные
            ohlcv = exchange.fetch_ohlcv(coin, '1h', limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # Получаем текущую цену
            ticker = exchange.fetch_ticker(coin)
            current_price = ticker['last']
            
            # Пытаемся загрузить ML модели
            try:
                minimum_detector = joblib.load('models/minimum_detector.pkl')
                maximum_detector = joblib.load('models/maximum_detector.pkl')
                scaler = joblib.load('models/scaler.pkl')
                feature_names = joblib.load('models/feature_names.pkl')
                
                # Создаем признаки для ML
                df_features = create_ml_features(df)
                
                # Берем последние данные для предсказания
                if len(df_features) > 0:
                    features = df_features.iloc[-1:][feature_names].values
                    features_scaled = scaler.transform(features)
                    
                    # Предсказания
                    min_prob = minimum_detector.predict_proba(features_scaled)[0][1]
                    max_prob = maximum_detector.predict_proba(features_scaled)[0][1]
                    
                    # Определяем сигнал с динамическими процентами
                    if min_prob > 0.7:
                        signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                        entry_price = current_price
                        
                        # Получаем динамические проценты
                        profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(min_prob, "LONG")
                        
                        take_profit = current_price * (1 + profit_pct)
                        stop_loss = current_price * (1 - loss_pct)
                        probability = min_prob * 100
                        
                    elif max_prob > 0.7:
                        signal_type = "🔴 ТОЧКА ВХОДА ШОРТ"
                        entry_price = current_price
                        
                        # Получаем динамические проценты
                        profit_pct, loss_pct, strength_text = calculate_dynamic_percentages(max_prob, "SHORT")
                        
                        take_profit = current_price * (1 - profit_pct)  # Для шорта
                        stop_loss = current_price * (1 + loss_pct)     # Для шорта
                        probability = max_prob * 100
                        
                    else:
                        signal_type = "⚪ ОЖИДАНИЕ"
                        entry_price = current_price
                        take_profit = current_price * 1.02  # Консервативный +2%
                        stop_loss = current_price * 0.98   # Консервативный -2%
                        probability = max(min_prob, max_prob) * 100
                        strength_text = "⚪ Слабый"
                else:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    take_profit = current_price * 1.02
                    stop_loss = current_price * 0.98
                    probability = 50.0
                    strength_text = "⚪ Слабый"
                    
                ml_status = "Активна"
                
            except Exception as e:
                # Fallback к простому анализу
                latest_close = df['close'].iloc[-1]
                ema_20_latest = df['ema_20'].iloc[-1]
                ema_50_latest = df['ema_50'].iloc[-1]
                
                if latest_close > ema_20_latest > ema_50_latest:
                    signal_type = "🟢 ТОЧКА ВХОДА ЛОНГ"
                    entry_price = current_price
                    take_profit = current_price * 1.035  # +3.5% для симуляции
                    stop_loss = current_price * 0.975    # -2.5% для симуляции
                    probability = 75.0
                    strength_text = "💪 Сильный"
                elif latest_close < ema_20_latest < ema_50_latest:
                    signal_type = "🔴 ТОЧКА ВХОДА ШОРТ"
                    entry_price = current_price
                    take_profit = current_price * 0.965  # -3.5% для шорта
                    stop_loss = current_price * 1.025    # +2.5% для шорта
                    probability = 75.0
                    strength_text = "💪 Сильный"
                else:
                    signal_type = "⚪ ОЖИДАНИЕ"
                    entry_price = current_price
                    take_profit = current_price * 1.02
                    stop_loss = current_price * 0.98
                    probability = 50.0
                    strength_text = "⚪ Слабый"
                    
                ml_status = "Симуляция"
            
            # Рассчитываем потенциал
            price_change_pct = ((take_profit - entry_price) / entry_price) * 100
            
            # Создаем данные сигнала для графика
            signal_data = {
                'symbol': coin,
                'signal_type': signal_type,
                'current_price': current_price,
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss,
                'probability': probability,
                'strength_text': strength_text,
                'price_change_pct': price_change_pct,
                'ml_status': ml_status
            }
            
            # Формируем сообщение в зависимости от типа сигнала
            if "ОЖИДАНИЕ" in signal_type and "падение" in strength_text:
                # Для сигналов ожидания с падением не показываем торговые уровни
                message = f"""
📊 **Анализ {coin}:**

🚨 {signal_type}

💰 **Цена:** ${current_price:,.4f}
📉 **Потенциальное падение:** -{abs(price_change_pct):.1f}%
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📉 Ожидание лучшего входа

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}
                """
            else:
                # Для обычных сигналов показываем полную информацию
                message = f"""
📊 **Анализ {coin}:**

🚨 {signal_type}

💰 **Цена:** ${current_price:,.4f}
📈 **Прогноз:** ${take_profit:,.4f} ({price_change_pct:+.1f}%)
🎯 **Вероятность:** {probability:.1f}% {strength_text}
📊 **Контекст:** 📈 Восходящий тренд

🎯 **ТОЧКИ ТОРГОВЛИ:**
• **Вход:** ${entry_price:,.4f}
• **Тейк-профит:** ${take_profit:,.4f}
• **Стоп-лосс:** ${stop_loss:,.4f}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M')} | 🤖 ML: {ml_status}
                """
            
            # Кнопка возврата в меню
            keyboard = [[InlineKeyboardButton("🔙 Назад в меню", callback_data="menu_back")]]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Создаем и отправляем график
            try:
                global use_advanced_chart
                logger.info(f"Создание графика для {coin}, продвинутый режим: {use_advanced_chart}")
                
                if use_advanced_chart:
                    logger.info("Создаю продвинутый график...")
                    chart_buffer = create_advanced_trading_chart(coin, df, signal_data)
                    chart_caption = f"📈 Продвинутый график {coin} в стиле TradingView\n🎯 С кнопками КУПИТЬ/ПРОДАТЬ и зонами тейк-профита"
                else:
                    logger.info("Создаю обычный график...")
                    chart_buffer = create_trading_chart(coin, df, signal_data)
                    chart_caption = f"📈 График {coin} с торговыми уровнями"
                
                if chart_buffer:
                    logger.info("График создан успешно, отправляю...")
                    # Отправляем график с текстом в caption (график сверху, текст снизу)
                    await context.bot.send_photo(
                        chat_id=query.message.chat_id,
                        photo=chart_buffer,
                        caption=message
                    )
                    logger.info("График с текстом отправлен успешно")
                else:
                    logger.error("Не удалось создать график, отправляю только текст")
                    await query.edit_message_text(message, reply_markup=reply_markup)
            except Exception as chart_error:
                logger.error(f"Ошибка создания графика: {chart_error}")
                import traceback
                logger.error(f"Подробности ошибки: {traceback.format_exc()}")
                await query.edit_message_text(message, reply_markup=reply_markup)
                
        except Exception as e:
            await query.edit_message_text(f"❌ Ошибка анализа {coin}: {str(e)}")
    
    elif query.data.startswith("auto_"):
        # Обработка кнопок автоматических сигналов
        if query.data == "auto_start":
            await handle_auto_start(query, context)
        elif query.data == "auto_stop":
            await handle_auto_stop(query, context)

async def start_command_from_callback(query, context):
    """Возврат в главное меню из callback"""
    global current_coin
    
    logger.info("🔄 Создание главного меню")
    logger.info(f"📱 Текущая монета: {current_coin}")
    
    # Создаем главное меню
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

# Глобальные переменные
current_coin = "BTC/USDT"
use_advanced_chart = True  # Использовать продвинутые графики по умолчанию

def main():
    """Основная функция"""
    print("🤖 Запуск ML Telegram Bot с автоматическими сигналами (v22.3+)")
    
    # Загружаем конфигурацию
    config = load_config()
    if not config:
        print("❌ Не удалось загрузить конфигурацию")
        return
    
    # Создаем приложение
    application = Application.builder().token(config["telegram_token"]).build()
    
    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("coins", coins_command))
    application.add_handler(CommandHandler("set_coin", set_coin_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("analyze", analyze_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("toggle_chart", toggle_chart_command))
    application.add_handler(CommandHandler("test_chart", test_chart_command))
    application.add_handler(CommandHandler("auto_start", auto_start_command))
    application.add_handler(CommandHandler("auto_stop", auto_stop_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    print("✅ Бот настроен успешно")
    print("🚀 Запускаю бота...")
    
    # Запускаем бота
    try:
        application.run_polling(drop_pending_updates=True)
    except KeyboardInterrupt:
        print("\n🛑 Бот остановлен пользователем")
    except Exception as e:
        print(f"❌ Ошибка запуска бота: {e}")

if __name__ == "__main__":
    main()


