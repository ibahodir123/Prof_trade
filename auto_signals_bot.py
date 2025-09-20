#!/usr/bin/env python3
"""
Отдельный бот для автоматических сигналов
Анализирует все монеты и отправляет топ сигналы каждые 30 минут
"""
import asyncio
import logging
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
from advanced_ema_analyzer import AdvancedEMAAnalyzer
from advanced_ml_trainer import AdvancedMLTrainer

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('auto_signals.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutoSignalsBot:
    def __init__(self):
        self.config = self.load_config()
        self.binance = None
        self.available_pairs = []
        self.scheduler = AsyncIOScheduler()
        self.entry_model = None
        self.exit_model = None
        self.scaler = None
        self.feature_names = None
        self.shooting_star_model = None
        self.shooting_star_scaler = None
        self.ema_analyzer = None
        
    def load_config(self):
        """Загрузка конфигурации бота"""
        try:
            import os
            if not os.path.exists('bot_config.json'):
                logger.error("❌ Файл конфигурации bot_config.json не найден")
                return None
                
            with open('bot_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Проверяем обязательные поля конфигурации
            required_fields = ['binance_api', 'telegram']
            for field in required_fields:
                if field not in config:
                    logger.error(f"❌ Отсутствует обязательное поле в конфигурации: {field}")
                    return None
                    
            # Проверяем поля Binance API
            binance_fields = ['api_key', 'secret_key']
            for field in binance_fields:
                if field not in config['binance_api']:
                    logger.error(f"❌ Отсутствует поле Binance API: {field}")
                    return None
                    
            # Проверяем поля Telegram
            telegram_fields = ['bot_token', 'chat_id']
            for field in telegram_fields:
                if field not in config['telegram']:
                    logger.error(f"❌ Отсутствует поле Telegram: {field}")
                    return None
            
            logger.info("✅ Конфигурация загружена и проверена")
            return config
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки конфигурации: {e}")
            return None
    
    def initialize_binance(self):
        """Инициализация Binance API"""
        try:
            self.binance = ccxt.binance({
                'apiKey': self.config['binance_api']['api_key'],
                'secret': self.config['binance_api']['secret_key'],
                'sandbox': False,
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,  # Увеличиваем окно приема
                }
            })
            logger.info("✅ Binance API инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Binance API: {e}")
    
    def load_models(self):
        """Загрузка ML моделей"""
        try:
            # Проверяем существование файлов моделей
            import os
            required_files = [
                'models/entry_model.pkl',
                'models/exit_model.pkl', 
                'models/ema_scaler.pkl',
                'models/feature_names.pkl'
            ]
            
            for file_path in required_files:
                if not os.path.exists(file_path):
                    logger.error(f"❌ Файл модели не найден: {file_path}")
                    return False
            
            # Загружаем новые модели (10 признаков)
            self.entry_model = joblib.load('models/entry_model.pkl')
            self.exit_model = joblib.load('models/exit_model.pkl')
            self.scaler = joblib.load('models/ema_scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            # Проверяем, что модели загружены корректно
            if not all([self.entry_model, self.exit_model, self.scaler, self.feature_names]):
                logger.error("❌ Одна или несколько моделей не загружены корректно")
                return False
                
            logger.info("✅ Основные ML модели загружены и проверены")
            return True
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки основных моделей: {e}")
            return False
        
        # Загружаем модель стреляющих звезд (опционально)
        try:
            self.shooting_star_model = load_model('simple_shooting_star_model.h5')
            self.shooting_star_scaler = joblib.load('simple_shooting_star_scaler.pkl')
            logger.info("✅ Модель стреляющих звезд загружена")
        except Exception as e:
            logger.warning(f"⚠️ Модель стреляющих звезд не загружена: {e}")
            self.shooting_star_model = None
            self.shooting_star_scaler = None
        
        # EMA анализатор отключен
        self.ema_analyzer = None
        
        return True
    
    async def get_available_pairs(self):
        """Получение списка доступных торговых пар"""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if not self.binance:
                    self.initialize_binance()
                
                # Синхронизируем время перед запросом
                await asyncio.sleep(1)
                
                # Используем популярные пары напрямую (без загрузки всех рынков)
                popular_pairs = [
                    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
                    'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT',
                    'LTC/USDT', 'LINK/USDT', 'UNI/USDT', 'ATOM/USDT', 'FIL/USDT'
                ]
                
                usdt_pairs = []
                for symbol in popular_pairs:
                    try:
                        # Проверяем доступность пары через тикер
                        ticker = self.binance.fetch_ticker(symbol)
                        if ticker and 'last' in ticker:
                            usdt_pairs.append(symbol)
                    except:
                        # Пара недоступна, пропускаем
                        continue
                
                self.available_pairs = usdt_pairs
                logger.info(f"✅ Найдено {len(usdt_pairs)} активных USDT пар")
                return usdt_pairs
                
            except Exception as e:
                logger.warning(f"Попытка {attempt + 1} неудачна: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)  # Пауза перед повторной попыткой
                else:
                    logger.error(f"Ошибка получения торговых пар после {max_retries} попыток: {e}")
                    return []
    
    def calculate_dynamic_percentages(self, signal_strength, signal_type):
        """Расчет динамических процентов на основе силы сигнала"""
        if signal_strength > 0.9:
            tp_percent = 8.0
            sl_percent = 4.0
        elif signal_strength > 0.8:
            tp_percent = 6.0
            sl_percent = 3.0
        elif signal_strength > 0.7:
            tp_percent = 5.0
            sl_percent = 2.5
        elif signal_strength > 0.6:
            tp_percent = 4.0
            sl_percent = 2.0
        else:
            tp_percent = 3.0
            sl_percent = 1.5
        
        return tp_percent, sl_percent
    
    async def analyze_coin_signal(self, symbol):
        """Анализ сигнала для конкретной монеты"""
        try:
            clean_symbol = symbol.replace(':USDT', '') if ':USDT' in symbol else symbol
            
            if not self.binance:
                self.initialize_binance()
            
            # Получаем исторические данные
            ohlcv = self.binance.fetch_ohlcv(symbol, '1h', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            if len(df) < 50:
                return None
            
            # Подготавливаем признаки
            features = self.prepare_features(df)
            
            if features is None or len(features) == 0:
                return None
            
            # Нормализуем признаки
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Получаем предсказания от новых моделей
            entry_prob = self.entry_model.predict_proba(features_scaled)[0][1]
            exit_prob = self.exit_model.predict_proba(features_scaled)[0][1]
            
            # Определяем сигнал
            signal = "⚪ ОЖИДАНИЕ"
            confidence = 0.0
            
            if entry_prob > 0.6:  # Высокая вероятность входа
                signal = "🟢 LONG"
                confidence = entry_prob
            elif exit_prob > 0.6:  # Высокая вероятность выхода
                signal = "🔴 SHORT"
                confidence = exit_prob
            else:
                signal = "⚪ ОЖИДАНИЕ"  # Недостаточная уверенность
            
            if signal != "⚪ ОЖИДАНИЕ":
                # Получаем текущую цену
                ticker = self.binance.fetch_ticker(symbol)
                current_price = ticker['last']
                
                # Рассчитываем TP и SL
                tp_percent, sl_percent = self.calculate_dynamic_percentages(confidence, signal)
                
                if "LONG" in signal:
                    tp_price = current_price * (1 + tp_percent / 100)
                    sl_price = current_price * (1 - sl_percent / 100)
                else:  # SHORT
                    tp_price = current_price * (1 - tp_percent / 100)
                    sl_price = current_price * (1 + sl_percent / 100)
                
                return {
                    'symbol': clean_symbol,
                    'signal': signal,
                    'confidence': confidence,
                    'current_price': current_price,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'tp_percent': tp_percent,
                    'sl_percent': sl_percent
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа {symbol}: {e}")
            return None
    
    def prepare_features(self, df):
        """Подготовка признаков для ML модели (10 признаков)"""
        try:
            if len(df) < 20:
                return np.zeros(10)
            
            # Базовые EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # Скорости EMA
            df['ema20_speed'] = df['ema_20'].diff(5) / df['ema_20']
            df['ema50_speed'] = df['ema_50'].diff(5) / df['ema_50']
            
            # Скорость цены относительно EMA 20
            df['price_speed_vs_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
            
            # Расстояния между EMA
            df['ema20_to_ema50'] = (df['ema_20'] - df['ema_50']) / df['ema_50']
            
            # Расстояние цены до EMA 20
            df['price_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
            
            # Угол тренда (упрощенный)
            df['trend_angle'] = np.arctan(df['ema_20'].diff(10) / df['ema_20']) * 180 / np.pi
            
            # Тип тренда (1=нисходящий, 2=восходящий, 3=боковой)
            df['trend_type'] = 1  # По умолчанию нисходящий
            df.loc[df['ema_20'] > df['ema_50'], 'trend_type'] = 2  # Восходящий
            df.loc[(df['ema_20'] - df['ema_50']).abs() < df['close'] * 0.01, 'trend_type'] = 3  # Боковой
            
            # Берем последние значения
            latest = df.iloc[-1]
            
            # Создаем массив из 10 признаков с проверкой на NaN
            features = np.array([
                float(latest['ema_20']) if pd.notna(latest['ema_20']) else 0.0,
                float(latest['ema_50']) if pd.notna(latest['ema_50']) else 0.0,
                float(latest['ema_100']) if pd.notna(latest['ema_100']) else 0.0,
                float(latest['ema20_speed']) if pd.notna(latest['ema20_speed']) else 0.0,
                float(latest['ema50_speed']) if pd.notna(latest['ema50_speed']) else 0.0,
                float(latest['price_speed_vs_ema20']) if pd.notna(latest['price_speed_vs_ema20']) else 0.0,
                float(latest['ema20_to_ema50']) if pd.notna(latest['ema20_to_ema50']) else 0.0,
                float(latest['price_to_ema20']) if pd.notna(latest['price_to_ema20']) else 0.0,
                float(latest['trend_angle']) if pd.notna(latest['trend_angle']) else 0.0,
                float(latest['trend_type']) if pd.notna(latest['trend_type']) else 1.0
            ])
            
            return features
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки признаков: {e}")
            return None
    
    
    async def send_telegram_message(self, message):
        """Отправка сообщения в Telegram"""
        try:
            import httpx
            
            url = f"https://api.telegram.org/bot{self.config['telegram_token']}/sendMessage"
            data = {
                'chat_id': self.config['chat_id'],
                'text': message,
                'parse_mode': 'HTML'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, data=data)
                if response.status_code == 200:
                    logger.info("Сообщение отправлено в Telegram")
                else:
                    logger.error(f"❌ Ошибка отправки в Telegram: {response.status_code}")
                    
        except Exception as e:
            logger.error(f"❌ Ошибка отправки сообщения: {e}")
    
    async def send_auto_signals(self):
        """Отправка автоматических сигналов"""
        try:
            logger.info("🤖 Отправляю автосигналы...")
            
            # Получаем список монет
            if not self.available_pairs:
                await self.get_available_pairs()
            
            coins_to_check = self.available_pairs
            logger.info(f"Анализирую {len(coins_to_check)} монет для автосигналов")
            
            all_signals = []
            analyzed_count = 0
            max_analysis_time = 300  # 5 минут максимум
            
            start_time = datetime.now()
            
            for coin in coins_to_check:
                if (datetime.now() - start_time).seconds > max_analysis_time:
                    logger.warning(f"⏰ Превышено время анализа ({max_analysis_time} сек)")
                    break
                
                try:
                    signal = await self.analyze_coin_signal(coin)
                    if signal:
                        all_signals.append(signal)
                    analyzed_count += 1
                    
                    if analyzed_count % 50 == 0:
                        logger.info(f"📈 Проанализировано {analyzed_count}/{len(coins_to_check)} монет")
                        
                except Exception as e:
                    logger.error(f"❌ Ошибка анализа {coin}: {e}")
                    continue
            
            # Сортируем сигналы по уверенности
            long_signals = [s for s in all_signals if "LONG" in s['signal']]
            short_signals = [s for s in all_signals if "SHORT" in s['signal']]
            
            long_signals.sort(key=lambda x: x['confidence'], reverse=True)
            short_signals.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Берем топ-5 сигналов каждого типа
            top_long = long_signals[:5]
            top_short = short_signals[:5]
            
            # Формируем сообщение
            message = f"🤖 <b>АВТОСИГНАЛЫ</b> ({datetime.now().strftime('%H:%M')})\n"
            message += f"📊 Проанализировано: {analyzed_count} монет\n"
            message += f"🟢 LONG: {len(long_signals)} | 🔴 SHORT: {len(short_signals)}\n\n"
            
            if top_long:
                message += "🟢 <b>ТОП LONG СИГНАЛЫ:</b>\n"
                for i, signal in enumerate(top_long, 1):
                    message += f"{i}. {signal['symbol']} - {signal['confidence']:.1%}\n"
                    message += f"   💰 Цена: ${signal['current_price']:.8f}\n"
                    message += f"   🎯 TP: ${signal['tp_price']:.8f} (+{signal['tp_percent']:.1f}%)\n"
                    message += f"   🛡️ SL: ${signal['sl_price']:.8f} (-{signal['sl_percent']:.1f}%)\n\n"
            
            if top_short:
                message += "🔴 <b>ТОП SHORT СИГНАЛЫ:</b>\n"
                for i, signal in enumerate(top_short, 1):
                    message += f"{i}. {signal['symbol']} - {signal['confidence']:.1%}\n"
                    message += f"   💰 Цена: ${signal['current_price']:.8f}\n"
                    message += f"   🎯 TP: ${signal['tp_price']:.8f} (-{signal['tp_percent']:.1f}%)\n"
                    message += f"   🛡️ SL: ${signal['sl_price']:.8f} (+{signal['sl_percent']:.1f}%)\n\n"
            
            if not top_long and not top_short:
                message += "⚪ <b>Нет четких сигналов</b>\n"
                message += "Рынок находится в неопределенном состоянии.\n"
                message += "Рекомендуется ожидание более четких сигналов."
            
            # Отправляем сообщение
            await self.send_telegram_message(message)
            logger.info(f"Автосигналы отправлены: {len(top_long)} LONG, {len(top_short)} SHORT")
            
        except Exception as e:
            logger.error(f"❌ Ошибка отправки автосигналов: {e}")
    
    def start_scheduler(self):
        """Запуск планировщика автосигналов"""
        try:
            # Добавляем задачу каждые 30 минут
            self.scheduler.add_job(
                self.send_auto_signals,
                trigger=IntervalTrigger(minutes=30),
                id='auto_signals',
                replace_existing=True
            )
            
            self.scheduler.start()
            logger.info("⏰ Планировщик автосигналов запущен (каждые 30 минут)")
            
        except Exception as e:
            logger.error(f"❌ Ошибка запуска планировщика: {e}")
    
    async def run(self):
        """Основной цикл работы бота"""
        try:
            logger.info("🚀 Запуск AutoSignals Bot")
            
            # Инициализация
            self.initialize_binance()
            
            # Проверяем загрузку моделей
            if not self.load_models():
                logger.error("❌ Не удалось загрузить основные модели. Завершение работы.")
                return
            
            await self.get_available_pairs()
            
            # Запускаем планировщик
            self.start_scheduler()
            
            logger.info("✅ AutoSignals Bot запущен успешно")
            
            # Отправляем первый сигнал через 1 минуту
            await asyncio.sleep(60)
            await self.send_auto_signals()
            
            # Основной цикл
            while True:
                await asyncio.sleep(3600)  # Проверяем каждый час
                
        except KeyboardInterrupt:
            logger.info("🛑 Остановка AutoSignals Bot")
        except Exception as e:
            logger.error(f"❌ Критическая ошибка: {e}")

async def main():
    """Главная функция"""
    bot = AutoSignalsBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())

