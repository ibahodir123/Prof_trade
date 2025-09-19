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
from ema_pattern_analyzer import EMAPatternAnalyzer
from ema_trend_trainer import EMATrendTrainer

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler('auto_signals.log'),
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
        self.min_detector = None
        self.max_detector = None
        self.scaler = None
        self.feature_names = None
        self.shooting_star_model = None
        self.shooting_star_scaler = None
        self.ema_analyzer = None
        
    def load_config(self):
        """Загрузка конфигурации бота"""
        try:
            with open('bot_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            logger.error(f"Ошибка загрузки конфигурации: {e}")
            return None
    
    def initialize_binance(self):
        """Инициализация Binance API"""
        try:
            self.binance = ccxt.binance({
                'apiKey': self.config['binance_api']['api_key'],
                'secret': self.config['binance_api']['secret_key'],
                'sandbox': False,
                'enableRateLimit': True,
            })
            logger.info("✅ Binance API инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации Binance API: {e}")
    
    def load_models(self):
        """Загрузка ML моделей"""
        try:
            # Загружаем основные модели
            self.min_detector = joblib.load('models/minimum_detector.pkl')
            self.max_detector = joblib.load('models/maximum_detector.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            logger.info("✅ Основные ML модели загружены")
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
        
        # Инициализируем EMA анализатор (опционально)
        try:
            self.ema_analyzer = EMAPatternAnalyzer()
            logger.info("✅ EMA анализатор инициализирован")
        except Exception as e:
            logger.warning(f"⚠️ EMA анализатор не инициализирован: {e}")
            self.ema_analyzer = None
        
        return True
    
    async def get_available_pairs(self):
        """Получение списка доступных торговых пар"""
        try:
            if not self.binance:
                self.initialize_binance()
            
            markets = self.binance.load_markets()
            usdt_pairs = []
            
            for symbol, market in markets.items():
                if market['quote'] == 'USDT' and market['active']:
                    # Проверяем объем торгов за 24ч
                    try:
                        ticker = self.binance.fetch_ticker(symbol)
                        volume_24h = ticker['quoteVolume']
                        
                        if volume_24h and volume_24h >= self.config['trading_settings']['min_volume_24h']:
                            usdt_pairs.append(symbol)
                    except:
                        continue
            
            self.available_pairs = usdt_pairs
            logger.info(f"✅ Найдено {len(usdt_pairs)} активных USDT пар")
            return usdt_pairs
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения торговых пар: {e}")
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
            
            # Получаем предсказания
            min_prob = self.min_detector.predict_proba(features_scaled)[0][1]
            max_prob = self.max_detector.predict_proba(features_scaled)[0][1]
            
            # Определяем сигнал
            diff = max_prob - min_prob
            signal = "⚪ ОЖИДАНИЕ"
            confidence = 0.0
            
            if diff > 0.02:      # Разница больше 2%
                if max_prob > 0.3:  # Достаточная уверенность
                    signal = "🔴 SHORT"
                    confidence = max_prob
            elif diff < -0.02:   # Разница меньше -2%
                if min_prob > 0.3:  # Достаточная уверенность
                    signal = "🟢 LONG"
                    confidence = min_prob
            else:
                signal = "⚪ ОЖИДАНИЕ"  # Разница менее 2%
            
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
        """Подготовка признаков для ML модели"""
        try:
            # Базовые признаки
            df['rsi'] = self.calculate_rsi(df['close'])
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # Дополнительные признаки
            df['price_change'] = df['close'].pct_change()
            df['volume_change'] = df['volume'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_open_ratio'] = df['close'] / df['open']
            
            # Скользящие средние
            df['sma_10'] = df['close'].rolling(10).mean()
            df['sma_20'] = df['close'].rolling(20).mean()
            
            # Волатильность
            df['volatility'] = df['price_change'].rolling(20).std()
            
            # Последние значения для модели
            features = []
            for col in self.feature_names:
                if col in df.columns:
                    features.append(df[col].iloc[-1])
                else:
                    features.append(0.0)
            
            return np.array(features)
            
        except Exception as e:
            logger.error(f"❌ Ошибка подготовки признаков: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Расчет RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
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
                    logger.info("✅ Сообщение отправлено в Telegram")
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
            logger.info(f"📊 Анализирую {len(coins_to_check)} монет для автосигналов")
            
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
            logger.info(f"✅ Автосигналы отправлены: {len(top_long)} LONG, {len(top_short)} SHORT")
            
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

