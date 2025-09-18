#!/usr/bin/env python3
"""
Сборщик исторических данных для обучения нейронной сети
Собирает данные всех монет с 1 января 2025 года
"""
import ccxt
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import time
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict, Tuple
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class HistoricalDataCollector:
    """Класс для сбора исторических данных всех монет"""
    
    def __init__(self):
        self.exchange = ccxt.binance({
            'options': {'adjustForTimeDifference': True}
        })
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime.now()
        self.timeframe = '1h'  # 1 час
        self.limit = 1000  # Максимум свечей за запрос
        
    def get_all_usdt_pairs(self) -> List[str]:
        """Получает все доступные USDT пары"""
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = []
            
            for symbol, market in markets.items():
                if (market['quote'] == 'USDT' and 
                    market['active'] and 
                    market['type'] == 'spot' and
                    not market['info'].get('isSpotTradingAllowed', True) == False):
                    usdt_pairs.append(symbol)
            
            logger.info(f"📊 Найдено {len(usdt_pairs)} USDT пар")
            return sorted(usdt_pairs)
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения пар: {e}")
            return []
    
    def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получает исторические данные для конкретной монеты"""
        try:
            # Конвертируем даты в миллисекунды
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while current_since < until:
                try:
                    # Получаем данные порциями
                    ohlcv = self.exchange.fetch_ohlcv(
                        symbol, 
                        self.timeframe, 
                        since=current_since, 
                        limit=self.limit
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_data.extend(ohlcv)
                    
                    # Обновляем время для следующей порции
                    current_since = ohlcv[-1][0] + 1
                    
                    # Небольшая задержка для избежания rate limit
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка получения данных {symbol} с {current_since}: {e}")
                    break
            
            if not all_data:
                return pd.DataFrame()
            
            # Создаем DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Фильтруем по датам
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            logger.info(f"✅ {symbol}: получено {len(df)} свечей с {df.index[0]} по {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Добавляет технические индикаторы"""
        if df.empty:
            return df
            
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
            
            # Volume indicators
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
            
            # Price momentum
            df['price_change_1h'] = df['close'].pct_change(1)
            df['price_change_4h'] = df['close'].pct_change(4)
            df['price_change_24h'] = df['close'].pct_change(24)
            
            # Volatility
            df['volatility'] = df['close'].rolling(window=24).std()
            
            return df.dropna()
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета индикаторов: {e}")
            return df
    
    def identify_shooting_stars(self, df: pd.DataFrame, lookforward_hours: int = 24) -> pd.DataFrame:
        """Определяет 'стреляющие' монеты - те, которые показали значительный рост в будущем"""
        if df.empty or len(df) < lookforward_hours:
            return df
            
        try:
            # Определяем пороги для "стреляющих" монет
            price_threshold = 0.15  # 15% рост
            volume_threshold = 2.0  # 2x объем
            
            # Создаем целевые переменные
            df['future_price_max'] = df['close'].rolling(window=lookforward_hours, min_periods=1).max()
            df['future_price_change'] = (df['future_price_max'] / df['close'] - 1) * 100
            
            df['future_volume_max'] = df['volume'].rolling(window=lookforward_hours, min_periods=1).max()
            df['future_volume_avg'] = df['volume'].rolling(window=lookforward_hours, min_periods=1).mean()
            df['future_volume_ratio'] = df['future_volume_max'] / df['future_volume_avg']
            
            # Бинарная классификация: стреляющая монета или нет
            df['is_shooting_star'] = (
                (df['future_price_change'] >= price_threshold) & 
                (df['future_volume_ratio'] >= volume_threshold)
            ).astype(int)
            
            # Мультиклассовая классификация по силе роста
            df['growth_category'] = 0  # Без роста
            df.loc[df['future_price_change'] >= 5, 'growth_category'] = 1    # Малый рост (5-15%)
            df.loc[df['future_price_change'] >= 15, 'growth_category'] = 2   # Средний рост (15-30%)
            df.loc[df['future_price_change'] >= 30, 'growth_category'] = 3   # Большой рост (30-50%)
            df.loc[df['future_price_change'] >= 50, 'growth_category'] = 4   # Огромный рост (50%+)
            
            logger.info(f"📊 Определено стреляющих моментов: {df['is_shooting_star'].sum()}")
            logger.info(f"📈 Распределение по категориям роста: {df['growth_category'].value_counts().to_dict()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка определения стреляющих монет: {e}")
            return df
    
    def collect_all_data(self, max_pairs: int = None) -> Dict[str, pd.DataFrame]:
        """Собирает данные для всех монет"""
        logger.info("🚀 Начинаю сбор исторических данных...")
        
        pairs = self.get_all_usdt_pairs()
        if max_pairs:
            pairs = pairs[:max_pairs]
        
        all_data = {}
        successful = 0
        failed = 0
        
        for i, pair in enumerate(pairs):
            try:
                logger.info(f"📊 Обрабатываю {pair} ({i+1}/{len(pairs)})")
                
                # Получаем данные
                df = self.get_historical_data(pair, self.start_date, self.end_date)
                
                if not df.empty:
                    # Добавляем индикаторы
                    df = self.calculate_technical_indicators(df)
                    
                    # Определяем стреляющие моменты
                    df = self.identify_shooting_stars(df)
                    
                    if not df.empty:
                        all_data[pair] = df
                        successful += 1
                        
                        # Сохраняем промежуточные результаты
                        if successful % 10 == 0:
                            self.save_data(all_data, f"data_batch_{successful}.json")
                    else:
                        failed += 1
                else:
                    failed += 1
                
                # Небольшая задержка между запросами
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Критическая ошибка для {pair}: {e}")
                failed += 1
                continue
        
        logger.info(f"✅ Сбор завершен: {successful} успешно, {failed} неудачно")
        return all_data
    
    def save_data(self, data: Dict[str, pd.DataFrame], filename: str = None):
        """Сохраняет данные в JSON файл"""
        if not filename:
            filename = f"historical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            # Конвертируем DataFrame в словарь для JSON
            json_data = {}
            for symbol, df in data.items():
                json_data[symbol] = {
                    'data': df.to_dict('records'),
                    'index': df.index.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'columns': df.columns.tolist()
                }
            
            with open(filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"💾 Данные сохранены в {filename}")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения данных: {e}")
    
    def load_data(self, filename: str) -> Dict[str, pd.DataFrame]:
        """Загружает данные из JSON файла"""
        try:
            with open(filename, 'r') as f:
                json_data = json.load(f)
            
            data = {}
            for symbol, symbol_data in json_data.items():
                df = pd.DataFrame(symbol_data['data'])
                df.index = pd.to_datetime(symbol_data['index'])
                data[symbol] = df
            
            logger.info(f"📂 Данные загружены из {filename}: {len(data)} монет")
            return data
            
        except Exception as e:
            logger.error(f"❌ Ошибка загрузки данных: {e}")
            return {}

def main():
    """Основная функция"""
    collector = HistoricalDataCollector()
    
    print("🎯 СБОР ИСТОРИЧЕСКИХ ДАННЫХ ДЛЯ ОБУЧЕНИЯ НЕЙРОННОЙ СЕТИ")
    print("=" * 60)
    print(f"📅 Период: {collector.start_date.strftime('%Y-%m-%d')} - {collector.end_date.strftime('%Y-%m-%d')}")
    print(f"⏰ Таймфрейм: {collector.timeframe}")
    print("=" * 60)
    
    # Собираем данные (ограничиваем для тестирования)
    data = collector.collect_all_data(max_pairs=50)  # Начнем с 50 монет
    
    if data:
        # Сохраняем финальные данные
        collector.save_data(data)
        
        # Статистика
        total_records = sum(len(df) for df in data.values())
        shooting_stars = sum(df['is_shooting_star'].sum() for df in data.values() if 'is_shooting_star' in df.columns)
        
        print("\n📊 СТАТИСТИКА СБОРА:")
        print(f"   - Монет обработано: {len(data)}")
        print(f"   - Всего записей: {total_records:,}")
        print(f"   - Стреляющих моментов: {shooting_stars:,}")
        print(f"   - Процент стреляющих: {(shooting_stars/total_records*100):.2f}%")
        
        return data
    else:
        print("❌ Не удалось собрать данные")
        return None

if __name__ == "__main__":
    data = main()


