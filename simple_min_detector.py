#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Упрощенный детектор минимумов для LONG позиций
Только 4 критерия с EMA20:
1. Скорость цены
2. Скорость EMA20  
3. Угол EMA20
4. Расстояние цена-EMA20
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import warnings
warnings.filterwarnings('ignore')

class SimpleMinDetector:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.minimums_database = []
        
    def get_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение данных с Binance"""
        try:
            print(f"📊 Загружаю {symbol} с {start_date.date()} по {end_date.date()}")
            exchange = ccxt.binance()
            since = int(start_date.timestamp() * 1000)
            end_ts = int(end_date.timestamp() * 1000)
            
            all_ohlcv = []
            current_since = since
            
            while current_since < end_ts:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                    if not ohlcv:
                        break
                    
                    # Фильтруем по дате
                    filtered_ohlcv = [candle for candle in ohlcv if candle[0] <= end_ts]
                    all_ohlcv.extend(filtered_ohlcv)
                    
                    if not ohlcv:
                        break
                        
                    current_since = ohlcv[-1][0] + 1
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"❌ Ошибка API: {e}")
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            # Удаляем дубликаты
            seen = set()
            unique_ohlcv = []
            for candle in all_ohlcv:
                if candle[0] not in seen:
                    seen.add(candle[0])
                    unique_ohlcv.append(candle)
            
            df = pd.DataFrame(unique_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            print(f"✅ Загружено {len(df)} свечей для {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_4_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет 4 критериев с EMA20"""
        try:
            # Добавляем EMA20
            df['ema20'] = df['close'].ewm(span=20).mean()
            
            # 1. Скорость цены (изменение за час в %)
            df['price_velocity'] = df['close'].pct_change() * 100
            
            # 2. Скорость EMA20 (изменение за час в %)
            df['ema20_velocity'] = df['ema20'].pct_change() * 100
            
            # 3. Угол EMA20 (изменение за 10 часов в %)
            df['ema20_angle'] = ((df['ema20'] / df['ema20'].shift(10)) - 1) * 100
            
            # 4. Расстояние цена-EMA20 (в %)
            df['distance_to_ema20'] = ((df['close'] - df['ema20']) / df['ema20']) * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка расчета критериев: {e}")
            return df
    
    def find_local_minimums(self, df: pd.DataFrame, lookback: int = 12) -> list:
        """Поиск локальных минимумов"""
        minimums = []
        
        for i in range(lookback, len(df) - lookback):
            current_low = df.iloc[i]['low']
            is_minimum = True
            
            # Проверяем что это самая низкая точка в окне
            for j in range(i - lookback, i + lookback + 1):
                if j != i and df.iloc[j]['low'] <= current_low:
                    is_minimum = False
                    break
            
            if is_minimum:
                minimums.append(i)
        
        return minimums
    
    def analyze_minimum(self, df: pd.DataFrame, min_idx: int) -> dict:
        """Анализ минимума и импульса перед ним"""
        try:
            min_time = df.index[min_idx]
            min_price = df.iloc[min_idx]['low']
            
            # Ищем начало импульса (последний локальный максимум перед минимумом)
            impulse_start_idx = min_idx
            max_price = min_price
            
            # Ищем в предыдущих 48 часах
            for i in range(max(0, min_idx - 48), min_idx):
                if df.iloc[i]['high'] > max_price:
                    max_price = df.iloc[i]['high']
                    impulse_start_idx = i
            
            # Процент падения
            fall_percent = ((min_price - max_price) / max_price) * 100
            
            # Только значимые падения (больше 2%)
            if fall_percent > -2:
                return None
            
            # Анализируем критерии в момент минимума
            min_data = df.iloc[min_idx]
            
            # Проверяем что данные корректны
            if pd.isna(min_data['ema20']) or pd.isna(min_data['price_velocity']):
                return None
            
            criteria = {
                'price_velocity': min_data['price_velocity'],
                'ema20_velocity': min_data['ema20_velocity'], 
                'ema20_angle': min_data['ema20_angle'],
                'distance_to_ema20': min_data['distance_to_ema20']
            }
            
            # Убираем NaN значения
            for key, value in criteria.items():
                if pd.isna(value) or np.isinf(value):
                    criteria[key] = 0.0
            
            # Проверяем что цена ниже EMA20 (признак падения)
            if criteria['distance_to_ema20'] > 0:
                return None
            
            # Анализируем что происходило после минимума (прибыльность)
            future_profit = 0
            if min_idx + 24 < len(df):
                future_max = df.iloc[min_idx:min_idx+24]['high'].max()
                future_profit = ((future_max - min_price) / min_price) * 100
            
            minimum_data = {
                'symbol': None,  # Заполним позже
                'time': min_time,
                'price': min_price,
                'fall_percent': fall_percent,
                'impulse_duration': min_idx - impulse_start_idx,
                'criteria': criteria,
                'future_profit_24h': future_profit,
                'is_profitable': future_profit > 2.0  # Минимум 2% прибыли
            }
            
            return minimum_data
            
        except Exception as e:
            print(f"❌ Ошибка анализа минимума: {e}")
            return None
    
    def find_minimums_in_symbol(self, symbol: str, start_date: datetime, end_date: datetime):
        """Поиск всех минимумов в символе"""
        print(f"\\n🔍 Анализирую минимумы в {symbol}")
        
        df = self.get_data(symbol, start_date, end_date)
        if df.empty:
            return
        
        # Рассчитываем критерии
        df = self.calculate_4_criteria(df)
        
        # Ищем локальные минимумы
        minimum_indices = self.find_local_minimums(df)
        print(f"   Найдено {len(minimum_indices)} потенциальных минимумов")
        
        # Анализируем каждый минимум
        valid_minimums = 0
        for min_idx in minimum_indices:
            minimum_data = self.analyze_minimum(df, min_idx)
            if minimum_data:
                minimum_data['symbol'] = symbol
                self.minimums_database.append(minimum_data)
                valid_minimums += 1
        
        print(f"   ✅ Добавлено {valid_minimums} валидных минимумов")
    
    def collect_minimums(self, start_date: datetime, end_date: datetime):
        """Сбор всех минимумов по всем символам"""
        print("🔍 СБОР МИНИМУМОВ ДЛЯ LONG ПОЗИЦИЙ")
        print(f"📅 Период: {start_date.date()} - {end_date.date()}")
        print("=" * 50)
        
        self.minimums_database = []
        
        for symbol in self.symbols:
            try:
                self.find_minimums_in_symbol(symbol, start_date, end_date)
                time.sleep(1)  # Пауза между символами
            except Exception as e:
                print(f"❌ Ошибка обработки {symbol}: {e}")
        
        # Сохраняем результаты
        self.save_results(start_date, end_date)
        self.print_statistics()
    
    def save_results(self, start_date: datetime, end_date: datetime):
        """Сохранение результатов"""
        try:
            # Конвертируем datetime для JSON
            minimums_for_json = []
            for minimum in self.minimums_database:
                min_copy = minimum.copy()
                min_copy['time'] = minimum['time'].isoformat()
                minimums_for_json.append(min_copy)
            
            filename = f"minimums_{start_date.strftime('%Y%m')}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(minimums_for_json, f, ensure_ascii=False, indent=2)
            
            print(f"\\n💾 Данные сохранены в {filename}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
    
    def print_statistics(self):
        """Вывод статистики"""
        if not self.minimums_database:
            print("❌ Минимумы не найдены!")
            return
        
        print(f"\\n📊 СТАТИСТИКА МИНИМУМОВ:")
        print(f"   Всего найдено: {len(self.minimums_database)}")
        
        # По символам
        symbol_stats = {}
        for minimum in self.minimums_database:
            symbol = minimum['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = 0
            symbol_stats[symbol] += 1
        
        print(f"\\n📈 По символам:")
        for symbol, count in symbol_stats.items():
            print(f"   {symbol}: {count} минимумов")
        
        # Прибыльность
        profitable = [m for m in self.minimums_database if m['is_profitable']]
        profit_rate = (len(profitable) / len(self.minimums_database)) * 100
        
        print(f"\\n💰 Прибыльность:")
        print(f"   Прибыльных минимумов: {len(profitable)}")
        print(f"   Процент прибыльных: {profit_rate:.1f}%")
        
        if profitable:
            avg_profit = np.mean([m['future_profit_24h'] for m in profitable])
            print(f"   Средняя прибыль: {avg_profit:.2f}%")
        
        # Статистика критериев
        print(f"\\n🔢 Средние значения критериев в минимумах:")
        criteria_stats = {}
        for criterion in ['price_velocity', 'ema20_velocity', 'ema20_angle', 'distance_to_ema20']:
            values = [m['criteria'][criterion] for m in self.minimums_database]
            criteria_stats[criterion] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
            
            print(f"   {criterion}:")
            print(f"     Среднее: {criteria_stats[criterion]['mean']:.3f}")
            print(f"     Мин: {criteria_stats[criterion]['min']:.3f}")
            print(f"     Макс: {criteria_stats[criterion]['max']:.3f}")

if __name__ == "__main__":
    detector = SimpleMinDetector()
    
    # Анализируем январь 2025
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31, 23, 59, 59)
    
    print("🎯 ДЕТЕКТОР МИНИМУМОВ - УПРОЩЕННАЯ ВЕРСИЯ")
    print("📊 4 критерия: скорость цены, скорость EMA20, угол EMA20, расстояние")
    print("🎯 Цель: найти минимумы для LONG позиций")
    
    detector.collect_minimums(start_date, end_date)




