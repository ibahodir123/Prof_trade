#!/usr/bin/env python3
"""
БАЗОВЫЙ ПОИСКОВИК ПАТТЕРНОВ
Простой подход: находим ВСЕ движения min→max и записываем 9 признаков
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
import ccxt
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class BasicPatternFinder:
    """Базовый поисковик паттернов движений"""
    
    def __init__(self):
        self.movements_data = []  # Все найденные движения
        self.min_movement = 0.01  # Минимум 1% для значимого движения
        
    def find_all_movements(self, symbols: List[str]) -> bool:
        """Поиск всех движений min→max для списка символов"""
        try:
            print(f"🔍 Ищу все движения min→max для {len(symbols)} символов...")
            
            for symbol in symbols:
                print(f"\n📊 Анализирую {symbol}...")
                
                # Получаем исторические данные
                df = self._get_historical_data(symbol)
                if df is None or len(df) < 100:
                    print(f"❌ Недостаточно данных для {symbol}")
                    continue
                
                # Подготавливаем 9 признаков
                df = self._prepare_features(df)
                if df is None:
                    print(f"❌ Ошибка подготовки признаков для {symbol}")
                    continue
                
                print(f"📈 Подготовлено {len(df)} свечей для {symbol}")
                
                # Ищем все движения
                movements = self._find_movements_in_data(df, symbol)
                print(f"✅ Найдено {len(movements)} движений в {symbol}")
                
                self.movements_data.extend(movements)
            
            total_movements = len(self.movements_data)
            print(f"\n🎯 ИТОГО найдено {total_movements} движений min→max")
            
            return total_movements > 0
            
        except Exception as e:
            print(f"❌ Ошибка поиска движений: {e}")
            return False

    def _find_movements_in_data(self, df: pd.DataFrame, symbol: str) -> List[Dict]:
        """Поиск всех движений min→max в данных"""
        try:
            movements = []
            
            # Простой алгоритм: ищем все локальные min и max
            prices = df['close'].values
            lookback = 6  # 6 часов lookback для локальных экстремумов
            
            # Находим все локальные минимумы и максимумы
            local_mins = []
            local_maxs = []
            
            for i in range(lookback, len(prices) - lookback):
                current_price = prices[i]
                
                # Проверяем локальный минимум
                is_min = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and prices[j] < current_price:
                        is_min = False
                        break
                
                if is_min:
                    local_mins.append({'idx': i, 'price': current_price})
                
                # Проверяем локальный максимум
                is_max = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and prices[j] > current_price:
                        is_max = False
                        break
                
                if is_max:
                    local_maxs.append({'idx': i, 'price': current_price})
            
            # Создаем движения min→max
            for min_point in local_mins:
                # Ищем следующий максимум после минимума
                for max_point in local_maxs:
                    if max_point['idx'] > min_point['idx']:
                        
                        # Проверяем значимость движения
                        movement_percent = (max_point['price'] - min_point['price']) / min_point['price']
                        
                        if movement_percent >= self.min_movement:
                            
                            # Создаем движение
                            movement = self._create_movement(
                                df, min_point['idx'], max_point['idx'], symbol, movement_percent
                            )
                            
                            if movement:
                                movements.append(movement)
                        
                        break  # Берем первый максимум после минимума
            
            return movements
            
        except Exception as e:
            print(f"❌ Ошибка поиска движений в данных: {e}")
            return []

    def _create_movement(self, df: pd.DataFrame, min_idx: int, max_idx: int, 
                        symbol: str, movement_percent: float) -> Optional[Dict]:
        """Создание записи о движении с 9 признаками"""
        try:
            min_row = df.iloc[min_idx]
            max_row = df.iloc[max_idx]
            
            # Извлекаем 9 групп признаков для точки минимума (вход LONG)
            min_features = self._extract_9_features(min_row)
            
            # Извлекаем 9 групп признаков для точки максимума (выход LONG)
            max_features = self._extract_9_features(max_row)
            
            if min_features is None or max_features is None:
                return None
            
            movement = {
                'symbol': symbol,
                'min_timestamp': str(min_row.name),
                'max_timestamp': str(max_row.name),
                'min_price': min_row['close'],
                'max_price': max_row['close'],
                'movement_percent': movement_percent * 100,  # В процентах
                'duration_hours': max_idx - min_idx,
                
                # 9 признаков в точке минимума (сигнал входа LONG)
                'min_features': {
                    'velocities': min_features['velocities'],
                    'accelerations': min_features['accelerations'],
                    'velocity_ratios': min_features['velocity_ratios'],
                    'distances': min_features['distances'],
                    'distance_changes': min_features['distance_changes'],
                    'angles': min_features['angles'],
                    'angle_changes': min_features['angle_changes'],
                    'ema_relationships': min_features['ema_relationships'],
                    'synchronizations': min_features['synchronizations']
                },
                
                # 9 признаков в точке максимума (сигнал выхода LONG)
                'max_features': {
                    'velocities': max_features['velocities'],
                    'accelerations': max_features['accelerations'],
                    'velocity_ratios': max_features['velocity_ratios'],
                    'distances': max_features['distances'],
                    'distance_changes': max_features['distance_changes'],
                    'angles': max_features['angles'],
                    'angle_changes': max_features['angle_changes'],
                    'ema_relationships': max_features['ema_relationships'],
                    'synchronizations': max_features['synchronizations']
                }
            }
            
            return movement
            
        except Exception as e:
            print(f"❌ Ошибка создания движения: {e}")
            return None

    def _extract_9_features(self, row) -> Optional[Dict]:
        """Извлечение 9 групп признаков из строки данных"""
        try:
            features = {
                # 1. Velocities (4 значения)
                'velocities': {
                    'price': float(row['price_velocity']),
                    'ema20': float(row['ema20_velocity']),
                    'ema50': float(row['ema50_velocity']),
                    'ema100': float(row['ema100_velocity'])
                },
                
                # 2. Accelerations (4 значения)
                'accelerations': {
                    'price': float(row['price_acceleration']),
                    'ema20': float(row['ema20_acceleration']),
                    'ema50': float(row['ema50_acceleration']),
                    'ema100': float(row['ema100_acceleration'])
                },
                
                # 3. Velocity ratios (3 значения)
                'velocity_ratios': {
                    'price_ema20': float(row['price_to_ema20_velocity_ratio']),
                    'price_ema50': float(row['price_to_ema50_velocity_ratio']),
                    'price_ema100': float(row['price_to_ema100_velocity_ratio'])
                },
                
                # 4. Distances (3 значения)
                'distances': {
                    'price_ema20': float(row['price_to_ema20_distance']),
                    'price_ema50': float(row['price_to_ema50_distance']),
                    'price_ema100': float(row['price_to_ema100_distance'])
                },
                
                # 5. Distance changes (3 значения)
                'distance_changes': {
                    'price_ema20': float(row['price_to_ema20_distance_change']),
                    'price_ema50': float(row['price_to_ema50_distance_change']),
                    'price_ema100': float(row['price_to_ema100_distance_change'])
                },
                
                # 6. Angles (3 значения)
                'angles': {
                    'ema20': float(row['ema20_angle']),
                    'ema50': float(row['ema50_angle']),
                    'ema100': float(row['ema100_angle'])
                },
                
                # 7. Angle changes (3 значения)
                'angle_changes': {
                    'ema20': float(row['ema20_angle_change']),
                    'ema50': float(row['ema50_angle_change']),
                    'ema100': float(row['ema100_angle_change'])
                },
                
                # 8. EMA relationships (3 значения)
                'ema_relationships': {
                    'ema20_ema50': float(row['ema20_to_ema50']),
                    'ema20_ema100': float(row['ema20_to_ema100']),
                    'ema50_ema100': float(row['ema50_to_ema100'])
                },
                
                # 9. Synchronizations (3 значения)
                'synchronizations': {
                    'price_ema20': float(row['price_ema20_sync']),
                    'price_ema50': float(row['price_ema50_sync']),
                    'price_ema100': float(row['price_ema100_sync'])
                }
            }
            
            # Проверяем на NaN и заменяем на 0
            for group_name, group_data in features.items():
                for key, value in group_data.items():
                    if np.isnan(value) or np.isinf(value):
                        features[group_name][key] = 0.0
            
            return features
            
        except Exception as e:
            print(f"❌ Ошибка извлечения признаков: {e}")
            return None

    def save_movements(self, filename: str = "movements_database.json") -> bool:
        """Сохранение всех найденных движений"""
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.movements_data, f, ensure_ascii=False, indent=2)
            
            print(f"💾 Сохранено {len(self.movements_data)} движений в {filename}")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Получение статистики по найденным движениям"""
        try:
            if not self.movements_data:
                return {}
            
            # Общая статистика
            total_movements = len(self.movements_data)
            movements_by_symbol = {}
            
            profits = []
            durations = []
            
            for movement in self.movements_data:
                symbol = movement['symbol']
                if symbol not in movements_by_symbol:
                    movements_by_symbol[symbol] = 0
                movements_by_symbol[symbol] += 1
                
                profits.append(movement['movement_percent'])
                durations.append(movement['duration_hours'])
            
            stats = {
                'total_movements': total_movements,
                'by_symbol': movements_by_symbol,
                'avg_profit_percent': np.mean(profits),
                'median_profit_percent': np.median(profits),
                'max_profit_percent': np.max(profits),
                'min_profit_percent': np.min(profits),
                'avg_duration_hours': np.mean(durations),
                'median_duration_hours': np.median(durations)
            }
            
            return stats
            
        except Exception as e:
            print(f"❌ Ошибка расчета статистики: {e}")
            return {}

    def _get_historical_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Получение исторических данных с 01.01.2025"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Дата начала: 01.01.2025
            start_date = datetime(2025, 1, 1, 0, 0, 0)
            since = int(start_date.timestamp() * 1000)
            
            # Получаем данные порциями
            all_ohlcv = []
            current_since = since
            max_per_request = 1000
            
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=max_per_request)
                if not ohlcv or len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                current_since = ohlcv[-1][0] + 3600000
                
                if current_since >= exchange.milliseconds():
                    break
                
                if len(all_ohlcv) > 20000:
                    break
            
            if not all_ohlcv:
                return None
            
            # Создаем DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения данных {symbol}: {e}")
            return None

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Подготовка всех 27 признаков"""
        try:
            # Рассчитываем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # 1. Velocity (скорость изменения)
            df['price_velocity'] = df['close'].pct_change()
            df['ema20_velocity'] = df['ema_20'].pct_change()
            df['ema50_velocity'] = df['ema_50'].pct_change()
            df['ema100_velocity'] = df['ema_100'].pct_change()
            
            # 2. Acceleration (ускорение)
            df['price_acceleration'] = df['price_velocity'].pct_change()
            df['ema20_acceleration'] = df['ema20_velocity'].pct_change()
            df['ema50_acceleration'] = df['ema50_velocity'].pct_change()
            df['ema100_acceleration'] = df['ema100_velocity'].pct_change()
            
            # 3. Velocity ratios
            df['price_to_ema20_velocity_ratio'] = df['price_velocity'] / (df['ema20_velocity'] + 1e-8)
            df['price_to_ema50_velocity_ratio'] = df['price_velocity'] / (df['ema50_velocity'] + 1e-8)
            df['price_to_ema100_velocity_ratio'] = df['price_velocity'] / (df['ema100_velocity'] + 1e-8)
            
            # 4. Distance to EMAs
            df['price_to_ema20_distance'] = (df['close'] - df['ema_20']) / df['close']
            df['price_to_ema50_distance'] = (df['close'] - df['ema_50']) / df['close']
            df['price_to_ema100_distance'] = (df['close'] - df['ema_100']) / df['close']
            
            # 5. Distance change
            df['price_to_ema20_distance_change'] = df['price_to_ema20_distance'].diff()
            df['price_to_ema50_distance_change'] = df['price_to_ema50_distance'].diff()
            df['price_to_ema100_distance_change'] = df['price_to_ema100_distance'].diff()
            
            # 6. EMA angles (наклон)
            df['ema20_angle'] = np.arctan(df['ema20_velocity']) * 180 / np.pi
            df['ema50_angle'] = np.arctan(df['ema50_velocity']) * 180 / np.pi
            df['ema100_angle'] = np.arctan(df['ema100_velocity']) * 180 / np.pi
            
            # 7. Angle change
            df['ema20_angle_change'] = df['ema20_angle'].diff()
            df['ema50_angle_change'] = df['ema50_angle'].diff()
            df['ema100_angle_change'] = df['ema100_angle'].diff()
            
            # 8. EMA relationships
            df['ema20_to_ema50'] = df['ema_20'] / (df['ema_50'] + 1e-8)
            df['ema20_to_ema100'] = df['ema_20'] / (df['ema_100'] + 1e-8)
            df['ema50_to_ema100'] = df['ema_50'] / (df['ema_100'] + 1e-8)
            
            # 9. Price-EMA synchronization
            window_size = 20
            df['price_ema20_sync'] = df['close'].rolling(window_size).corr(df['ema_20'])
            df['price_ema50_sync'] = df['close'].rolling(window_size).corr(df['ema_50'])
            df['price_ema100_sync'] = df['close'].rolling(window_size).corr(df['ema_100'])
            
            # Очищаем данные
            df = df.dropna()
            df = df.replace([np.inf, -np.inf], 0)
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка подготовки признаков: {e}")
            return None

if __name__ == "__main__":
    # Запуск базового поисковика
    finder = BasicPatternFinder()
    
    print("🎯 БАЗОВЫЙ ПОИСКОВИК ДВИЖЕНИЙ MIN→MAX")
    print("====================================")
    
    # Список монет для анализа
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    
    # Ищем все движения
    print("🔍 ШАГ 1: Поиск всех движений...")
    if finder.find_all_movements(symbols):
        
        # Показываем статистику
        print("\n📊 ШАГ 2: Статистика движений...")
        stats = finder.get_statistics()
        
        print(f"📈 Всего движений: {stats['total_movements']}")
        print(f"📊 По символам: {stats['by_symbol']}")
        print(f"💰 Средняя прибыль: {stats['avg_profit_percent']:.2f}%")
        print(f"📏 Медианная прибыль: {stats['median_profit_percent']:.2f}%")
        print(f"🚀 Максимальная прибыль: {stats['max_profit_percent']:.2f}%")
        print(f"⏰ Средняя длительность: {stats['avg_duration_hours']:.1f} часов")
        
        # Сохраняем результаты
        print("\n💾 ШАГ 3: Сохранение результатов...")
        finder.save_movements("movements_database.json")
        
        print("\n✅ ГОТОВО! База движений создана.")
        print("Теперь можно анализировать закономерности в 9 признаках.")
    else:
        print("❌ Не удалось найти движения")
