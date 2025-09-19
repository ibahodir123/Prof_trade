#!/usr/bin/env python3
"""
Продвинутый EMA анализатор с ML логикой
Основан на скоростях, расстояниях и углах EMA линий
Только LONG сигналы для всех типов трендов
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import ccxt

logger = logging.getLogger(__name__)

class AdvancedEMAAnalyzer:
    """Продвинутый анализатор EMA с ML логикой"""
    
    def __init__(self):
        self.ema_periods = [20, 50, 100]
        self.min_data_points = 200  # Минимум данных для анализа
        
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Расчет EMA"""
        return prices.ewm(span=period, adjust=False).mean()
    
    def calculate_ema_speeds(self, ema_values: pd.Series, window: int = 5) -> pd.Series:
        """Расчет скорости EMA линии"""
        # Скорость = изменение угла наклона EMA за период
        ema_diff = ema_values.diff(window)
        ema_speed = ema_diff / ema_values.shift(window)
        return ema_speed.fillna(0)
    
    def calculate_price_speed(self, prices: pd.Series, ema_values: pd.Series, window: int = 5) -> pd.Series:
        """Расчет скорости цены относительно EMA"""
        price_ema_ratio = prices / ema_values
        speed = price_ema_ratio.diff(window) / price_ema_ratio.shift(window)
        return speed.fillna(0)
    
    def calculate_distances(self, prices: pd.Series, ema_20: pd.Series, ema_50: pd.Series, ema_100: pd.Series) -> Dict[str, pd.Series]:
        """Расчет всех расстояний между линиями и ценой"""
        distances = {
            'price_to_ema20': abs(prices - ema_20) / prices,
            'price_to_ema50': abs(prices - ema_50) / prices,
            'price_to_ema100': abs(prices - ema_100) / prices,
            'ema20_to_ema50': abs(ema_20 - ema_50) / ema_20,
            'ema50_to_ema100': abs(ema_50 - ema_100) / ema_50,
            'ema20_to_ema100': abs(ema_20 - ema_100) / ema_20
        }
        return distances
    
    def calculate_trend_angle(self, ema_20: pd.Series, ema_50: pd.Series, ema_100: pd.Series) -> pd.Series:
        """Расчет угла тренда в градусах"""
        # Угол тренда основан на наклоне EMA 20 относительно EMA 100
        ema_slope = ema_20.diff(20) / ema_100  # Наклон за 20 периодов
        trend_angle = np.arctan(ema_slope) * 180 / np.pi  # В градусах
        return trend_angle.fillna(0)
    
    def calculate_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех EMA признаков для ML"""
        # EMA линии
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        
        # Скорости EMA
        df['ema20_speed'] = df['ema20'].diff(5) / df['ema20'].shift(5)
        df['ema50_speed'] = df['ema50'].diff(5) / df['ema50'].shift(5)
        df['ema100_speed'] = df['ema100'].diff(5) / df['ema100'].shift(5)
        
        # Скорость цены относительно EMA
        df['price_speed_vs_ema20'] = (df['close'] / df['ema20']).diff(5)
        df['price_speed_vs_ema50'] = (df['close'] / df['ema50']).diff(5)
        df['price_speed_vs_ema100'] = (df['close'] / df['ema100']).diff(5)
        
        # Расстояния между EMA
        df['ema20_to_ema50'] = abs(df['ema20'] - df['ema50']) / df['ema20']
        df['ema50_to_ema100'] = abs(df['ema50'] - df['ema100']) / df['ema50']
        df['ema20_to_ema100'] = abs(df['ema20'] - df['ema100']) / df['ema20']
        
        # Расстояния от цены до EMA
        df['price_to_ema20'] = abs(df['close'] - df['ema20']) / df['close']
        df['price_to_ema50'] = abs(df['close'] - df['ema50']) / df['close']
        df['price_to_ema100'] = abs(df['close'] - df['ema100']) / df['close']
        
        # Угол тренда
        ema_slope = df['ema20'].diff(20) / df['ema100']
        df['trend_angle'] = np.arctan(ema_slope) * 180 / np.pi
        
        # Тип тренда (кодирование)
        df['trend_type'] = 0  # боковой
        df.loc[df['ema20'] > df['ema50'], 'trend_type'] = 1  # восходящий
        df.loc[df['ema20'] < df['ema50'], 'trend_type'] = -1  # нисходящий
        
        # Фаза рынка
        df['market_phase'] = 0  # коррекция
        df.loc[df['close'] > df['ema20'], 'market_phase'] = 1  # импульс
        
        return df.fillna(0)
    
    def determine_trend_type(self, ema_20: pd.Series, ema_50: pd.Series, ema_100: pd.Series) -> str:
        """Определение типа тренда по расположению EMA"""
        latest_ema20 = ema_20.iloc[-1]
        latest_ema50 = ema_50.iloc[-1]
        latest_ema100 = ema_100.iloc[-1]
        
        if latest_ema20 > latest_ema50 > latest_ema100:
            return "восходящий"
        elif latest_ema20 < latest_ema50 < latest_ema100:
            return "нисходящий"
        else:
            return "боковой"
    
    def determine_market_phase(self, prices: pd.Series, ema_20: pd.Series, trend_type: str) -> str:
        """Определение фазы рынка (импульс/коррекция)"""
        current_price = prices.iloc[-1]
        current_ema20 = ema_20.iloc[-1]
        
        if trend_type == "нисходящий":
            # В нисходящем тренде: цена ниже EMA20 = импульс, выше = коррекция
            return "импульс" if current_price < current_ema20 else "коррекция"
        elif trend_type == "восходящий":
            # В восходящем тренде: цена выше EMA20 = импульс, ниже = коррекция
            return "импульс" if current_price > current_ema20 else "коррекция"
        else:
            # В боковом тренде: определяем по расстоянию до EMA20
            distance = abs(current_price - current_ema20) / current_ema20
            return "импульс" if distance > 0.01 else "коррекция"
    
    def find_entry_points(self, distances: Dict[str, pd.Series], prices: pd.Series, trend_type: str, window: int = 10) -> List[int]:
        """Поиск точек входа в зависимости от типа тренда"""
        entry_points = []
        
        for dist_name, dist_values in distances.items():
            if 'price_to' in dist_name:  # Только расстояния цена-EMA
                
                if trend_type == "нисходящий" or trend_type == "боковой":
                    # В нисходящем и боковом: максимальные расстояния = минимумы цены = LONG вход
                    local_maxima = []
                    for i in range(window, len(dist_values) - window):
                        if dist_values.iloc[i] == dist_values.iloc[i-window:i+window+1].max():
                            local_maxima.append(i)
                    
                    # Проверяем, что это минимум цены
                    for idx in local_maxima:
                        if idx < len(prices):
                            price_window = prices.iloc[max(0, idx-window):min(len(prices), idx+window+1)]
                            if prices.iloc[idx] == price_window.min():
                                entry_points.append(idx)
                
                elif trend_type == "восходящий":
                    # В восходящем: минимальные расстояния = разные ситуации (приближение, пересечение) = LONG вход
                    local_minima = []
                    for i in range(window, len(dist_values) - window):
                        if dist_values.iloc[i] == dist_values.iloc[i-window:i+window+1].min():
                            local_minima.append(i)
                    
                    # Проверяем разные ситуации при минимальных расстояниях:
                    # 1. Приближение к EMA (минимум цены)
                    # 2. Пересечение EMA линий
                    # 3. Коррекционные сближения
                    for idx in local_minima:
                        if idx < len(prices):
                            current_distance = dist_values.iloc[idx]
                            price_window = prices.iloc[max(0, idx-window):min(len(prices), idx+window+1)]
                            
                            # Разные ситуации при минимальных расстояниях в восходящем тренде:
                            # 1. Приближение к EMA линии
                            # 2. Пересечение EMA линии  
                            # 3. Касание EMA линии
                            # 4. Отскок от EMA линии
                            if (prices.iloc[idx] == price_window.min() or  # Минимум цены (приближение)
                                current_distance < 0.001 or  # Очень близко к EMA (пересечение/касание)
                                current_distance < dist_values.iloc[idx-window:idx+window+1].quantile(0.1)):  # Локальный минимум (отскок)
                                entry_points.append(idx)
        
        return sorted(list(set(entry_points)))
    
    def find_exit_points(self, distances: Dict[str, pd.Series], prices: pd.Series, window: int = 10) -> List[int]:
        """Поиск точек выхода (минимальные расстояния = разные ситуации)"""
        exit_points = []
        
        # Ищем локальные минимумы в расстояниях
        for dist_name, dist_values in distances.items():
            if 'price_to' in dist_name:  # Только расстояния цена-EMA
                # Находим локальные минимумы расстояний
                local_minima = []
                for i in range(window, len(dist_values) - window):
                    if dist_values.iloc[i] == dist_values.iloc[i-window:i+window+1].min():
                        local_minima.append(i)
                
                # Анализируем разные ситуации при минимальных расстояниях
                for idx in local_minima:
                    if idx < len(prices):
                        current_price = prices.iloc[idx]
                        price_window = prices.iloc[max(0, idx-window):min(len(prices), idx+window+1)]
                        
                        # Разные ситуации при минимальных расстояниях:
                        # 1. Максимумы цены (тейк профит)
                        # 2. Коррекционные сближения
                        # 3. Пересечения EMA линий
                        
                        # Проверяем, что это значимое событие
                        if (current_price == price_window.max() or  # Максимум цены
                            current_price == price_window.min() or  # Минимум цены (коррекция)
                            dist_values.iloc[idx] < 0.001):  # Очень близко к EMA (пересечение)
                            exit_points.append(idx)
        
        return sorted(list(set(exit_points)))
    
    def analyze_coin(self, symbol: str, ohlcv_data: Optional[List] = None) -> Dict:
        """Полный анализ монеты с EMA логикой"""
        try:
            if ohlcv_data is None:
                # Получаем данные через Binance API
                exchange = ccxt.binance()
                ohlcv_data = exchange.fetch_ohlcv(symbol, '1h', limit=self.min_data_points)
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            if len(df) < self.min_data_points:
                return {
                    'symbol': symbol,
                    'signal': 'ОЖИДАНИЕ',
                    'reason': 'Недостаточно данных для анализа',
                    'trend_type': 'неопределен',
                    'market_phase': 'неопределена',
                    'trend_angle': 0,
                    'entry_points': [],
                    'exit_points': []
                }
            
            # Расчет EMA
            df['ema20'] = self.calculate_ema(df['close'], 20)
            df['ema50'] = self.calculate_ema(df['close'], 50)
            df['ema100'] = self.calculate_ema(df['close'], 100)
            
            # Расчет скоростей
            df['ema20_speed'] = self.calculate_ema_speeds(df['ema20'])
            df['ema50_speed'] = self.calculate_ema_speeds(df['ema50'])
            df['ema100_speed'] = self.calculate_ema_speeds(df['ema100'])
            df['price_speed'] = self.calculate_price_speed(df['close'], df['ema20'])
            
            # Расчет расстояний
            distances = self.calculate_distances(df['close'], df['ema20'], df['ema50'], df['ema100'])
            
            # Расчет угла тренда
            df['trend_angle'] = self.calculate_trend_angle(df['ema20'], df['ema50'], df['ema100'])
            
            # Определение тренда и фазы
            trend_type = self.determine_trend_type(df['ema20'], df['ema50'], df['ema100'])
            market_phase = self.determine_market_phase(df['close'], df['ema20'], trend_type)
            
            # Поиск точек входа и выхода
            entry_points = self.find_entry_points(distances, df['close'], trend_type)
            exit_points = self.find_exit_points(distances, df['close'])
            
            # Генерация сигнала
            signal = self.generate_signal(df, trend_type, market_phase, entry_points, exit_points)
            
            return {
                'symbol': symbol,
                'signal': signal['type'],
                'reason': signal['reason'],
                'trend_type': trend_type,
                'market_phase': market_phase,
                'trend_angle': df['trend_angle'].iloc[-1],
                'entry_points': entry_points[-5:] if entry_points else [],  # Последние 5 точек
                'exit_points': exit_points[-5:] if exit_points else [],    # Последние 5 точек
                'current_price': df['close'].iloc[-1],
                'ema20': df['ema20'].iloc[-1],
                'ema50': df['ema50'].iloc[-1],
                'ema100': df['ema100'].iloc[-1],
                'ema20_speed': df['ema20_speed'].iloc[-1],
                'price_speed': df['price_speed'].iloc[-1]
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return {
                'symbol': symbol,
                'signal': 'ОШИБКА',
                'reason': f'Ошибка анализа: {str(e)}',
                'trend_type': 'неопределен',
                'market_phase': 'неопределена',
                'trend_angle': 0,
                'entry_points': [],
                'exit_points': []
            }
    
    def generate_signal(self, df: pd.DataFrame, trend_type: str, market_phase: str, 
                       entry_points: List[int], exit_points: List[int]) -> Dict[str, str]:
        """Генерация торгового сигнала с универсальной логикой"""
        current_price = df['close'].iloc[-1]
        ema20 = df['ema20'].iloc[-1]
        ema50 = df['ema50'].iloc[-1]
        ema100 = df['ema100'].iloc[-1]
        
        # Логика по типам трендов:
        # НИСХОДЯЩИЙ/БОКОВОЙ: максимальные расстояния = точки входа (минимумы цены)
        # ВОСХОДЯЩИЙ: минимальные расстояния = точки входа (минимумы во время коррекции)
        # Минимальные расстояния = разные ситуации (максимумы, коррекции, пересечения)
        
        # Только LONG сигналы!
        if trend_type == "нисходящий":
            if market_phase == "импульс":
                # В нисходящем импульсе: максимальные расстояния = минимумы цены = LONG вход
                if entry_points and len(entry_points) > 0:
                    return {
                        'type': 'LONG',
                        'reason': f'НИСХОДЯЩИЙ ИМПУЛЬС: максимальное расстояние = минимум цены = LONG вход'
                    }
            else:  # коррекция
                # В коррекции: минимальные расстояния = разные ситуации = анализ выхода
                if exit_points and len(exit_points) > 0:
                    return {
                        'type': 'ТЕЙК ПРОФИТ',
                        'reason': f'НИСХОДЯЩИЙ КОРРЕКЦИЯ: минимальное расстояние = разные ситуации = выход'
                    }
        
        elif trend_type == "восходящий":
            if market_phase == "коррекция":
                # В восходящей коррекции: минимальные расстояния = разные ситуации (приближение, пересечение) = LONG вход
                if entry_points and len(entry_points) > 0:
                    return {
                        'type': 'LONG',
                        'reason': f'ВОСХОДЯЩИЙ КОРРЕКЦИЯ: минимальное расстояние = разные ситуации (приближение, пересечение, касание, отскок) = LONG вход'
                    }
            else:  # импульс
                # В восходящем импульсе: максимальные расстояния = выход при максимуме импульса = анализ выхода
                if exit_points and len(exit_points) > 0:
                    return {
                        'type': 'ТЕЙК ПРОФИТ',
                        'reason': f'ВОСХОДЯЩИЙ ИМПУЛЬС: максимальное расстояние = выход при максимуме импульса = выход'
                    }
        
        elif trend_type == "боковой":
            # В боковом тренде: максимальные расстояния = минимумы цены = LONG вход
            if entry_points and len(entry_points) > 0:
                return {
                    'type': 'LONG',
                    'reason': f'БОКОВОЙ ТРЕНД: максимальное расстояние = минимум цены = LONG вход'
                }
            # Минимальные расстояния = разные ситуации = анализ выхода
            elif exit_points and len(exit_points) > 0:
                return {
                    'type': 'ТЕЙК ПРОФИТ',
                    'reason': f'БОКОВОЙ ТРЕНД: минимальное расстояние = разные ситуации = выход'
                }
        
        return {
            'type': 'ОЖИДАНИЕ',
            'reason': 'Нет четкого сигнала для LONG позиции'
        }

# Пример использования
if __name__ == "__main__":
    analyzer = AdvancedEMAAnalyzer()
    
    # Тест анализа
    result = analyzer.analyze_coin("BTC/USDT")
    print("=== РЕЗУЛЬТАТ АНАЛИЗА ===")
    print(f"Монета: {result['symbol']}")
    print(f"Сигнал: {result['signal']}")
    print(f"Причина: {result['reason']}")
    print(f"Тренд: {result['trend_type']}")
    print(f"Фаза: {result['market_phase']}")
    print(f"Угол тренда: {result['trend_angle']:.2f}°")
    print(f"Текущая цена: ${result['current_price']:.8f}")
    print(f"EMA 20: ${result['ema20']:.8f}")
    print(f"EMA 50: ${result['ema50']:.8f}")
    print(f"EMA 100: ${result['ema100']:.8f}")
    print(f"Скорость EMA 20: {result['ema20_speed']:.6f}")
    print(f"Скорость цены: {result['price_speed']:.6f}")
