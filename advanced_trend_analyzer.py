"""
Продвинутый анализатор трендов с импульсами и коррекциями
Реализует логику:
1. Определение трендов (восходящий/нисходящий/флет)
2. Выявление импульсов и коррекций
3. Точки входа/выхода на основе рыночных структур
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class TrendAnalyzer:
    """Анализатор трендов с импульсами и коррекциями"""
    
    def __init__(self):
        self.trend_periods = {
            'short': 20,    # Краткосрочный тренд
            'medium': 50,   # Среднесрочный тренд  
            'long': 100     # Долгосрочный тренд
        }
        
        self.impulse_threshold = 0.02  # 2% для определения импульса
        self.correction_threshold = 0.015  # 1.5% для коррекции
        
    def analyze_trend(self, df: pd.DataFrame) -> Dict:
        """
        Основной анализ тренда с импульсами и коррекциями
        
        Args:
            df: DataFrame с OHLCV данными
            
        Returns:
            Dict с результатами анализа
        """
        try:
            # 1. Определяем общий тренд
            trend_direction = self._detect_trend_direction(df)
            
            # 2. Находим импульсы и коррекции
            impulses, corrections = self._find_impulses_corrections(df, trend_direction)
            
            # 3. Определяем текущую фазу
            current_phase = self._get_current_phase(df, impulses, corrections)
            
            # 4. Генерируем сигналы
            signal = self._generate_signal(df, trend_direction, current_phase, impulses, corrections)
            
            return {
                'trend_direction': trend_direction,
                'current_phase': current_phase,
                'impulses': impulses,
                'corrections': corrections,
                'signal': signal,
                'support_resistance': self._find_support_resistance(df),
                'analysis_timestamp': df.index[-1]
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа тренда: {e}")
            return {'error': str(e)}
    
    def _detect_trend_direction(self, df: pd.DataFrame) -> str:
        """Определение направления тренда"""
        
        # EMA для разных периодов
        ema_20 = df['close'].ewm(span=20).mean()
        ema_50 = df['close'].ewm(span=50).mean()
        ema_100 = df['close'].ewm(span=100).mean()
        
        current_price = df['close'].iloc[-1]
        
        # Логика определения тренда
        if (current_price > ema_20.iloc[-1] > ema_50.iloc[-1] > ema_100.iloc[-1]):
            return "ВОСХОДЯЩИЙ"
        elif (current_price < ema_20.iloc[-1] < ema_50.iloc[-1] < ema_100.iloc[-1]):
            return "НИСХОДЯЩИЙ"
        else:
            # Проверяем боковое движение
            price_range = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
            if price_range.iloc[-1] < 0.1:  # Менее 10% волатильности
                return "ФЛЕТ"
            else:
                return "НЕОПРЕДЕЛЕННЫЙ"
    
    def _find_impulses_corrections(self, df: pd.DataFrame, trend: str) -> Tuple[List, List]:
        """Поиск импульсов и коррекций"""
        
        impulses = []
        corrections = []
        
        # Скользящее окно для анализа
        window = 10
        
        for i in range(window, len(df)):
            current_range = df.iloc[i-window:i]
            
            # Определяем движение в окне
            price_change = (current_range['close'].iloc[-1] - current_range['close'].iloc[0]) / current_range['close'].iloc[0]
            
            if abs(price_change) > self.impulse_threshold:
                # Это импульс
                direction = "UP" if price_change > 0 else "DOWN"
                
                # Проверяем соответствие тренду
                if (trend == "ВОСХОДЯЩИЙ" and direction == "UP") or \
                   (trend == "НИСХОДЯЩИЙ" and direction == "DOWN"):
                    
                    impulses.append({
                        'start_idx': i - window,
                        'end_idx': i,
                        'direction': direction,
                        'strength': abs(price_change),
                        'start_price': current_range['close'].iloc[0],
                        'end_price': current_range['close'].iloc[-1],
                        'timestamp': df.index[i]
                    })
            
            elif abs(price_change) > self.correction_threshold:
                # Это коррекция
                direction = "UP" if price_change > 0 else "DOWN"
                
                # Коррекция идет против тренда
                if (trend == "ВОСХОДЯЩИЙ" and direction == "DOWN") or \
                   (trend == "НИСХОДЯЩИЙ" and direction == "UP"):
                    
                    corrections.append({
                        'start_idx': i - window,
                        'end_idx': i,
                        'direction': direction,
                        'strength': abs(price_change),
                        'start_price': current_range['close'].iloc[0],
                        'end_price': current_range['close'].iloc[-1],
                        'timestamp': df.index[i]
                    })
        
        return impulses, corrections
    
    def _get_current_phase(self, df: pd.DataFrame, impulses: List, corrections: List) -> str:
        """Определение текущей фазы рынка"""
        
        if not impulses and not corrections:
            return "НЕОПРЕДЕЛЕННАЯ"
        
        # Берем последние движения
        last_impulse = impulses[-1] if impulses else None
        last_correction = corrections[-1] if corrections else None
        
        current_idx = len(df) - 1
        
        if last_impulse and current_idx - last_impulse['end_idx'] < 5:
            return "ИМПУЛЬС"
        elif last_correction and current_idx - last_correction['end_idx'] < 5:
            return "КОРРЕКЦИЯ"
        else:
            return "КОНСОЛИДАЦИЯ"
    
    def _generate_signal(self, df: pd.DataFrame, trend: str, phase: str, 
                        impulses: List, corrections: List) -> Dict:
        """Генерация торгового сигнала"""
        
        current_price = df['close'].iloc[-1]
        
        # Логика сигналов на основе вашей концепции
        if trend == "ВОСХОДЯЩИЙ":
            return self._handle_uptrend_signal(df, phase, current_price, impulses, corrections)
        elif trend == "НИСХОДЯЩИЙ":
            return self._handle_downtrend_signal(df, phase, current_price, impulses, corrections)
        elif trend == "ФЛЕТ":
            return self._handle_range_signal(df, phase, current_price, impulses, corrections)
        else:
            return {"type": "ОЖИДАНИЕ", "reason": "Неопределенный тренд"}
    
    def _handle_uptrend_signal(self, df: pd.DataFrame, phase: str, current_price: float,
                              impulses: List, corrections: List) -> Dict:
        """Обработка сигналов в восходящем тренде"""
        
        if phase == "КОРРЕКЦИЯ" and corrections:
            # Входим в коррекции на минимумах
            last_correction = corrections[-1]
            if current_price <= last_correction['end_price'] * 1.01:  # В пределах 1%
                return {
                    "type": "LONG",
                    "reason": "Вход в коррекции восходящего тренда",
                    "entry_price": current_price,
                    "take_profit": self._calculate_uptrend_tp(df, current_price),
                    "stop_loss": self._calculate_uptrend_sl(df, current_price),
                    "confidence": 0.8
                }
        
        elif phase == "ИМПУЛЬС":
            # В восходящем тренде импульсы - это рост, не входим
            return {
                "type": "ОЖИДАНИЕ", 
                "reason": "Импульс роста в восходящем тренде - ждем коррекции"
            }
        
        return {"type": "ОЖИДАНИЕ", "reason": "Ожидание подходящего момента"}
    
    def _handle_downtrend_signal(self, df: pd.DataFrame, phase: str, current_price: float,
                                impulses: List, corrections: List) -> Dict:
        """Обработка сигналов в нисходящем тренде"""
        
        if phase == "КОРРЕКЦИЯ" and corrections:
            # В нисходящем тренде коррекции - это рост, входим в SHORT на максимумах
            last_correction = corrections[-1]
            if current_price >= last_correction['end_price'] * 0.99:  # В пределах 1%
                return {
                    "type": "SHORT",
                    "reason": "Вход в коррекции нисходящего тренда",
                    "entry_price": current_price,
                    "take_profit": self._calculate_downtrend_tp(df, current_price),
                    "stop_loss": self._calculate_downtrend_sl(df, current_price),
                    "confidence": 0.8
                }
        
        elif phase == "ИМПУЛЬС":
            # В нисходящем тренде импульсы - это падение, не входим
            return {
                "type": "ОЖИДАНИЕ",
                "reason": "Импульс падения в нисходящем тренде - ждем коррекции"
            }
        
        return {"type": "ОЖИДАНИЕ", "reason": "Ожидание подходящего момента"}
    
    def _handle_range_signal(self, df: pd.DataFrame, phase: str, current_price: float,
                            impulses: List, corrections: List) -> Dict:
        """Обработка сигналов во флете"""
        
        support, resistance = self._find_support_resistance(df)
        
        # Входим от поддержки на LONG, от сопротивления на SHORT
        if current_price <= support * 1.005:  # В пределах 0.5% от поддержки
            return {
                "type": "LONG",
                "reason": "Отскок от поддержки во флете",
                "entry_price": current_price,
                "take_profit": resistance,
                "stop_loss": support * 0.98,
                "confidence": 0.6
            }
        elif current_price >= resistance * 0.995:  # В пределах 0.5% от сопротивления
            return {
                "type": "SHORT", 
                "reason": "Отскок от сопротивления во флете",
                "entry_price": current_price,
                "take_profit": support,
                "stop_loss": resistance * 1.02,
                "confidence": 0.6
            }
        
        return {"type": "ОЖИДАНИЕ", "reason": "Флет - ожидание пробоя или отскока"}
    
    def _calculate_uptrend_tp(self, df: pd.DataFrame, entry_price: float) -> float:
        """Расчет Take Profit для восходящего тренда"""
        # TP на 3-5% выше входа
        return entry_price * 1.04
    
    def _calculate_uptrend_sl(self, df: pd.DataFrame, entry_price: float) -> float:
        """Расчет Stop Loss для восходящего тренда"""
        # SL на 2-3% ниже входа
        return entry_price * 0.97
    
    def _calculate_downtrend_tp(self, df: pd.DataFrame, entry_price: float) -> float:
        """Расчет Take Profit для нисходящего тренда"""
        # TP на 3-5% ниже входа
        return entry_price * 0.96
    
    def _calculate_downtrend_sl(self, df: pd.DataFrame, entry_price: float) -> float:
        """Расчет Stop Loss для нисходящего тренда"""
        # SL на 2-3% выше входа
        return entry_price * 1.03
    
    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Поиск уровней поддержки и сопротивления"""
        
        # Используем последние 50 свечей
        recent_data = df.tail(50)
        
        # Простой метод: минимумы и максимумы
        support = recent_data['low'].min()
        resistance = recent_data['high'].max()
        
        return support, resistance

def test_trend_analyzer():
    """Тестирование анализатора трендов"""
    
    # Создаем тестовые данные
    dates = pd.date_range('2025-01-01', periods=200, freq='1H')
    
    # Симулируем восходящий тренд с коррекциями
    trend_data = []
    base_price = 100
    
    for i in range(200):
        if i < 50:  # Восходящий тренд
            price = base_price + i * 0.5 + np.random.normal(0, 1)
        elif i < 80:  # Коррекция
            price = base_price + 50 * 0.5 - (i - 50) * 0.3 + np.random.normal(0, 1)
        elif i < 130:  # Продолжение тренда
            price = base_price + 50 * 0.5 - 30 * 0.3 + (i - 80) * 0.4 + np.random.normal(0, 1)
        else:  # Флет
            price = base_price + 50 * 0.5 - 30 * 0.3 + 50 * 0.4 + np.random.normal(0, 0.5)
        
        trend_data.append({
            'open': price + np.random.normal(0, 0.5),
            'high': price + abs(np.random.normal(0, 1)),
            'low': price - abs(np.random.normal(0, 1)),
            'close': price,
            'volume': np.random.uniform(1000, 10000)
        })
    
    df = pd.DataFrame(trend_data, index=dates)
    
    # Тестируем анализатор
    analyzer = TrendAnalyzer()
    result = analyzer.analyze_trend(df)
    
    print("🎯 Результаты анализа тренда:")
    print(f"Направление тренда: {result['trend_direction']}")
    print(f"Текущая фаза: {result['current_phase']}")
    print(f"Сигнал: {result['signal']['type']}")
    print(f"Причина: {result['signal']['reason']}")
    
    if 'entry_price' in result['signal']:
        print(f"Цена входа: {result['signal']['entry_price']:.4f}")
        print(f"Take Profit: {result['signal']['take_profit']:.4f}")
        print(f"Stop Loss: {result['signal']['stop_loss']:.4f}")
    
    return result

if __name__ == "__main__":
    test_trend_analyzer()




