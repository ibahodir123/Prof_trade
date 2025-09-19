"""
EMA Pattern Analyzer - Анализатор паттернов EMA
Анализирует тренды, импульсы и коррекции на основе EMA 20, 50, 100
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class EMAPatternAnalyzer:
    """Анализатор паттернов EMA для торговых сигналов"""
    
    def __init__(self):
        self.ema_periods = [20, 50, 100]
        self.logger = logger
        
    def calculate_ema(self, prices: List[float], period: int) -> List[float]:
        """Вычисление EMA для заданного периода"""
        if len(prices) < period:
            return [np.nan] * len(prices)
            
        ema_values = []
        multiplier = 2 / (period + 1)
        
        # Первое значение - простое среднее
        ema_values.append(sum(prices[:period]) / period)
        
        # Остальные значения
        for i in range(period, len(prices)):
            ema = (prices[i] * multiplier) + (ema_values[-1] * (1 - multiplier))
            ema_values.append(ema)
            
        return [np.nan] * (period - 1) + ema_values
    
    def detect_trend(self, ema20: float, ema50: float, ema100: float, price: float) -> str:
        """Определение тренда на основе EMA"""
        if pd.isna(ema20) or pd.isna(ema50) or pd.isna(ema100):
            return "НЕИЗВЕСТНО"
            
        # Восходящий тренд: EMA20 > EMA50 > EMA100 и цена выше EMA20
        if ema20 > ema50 > ema100 and price > ema20:
            return "ВОСХОДЯЩИЙ"
        # Нисходящий тренд: EMA20 < EMA50 < EMA100 и цена ниже EMA20
        elif ema20 < ema50 < ema100 and price < ema20:
            return "НИСХОДЯЩИЙ"
        # Боковой тренд
        else:
            return "БОКОВОЙ"
    
    def analyze_coin(self, symbol: str, ohlcv_data: List[List]) -> Dict:
        """Полный анализ монеты на основе EMA паттернов"""
        try:
            if len(ohlcv_data) < 100:
                return {"error": "Недостаточно данных для анализа"}
            
            # Извлекаем цены закрытия
            prices = [float(candle[4]) for candle in ohlcv_data]
            
            # Вычисляем EMA
            ema20 = self.calculate_ema(prices, 20)
            ema50 = self.calculate_ema(prices, 50)
            ema100 = self.calculate_ema(prices, 100)
            
            # Текущие значения
            current_price = prices[-1]
            current_ema20 = ema20[-1]
            current_ema50 = ema50[-1]
            current_ema100 = ema100[-1]
            
            # Анализ тренда
            trend = self.detect_trend(current_ema20, current_ema50, current_ema100, current_price)
            
            # Генерируем простой сигнал
            signal_type = "ОЖИДАНИЕ"
            if trend == "ВОСХОДЯЩИЙ":
                signal_type = "LONG"
            elif trend == "НИСХОДЯЩИЙ":
                signal_type = "SHORT"
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "ema20": current_ema20,
                "ema50": current_ema50,
                "ema100": current_ema100,
                "trend": trend,
                "signal_type": signal_type,
                "confidence": 70 if trend != "БОКОВОЙ" else 30
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа EMA для {symbol}: {e}")
            return {"error": str(e)}