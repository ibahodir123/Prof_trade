#!/usr/bin/env python3
"""
Продвинутый анализатор EMA паттернов
Анализирует EMA 20, 50, 100 с учетом скоростей, расстояний и углов трендов
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class AdvancedEMAAnalyzer:
    def __init__(self):
        self.ema_periods = [20, 50, 100]

    def calculate_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Вычисление всех EMA признаков для ML"""
        df = df.copy()

        # Расчет EMA
        for period in self.ema_periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Скорости EMA
        for period in self.ema_periods:
            df[f'ema{period}_speed'] = df[f'ema_{period}'].pct_change()

        # Скорости цены относительно EMA
        for period in self.ema_periods:
            df[f'price_speed_vs_ema{period}'] = (df['close'] / df[f'ema_{period}']).pct_change()

        # Расстояния между EMA линиями
        df['ema20_to_ema50'] = df['ema_20'] - df['ema_50']
        df['ema50_to_ema100'] = df['ema_50'] - df['ema_100']
        df['ema20_to_ema100'] = df['ema_20'] - df['ema_100']

        # Расстояния от цены до EMA
        for period in self.ema_periods:
            df[f'price_to_ema{period}'] = df['close'] - df[f'ema_{period}']

        # Угол тренда
        df['trend_angle'] = np.arctan(df['ema_20'].diff() / df['ema_20']) * 180 / np.pi

        # Тип тренда
        df['trend_type'] = self._determine_trend_type(df)

        # Фаза рынка
        df['market_phase'] = self._determine_market_phase(df)

        return df

    def _determine_trend_type(self, df: pd.DataFrame) -> int:
        """Определение типа тренда: 0-нисходящий, 1-боковой, 2-восходящий"""
        if len(df) < 10:
            return 1

        ema20_trend = df['ema_20'].iloc[-1] - df['ema_20'].iloc[-10]
        ema50_trend = df['ema_50'].iloc[-1] - df['ema_50'].iloc[-10]

        if ema20_trend > 0.02 and ema50_trend > 0.01:
            return 2  # Восходящий
        elif ema20_trend < -0.02 and ema50_trend < -0.01:
            return 0  # Нисходящий
        else:
            return 1  # Боковой

    def _determine_market_phase(self, df: pd.DataFrame) -> int:
        """Определение фазы рынка: 0-коррекция, 1-импульс"""
        if len(df) < 5:
            return 1

        price_change = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]
        return 1 if abs(price_change) > 0.01 else 0

    def analyze_coin(self, symbol: str, ohlcv_data: List) -> Dict[str, Any]:
        """Анализ монеты с продвинутой EMA логикой"""
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.calculate_ema_features(df)

            current_price = df['close'].iloc[-1]
            trend_type = df['trend_type'].iloc[-1]
            market_phase = df['market_phase'].iloc[-1]

            # Генерация сигнала
            signal = self._generate_signal(df, trend_type, market_phase)

            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal,
                'trend_type': trend_type,
                'market_phase': market_phase,
                'df': df
            }

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return {'symbol': symbol, 'signal': 'ОШИБКА', 'error': str(e)}

    def _generate_signal(self, df: pd.DataFrame, trend_type: int, market_phase: int) -> str:
        """Генерация торгового сигнала"""
        if len(df) < 20:
            return 'ОЖИДАНИЕ'

        # Простая логика для демонстрации
        if trend_type == 2:  # Восходящий тренд
            return 'LONG'
        elif trend_type == 0:  # Нисходящий тренд
            return 'ОЖИДАНИЕ'
        else:  # Боковой тренд
            return 'LONG' if market_phase == 0 else 'ОЖИДАНИЕ'