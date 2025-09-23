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

# Константы для анализа трендов
class TrendConstants:
    # Пороги для определения тренда
    UPTREND_EMA20_THRESHOLD = 0.02      # 2% рост EMA 20
    UPTREND_EMA50_THRESHOLD = 0.01      # 1% рост EMA 50
    DOWNTREND_EMA20_THRESHOLD = -0.02   # 2% падение EMA 20
    DOWNTREND_EMA50_THRESHOLD = -0.01   # 1% падение EMA 50
    
    # Пороги для определения фазы рынка
    MARKET_PHASE_THRESHOLD = 0.01       # 1% изменение цены для импульса
    
    # Минимальное количество свечей для анализа
    MIN_CANDLES_TREND = 10
    MIN_CANDLES_PHASE = 5
    
    # Типы трендов
    TREND_DOWN = 0
    TREND_SIDEWAYS = 1
    TREND_UP = 2
    
    # Фазы рынка
    PHASE_CORRECTION = 0
    PHASE_IMPULSE = 1

class AdvancedEMAAnalyzer:
    def __init__(self):
        self.ema_periods = [20, 50, 100]
        self.constants = TrendConstants()

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
        if len(df) < self.constants.MIN_CANDLES_TREND:
            return self.constants.TREND_SIDEWAYS

        ema20_trend = df['ema_20'].iloc[-1] - df['ema_20'].iloc[-self.constants.MIN_CANDLES_TREND]
        ema50_trend = df['ema_50'].iloc[-1] - df['ema_50'].iloc[-self.constants.MIN_CANDLES_TREND]

        if (ema20_trend > self.constants.UPTREND_EMA20_THRESHOLD and 
            ema50_trend > self.constants.UPTREND_EMA50_THRESHOLD):
            return self.constants.TREND_UP  # Восходящий
        elif (ema20_trend < self.constants.DOWNTREND_EMA20_THRESHOLD and 
              ema50_trend < self.constants.DOWNTREND_EMA50_THRESHOLD):
            return self.constants.TREND_DOWN  # Нисходящий
        else:
            return self.constants.TREND_SIDEWAYS  # Боковой

    def _determine_market_phase(self, df: pd.DataFrame) -> int:
        """Определение фазы рынка: 0-коррекция, 1-импульс"""
        if len(df) < self.constants.MIN_CANDLES_PHASE:
            return self.constants.PHASE_IMPULSE

        price_change = (df['close'].iloc[-1] - df['close'].iloc[-self.constants.MIN_CANDLES_PHASE]) / df['close'].iloc[-self.constants.MIN_CANDLES_PHASE]
        return (self.constants.PHASE_IMPULSE if abs(price_change) > self.constants.MARKET_PHASE_THRESHOLD 
                else self.constants.PHASE_CORRECTION)

    def analyze_coin(self, symbol: str, ohlcv_data: List, ml_trainer=None) -> Dict[str, Any]:
        """Анализ монеты с продвинутой EMA логикой и ML"""
        try:
            df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df = self.calculate_ema_features(df)

            current_price = df['close'].iloc[-1]
            trend_type = df['trend_type'].iloc[-1]
            market_phase = df['market_phase'].iloc[-1]

            # Попытка использовать ML для более точного анализа
            signal = '⚪ ОЖИДАНИЕ'
            confidence = 0.0
            ml_entry_prob = 0.0
            ml_exit_prob = 0.0
            
            if ml_trainer:
                try:
                    features = self.extract_ml_features(df)
                    ml_entry_prob, ml_exit_prob = ml_trainer.predict_entry_exit(features)
                    
                    # Используем ML вероятности для принятия решения
                    if ml_entry_prob > 0.7:  # Высокая вероятность входа
                        signal = 'LONG'
                        confidence = ml_entry_prob * 100
                    elif ml_entry_prob > 0.5:  # Средняя вероятность
                        signal = 'LONG'
                        confidence = ml_entry_prob * 100
                    else:
                        signal = '⚪ ОЖИДАНИЕ'
                        confidence = (1 - ml_entry_prob) * 100
                        
                except Exception as ml_error:
                    logger.warning(f"Ошибка ML анализа для {symbol}: {ml_error}")
                    # Fallback к простой логике
                    signal = self._generate_signal(df, trend_type, market_phase)
                    confidence = 50.0
            else:
                # Простая логика без ML
                signal = self._generate_signal(df, trend_type, market_phase)
                confidence = 50.0

            # Определяем тренд и фазу для отображения
            trend_name = ['Ниcходящий', 'Боковой', 'Восходящий'][trend_type] if trend_type in [0, 1, 2] else 'Не определен'
            phase_name = ['Коррекция', 'Импульс'][market_phase] if market_phase in [0, 1] else 'Не определена'

            return {
                'symbol': symbol,
                'current_price': current_price,
                'signal': signal,
                'confidence': confidence,
                'trend_type': trend_type,
                'trend_name': trend_name,
                'market_phase': market_phase,
                'phase_name': phase_name,
                'ml_entry_prob': ml_entry_prob,
                'ml_exit_prob': ml_exit_prob,
                'df': df
            }

        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return {'symbol': symbol, 'signal': 'ОШИБКА', 'error': str(e)}

    def extract_ml_features(self, df: pd.DataFrame) -> np.ndarray:
        """Извлечение признаков для ML модели (ровно 10 признаков)"""
        if len(df) < 20:
            return np.zeros(10)
        
        # Берем последние значения
        latest = df.iloc[-1]
        
        # Создаем массив из 10 признаков с проверкой на NaN
        features = np.array([
            float(latest['ema_20']) if pd.notna(latest['ema_20']) else 0.0,                    # 1. EMA 20
            float(latest['ema_50']) if pd.notna(latest['ema_50']) else 0.0,                    # 2. EMA 50
            float(latest['ema_100']) if pd.notna(latest['ema_100']) else 0.0,                  # 3. EMA 100
            float(latest['ema20_speed']) if pd.notna(latest['ema20_speed']) else 0.0,          # 4. Скорость EMA 20
            float(latest['ema50_speed']) if pd.notna(latest['ema50_speed']) else 0.0,          # 5. Скорость EMA 50
            float(latest['price_speed_vs_ema20']) if pd.notna(latest['price_speed_vs_ema20']) else 0.0,  # 6. Скорость цены vs EMA 20
            float(latest['ema20_to_ema50']) if pd.notna(latest['ema20_to_ema50']) else 0.0,    # 7. Расстояние EMA 20-50
            float(latest['price_to_ema20']) if pd.notna(latest['price_to_ema20']) else 0.0,    # 8. Расстояние цена-EMA 20
            float(latest['trend_angle']) if pd.notna(latest['trend_angle']) else 0.0,          # 9. Угол тренда
            float(latest['trend_type']) if pd.notna(latest['trend_type']) else 1.0             # 10. Тип тренда
        ])
        
        # Проверяем, что получили ровно 10 признаков
        if len(features) != 10:
            logger.warning(f"Количество признаков не совпадает: ожидается 10, получено {len(features)}")
            # Если что-то пошло не так, возвращаем нулевой массив
            return np.zeros(10)
        
        return features

    def _generate_signal(self, df: pd.DataFrame, trend_type: int, market_phase: int) -> str:
        """Генерация торгового сигнала"""
        if len(df) < 20:
            return '⚪ ОЖИДАНИЕ'

        # Простая логика для демонстрации
        if trend_type == 2:  # Восходящий тренд
            return 'LONG'
        elif trend_type == 0:  # Нисходящий тренд
            return '⚪ ОЖИДАНИЕ'
        else:  # Боковой тренд
            return 'LONG' if market_phase == 0 else '⚪ ОЖИДАНИЕ'