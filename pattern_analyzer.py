#!/usr/bin/env python3
"""
ДЕТАЛЬНЫЙ АНАЛИЗАТОР ПАТТЕРНОВ ДЛЯ LONG ПОЗИЦИЙ
Анализирует все импульсы и коррекции во всех типах трендов
Создает базу паттернов на основе 9 групп признаков
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import ccxt

logger = logging.getLogger(__name__)

class TrendType(Enum):
    """Типы трендов"""
    DOWNTREND = "нисходящий"
    SIDEWAYS = "боковой" 
    UPTREND = "восходящий"

class MovementType(Enum):
    """Типы движений"""
    IMPULSE_DOWN = "импульс_вниз"
    IMPULSE_UP = "импульс_вверх"
    CORRECTION_UP = "коррекция_вверх"
    CORRECTION_DOWN = "коррекция_вниз"

@dataclass
class PatternPoint:
    """Точка паттерна с 9 признаками"""
    timestamp: str
    trend_type: TrendType
    movement_type: MovementType
    phase: str  # "начало" или "конец"
    price: float
    
    # 9 групп признаков (27 значений)
    velocities: Dict[str, float]  # 4 значения: price, ema20, ema50, ema100
    accelerations: Dict[str, float]  # 4 значения
    velocity_ratios: Dict[str, float]  # 3 значения
    distances: Dict[str, float]  # 3 значения
    distance_changes: Dict[str, float]  # 3 значения
    angles: Dict[str, float]  # 3 значения
    angle_changes: Dict[str, float]  # 3 значения
    ema_relationships: Dict[str, float]  # 3 значения
    synchronizations: Dict[str, float]  # 3 значения
    
    # Результат движения (для машинного обучения)
    price_change_percent: float  # Изменение цены от начала до конца движения
    movement_duration_hours: int  # Длительность движения

@dataclass
class MovementPattern:
    """Полный паттерн движения (начало + конец)"""
    start_point: PatternPoint
    end_point: PatternPoint
    trend_type: TrendType
    movement_type: MovementType
    profit_percent: float  # Прибыль для LONG позиции
    duration_hours: int

class DetailedPatternAnalyzer:
    """Детальный анализатор паттернов"""
    
    def __init__(self):
        self.patterns_database: List[MovementPattern] = []
        self.trend_detection_period = 30  # Уменьшаем период для определения тренда
        self.impulse_threshold = 0.008  # 0.8% минимальное движение для импульса (было 1.5%)
        self.correction_threshold = 0.005  # 0.5% минимальное движение для коррекции (было 0.8%)
        
    def analyze_historical_data(self, symbol: str, days: int = 90) -> bool:
        """Анализ исторических данных для поиска всех паттернов"""
        try:
            # Очищаем базу паттернов для нового символа
            patterns_before = len(self.patterns_database)
            logger.info(f"🔍 Начинаю СТРУКТУРНЫЙ анализ {symbol} с 01.01.2025...")
            print(f"🔍 ДЕБАГ: Паттернов до анализа {symbol}: {patterns_before}")
            
            # Получаем исторические данные с 01.01.2025
            df = self._get_historical_data(symbol, days)
            if df is None or len(df) < 100:
                logger.error(f"❌ Недостаточно данных для {symbol}")
                return False
            
            # Подготавливаем все признаки
            df = self._prepare_all_features(df)
            if df is None:
                return False
            
            logger.info(f"📊 Подготовлено {len(df)} свечей для анализа")
            
            # НОВЫЙ ПОДХОД: Структурный анализ
            self._structural_analysis(df, symbol)
            
            logger.info(f"✅ Найдено {len(self.patterns_database)} паттернов для {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа {symbol}: {e}")
            return False

    def _structural_analysis(self, df: pd.DataFrame, symbol: str):
        """Структурный анализ: сначала экстремумы, потом тренды, потом паттерны"""
        try:
            logger.info("🔍 Ищу все экстремумы...")
            print(f"🔍 ДЕБАГ: Начинаю структурный анализ. Данных: {len(df)} свечей")
            
            # 1. Находим ВСЕ значимые экстремумы
            major_highs, major_lows = self._find_major_extremes(df)
            
            logger.info(f"📊 Найдено: {len(major_highs)} максимумов, {len(major_lows)} минимумов")
            print(f"🔍 ДЕБАГ: Найдено {len(major_highs)} максимумов, {len(major_lows)} минимумов")
            
            # 2. Создаем последовательность экстремумов
            extremes_sequence = self._create_extremes_sequence(major_highs, major_lows)
            
            logger.info(f"📈 Создана последовательность из {len(extremes_sequence)} экстремумов")
            
            # 3. Размечаем тренды между экстремумами
            trends = self._identify_trends_from_extremes(extremes_sequence, df)
            
            logger.info(f"📊 Идентифицировано {len(trends)} трендов")
            
            # 4. Анализируем каждый тренд на предмет паттернов
            for trend in trends:
                self._analyze_trend_patterns(trend, df, symbol)
                
        except Exception as e:
            logger.error(f"❌ Ошибка структурного анализа: {e}")

    def _find_major_extremes(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Поиск значимых экстремумов (не мелких колебаний)"""
        try:
            major_highs = []
            major_lows = []
            
            prices = df['close'].values
            min_significance = 0.01  # 1% минимальное отличие для значимого экстремума (было 2%)
            lookback = 12  # 12 часов lookback для фильтрации (было 24)
            
            for i in range(lookback, len(prices) - lookback):
                current_price = prices[i]
                
                # Проверяем локальный максимум
                is_major_high = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i:
                        # Должен быть выше на минимум 2%
                        if prices[j] >= current_price * (1 - min_significance):
                            is_major_high = False
                            break
                
                if is_major_high:
                    major_highs.append({
                        'idx': i,
                        'price': current_price,
                        'timestamp': df.index[i],
                        'type': 'high'
                    })
                
                # Проверяем локальный минимум
                is_major_low = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i:
                        # Должен быть ниже на минимум 2%
                        if prices[j] <= current_price * (1 + min_significance):
                            is_major_low = False
                            break
                
                if is_major_low:
                    major_lows.append({
                        'idx': i,
                        'price': current_price,
                        'timestamp': df.index[i],
                        'type': 'low'
                    })
            
            return major_highs, major_lows
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска экстремумов: {e}")
            return [], []

    def _create_extremes_sequence(self, highs: List[Dict], lows: List[Dict]) -> List[Dict]:
        """Создание хронологической последовательности экстремумов"""
        try:
            # Объединяем и сортируем по времени
            all_extremes = highs + lows
            all_extremes.sort(key=lambda x: x['idx'])
            
            # Фильтруем: убираем подряд идущие однотипные экстремумы
            filtered_sequence = []
            last_type = None
            
            for extreme in all_extremes:
                if extreme['type'] != last_type:
                    filtered_sequence.append(extreme)
                    last_type = extreme['type']
                else:
                    # Если тот же тип, берем более экстремальный
                    if extreme['type'] == 'high' and extreme['price'] > filtered_sequence[-1]['price']:
                        filtered_sequence[-1] = extreme
                    elif extreme['type'] == 'low' and extreme['price'] < filtered_sequence[-1]['price']:
                        filtered_sequence[-1] = extreme
            
            return filtered_sequence
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания последовательности: {e}")
            return []

    def _identify_trends_from_extremes(self, extremes: List[Dict], df: pd.DataFrame) -> List[Dict]:
        """Идентификация трендов на основе последовательности экстремумов"""
        try:
            trends = []
            
            for i in range(len(extremes) - 1):
                current_extreme = extremes[i]
                next_extreme = extremes[i + 1]
                
                # Определяем тип тренда
                if current_extreme['type'] == 'high' and next_extreme['type'] == 'low':
                    trend_type = TrendType.DOWNTREND
                elif current_extreme['type'] == 'low' and next_extreme['type'] == 'high':
                    trend_type = TrendType.UPTREND
                else:
                    continue  # Пропускаем некорректные последовательности
                
                # Вычисляем изменение цены
                price_change = (next_extreme['price'] - current_extreme['price']) / current_extreme['price'] * 100
                
                trend = {
                    'start_idx': current_extreme['idx'],
                    'end_idx': next_extreme['idx'],
                    'start_price': current_extreme['price'],
                    'end_price': next_extreme['price'],
                    'trend_type': trend_type,
                    'price_change_percent': price_change,
                    'duration_hours': next_extreme['idx'] - current_extreme['idx']
                }
                
                trends.append(trend)
            
            # Логируем статистику трендов
            uptrends = sum(1 for t in trends if t['trend_type'] == TrendType.UPTREND)
            downtrends = sum(1 for t in trends if t['trend_type'] == TrendType.DOWNTREND)
            
            logger.info(f"📊 Тренды: {uptrends} восходящих, {downtrends} нисходящих")
            print(f"🔍 ДЕБАГ: Тренды: {uptrends} восходящих, {downtrends} нисходящих")
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ Ошибка идентификации трендов: {e}")
            return []

    def _analyze_trend_patterns(self, trend: Dict, df: pd.DataFrame, symbol: str):
        """Анализ паттернов внутри конкретного тренда"""
        try:
            start_idx = trend['start_idx']
            end_idx = trend['end_idx']
            trend_type = trend['trend_type']
            
            # Получаем данные тренда
            trend_df = df.iloc[start_idx:end_idx + 1].copy()
            
            if len(trend_df) < 5:  # Слишком короткий тренд
                return
            
            # Ищем импульсы и коррекции внутри тренда
            if trend_type == TrendType.UPTREND:
                self._find_uptrend_patterns(trend_df, start_idx, symbol, trend)
            elif trend_type == TrendType.DOWNTREND:
                self._find_downtrend_patterns(trend_df, start_idx, symbol, trend)
                
        except Exception as e:
            logger.error(f"❌ Ошибка анализа паттернов тренда: {e}")

    def _find_uptrend_patterns(self, trend_df: pd.DataFrame, global_start_idx: int, symbol: str, trend: Dict):
        """Поиск паттернов в восходящем тренде"""
        try:
            # В восходящем тренде ищем:
            # 1. Основной импульс вверх (весь тренд)
            # 2. Коррекции вниз внутри тренда
            
            # Основной импульс
            if abs(trend['price_change_percent']) >= self.impulse_threshold * 100:
                pattern = self._create_movement_pattern(
                    trend_df, 0, len(trend_df) - 1,
                    TrendType.UPTREND, MovementType.IMPULSE_UP,
                    trend['price_change_percent'], symbol
                )
                if pattern:
                    self.patterns_database.append(pattern)
            
            # Ищем внутренние коррекции
            internal_highs, internal_lows = self._find_local_extremes(trend_df)
            
            for i in range(len(internal_highs) - 1):
                for low in internal_lows:
                    if internal_highs[i]['idx'] < low['idx'] < internal_highs[i + 1]['idx']:
                        # Коррекция от высокого к низкому
                        correction_change = (low['price'] - internal_highs[i]['price']) / internal_highs[i]['price'] * 100
                        
                        if abs(correction_change) >= self.correction_threshold * 100:
                            pattern = self._create_movement_pattern(
                                trend_df, internal_highs[i]['idx'], low['idx'],
                                TrendType.UPTREND, MovementType.CORRECTION_DOWN,
                                abs(correction_change), symbol
                            )
                            if pattern:
                                self.patterns_database.append(pattern)
                        break
                        
        except Exception as e:
            logger.error(f"❌ Ошибка поиска паттернов восходящего тренда: {e}")

    def _find_downtrend_patterns(self, trend_df: pd.DataFrame, global_start_idx: int, symbol: str, trend: Dict):
        """Поиск паттернов в нисходящем тренде"""
        try:
            # В нисходящем тренде ищем:
            # 1. Основной импульс вниз (весь тренд)  
            # 2. Коррекции вверх внутри тренда (возможности для LONG)
            
            # Основной импульс вниз
            if abs(trend['price_change_percent']) >= self.impulse_threshold * 100:
                pattern = self._create_movement_pattern(
                    trend_df, 0, len(trend_df) - 1,
                    TrendType.DOWNTREND, MovementType.IMPULSE_DOWN,
                    abs(trend['price_change_percent']), symbol
                )
                if pattern:
                    self.patterns_database.append(pattern)
            
            # Ищем внутренние коррекции вверх (LONG возможности)
            internal_highs, internal_lows = self._find_local_extremes(trend_df)
            
            for i in range(len(internal_lows) - 1):
                for high in internal_highs:
                    if internal_lows[i]['idx'] < high['idx'] < internal_lows[i + 1]['idx']:
                        # Коррекция от низкого к высокому (LONG возможность)
                        correction_profit = (high['price'] - internal_lows[i]['price']) / internal_lows[i]['price'] * 100
                        
                        if correction_profit >= self.correction_threshold * 100:
                            pattern = self._create_movement_pattern(
                                trend_df, internal_lows[i]['idx'], high['idx'],
                                TrendType.DOWNTREND, MovementType.CORRECTION_UP,
                                correction_profit, symbol
                            )
                            if pattern:
                                self.patterns_database.append(pattern)
                        break
                        
        except Exception as e:
            logger.error(f"❌ Ошибка поиска паттернов нисходящего тренда: {e}")

    def _classify_trend(self, df: pd.DataFrame, current_idx: int) -> TrendType:
        """Классификация типа тренда на основе 9 признаков"""
        try:
            # Проверяем границы индекса
            if current_idx < 0 or current_idx >= len(df):
                return TrendType.SIDEWAYS
            
            # Берем текущие значения признаков
            current_row = df.iloc[current_idx]
            
            # Анализируем EMA направления
            ema20_angle = current_row['ema20_angle']
            ema50_angle = current_row['ema50_angle'] 
            ema100_angle = current_row['ema100_angle']
            
            # Анализируем скорости
            price_velocity = current_row['price_velocity']
            ema20_velocity = current_row['ema20_velocity']
            ema50_velocity = current_row['ema50_velocity']
            
            # Анализируем синхронизацию
            sync_20 = current_row['price_ema20_sync']
            sync_50 = current_row['price_ema50_sync']
            
            # Критерии для восходящего тренда (смягченные)
            uptrend_signals = 0
            if ema20_angle > 5 or ema50_angle > 3: uptrend_signals += 1  # Снижены пороги
            if price_velocity > 0.005 or ema20_velocity > 0.003: uptrend_signals += 1  # Снижены пороги
            if sync_20 > 0.4 or sync_50 > 0.3: uptrend_signals += 1  # Снижены пороги
            if current_row['ema20_to_ema50'] > 1.001: uptrend_signals += 1
            if current_row['price_to_ema20_distance'] > 0: uptrend_signals += 1  # Цена выше EMA20
            
            # Критерии для нисходящего тренда (смягченные)
            downtrend_signals = 0
            if ema20_angle < -5 or ema50_angle < -3: downtrend_signals += 1  # Снижены пороги
            if price_velocity < -0.005 or ema20_velocity < -0.003: downtrend_signals += 1  # Снижены пороги
            if sync_20 > 0.4 or sync_50 > 0.3: downtrend_signals += 1  # Синхронно вниз
            if current_row['ema20_to_ema50'] < 0.999: downtrend_signals += 1
            if current_row['price_to_ema20_distance'] < 0: downtrend_signals += 1  # Цена ниже EMA20
            
            # Определяем тип тренда (снижены пороги)
            if uptrend_signals >= 3:
                trend_result = TrendType.UPTREND
            elif downtrend_signals >= 3:
                trend_result = TrendType.DOWNTREND
            else:
                trend_result = TrendType.SIDEWAYS
            
            # Логируем каждый 100-й результат для отладки
            if current_idx % 100 == 0:
                logger.info(f"🔍 Период {current_idx}: {trend_result.value} (up:{uptrend_signals}, down:{downtrend_signals})")
            
            return trend_result
                
        except Exception as e:
            logger.error(f"❌ Ошибка классификации тренда: {e}")
            return TrendType.SIDEWAYS

    def _analyze_downtrend_movements(self, df: pd.DataFrame, current_idx: int, symbol: str):
        """Анализ движений в нисходящем тренде"""
        try:
            # В нисходящем тренде ищем:
            # 1. Импульсы вниз (для понимания структуры)
            # 2. Коррекции вверх (возможности для LONG)
            
            window_start = max(0, current_idx - 20)
            window_end = min(len(df), current_idx + 10)
            window_df = df.iloc[window_start:window_end].copy()
            
            # Ищем локальные минимумы и максимумы
            highs, lows = self._find_local_extremes(window_df)
            
            # Анализируем каждое движение
            for i in range(len(lows) - 1):
                current_low = lows[i]
                next_high = None
                next_low = None
                
                # Находим следующий максимум после текущего минимума
                for high in highs:
                    if high['idx'] > current_low['idx']:
                        next_high = high
                        break
                
                # Находим следующий минимум после максимума
                if next_high:
                    for low in lows:
                        if low['idx'] > next_high['idx']:
                            next_low = low
                            break
                
                # Анализируем коррекцию вверх (LONG возможность)
                if next_high:
                    correction_up_profit = (next_high['price'] - current_low['price']) / current_low['price'] * 100
                    
                    if correction_up_profit >= self.correction_threshold * 100:
                        # Сохраняем паттерн коррекции вверх
                        pattern = self._create_movement_pattern(
                            window_df, current_low['idx'], next_high['idx'],
                            TrendType.DOWNTREND, MovementType.CORRECTION_UP,
                            correction_up_profit, symbol
                        )
                        if pattern:
                            self.patterns_database.append(pattern)
                
                # Анализируем импульс вниз (для понимания структуры)
                if next_high and next_low:
                    impulse_down_change = (next_low['price'] - next_high['price']) / next_high['price'] * 100
                    
                    if abs(impulse_down_change) >= self.impulse_threshold * 100:
                        # Сохраняем паттерн импульса вниз
                        pattern = self._create_movement_pattern(
                            window_df, next_high['idx'], next_low['idx'],
                            TrendType.DOWNTREND, MovementType.IMPULSE_DOWN,
                            impulse_down_change, symbol
                        )
                        if pattern:
                            self.patterns_database.append(pattern)
                            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа нисходящего тренда: {e}")

    def _analyze_uptrend_movements(self, df: pd.DataFrame, current_idx: int, symbol: str):
        """Анализ движений в восходящем тренде"""
        try:
            # В восходящем тренде ищем:
            # 1. Импульсы вверх (LONG возможности)
            # 2. Коррекции вниз (точки доливки LONG)
            
            window_start = max(0, current_idx - 20)
            window_end = min(len(df), current_idx + 10)
            window_df = df.iloc[window_start:window_end].copy()
            
            # Ищем локальные минимумы и максимумы
            highs, lows = self._find_local_extremes(window_df)
            
            # Анализируем каждое движение
            for i in range(len(lows) - 1):
                current_low = lows[i]
                next_high = None
                next_low = None
                
                # Находим следующий максимум
                for high in highs:
                    if high['idx'] > current_low['idx']:
                        next_high = high
                        break
                
                # Находим следующий минимум
                if next_high:
                    for low in lows:
                        if low['idx'] > next_high['idx']:
                            next_low = low
                            break
                
                # Анализируем импульс вверх (LONG возможность)
                if next_high:
                    impulse_up_profit = (next_high['price'] - current_low['price']) / current_low['price'] * 100
                    
                    if impulse_up_profit >= self.impulse_threshold * 100:
                        # Сохраняем паттерн импульса вверх
                        pattern = self._create_movement_pattern(
                            window_df, current_low['idx'], next_high['idx'],
                            TrendType.UPTREND, MovementType.IMPULSE_UP,
                            impulse_up_profit, symbol
                        )
                        if pattern:
                            self.patterns_database.append(pattern)
                
                # Анализируем коррекцию вниз (точка доливки LONG)
                if next_high and next_low:
                    correction_down_change = (next_low['price'] - next_high['price']) / next_high['price'] * 100
                    
                    if abs(correction_down_change) >= self.correction_threshold * 100:
                        # Сохраняем конец коррекции как точку входа LONG
                        pattern = self._create_movement_pattern(
                            window_df, next_high['idx'], next_low['idx'],
                            TrendType.UPTREND, MovementType.CORRECTION_DOWN,
                            abs(correction_down_change), symbol
                        )
                        if pattern:
                            self.patterns_database.append(pattern)
                            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа восходящего тренда: {e}")

    def _analyze_sideways_movements(self, df: pd.DataFrame, current_idx: int, symbol: str):
        """Анализ движений в боковом тренде"""
        try:
            # В боковом тренде ищем:
            # 1. Импульсы от поддержки к сопротивлению (LONG возможности)
            # 2. Откаты от сопротивления к поддержке (подготовка к LONG)
            
            window_start = max(0, current_idx - 30)  # Больше окно для бокового тренда
            window_end = min(len(df), current_idx + 10)
            window_df = df.iloc[window_start:window_end].copy()
            
            # Определяем уровни поддержки и сопротивления
            support_level, resistance_level = self._find_support_resistance(window_df)
            
            if support_level is None or resistance_level is None:
                return
            
            # Ищем локальные экстремумы
            highs, lows = self._find_local_extremes(window_df)
            
            # Анализируем движения от поддержки к сопротивлению
            for low in lows:
                if abs(low['price'] - support_level) / support_level < 0.01:  # Близко к поддержке
                    
                    # Ищем ближайший максимум
                    for high in highs:
                        if high['idx'] > low['idx'] and abs(high['price'] - resistance_level) / resistance_level < 0.015:
                            
                            # Движение от поддержки к сопротивлению (LONG возможность)
                            sideways_profit = (high['price'] - low['price']) / low['price'] * 100
                            
                            if sideways_profit >= self.correction_threshold * 100:
                                pattern = self._create_movement_pattern(
                                    window_df, low['idx'], high['idx'],
                                    TrendType.SIDEWAYS, MovementType.IMPULSE_UP,
                                    sideways_profit, symbol
                                )
                                if pattern:
                                    self.patterns_database.append(pattern)
                            break
                            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа бокового тренда: {e}")

    def _find_local_extremes(self, df: pd.DataFrame) -> Tuple[List[Dict], List[Dict]]:
        """Поиск локальных максимумов и минимумов"""
        try:
            highs = []
            lows = []
            
            prices = df['close'].values
            
            # Параметры для поиска экстремумов (уменьшены для большей чувствительности)
            lookback = 2  # Период для сравнения (было 3)
            
            for i in range(lookback, len(prices) - lookback):
                current_price = prices[i]
                
                # Проверяем локальный максимум
                is_high = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and prices[j] >= current_price:
                        is_high = False
                        break
                
                if is_high:
                    highs.append({
                        'idx': i,
                        'price': current_price,
                        'timestamp': df.index[i]
                    })
                
                # Проверяем локальный минимум
                is_low = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and prices[j] <= current_price:
                        is_low = False
                        break
                
                if is_low:
                    lows.append({
                        'idx': i,
                        'price': current_price,
                        'timestamp': df.index[i]
                    })
            
            return highs, lows
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска экстремумов: {e}")
            return [], []

    def _find_support_resistance(self, df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
        """Поиск уровней поддержки и сопротивления в боковом тренде"""
        try:
            # Проверяем, что данных достаточно
            if len(df) < 10:
                return None, None
                
            prices = df['close'].values
            
            # Проверяем, что массив не пустой
            if len(prices) == 0:
                return None, None
            
            # Простой метод: квантили
            support_level = np.percentile(prices, 20)
            resistance_level = np.percentile(prices, 80)
            
            # Проверяем, что уровни достаточно далеко друг от друга
            if (resistance_level - support_level) / support_level > 0.03:  # Минимум 3% канал
                return support_level, resistance_level
            
            return None, None
            
        except Exception as e:
            logger.error(f"❌ Ошибка поиска поддержки/сопротивления: {e}")
            return None, None

    def _create_movement_pattern(self, df: pd.DataFrame, start_idx: int, end_idx: int,
                                trend_type: TrendType, movement_type: MovementType,
                                profit_percent: float, symbol: str) -> Optional[MovementPattern]:
        """Создание паттерна движения с 9 признаками"""
        try:
            start_row = df.iloc[start_idx]
            end_row = df.iloc[end_idx]
            
            # Создаем точку начала движения
            start_point = PatternPoint(
                timestamp=str(start_row.name),
                trend_type=trend_type,
                movement_type=movement_type,
                phase="начало",
                price=start_row['close'],
                velocities={
                    'price': start_row['price_velocity'],
                    'ema20': start_row['ema20_velocity'],
                    'ema50': start_row['ema50_velocity'],
                    'ema100': start_row['ema100_velocity']
                },
                accelerations={
                    'price': start_row['price_acceleration'],
                    'ema20': start_row['ema20_acceleration'],
                    'ema50': start_row['ema50_acceleration'],
                    'ema100': start_row['ema100_acceleration']
                },
                velocity_ratios={
                    'price_ema20': start_row['price_to_ema20_velocity_ratio'],
                    'price_ema50': start_row['price_to_ema50_velocity_ratio'],
                    'price_ema100': start_row['price_to_ema100_velocity_ratio']
                },
                distances={
                    'price_ema20': start_row['price_to_ema20_distance'],
                    'price_ema50': start_row['price_to_ema50_distance'],
                    'price_ema100': start_row['price_to_ema100_distance']
                },
                distance_changes={
                    'price_ema20': start_row['price_to_ema20_distance_change'],
                    'price_ema50': start_row['price_to_ema50_distance_change'],
                    'price_ema100': start_row['price_to_ema100_distance_change']
                },
                angles={
                    'ema20': start_row['ema20_angle'],
                    'ema50': start_row['ema50_angle'],
                    'ema100': start_row['ema100_angle']
                },
                angle_changes={
                    'ema20': start_row['ema20_angle_change'],
                    'ema50': start_row['ema50_angle_change'],
                    'ema100': start_row['ema100_angle_change']
                },
                ema_relationships={
                    'ema20_ema50': start_row['ema20_to_ema50'],
                    'ema20_ema100': start_row['ema20_to_ema100'],
                    'ema50_ema100': start_row['ema50_to_ema100']
                },
                synchronizations={
                    'price_ema20': start_row['price_ema20_sync'],
                    'price_ema50': start_row['price_ema50_sync'],
                    'price_ema100': start_row['price_ema100_sync']
                },
                price_change_percent=0.0,  # Начало движения
                movement_duration_hours=0
            )
            
            # Создаем точку конца движения аналогично
            end_point = PatternPoint(
                timestamp=str(end_row.name),
                trend_type=trend_type,
                movement_type=movement_type,
                phase="конец",
                price=end_row['close'],
                velocities={
                    'price': end_row['price_velocity'],
                    'ema20': end_row['ema20_velocity'],
                    'ema50': end_row['ema50_velocity'],
                    'ema100': end_row['ema100_velocity']
                },
                accelerations={
                    'price': end_row['price_acceleration'],
                    'ema20': end_row['ema20_acceleration'],
                    'ema50': end_row['ema50_acceleration'],
                    'ema100': end_row['ema100_acceleration']
                },
                velocity_ratios={
                    'price_ema20': end_row['price_to_ema20_velocity_ratio'],
                    'price_ema50': end_row['price_to_ema50_velocity_ratio'],
                    'price_ema100': end_row['price_to_ema100_velocity_ratio']
                },
                distances={
                    'price_ema20': end_row['price_to_ema20_distance'],
                    'price_ema50': end_row['price_to_ema50_distance'],
                    'price_ema100': end_row['price_to_ema100_distance']
                },
                distance_changes={
                    'price_ema20': end_row['price_to_ema20_distance_change'],
                    'price_ema50': end_row['price_to_ema50_distance_change'],
                    'price_ema100': end_row['price_to_ema100_distance_change']
                },
                angles={
                    'ema20': end_row['ema20_angle'],
                    'ema50': end_row['ema50_angle'],
                    'ema100': end_row['ema100_angle']
                },
                angle_changes={
                    'ema20': end_row['ema20_angle_change'],
                    'ema50': end_row['ema50_angle_change'],
                    'ema100': end_row['ema100_angle_change']
                },
                ema_relationships={
                    'ema20_ema50': end_row['ema20_to_ema50'],
                    'ema20_ema100': end_row['ema20_to_ema100'],
                    'ema50_ema100': end_row['ema50_to_ema100']
                },
                synchronizations={
                    'price_ema20': end_row['price_ema20_sync'],
                    'price_ema50': end_row['price_ema50_sync'],
                    'price_ema100': end_row['price_ema100_sync']
                },
                price_change_percent=profit_percent,
                movement_duration_hours=end_idx - start_idx
            )
            
            # Создаем полный паттерн движения
            pattern = MovementPattern(
                start_point=start_point,
                end_point=end_point,
                trend_type=trend_type,
                movement_type=movement_type,
                profit_percent=profit_percent,
                duration_hours=end_idx - start_idx
            )
            
            return pattern
            
        except Exception as e:
            logger.error(f"❌ Ошибка создания паттерна: {e}")
            return None

    def _get_historical_data(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        """Получение исторических данных с 01.01.2025"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            
            # Устанавливаем дату начала: 01.01.2025 00:00:00 UTC
            from datetime import datetime
            start_date = datetime(2025, 1, 1, 0, 0, 0)
            since = int(start_date.timestamp() * 1000)
            
            # Получаем данные с 01.01.2025 до сегодня
            # На случай большого объема данных - делаем несколько запросов
            all_ohlcv = []
            current_since = since
            max_per_request = 1000  # Безопасный лимит за запрос
            
            while True:
                ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=max_per_request)
                if not ohlcv or len(ohlcv) == 0:
                    break
                
                all_ohlcv.extend(ohlcv)
                
                # Обновляем время начала для следующего запроса
                current_since = ohlcv[-1][0] + 3600000  # +1 час в миллисекундах
                
                # Проверяем, не достигли ли текущего времени
                if current_since >= exchange.milliseconds():
                    break
                
                # Защита от бесконечного цикла
                if len(all_ohlcv) > 20000:  # Максимум ~2 года данных
                    break
            
            ohlcv = all_ohlcv
            
            if not ohlcv:
                logger.error(f"❌ Нет данных для {symbol}")
                return None
            
            logger.info(f"📊 Загружено {len(ohlcv)} свечей для {symbol} с 01.01.2025")
            
            # Создаем DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Логируем период данных
            logger.info(f"📅 Период данных: {df.index[0]} - {df.index[-1]}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных {symbol}: {e}")
            return None

    def _prepare_all_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
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
            logger.error(f"❌ Ошибка подготовки признаков: {e}")
            return None

    def save_patterns_to_file(self, filename: str = "patterns_database.json"):
        """Сохранение базы паттернов в файл"""
        try:
            import json
            
            # Конвертируем в JSON-совместимый формат
            patterns_data = []
            for pattern in self.patterns_database:
                pattern_dict = {
                    'trend_type': pattern.trend_type.value,
                    'movement_type': pattern.movement_type.value,
                    'profit_percent': pattern.profit_percent,
                    'duration_hours': pattern.duration_hours,
                    'start_point': {
                        'timestamp': pattern.start_point.timestamp,
                        'price': pattern.start_point.price,
                        'velocities': pattern.start_point.velocities,
                        'accelerations': pattern.start_point.accelerations,
                        'velocity_ratios': pattern.start_point.velocity_ratios,
                        'distances': pattern.start_point.distances,
                        'distance_changes': pattern.start_point.distance_changes,
                        'angles': pattern.start_point.angles,
                        'angle_changes': pattern.start_point.angle_changes,
                        'ema_relationships': pattern.start_point.ema_relationships,
                        'synchronizations': pattern.start_point.synchronizations
                    },
                    'end_point': {
                        'timestamp': pattern.end_point.timestamp,
                        'price': pattern.end_point.price,
                        'velocities': pattern.end_point.velocities,
                        'accelerations': pattern.end_point.accelerations,
                        'velocity_ratios': pattern.end_point.velocity_ratios,
                        'distances': pattern.end_point.distances,
                        'distance_changes': pattern.end_point.distance_changes,
                        'angles': pattern.end_point.angles,
                        'angle_changes': pattern.end_point.angle_changes,
                        'ema_relationships': pattern.end_point.ema_relationships,
                        'synchronizations': pattern.end_point.synchronizations
                    }
                }
                patterns_data.append(pattern_dict)
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"✅ Сохранено {len(patterns_data)} паттернов в {filename}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения паттернов: {e}")
            return False

    def get_statistics(self) -> Dict:
        """Получение статистики по найденным паттернам"""
        try:
            stats = {
                'total_patterns': len(self.patterns_database),
                'by_trend_type': {},
                'by_movement_type': {},
                'avg_profit_by_trend': {},
                'avg_duration_by_trend': {}
            }
            
            # Статистика по типам трендов
            for trend_type in TrendType:
                trend_patterns = [p for p in self.patterns_database if p.trend_type == trend_type]
                stats['by_trend_type'][trend_type.value] = len(trend_patterns)
                
                if trend_patterns:
                    profits = [p.profit_percent for p in trend_patterns if p.profit_percent > 0]
                    durations = [p.duration_hours for p in trend_patterns]
                    
                    stats['avg_profit_by_trend'][trend_type.value] = np.mean(profits) if profits else 0
                    stats['avg_duration_by_trend'][trend_type.value] = np.mean(durations) if durations else 0
            
            # Статистика по типам движений
            for movement_type in MovementType:
                movement_patterns = [p for p in self.patterns_database if p.movement_type == movement_type]
                stats['by_movement_type'][movement_type.value] = len(movement_patterns)
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Ошибка расчета статистики: {e}")
            return {}

if __name__ == "__main__":
    # Пример использования
    analyzer = DetailedPatternAnalyzer()
    
    # Анализируем несколько монет
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT']
    
    for symbol in symbols:
        print(f"🔍 Анализирую {symbol}...")
        success = analyzer.analyze_historical_data(symbol, days=365)  # Год данных
        if success:
            print(f"✅ {symbol} проанализирован")
        else:
            print(f"❌ Ошибка анализа {symbol}")
    
    # Получаем статистику
    stats = analyzer.get_statistics()
    print(f"\n📊 Статистика:")
    print(f"Всего паттернов: {stats['total_patterns']}")
    print(f"По трендам: {stats['by_trend_type']}")
    print(f"По движениям: {stats['by_movement_type']}")
    
    # Сохраняем в файл
    analyzer.save_patterns_to_file()
    print("💾 База паттернов сохранена!")
