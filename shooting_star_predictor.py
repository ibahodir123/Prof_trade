#!/usr/bin/env python3
"""
Предиктор стреляющих звезд
Анализирует монеты и предсказывает, когда они "выстрелят" (резко вырастут)
"""

import pandas as pd
import numpy as np
import ccxt
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ShootingStarPredictor:
    def __init__(self):
        self.exchange = ccxt.binance({
            'sandbox': False,
            'enableRateLimit': True,
        })
        
    def get_binance_data(self, symbol: str, timeframe: str = '1h', limit: int = 100) -> pd.DataFrame:
        """Получает данные с Binance"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Конвертируем в числовые типы
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna()
        except Exception as e:
            logger.error(f"Ошибка получения данных {symbol}: {e}")
            return None
    
    def analyze_shooting_potential(self, symbol: str) -> Dict[str, Any]:
        """Анализирует потенциал монеты для "выстрела" """
        try:
            df = self.get_binance_data(symbol, '1h', 100)
            if df is None or len(df) < 50:
                return None
            
            current_price = df['close'].iloc[-1]
            
            # Расчет EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            
            ema20 = df['ema_20'].iloc[-1]
            ema50 = df['ema_50'].iloc[-1]
            
            # 1. Анализ консолидации (боковое движение)
            recent_high = df['high'].iloc[-10:].max()
            recent_low = df['low'].iloc[-10:].min()
            consolidation_range = (recent_high - recent_low) / current_price
            
            # 2. Анализ объема
            avg_volume = df['volume'].iloc[-20:].mean()
            recent_volume = df['volume'].iloc[-3:].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # 3. Анализ волатильности (ИСПРАВЛЕНО - защита от деления на ноль)
            price_mean = df['close'].iloc[-10:].mean()
            volatility = df['close'].iloc[-10:].std() / price_mean if price_mean > 0 else 0.0
            
            # 4. Анализ EMA сближения
            ema_distance = abs(ema20 - ema50) / current_price
            
            # 5. Анализ поддержки (ИСПРАВЛЕНО - защита от деления на ноль)
            range_size = recent_high - recent_low
            support_strength = (current_price - recent_low) / range_size if range_size > 0 else 0.5
            
            # 6. Анализ тренда
            price_change_5 = (current_price - df['close'].iloc[-5]) / df['close'].iloc[-5]
            price_change_10 = (current_price - df['close'].iloc[-10]) / df['close'].iloc[-10]
            
            # Рассчитываем вероятность "выстрела"
            probability = 0.0
            conditions = []
            
            # Идеальные условия для выстрела
            
            # 1. Узкая консолидация (цена сжалась в узком диапазоне)
            if consolidation_range < 0.05:  # Узкая консолидация
                probability += 0.2
                conditions.append("✅ Узкая консолидация")
            else:
                conditions.append(f"❌ Широкая консолидация ({consolidation_range:.3f})")
            
            # 2. ИСПРАВЛЕНО - Умеренный объем (накопление перед выстрелом)
            if 0.5 <= volume_ratio <= 1.5:  # Умеренный объем (накопление)
                probability += 0.2
                conditions.append("✅ Умеренный объем (накопление)")
            else:
                conditions.append(f"❌ Экстремальный объем ({volume_ratio:.2f})")
            
            # 3. Низкая волатильность (сжатие перед выстрелом)
            if volatility < 0.03:  # Низкая волатильность
                probability += 0.2
                conditions.append("✅ Низкая волатильность")
            else:
                conditions.append(f"❌ Высокая волатильность ({volatility:.3f})")
            
            # 4. EMA сближены (готовность к движению)
            if ema_distance < 0.02:  # EMA близко
                probability += 0.2
                conditions.append("✅ EMA сближены")
            else:
                conditions.append(f"❌ EMA далеко ({ema_distance:.3f})")
            
            # 5. ИСПРАВЛЕНО - Близко к поддержке (хорошо для отскока)
            if support_strength < 0.7:  # Близко к поддержке (хорошо для отскока)
                probability += 0.2
                conditions.append("✅ Близко к поддержке")
            else:
                conditions.append(f"❌ Далеко от поддержки ({support_strength:.2f})")
            
            # Дополнительные факторы
            if price_change_5 > -0.02:  # Не падает сильно
                probability += 0.1
                conditions.append("✅ Стабильная цена")
            
            if current_price > ema20 > ema50:  # Восходящий тренд
                probability += 0.1
                conditions.append("✅ Восходящий тренд")
            
            # Ограничиваем вероятность
            probability = min(1.0, probability)
            
            return {
                'symbol': symbol,
                'current_price': current_price,
                'probability': probability,
                'conditions': conditions,
                'consolidation_range': consolidation_range,
                'volume_ratio': volume_ratio,
                'volatility': volatility,
                'ema_distance': ema_distance,
                'support_strength': support_strength,
                'predicted_change': f"+{probability*50:.0f}%"
            }
            
        except Exception as e:
            logger.error(f"Ошибка анализа {symbol}: {e}")
            return None
    
    def find_shooting_stars(self, symbols: List[str], min_probability: float = 0.4) -> List[Dict[str, Any]]:
        """Находит монеты с высоким потенциалом выстрела"""
        shooting_stars = []
        
        # Дедуплицируем символы (убираем дубликаты)
        unique_symbols = []
        seen = set()
        for symbol in symbols:
            # Нормализуем символ (убираем :USDT суффикс)
            normalized = symbol.replace(':USDT', '')
            if normalized not in seen:
                unique_symbols.append(symbol)
                seen.add(normalized)
        
        logger.info(f"📊 Анализирую {len(unique_symbols)} уникальных символов (из {len(symbols)} исходных)")
        
        for symbol in unique_symbols:
            try:
                result = self.analyze_shooting_potential(symbol)
                if result and result['probability'] >= min_probability:
                    shooting_stars.append(result)
                    logger.info(f"🚀 {symbol}: вероятность выстрела {result['probability']:.2f}")
            except Exception as e:
                logger.error(f"Ошибка анализа {symbol}: {e}")
                continue
        
        # Сортируем по вероятности
        shooting_stars.sort(key=lambda x: x['probability'], reverse=True)
        return shooting_stars
    
    def get_available_pairs(self) -> List[str]:
        """Получает список доступных USDT пар"""
        try:
            markets = self.exchange.load_markets()
            usdt_pairs = []
            for symbol, market in markets.items():
                if market['quote'] == 'USDT' and market['active']:
                    usdt_pairs.append(symbol)
            return sorted(usdt_pairs)
        except Exception as e:
            logger.error(f"Ошибка получения пар: {e}")
            return []

# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = ShootingStarPredictor()
    
    # Получаем список монет
    symbols = predictor.get_available_pairs()[:20]  # Первые 20 монет
    
    # Ищем стреляющие звезды
    shooting_stars = predictor.find_shooting_stars(symbols)
    
    print(f"\n🚀 Найдено {len(shooting_stars)} потенциальных стреляющих звезд:")
    for star in shooting_stars:
        print(f"\n{star['symbol']} - Вероятность: {star['probability']:.2f}")
        print(f"Цена: ${star['current_price']:.6f}")
        print("Условия:")
        for condition in star['conditions']:
            print(f"  {condition}")