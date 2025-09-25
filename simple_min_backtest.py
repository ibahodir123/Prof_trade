#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Простой бэктест для тестирования детектора минимумов
Обучение: январь 2025
Тест: февраль 2025
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import pickle
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

class SimpleMinimumBacktest:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.model = None
        self.feature_names = None
        self.feature_weights = None
        
        # Параметры торговли
        self.initial_balance = 1000.0
        self.current_balance = self.initial_balance
        self.position_size_percent = 0.2  # 20% от баланса на позицию
        self.max_positions = 2
        
        self.trades = []
        self.positions = {}
        
        # Параметры TP/SL
        self.take_profit_percent = 5.0  # 5% прибыли
        self.stop_loss_percent = 2.0    # 2% убытка
        
    def load_model(self, model_filename: str = "minimum_detector_model.pkl") -> bool:
        """Загрузка обученной модели"""
        try:
            print(f"🧠 Загружаю модель {model_filename}")
            
            with open(model_filename, 'rb') as f:
                self.model = pickle.load(f)
            
            # Загружаем метаданные
            metadata_filename = model_filename.replace('.pkl', '_metadata.json')
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.feature_names = metadata['feature_names']
            self.feature_weights = metadata['feature_weights']
            
            print(f"✅ Модель загружена")
            print(f"📊 Признаки: {', '.join(self.feature_names)}")
            print(f"⚖️ Веса загружены")
            
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def get_test_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение тестовых данных"""
        try:
            print(f"📊 Загружаю тестовые данные {symbol}")
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
            
            print(f"   ✅ Загружено {len(df)} свечей")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка загрузки данных {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_4_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет 4 критериев (точно как в детекторе)"""
        try:
            # Добавляем EMA20
            df['ema20'] = df['close'].ewm(span=20).mean()
            
            # 1. Скорость цены
            df['price_velocity'] = df['close'].pct_change() * 100
            
            # 2. Скорость EMA20
            df['ema20_velocity'] = df['ema20'].pct_change() * 100
            
            # 3. Угол EMA20
            df['ema20_angle'] = ((df['ema20'] / df['ema20'].shift(10)) - 1) * 100
            
            # 4. Расстояние цена-EMA20
            df['distance_to_ema20'] = ((df['close'] - df['ema20']) / df['ema20']) * 100
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка расчета критериев: {e}")
            return df
    
    def predict_minimum(self, criteria_values: dict) -> Dict[str, Any]:
        """Предсказание минимума с помощью модели"""
        try:
            # Подготавливаем вектор признаков
            feature_vector = []
            for name in self.feature_names:
                value = criteria_values.get(name, 0.0)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Предсказание
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            return {
                'is_minimum': bool(prediction),
                'probability': float(probabilities[1]),
                'criteria': criteria_values
            }
            
        except Exception as e:
            return {'error': f'Ошибка предсказания: {e}'}
    
    def analyze_signal(self, symbol: str, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Анализ сигнала в текущий момент"""
        try:
            # Нужно минимум данных для расчета критериев
            if current_idx < 20:
                return {'signal': 'WAIT', 'reason': 'Недостаточно данных'}
            
            # Берем данные до текущего момента
            historical_data = df.iloc[:current_idx + 1].copy()
            
            # Рассчитываем критерии
            historical_data = self.calculate_4_criteria(historical_data)
            
            # Текущие значения критериев
            current_data = historical_data.iloc[-1]
            
            criteria = {
                'price_velocity': current_data['price_velocity'],
                'ema20_velocity': current_data['ema20_velocity'],
                'ema20_angle': current_data['ema20_angle'], 
                'distance_to_ema20': current_data['distance_to_ema20']
            }
            
            # Проверяем корректность данных
            if any(pd.isna(value) for value in criteria.values()):
                return {'signal': 'WAIT', 'reason': 'Некорректные данные'}
            
            # Предсказание минимума
            prediction = self.predict_minimum(criteria)
            
            if 'error' in prediction:
                return {'signal': 'WAIT', 'reason': prediction['error']}
            
            # Генерируем сигнал
            signal = 'WAIT'
            confidence = prediction['probability']
            
            # Условия для LONG сигнала:
            # 1. Модель предсказывает минимум
            # 2. Цена ниже EMA20 (падение)
            # 3. Высокая вероятность (>60%)
            if (prediction['is_minimum'] and 
                criteria['distance_to_ema20'] < -1.0 and  # Цена ниже EMA20 минимум на 1%
                confidence > 0.6):
                signal = 'LONG'
            
            return {
                'signal': signal,
                'confidence': confidence,
                'price': float(current_data['close']),
                'criteria': criteria,
                'prediction': prediction
            }
            
        except Exception as e:
            return {'signal': 'WAIT', 'reason': f'Ошибка анализа: {e}'}
    
    def run_backtest(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Запуск бэктеста"""
        print(f"\\n🧪 ЗАПУСК БЭКТЕСТА")
        print(f"📅 Период: {start_date.date()} - {end_date.date()}")
        print("=" * 50)
        
        # Сброс состояния
        self.current_balance = self.initial_balance
        self.trades = []
        self.positions = {}
        
        # Статистика сигналов
        signals_stats = {
            'total_analyzed': 0,
            'long_signals': 0,
            'wait_signals': 0,
            'positions_opened': 0,
            'positions_closed': 0
        }
        
        # Загружаем данные для всех символов
        all_data = {}
        for symbol in self.symbols:
            df = self.get_test_data(symbol, start_date, end_date)
            if not df.empty:
                all_data[symbol] = df
            time.sleep(1)
        
        if not all_data:
            return {'error': 'Нет данных для тестирования'}
        
        # Находим общие временные точки
        all_timestamps = set()
        for df in all_data.values():
            all_timestamps.update(df.index)
        timestamps = sorted(list(all_timestamps))
        
        print(f"📊 Анализируем {len(timestamps)} временных точек")
        
        # Основной цикл бэктеста
        for i, timestamp in enumerate(timestamps):
            # Показываем прогресс
            if i % 100 == 0:
                progress = (i / len(timestamps)) * 100
                print(f"   Прогресс: {progress:.1f}% | Баланс: ${self.current_balance:.2f} | Позиций: {len(self.positions)}")
            
            for symbol, df in all_data.items():
                if timestamp not in df.index:
                    continue
                
                current_price = float(df.loc[timestamp, 'close'])
                current_idx = df.index.get_loc(timestamp)
                
                # Проверяем закрытие позиций
                if symbol in self.positions:
                    position = self.positions[symbol]
                    should_close = False
                    close_reason = ""
                    
                    # Проверяем TP/SL
                    if current_price >= position['take_profit']:
                        should_close = True
                        close_reason = "Take Profit"
                    elif current_price <= position['stop_loss']:
                        should_close = True
                        close_reason = "Stop Loss"
                    
                    if should_close:
                        # Закрываем позицию
                        pnl = (current_price - position['entry_price']) * position['size']
                        self.current_balance += pnl
                        
                        pnl_percent = (pnl / (position['entry_price'] * position['size'])) * 100
                        
                        trade = {
                            'symbol': symbol,
                            'entry_time': position['entry_time'],
                            'exit_time': timestamp,
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'size': position['size'],
                            'pnl': pnl,
                            'pnl_percent': pnl_percent,
                            'close_reason': close_reason
                        }
                        
                        self.trades.append(trade)
                        del self.positions[symbol]
                        signals_stats['positions_closed'] += 1
                
                # Ищем новые сигналы
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    signal_data = self.analyze_signal(symbol, df, current_idx)
                    signals_stats['total_analyzed'] += 1
                    
                    if signal_data['signal'] == 'LONG':
                        signals_stats['long_signals'] += 1
                        
                        # Открываем позицию
                        position_value = self.current_balance * self.position_size_percent
                        size = position_value / current_price
                        
                        take_profit = current_price * (1 + self.take_profit_percent / 100)
                        stop_loss = current_price * (1 - self.stop_loss_percent / 100)
                        
                        self.positions[symbol] = {
                            'entry_time': timestamp,
                            'entry_price': current_price,
                            'size': size,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'confidence': signal_data['confidence']
                        }
                        
                        signals_stats['positions_opened'] += 1
                        
                    else:
                        signals_stats['wait_signals'] += 1
        
        # Закрываем оставшиеся позиции
        for symbol in list(self.positions.keys()):
            if symbol in all_data:
                last_price = float(all_data[symbol].iloc[-1]['close'])
                position = self.positions[symbol]
                
                pnl = (last_price - position['entry_price']) * position['size']
                self.current_balance += pnl
                pnl_percent = (pnl / (position['entry_price'] * position['size'])) * 100
                
                trade = {
                    'symbol': symbol,
                    'entry_time': position['entry_time'],
                    'exit_time': timestamps[-1],
                    'entry_price': position['entry_price'],
                    'exit_price': last_price,
                    'size': position['size'],
                    'pnl': pnl,
                    'pnl_percent': pnl_percent,
                    'close_reason': 'End of period'
                }
                
                self.trades.append(trade)
        
        # Подготавливаем результаты
        return self.calculate_results(signals_stats)
    
    def calculate_results(self, signals_stats: dict) -> Dict[str, Any]:
        """Расчет итоговых результатов"""
        if not self.trades:
            return {
                'error': 'Сделок не было',
                'signals_stats': signals_stats
            }
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        # Статистика по символам
        symbol_stats = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                symbol_stats[symbol]['wins'] += 1
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': total_return,
            'total_pnl': self.current_balance - self.initial_balance,
            'trades': self.trades,
            'signals_stats': signals_stats,
            'symbol_stats': symbol_stats
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Вывод результатов"""
        if 'error' in results:
            print(f"\\n❌ {results['error']}")
            if 'signals_stats' in results:
                stats = results['signals_stats']
                print(f"\\n📊 Статистика сигналов:")
                print(f"   Проанализировано: {stats['total_analyzed']}")
                print(f"   LONG сигналов: {stats['long_signals']}")
                print(f"   Позиций открыто: {stats['positions_opened']}")
            return
        
        print(f"\\n🎉 РЕЗУЛЬТАТЫ БЭКТЕСТА")
        print("=" * 40)
        print(f"💰 Стартовый баланс: ${results['initial_balance']:,.2f}")
        print(f"💵 Финальный баланс: ${results['final_balance']:,.2f}")
        print(f"📈 Доходность: {results['total_return']:+.2f}%")
        print(f"💸 P&L: ${results['total_pnl']:+,.2f}")
        print()
        print(f"📊 Всего сделок: {results['total_trades']}")
        print(f"✅ Прибыльных: {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"❌ Убыточных: {results['losing_trades']} ({100-results['win_rate']:.1f}%)")
        
        print(f"\\n📊 Статистика сигналов:")
        stats = results['signals_stats']
        print(f"   Проанализировано: {stats['total_analyzed']}")
        print(f"   LONG сигналов: {stats['long_signals']}")
        print(f"   Позиций открыто: {stats['positions_opened']}")
        print(f"   Позиций закрыто: {stats['positions_closed']}")
        
        if results['symbol_stats']:
            print(f"\\n🏆 По символам:")
            for symbol, stats in results['symbol_stats'].items():
                wr = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                print(f"   {symbol}: {stats['trades']} сделок, ${stats['pnl']:+.2f}, WR: {wr:.1f}%")

if __name__ == "__main__":
    backtest = SimpleMinimumBacktest()
    
    print("🧪 ПРОСТОЙ БЭКТЕСТ ДЕТЕКТОРА МИНИМУМОВ")
    print("🎯 Модель обучена на январе, тестируется на феврале")
    print("=" * 55)
    
    # Загружаем модель
    if backtest.load_model():
        # Запускаем бэктест на феврале 2025
        start_date = datetime(2025, 2, 1)
        end_date = datetime(2025, 2, 28, 23, 59, 59)
        
        results = backtest.run_backtest(start_date, end_date)
        backtest.print_results(results)
        
        # Сохраняем результаты
        with open('minimum_backtest_results.json', 'w', encoding='utf-8') as f:
            # Конвертируем datetime для JSON
            results_copy = results.copy()
            if 'trades' in results_copy:
                for trade in results_copy['trades']:
                    if 'entry_time' in trade:
                        trade['entry_time'] = trade['entry_time'].isoformat()
                    if 'exit_time' in trade:
                        trade['exit_time'] = trade['exit_time'].isoformat()
            
            json.dump(results_copy, f, ensure_ascii=False, indent=2)
        
        print(f"\\n💾 Результаты сохранены в minimum_backtest_results.json")
        
    else:
        print("❌ Сначала запустите:")
        print("1. simple_min_detector.py (сбор данных)")
        print("2. weighted_ml_trainer.py (обучение модели)")






