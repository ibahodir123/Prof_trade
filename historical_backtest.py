#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Правильный бэктест: тестируем модель (обученную на 2020-2024) на данных 2025 года
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import json
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class HistoricalBacktest:
    def __init__(self):
        self.test_start = datetime(2025, 1, 1)
        self.test_end = datetime.now()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        
        self.initial_balance = 1000.0
        self.current_balance = self.initial_balance
        self.position_size_percent = 0.1
        self.max_positions = 3
        
        self.trades = []
        self.positions = {}
        
        self.model = None
        self.feature_names = None
        
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            print("🧠 Загружаю обученную модель...")
            
            with open('historical_model_2020_2024.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('historical_feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            print(f"✅ Модель загружена: {len(self.feature_names)} признаков")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка загрузки модели: {e}")
            return False
    
    def get_test_data(self, symbol: str) -> pd.DataFrame:
        """Получение тестовых данных 2025 года"""
        try:
            print(f"📊 Загружаю данные {symbol} за 2025 год...")
            exchange = ccxt.binance()
            since = int(self.test_start.timestamp() * 1000)
            end_ts = int(self.test_end.timestamp() * 1000)
            
            all_ohlcv = []
            current_since = since
            
            while current_since < end_ts:
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=current_since, limit=1000)
                    if not ohlcv:
                        break
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    if current_since > end_ts:
                        break
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            # Фильтруем по конечной дате
            all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end_ts]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            print(f"✅ Загружено {len(df)} свечей для {symbol}")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def prepare_ml_features(self, df: pd.DataFrame, symbol: str) -> dict:
        """Подготовка 27 ML признаков (точно такая же как в trainer)"""
        try:
            if len(df) < 100:
                return None
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            # Берем последние значения
            latest = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 1. Velocity группа (3 признака)
            price_velocity = (latest['close'] - prev['close']) / prev['close']
            ema20_velocity = (latest['ema_20'] - prev['ema_20']) / prev['ema_20']
            ema50_velocity = (latest['ema_50'] - prev['ema_50']) / prev['ema_50']
            
            # 2. Acceleration группа (3 признака)  
            price_accel = price_velocity - ((prev['close'] - df.iloc[-3]['close']) / df.iloc[-3]['close'])
            ema20_accel = ema20_velocity - ((prev['ema_20'] - df.iloc[-3]['ema_20']) / df.iloc[-3]['ema_20'])
            ema50_accel = ema50_velocity - ((prev['ema_50'] - df.iloc[-3]['ema_50']) / df.iloc[-3]['ema_50'])
            
            # 3. Velocity Ratios группа (3 признака)
            velocity_ratio_20_50 = ema20_velocity / ema50_velocity if ema50_velocity != 0 else 0
            velocity_ratio_price_20 = price_velocity / ema20_velocity if ema20_velocity != 0 else 0
            velocity_ratio_price_50 = price_velocity / ema50_velocity if ema50_velocity != 0 else 0
            
            # 4. Distance to EMAs группа (3 признака)
            distance_to_ema20 = (latest['close'] - latest['ema_20']) / latest['ema_20']
            distance_to_ema50 = (latest['close'] - latest['ema_50']) / latest['ema_50']
            distance_to_ema100 = (latest['close'] - latest['ema_100']) / latest['ema_100']
            
            # 5. Distance Changes группа (3 признака)
            prev_dist_20 = (prev['close'] - prev['ema_20']) / prev['ema_20']
            prev_dist_50 = (prev['close'] - prev['ema_50']) / prev['ema_50']
            prev_dist_100 = (prev['close'] - prev['ema_100']) / prev['ema_100']
            
            distance_change_20 = distance_to_ema20 - prev_dist_20
            distance_change_50 = distance_to_ema50 - prev_dist_50
            distance_change_100 = distance_to_ema100 - prev_dist_100
            
            # 6. EMA Angles группа (3 признака)
            ema20_angle = (latest['ema_20'] - df.iloc[-10]['ema_20']) / df.iloc[-10]['ema_20']
            ema50_angle = (latest['ema_50'] - df.iloc[-10]['ema_50']) / df.iloc[-10]['ema_50']
            ema100_angle = (latest['ema_100'] - df.iloc[-10]['ema_100']) / df.iloc[-10]['ema_100']
            
            # 7. Angle Changes группа (3 признака)
            prev_ema20_angle = (prev['ema_20'] - df.iloc[-11]['ema_20']) / df.iloc[-11]['ema_20']
            prev_ema50_angle = (prev['ema_50'] - df.iloc[-11]['ema_50']) / df.iloc[-11]['ema_50']
            prev_ema100_angle = (prev['ema_100'] - df.iloc[-11]['ema_100']) / df.iloc[-11]['ema_100']
            
            ema20_angle_change = ema20_angle - prev_ema20_angle
            ema50_angle_change = ema50_angle - prev_ema50_angle
            ema100_angle_change = ema100_angle - prev_ema100_angle
            
            # 8. EMA Relationships группа (3 признака)
            ema20_vs_50 = (latest['ema_20'] - latest['ema_50']) / latest['ema_50']
            ema50_vs_100 = (latest['ema_50'] - latest['ema_100']) / latest['ema_100']
            ema20_vs_100 = (latest['ema_20'] - latest['ema_100']) / latest['ema_100']
            
            # 9. Synchronization группа (6 признаков)
            price_ema20_sync = 1 if (price_velocity > 0) == (ema20_velocity > 0) else 0
            price_ema50_sync = 1 if (price_velocity > 0) == (ema50_velocity > 0) else 0
            ema20_ema50_sync = 1 if (ema20_velocity > 0) == (ema50_velocity > 0) else 0
            all_up_sync = 1 if all([price_velocity > 0, ema20_velocity > 0, ema50_velocity > 0]) else 0
            all_down_sync = 1 if all([price_velocity < 0, ema20_velocity < 0, ema50_velocity < 0]) else 0
            mixed_signals = 1 if len(set([price_velocity > 0, ema20_velocity > 0, ema50_velocity > 0])) > 1 else 0
            
            features = {
                # Velocity группа
                'price_velocity': price_velocity,
                'ema20_velocity': ema20_velocity,
                'ema50_velocity': ema50_velocity,
                
                # Acceleration группа
                'price_acceleration': price_accel,
                'ema20_acceleration': ema20_accel,
                'ema50_acceleration': ema50_accel,
                
                # Velocity Ratios группа
                'velocity_ratio_20_50': velocity_ratio_20_50,
                'velocity_ratio_price_20': velocity_ratio_price_20,
                'velocity_ratio_price_50': velocity_ratio_price_50,
                
                # Distance to EMAs группа
                'distance_to_ema20': distance_to_ema20,
                'distance_to_ema50': distance_to_ema50,
                'distance_to_ema100': distance_to_ema100,
                
                # Distance Changes группа
                'distance_change_20': distance_change_20,
                'distance_change_50': distance_change_50,
                'distance_change_100': distance_change_100,
                
                # EMA Angles группа
                'ema20_angle': ema20_angle,
                'ema50_angle': ema50_angle,
                'ema100_angle': ema100_angle,
                
                # Angle Changes группа
                'ema20_angle_change': ema20_angle_change,
                'ema50_angle_change': ema50_angle_change,
                'ema100_angle_change': ema100_angle_change,
                
                # EMA Relationships группа
                'ema20_vs_50': ema20_vs_50,
                'ema50_vs_100': ema50_vs_100,
                'ema20_vs_100': ema20_vs_100,
                
                # Synchronization группа
                'price_ema20_sync': price_ema20_sync,
                'price_ema50_sync': price_ema50_sync,
                'ema20_ema50_sync': ema20_ema50_sync,
                'all_up_sync': all_up_sync,
                'all_down_sync': all_down_sync,
                'mixed_signals': mixed_signals
            }
            
            return features
            
        except Exception as e:
            return None
    
    def predict_with_model(self, features: dict) -> Dict[str, Any]:
        """Предсказание с помощью обученной модели"""
        try:
            # Преобразуем в вектор
            feature_vector = []
            for name in self.feature_names:
                if name in features:
                    value = features[name]
                    if np.isnan(value) or np.isinf(value):
                        value = 0.0
                    feature_vector.append(value)
                else:
                    feature_vector.append(0.0)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            # Предсказание
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            # Названия классов
            class_names = ['Малое (1-3%)', 'Среднее (3-7%)', 'Крупное (7%+)']
            
            result = {
                'prediction': class_names[prediction],
                'probabilities': {
                    'small': probabilities[0],
                    'medium': probabilities[1], 
                    'large': probabilities[2]
                }
            }
            
            return result
            
        except Exception as e:
            return None
    
    def analyze_signal_for_backtest(self, symbol: str, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Анализ сигнала на основе исторической модели"""
        try:
            historical_data = df.iloc[:current_idx + 1].copy()
            if len(historical_data) < 100:
                return {'signal': 'WAIT', 'confidence': 0}
            
            # Подготавливаем признаки
            features = self.prepare_ml_features(historical_data, symbol)
            if not features:
                return {'signal': 'WAIT', 'confidence': 0}
            
            # Получаем предсказание
            prediction = self.predict_with_model(features)
            if not prediction:
                return {'signal': 'WAIT', 'confidence': 0}
            
            probabilities = prediction['probabilities']
            
            # Генерируем сигнал
            signal = 'WAIT'
            confidence = 0
            
            medium_prob = probabilities['medium']
            large_prob = probabilities['large']
            
            # Более строгие пороги для честного теста
            if medium_prob > 0.4 or large_prob > 0.3:
                signal = 'LONG'
                confidence = int((medium_prob + large_prob) * 100)
            
            return {
                'signal': signal,
                'confidence': confidence,
                'price': float(historical_data.iloc[-1]['close']),
                'prediction': prediction['prediction'],
                'probabilities': probabilities
            }
            
        except Exception as e:
            return {'signal': 'WAIT', 'confidence': 0}
    
    def run_backtest(self) -> Dict[str, Any]:
        """Запуск правильного бэктеста"""
        print("\\n🧪 ЗАПУСК ПРАВИЛЬНОГО БЭКТЕСТА")
        print("📅 Тестируем модель (2020-2024) на данных 2025 года")
        print("=" * 60)
        
        # Статистика по сигналам
        signals_stats = {
            'total_signals': 0,
            'long_signals': 0,
            'wait_signals': 0,
            'trades_opened': 0
        }
        
        # Загружаем тестовые данные
        historical_data = {}
        for symbol in self.symbols:
            df = self.get_test_data(symbol)
            if not df.empty:
                historical_data[symbol] = df
            time.sleep(1)
        
        if not historical_data:
            return {'error': 'Нет данных для тестирования'}
        
        # Находим общие временные точки
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index)
        timestamps = sorted(list(all_timestamps))
        
        print(f"📊 Анализируем {len(timestamps)} временных точек...")
        
        # Основной цикл бэктеста
        progress_counter = 0
        for i, timestamp in enumerate(timestamps):
            progress_counter += 1
            
            if progress_counter % 500 == 0:
                progress = (i / len(timestamps)) * 100
                print(f"   Прогресс: {progress:.1f}% | Сделок: {len(self.trades)} | Баланс: ${self.current_balance:.2f}")
            
            for symbol, df in historical_data.items():
                if timestamp not in df.index:
                    continue
                
                current_price = float(df.loc[timestamp, 'close'])
                current_idx = df.index.get_loc(timestamp)
                
                # Проверяем закрытие позиций
                if symbol in self.positions:
                    position = self.positions[symbol]
                    should_close = False
                    
                    if position['side'] == 'LONG':
                        if current_price >= position['take_profit'] or current_price <= position['stop_loss']:
                            should_close = True
                    
                    if should_close:
                        # Расчет PnL
                        pnl = (current_price - position['entry_price']) * position['size']
                        self.current_balance += pnl
                        
                        self.trades.append({
                            'symbol': symbol,
                            'side': position['side'],
                            'entry_price': position['entry_price'],
                            'exit_price': current_price,
                            'pnl': pnl,
                            'pnl_percent': (pnl / (position['entry_price'] * position['size'])) * 100,
                            'timestamp': timestamp
                        })
                        
                        del self.positions[symbol]
                
                # Ищем новые сигналы
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    signal_data = self.analyze_signal_for_backtest(symbol, df, current_idx)
                    signals_stats['total_signals'] += 1
                    
                    if signal_data['signal'] == 'LONG':
                        signals_stats['long_signals'] += 1
                    else:
                        signals_stats['wait_signals'] += 1
                    
                    # Открываем позицию
                    if signal_data['signal'] == 'LONG' and signal_data['confidence'] >= 50:
                        position_value = self.current_balance * self.position_size_percent
                        size = position_value / current_price
                        
                        # TP/SL уровни
                        if signal_data['confidence'] >= 70:
                            profit_pct, loss_pct = 0.05, 0.02  # 5% прибыль, 2% убыток
                        else:
                            profit_pct, loss_pct = 0.04, 0.015  # 4% прибыль, 1.5% убыток
                        
                        take_profit = current_price * (1 + profit_pct)
                        stop_loss = current_price * (1 - loss_pct)
                        
                        self.positions[symbol] = {
                            'side': 'LONG',
                            'entry_price': current_price,
                            'size': size,
                            'take_profit': take_profit,
                            'stop_loss': stop_loss,
                            'timestamp': timestamp
                        }
                        signals_stats['trades_opened'] += 1
        
        # Закрываем оставшиеся позиции
        for symbol in list(self.positions.keys()):
            last_price = float(historical_data[symbol].iloc[-1]['close'])
            position = self.positions[symbol]
            
            pnl = (last_price - position['entry_price']) * position['size']
            self.current_balance += pnl
            
            self.trades.append({
                'symbol': symbol,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': last_price,
                'pnl': pnl,
                'pnl_percent': (pnl / (position['entry_price'] * position['size'])) * 100,
                'timestamp': timestamps[-1]
            })
        
        # Выводим статистику
        print(f"\\n📊 СТАТИСТИКА СИГНАЛОВ:")
        print(f"   Всего сигналов: {signals_stats['total_signals']}")
        print(f"   LONG сигналов: {signals_stats['long_signals']}")
        print(f"   WAIT сигналов: {signals_stats['wait_signals']}")
        print(f"   Позиций открыто: {signals_stats['trades_opened']}")
        
        # Расчет результатов
        if not self.trades:
            return {
                'error': 'Сделок не было',
                'signals_stats': signals_stats
            }
        
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_return': total_return,
            'total_pnl': self.current_balance - self.initial_balance,
            'trades': self.trades,
            'signals_stats': signals_stats
        }

    def print_results(self, results: Dict[str, Any]):
        """Красивый вывод результатов"""
        if 'error' in results:
            print(f"\\n❌ {results['error']}")
            if 'signals_stats' in results:
                stats = results['signals_stats']
                print(f"\\n📊 Статистика сигналов:")
                print(f"   Проанализировано: {stats['total_signals']}")
                print(f"   LONG: {stats['long_signals']}")
                print(f"   Открыто позиций: {stats['trades_opened']}")
            return
        
        print(f"\\n🎉 РЕЗУЛЬТАТЫ ПРАВИЛЬНОГО БЭКТЕСТА")
        print("=" * 50)
        print(f"📅 Период: {self.test_start.date()} - {self.test_end.date()}")
        print(f"🧠 Модель: Обучена на 2020-2024, тестируется на 2025")
        print()
        print(f"💰 Стартовый капитал: ${results['initial_balance']:,.2f}")
        print(f"💵 Финальный капитал: ${results['final_balance']:,.2f}")
        print(f"📈 Доходность: {results['total_return']:+.2f}%")
        print(f"💸 P&L: ${results['total_pnl']:+,.2f}")
        print()
        print(f"📊 Всего сделок: {results['total_trades']}")
        print(f"✅ Прибыльных: {results['winning_trades']} ({results['win_rate']:.1f}%)")
        print(f"❌ Убыточных: {results['total_trades'] - results['winning_trades']} ({100-results['win_rate']:.1f}%)")
        
        # Статистика по монетам
        if results['trades']:
            symbol_stats = {}
            for trade in results['trades']:
                symbol = trade['symbol']
                if symbol not in symbol_stats:
                    symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
                symbol_stats[symbol]['trades'] += 1
                symbol_stats[symbol]['pnl'] += trade['pnl']
                if trade['pnl'] > 0:
                    symbol_stats[symbol]['wins'] += 1
            
            print(f"\\n🏆 Статистика по монетам:")
            for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
                wr = (stats['wins'] / stats['trades']) * 100 if stats['trades'] > 0 else 0
                print(f"   {symbol}: {stats['trades']} сделок, ${stats['pnl']:+.2f}, WR: {wr:.1f}%")

if __name__ == "__main__":
    backtest = HistoricalBacktest()
    
    # Загружаем модель
    if not backtest.load_model():
        print("❌ Сначала запустите historical_ml_trainer.py для обучения модели!")
        exit(1)
    
    # Запускаем бэктест
    results = backtest.run_backtest()
    
    # Выводим результаты
    backtest.print_results(results)
    
    # Сохраняем результаты
    with open('historical_backtest_results.json', 'w', encoding='utf-8') as f:
        # Конвертируем datetime для JSON
        results_for_json = results.copy()
        if 'trades' in results_for_json:
            for trade in results_for_json['trades']:
                if 'timestamp' in trade:
                    trade['timestamp'] = trade['timestamp'].isoformat()
        json.dump(results_for_json, f, ensure_ascii=False, indent=2)
    
    print(f"\\n💾 Результаты сохранены в 'historical_backtest_results.json'")




