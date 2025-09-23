#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🎯 РЕАЛЬНЫЙ БЭКТЕСТЕР
Бэктестинг с учетом реальных торговых условий
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

class RealBacktester:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        
        # Загружаем модели
        self.minimum_model = None
        self.maximum_model = None
        self.minimum_scaler = None
        self.maximum_scaler = None
        self.minimum_features = None
        self.maximum_features = None
        
        # Реальные торговые параметры
        self.initial_balance = 10000      # Начальный баланс $10,000
        self.position_size = 0.1         # 10% от баланса на сделку
        self.commission = 0.001          # Комиссия Binance 0.1%
        self.slippage = 0.0005           # Проскальзывание 0.05%
        self.max_drawdown = 0.2          # Максимальная просадка 20%
        self.stop_loss = 0.03            # Стоп-лосс 3%
        self.take_profit = 0.06          # Тейк-профит 6%
        
        # Результаты
        self.trades = []
        self.balance_history = []
        self.current_balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.max_drawdown_reached = 0
        
        print("🎯 РЕАЛЬНЫЙ БЭКТЕСТЕР")
        print("📊 С учетом реальных торговых условий")
        print("=" * 50)
    
    def load_models(self):
        """Загрузка моделей"""
        print("📂 Загружаю модели...")
        
        try:
            with open('historical_models/minimum_model.pkl', 'rb') as f:
                self.minimum_model = pickle.load(f)
            
            with open('historical_models/maximum_model.pkl', 'rb') as f:
                self.maximum_model = pickle.load(f)
            
            with open('historical_models/minimum_scaler.pkl', 'rb') as f:
                self.minimum_scaler = pickle.load(f)
            
            with open('historical_models/maximum_scaler.pkl', 'rb') as f:
                self.maximum_scaler = pickle.load(f)
            
            with open('historical_models/minimum_features.pkl', 'rb') as f:
                self.minimum_features = pickle.load(f)
            
            with open('historical_models/maximum_features.pkl', 'rb') as f:
                self.maximum_features = pickle.load(f)
            
            print("   ✅ Модели загружены")
            return True
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return False
    
    def get_real_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Получение реальных данных"""
        try:
            print(f"📊 Загружаю {symbol} с {start_date.strftime('%d.%m.%Y')} по {end_date.strftime('%d.%m.%Y')}...")
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
                    
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1
                    
                    if current_since > end_ts:
                        break
                        
                    time.sleep(0.1)
                    
                except Exception as e:
                    break
            
            if not all_ohlcv:
                return pd.DataFrame()
            
            all_ohlcv = [candle for candle in all_ohlcv if candle[0] <= end_ts]
            
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Добавляем EMA
            df['ema_20'] = df['close'].ewm(span=20).mean()
            df['ema_50'] = df['close'].ewm(span=50).mean()
            df['ema_100'] = df['close'].ewm(span=100).mean()
            
            print(f"   ✅ {len(df)} свечей")
            return df
            
        except Exception as e:
            print(f"   ❌ Ошибка: {e}")
            return pd.DataFrame()
    
    def prepare_features(self, df: pd.DataFrame, idx: int, is_minimum: bool = True) -> dict:
        """Подготовка признаков"""
        try:
            if idx < 50 or idx >= len(df) - 6:
                return None
            
            current = df.iloc[idx]
            prev = df.iloc[idx - 1]
            prev_2 = df.iloc[idx - 2]
            
            if is_minimum:
                # Признаки для минимума
                features = {
                    'price_velocity': (current['close'] - prev['close']) / prev['close'],
                    'ema20_velocity': (current['ema_20'] - prev['ema_20']) / prev['ema_20'],
                    'ema50_velocity': (current['ema_50'] - prev['ema_50']) / prev['ema_50'],
                    'ema100_velocity': (current['ema_100'] - prev['ema_100']) / prev['ema_100'],
                    'price_acceleration': ((current['close'] - prev['close']) / prev['close']) - 
                                       ((prev['close'] - prev_2['close']) / prev_2['close']),
                    'price_distance_ema20': (current['close'] - current['ema_20']) / current['ema_20'],
                    'price_distance_ema50': (current['close'] - current['ema_50']) / current['ema_50'],
                    'price_distance_ema100': (current['close'] - current['ema_100']) / current['ema_100'],
                    'ema20_angle': ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) * 100,
                    'ema50_angle': ((current['ema_50'] - prev['ema_50']) / prev['ema_50']) * 100,
                    'ema100_angle': ((current['ema_100'] - prev['ema_100']) / prev['ema_100']) * 100,
                    'volatility': df['close'].iloc[idx-20:idx].std() / df['close'].iloc[idx-20:idx].mean(),
                    'volume_ratio': current['volume'] / df['volume'].iloc[idx-20:idx].mean() if df['volume'].iloc[idx-20:idx].mean() > 0 else 1,
                    'distance_change_ema20': (current['close'] - current['ema_20']) / current['ema_20'] - 
                                           (prev['close'] - prev['ema_20']) / prev['ema_20'],
                    'distance_change_ema50': (current['close'] - current['ema_50']) / current['ema_50'] - 
                                           (prev['close'] - prev['ema_50']) / prev['ema_50'],
                    'velocity_ratio_price_ema20': (current['close'] - prev['close']) / prev['close'] / 
                                                ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) if (current['ema_20'] - prev['ema_20']) / prev['ema_20'] != 0 else 0,
                    'velocity_ratio_ema20_ema50': ((current['ema_20'] - prev['ema_20']) / prev['ema_20']) / 
                                                ((current['ema_50'] - prev['ema_50']) / prev['ema_50']) if (current['ema_50'] - prev['ema_50']) / prev['ema_50'] != 0 else 0,
                    'ema20_to_ema50': (current['ema_20'] - current['ema_50']) / current['ema_50'],
                    'ema50_to_ema100': (current['ema_50'] - current['ema_100']) / current['ema_100'],
                    'ema20_to_ema100': (current['ema_20'] - current['ema_100']) / current['ema_100']
                }
            else:
                # Признаки для максимума (расширенные)
                price_velocity = (current['close'] - prev['close']) / prev['close']
                ema20_velocity = (current['ema_20'] - prev['ema_20']) / prev['ema_20']
                ema50_velocity = (current['ema_50'] - prev['ema_50']) / prev['ema_50']
                ema100_velocity = (current['ema_100'] - prev['ema_100']) / prev['ema_100']
                
                features = {
                    'price_velocity': price_velocity,
                    'ema20_velocity': ema20_velocity,
                    'ema50_velocity': ema50_velocity,
                    'ema100_velocity': ema100_velocity,
                    'price_acceleration': price_velocity - ((prev['close'] - prev_2['close']) / prev_2['close']),
                    'price_distance_ema20': (current['close'] - current['ema_20']) / current['ema_20'],
                    'price_distance_ema50': (current['close'] - current['ema_50']) / current['ema_50'],
                    'price_distance_ema100': (current['close'] - current['ema_100']) / current['ema_100'],
                    'ema20_angle': ema20_velocity * 100,
                    'ema50_angle': ema50_velocity * 100,
                    'ema100_angle': ema100_velocity * 100,
                    'volatility': df['close'].iloc[idx-20:idx].std() / df['close'].iloc[idx-20:idx].mean(),
                    'volume_ratio': current['volume'] / df['volume'].iloc[idx-20:idx].mean() if df['volume'].iloc[idx-20:idx].mean() > 0 else 1,
                    'distance_change_ema20': (current['close'] - current['ema_20']) / current['ema_20'] - 
                                           (prev['close'] - prev['ema_20']) / prev['ema_20'],
                    'distance_change_ema50': (current['close'] - current['ema_50']) / current['ema_50'] - 
                                           (prev['close'] - prev['ema_50']) / prev['ema_50'],
                    'velocity_ratio_price_ema20': price_velocity / ema20_velocity if ema20_velocity != 0 else 0,
                    'velocity_ratio_ema20_ema50': ema20_velocity / ema50_velocity if ema50_velocity != 0 else 0,
                    'velocity_ratio_ema50_ema100': ema50_velocity / ema100_velocity if ema100_velocity != 0 else 0,
                    'ema20_to_ema50': (current['ema_20'] - current['ema_50']) / current['ema_50'],
                    'ema50_to_ema100': (current['ema_50'] - current['ema_100']) / current['ema_100'],
                    'ema20_to_ema100': (current['ema_20'] - current['ema_100']) / current['ema_100'],
                    'ema20_slope': (current['ema_20'] - df.iloc[idx-5]['ema_20']) / 5,
                    'ema50_slope': (current['ema_50'] - df.iloc[idx-5]['ema_50']) / 5,
                    'ema100_slope': (current['ema_100'] - df.iloc[idx-5]['ema_100']) / 5,
                    'ema20_ema50_ratio': current['ema_20'] / current['ema_50'],
                    'ema50_ema100_ratio': current['ema_50'] / current['ema_100'],
                    'ema20_ema100_ratio': current['ema_20'] / current['ema_100']
                }
            
            return features
            
        except Exception:
            return None
    
    def predict_signal(self, features_dict, is_minimum: bool = True):
        """Прогнозирование сигнала"""
        try:
            if is_minimum:
                features_list = [features_dict.get(name, 0) for name in self.minimum_features]
                features_array = np.array(features_list).reshape(1, -1)
                features_scaled = self.minimum_scaler.transform(features_array)
                prediction = self.minimum_model.predict(features_scaled)[0]
                probability = self.minimum_model.predict_proba(features_scaled)[0]
            else:
                features_list = [features_dict.get(name, 0) for name in self.maximum_features]
                features_array = np.array(features_list).reshape(1, -1)
                features_scaled = self.maximum_scaler.transform(features_array)
                prediction = self.maximum_model.predict(features_scaled)[0]
                probability = self.maximum_model.predict_proba(features_scaled)[0]
            
            return {
                'prediction': prediction,
                'confidence': max(probability),
                'is_good_signal': prediction == 1
            }
        except Exception:
            return None
    
    def calculate_position_size(self):
        """Расчет размера позиции"""
        return self.current_balance * self.position_size
    
    def apply_costs(self, price: float, is_entry: bool = True) -> float:
        """Применение комиссий и проскальзывания"""
        if is_entry:
            # При входе: цена + проскальзывание + комиссия
            return price * (1 + self.slippage + self.commission)
        else:
            # При выходе: цена - проскальзывание - комиссия
            return price * (1 - self.slippage - self.commission)
    
    def check_risk_management(self, entry_price: float, current_price: float) -> str:
        """Проверка управления рисками"""
        # Стоп-лосс
        if current_price <= entry_price * (1 - self.stop_loss):
            return 'stop_loss'
        
        # Тейк-профит
        if current_price >= entry_price * (1 + self.take_profit):
            return 'take_profit'
        
        # Максимальная просадка
        current_drawdown = (self.max_balance - self.current_balance) / self.max_balance
        if current_drawdown >= self.max_drawdown:
            return 'max_drawdown'
        
        return 'continue'
    
    def simulate_real_trading(self, df: pd.DataFrame, symbol: str):
        """Симуляция реальной торговли"""
        print(f"🎯 Симулирую реальную торговлю {symbol}...")
        
        current_position = None
        symbol_trades = []
        
        for i in range(100, len(df) - 10):
            current_time = df.index[i]
            current_price = df.iloc[i]['close']
            
            # Обновляем максимальный баланс
            if self.current_balance > self.max_balance:
                self.max_balance = self.current_balance
            
            # Проверяем управление рисками
            if current_position:
                risk_action = self.check_risk_management(current_position['entry_price'], current_price)
                
                if risk_action != 'continue':
                    # Принудительный выход
                    exit_price = self.apply_costs(current_price, is_entry=False)
                    position_size = current_position['position_size']
                    
                    profit_loss = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    profit_amount = position_size * profit_loss
                    
                    self.current_balance += profit_amount
                    
                    trade = {
                        'symbol': symbol,
                        'entry_time': current_position['entry_time'],
                        'exit_time': current_time,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'position_size': position_size,
                        'profit_loss': profit_loss,
                        'profit_amount': profit_amount,
                        'balance_after': self.current_balance,
                        'exit_reason': risk_action,
                        'duration_hours': (current_time - current_position['entry_time']).total_seconds() / 3600
                    }
                    
                    symbol_trades.append(trade)
                    current_position = None
            
            # Поиск сигнала входа
            if current_position is None:
                # Проверяем минимум
                is_minimum = True
                for j in range(max(0, i-5), min(len(df), i+6)):
                    if j != i and df.iloc[j]['low'] <= df.iloc[i]['low']:
                        is_minimum = False
                        break
                
                if is_minimum:
                    features = self.prepare_features(df, i, is_minimum=True)
                    if features:
                        prediction = self.predict_signal(features, is_minimum=True)
                        if prediction and prediction['is_good_signal'] and prediction['confidence'] > 0.7:
                            # Входим в позицию
                            entry_price = self.apply_costs(current_price, is_entry=True)
                            position_size = self.calculate_position_size()
                            
                            if position_size > 0:
                                current_position = {
                                    'entry_time': current_time,
                                    'entry_price': entry_price,
                                    'position_size': position_size,
                                    'confidence': prediction['confidence']
                                }
            
            # Поиск сигнала выхода
            if current_position:
                # Проверяем максимум
                is_maximum = True
                for j in range(max(0, i-5), min(len(df), i+6)):
                    if j != i and df.iloc[j]['high'] >= df.iloc[i]['high']:
                        is_maximum = False
                        break
                
                if is_maximum:
                    features = self.prepare_features(df, i, is_minimum=False)
                    if features:
                        prediction = self.predict_signal(features, is_minimum=False)
                        if prediction and prediction['is_good_signal'] and prediction['confidence'] > 0.7:
                            # Выходим из позиции
                            exit_price = self.apply_costs(current_price, is_entry=False)
                            position_size = current_position['position_size']
                            
                            profit_loss = (exit_price - current_position['entry_price']) / current_position['entry_price']
                            profit_amount = position_size * profit_loss
                            
                            self.current_balance += profit_amount
                            
                            trade = {
                                'symbol': symbol,
                                'entry_time': current_position['entry_time'],
                                'exit_time': current_time,
                                'entry_price': current_position['entry_price'],
                                'exit_price': exit_price,
                                'position_size': position_size,
                                'profit_loss': profit_loss,
                                'profit_amount': profit_amount,
                                'balance_after': self.current_balance,
                                'exit_reason': 'signal',
                                'duration_hours': (current_time - current_position['entry_time']).total_seconds() / 3600
                            }
                            
                            symbol_trades.append(trade)
                            current_position = None
            
            # Записываем историю баланса
            self.balance_history.append({
                'time': current_time,
                'balance': self.current_balance,
                'symbol': symbol
            })
        
        print(f"   ✅ {len(symbol_trades)} сделок")
        return symbol_trades
    
    def run_real_backtest(self, start_date: datetime, end_date: datetime):
        """Запуск реального бэктестинга"""
        print("🚀 ЗАПУСК РЕАЛЬНОГО БЭКТЕСТИНГА")
        print("=" * 50)
        
        # 1. Загружаем модели
        if not self.load_models():
            return
        
        # 2. Сбрасываем баланс
        self.current_balance = self.initial_balance
        self.max_balance = self.initial_balance
        self.balance_history = []
        self.trades = []
        
        print(f"💰 Начальный баланс: ${self.initial_balance:,}")
        print(f"📊 Размер позиции: {self.position_size*100}% от баланса")
        print(f"💸 Комиссия: {self.commission*100}%")
        print(f"📉 Проскальзывание: {self.slippage*100}%")
        print(f"🛡️ Стоп-лосс: {self.stop_loss*100}%")
        print(f"🎯 Тейк-профит: {self.take_profit*100}%")
        
        # 3. Тестируем каждый символ
        for i, symbol in enumerate(self.symbols):
            print(f"\n[{i+1}/{len(self.symbols)}] {symbol}")
            
            df = self.get_real_data(symbol, start_date, end_date)
            if df.empty:
                print(f"   ❌ Нет данных для {symbol}")
                continue
            
            trades = self.simulate_real_trading(df, symbol)
            self.trades.extend(trades)
            
            time.sleep(1)
        
        # 4. Анализируем результаты
        self.analyze_results()
        
        # 5. Сохраняем результаты
        self.save_results()
        
        print(f"\n✅ РЕАЛЬНЫЙ БЭКТЕСТИНГ ЗАВЕРШЕН!")
    
    def analyze_results(self):
        """Анализ результатов"""
        print(f"\n📊 АНАЛИЗ РЕЗУЛЬТАТОВ:")
        print("-" * 30)
        
        if not self.trades:
            print("❌ Нет сделок для анализа")
            return
        
        # Основная статистика
        total_trades = len(self.trades)
        profitable_trades = [t for t in self.trades if t['profit_amount'] > 0]
        losing_trades = [t for t in self.trades if t['profit_amount'] < 0]
        
        win_rate = len(profitable_trades) / total_trades * 100
        total_profit = sum(t['profit_amount'] for t in self.trades)
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        
        avg_profit = np.mean([t['profit_amount'] for t in self.trades])
        avg_duration = np.mean([t['duration_hours'] for t in self.trades])
        
        max_profit = max(t['profit_amount'] for t in self.trades)
        max_loss = min(t['profit_amount'] for t in self.trades)
        
        # Просадка
        max_drawdown = (self.max_balance - min(h['balance'] for h in self.balance_history)) / self.max_balance * 100
        
        print(f"💰 Финальный баланс: ${self.current_balance:,.2f}")
        print(f"📈 Общая прибыль: ${total_profit:,.2f}")
        print(f"📊 Общая доходность: {total_return:.2f}%")
        print(f"🎯 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных: {len(profitable_trades)} ({win_rate:.1f}%)")
        print(f"❌ Убыточных: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"📈 Средняя прибыль: ${avg_profit:.2f}")
        print(f"⏱️ Средняя длительность: {avg_duration:.1f} часов")
        print(f"🎯 Максимальная прибыль: ${max_profit:.2f}")
        print(f"📉 Максимальный убыток: ${max_loss:.2f}")
        print(f"📊 Максимальная просадка: {max_drawdown:.2f}%")
        
        # Статистика по символам
        print(f"\n📊 ПО СИМВОЛАМ:")
        symbol_stats = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'profit': 0, 'profitable': 0}
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['profit'] += trade['profit_amount']
            if trade['profit_amount'] > 0:
                symbol_stats[symbol]['profitable'] += 1
        
        for symbol, stats in symbol_stats.items():
            win_rate_symbol = stats['profitable'] / stats['trades'] * 100
            print(f"   {symbol}: {stats['trades']} сделок, ${stats['profit']:.2f} прибыль, {win_rate_symbol:.1f}% win rate")
    
    def save_results(self):
        """Сохранение результатов"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results = {
            'backtest_info': {
                'start_date': '2025-01-01',
                'end_date': '2025-09-23',
                'initial_balance': self.initial_balance,
                'final_balance': self.current_balance,
                'total_return': (self.current_balance - self.initial_balance) / self.initial_balance * 100,
                'total_trades': len(self.trades),
                'position_size': self.position_size,
                'commission': self.commission,
                'slippage': self.slippage,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit
            },
            'trades': self.trades,
            'balance_history': self.balance_history
        }
        
        results_file = f'real_backtest_results_{timestamp}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"💾 Результаты сохранены: {results_file}")

if __name__ == "__main__":
    backtester = RealBacktester()
    
    # Запускаем бэктест на данных 2025 года
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 9, 23)
    
    backtester.run_real_backtest(start_date, end_date)
