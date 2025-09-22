#!/usr/bin/env python3
"""
🎯 БЭКТЕСТЕР ПРИБЫЛЬНОСТИ БОТА
Тестирует прибыльность торгового бота на исторических данных
Период: 01.01.2025 - текущая дата
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
import logging
from typing import Dict, List, Tuple, Any
import time

# Импортируем компоненты бота
from advanced_ema_analyzer import AdvancedEMAAnalyzer
from advanced_ml_trainer import AdvancedMLTrainer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('backtest_results.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProfitabilityBacktester:
    def __init__(self):
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime.now()
        self.initial_balance = 1000.0  # Стартовый капитал $1000
        self.current_balance = self.initial_balance
        self.trades = []
        self.positions = {}
        
        # Инициализация компонентов бота
        self.ema_analyzer = AdvancedEMAAnalyzer()
        self.ml_trainer = AdvancedMLTrainer()
        
        # Загружаем обученные модели
        try:
            self.ml_trainer.load_models()
            logger.info("✅ ML модели загружены")
        except Exception as e:
            logger.warning(f"⚠️ Не удалось загрузить ML модели: {e}")
            
        # Настройки торговли
        self.position_size_percent = 0.1  # 10% от баланса на сделку
        self.max_positions = 5  # Максимум 5 открытых позиций
        
    def get_historical_data(self, symbol: str, timeframe='1h') -> pd.DataFrame:
        """Получение исторических данных с Binance"""
        try:
            exchange = ccxt.binance()
            
            # Конвертируем даты в миллисекунды
            since = int(self.start_date.timestamp() * 1000)
            
            logger.info(f"📥 Загружаю данные {symbol} с {self.start_date.strftime('%Y-%m-%d')}...")
            
            all_ohlcv = []
            current_since = since
            
            while current_since < int(self.end_date.timestamp() * 1000):
                try:
                    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                    if not ohlcv:
                        break
                        
                    all_ohlcv.extend(ohlcv)
                    current_since = ohlcv[-1][0] + 1  # Следующая свеча
                    
                    # Пауза чтобы не превысить лимиты API
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"❌ Ошибка загрузки данных {symbol}: {e}")
                    break
            
            if not all_ohlcv:
                logger.error(f"❌ Нет данных для {symbol}")
                return pd.DataFrame()
            
            # Создаем DataFrame
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            logger.info(f"✅ Загружено {len(df)} свечей для {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Ошибка получения данных {symbol}: {e}")
            return pd.DataFrame()
    
    def analyze_signal(self, symbol: str, df: pd.DataFrame, current_idx: int) -> Dict[str, Any]:
        """Анализ сигнала на конкретной свече (точно как в боте)"""
        try:
            # Берем данные до текущего момента (имитируем реальное время)
            historical_data = df.iloc[:current_idx + 1].copy()
            
            if len(historical_data) < 100:  # Нужно минимум 100 свечей для EMA
                return {'signal': 'WAIT', 'confidence': 0}
            
            # Конвертируем в формат для анализатора
            ohlcv_data = []
            for idx, row in historical_data.iterrows():
                ohlcv_data.append([
                    int(idx.timestamp() * 1000),  # timestamp
                    float(row['open']),
                    float(row['high']),
                    float(row['low']),
                    float(row['close']),
                    float(row['volume'])
                ])
            
            # Анализ с EMA + ML (точно как в боте)
            analysis = self.ema_analyzer.analyze_coin(symbol, ohlcv_data, self.ml_trainer)
            
            return {
                'signal': analysis.get('signal', 'WAIT'),
                'confidence': analysis.get('confidence', 0),
                'ml_entry_prob': analysis.get('ml_entry_prob', 0),
                'ml_exit_prob': analysis.get('ml_exit_prob', 0),
                'price': float(historical_data.iloc[-1]['close'])
            }
            
        except Exception as e:
            logger.error(f"❌ Ошибка анализа {symbol}: {e}")
            return {'signal': 'WAIT', 'confidence': 0}
    
    def calculate_position_size(self, price: float) -> float:
        """Расчет размера позиции"""
        position_value = self.current_balance * self.position_size_percent
        return position_value / price
    
    def open_position(self, symbol: str, signal: str, price: float, confidence: float, timestamp: datetime):
        """Открытие позиции"""
        if len(self.positions) >= self.max_positions:
            return False
            
        if symbol in self.positions:
            return False  # Уже есть позиция по этому символу
            
        size = self.calculate_position_size(price)
        
        # Расчет динамических уровней (как в боте)
        if confidence >= 80:
            profit_pct = 0.05  # +5%
            loss_pct = 0.025   # -2.5%
        elif confidence >= 60:
            profit_pct = 0.04  # +4%
            loss_pct = 0.02    # -2%
        elif confidence >= 40:
            profit_pct = 0.03  # +3%
            loss_pct = 0.015   # -1.5%
        else:
            profit_pct = 0.02  # +2%
            loss_pct = 0.01    # -1%
        
        if signal == 'LONG':
            take_profit = price * (1 + profit_pct)
            stop_loss = price * (1 - loss_pct)
        else:  # SHORT
            take_profit = price * (1 - profit_pct)
            stop_loss = price * (1 + loss_pct)
        
        position = {
            'symbol': symbol,
            'side': signal,
            'entry_price': price,
            'size': size,
            'take_profit': take_profit,
            'stop_loss': stop_loss,
            'timestamp': timestamp,
            'confidence': confidence
        }
        
        self.positions[symbol] = position
        logger.info(f"🔓 Открыта позиция {signal} {symbol} @ ${price:.4f} (TP: ${take_profit:.4f}, SL: ${stop_loss:.4f})")
        return True
    
    def check_position_exit(self, symbol: str, current_price: float, timestamp: datetime) -> bool:
        """Проверка условий закрытия позиции"""
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        take_profit = position['take_profit']
        stop_loss = position['stop_loss']
        side = position['side']
        
        should_close = False
        exit_reason = ""
        
        if side == 'LONG':
            if current_price >= take_profit:
                should_close = True
                exit_reason = "Take Profit"
            elif current_price <= stop_loss:
                should_close = True
                exit_reason = "Stop Loss"
        else:  # SHORT
            if current_price <= take_profit:
                should_close = True
                exit_reason = "Take Profit"
            elif current_price >= stop_loss:
                should_close = True
                exit_reason = "Stop Loss"
        
        if should_close:
            self.close_position(symbol, current_price, timestamp, exit_reason)
            return True
            
        return False
    
    def close_position(self, symbol: str, exit_price: float, timestamp: datetime, reason: str):
        """Закрытие позиции"""
        if symbol not in self.positions:
            return
            
        position = self.positions[symbol]
        entry_price = position['entry_price']
        size = position['size']
        side = position['side']
        
        # Расчет PnL
        if side == 'LONG':
            pnl = (exit_price - entry_price) * size
        else:  # SHORT
            pnl = (entry_price - exit_price) * size
        
        pnl_percent = (pnl / (entry_price * size)) * 100
        
        # Обновляем баланс
        self.current_balance += pnl
        
        # Записываем сделку
        trade = {
            'symbol': symbol,
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'size': size,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'entry_time': position['timestamp'],
            'exit_time': timestamp,
            'duration': timestamp - position['timestamp'],
            'exit_reason': reason,
            'confidence': position['confidence']
        }
        
        self.trades.append(trade)
        
        # Удаляем позицию
        del self.positions[symbol]
        
        logger.info(f"🔒 Закрыта позиция {side} {symbol} @ ${exit_price:.4f} | PnL: ${pnl:.2f} ({pnl_percent:+.2f}%) | {reason}")
    
    def run_backtest(self, symbols: List[str]):
        """Запуск бэктестинга"""
        logger.info(f"🚀 ЗАПУСК БЭКТЕСТИНГА")
        logger.info(f"📅 Период: {self.start_date.strftime('%Y-%m-%d')} - {self.end_date.strftime('%Y-%m-%d')}")
        logger.info(f"💰 Стартовый капитал: ${self.initial_balance}")
        logger.info(f"🪙 Тестируемые пары: {', '.join(symbols)}")
        
        # Загружаем данные для всех символов
        historical_data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty:
                historical_data[symbol] = df
            else:
                logger.warning(f"⚠️ Пропускаю {symbol} - нет данных")
        
        if not historical_data:
            logger.error("❌ Нет данных для тестирования!")
            return
        
        # Находим общий временной период
        all_timestamps = set()
        for df in historical_data.values():
            all_timestamps.update(df.index)
        
        timestamps = sorted(list(all_timestamps))
        logger.info(f"📊 Обрабатываю {len(timestamps)} временных точек...")
        
        # Основной цикл бэктестинга
        for i, timestamp in enumerate(timestamps):
            if i % 1000 == 0:
                progress = (i / len(timestamps)) * 100
                logger.info(f"⏳ Прогресс: {progress:.1f}% | Баланс: ${self.current_balance:.2f}")
            
            # Проверяем каждый символ на этой временной отметке
            for symbol, df in historical_data.items():
                if timestamp not in df.index:
                    continue
                
                current_price = float(df.loc[timestamp, 'close'])
                current_idx = df.index.get_loc(timestamp)
                
                # Проверяем закрытие существующих позиций
                self.check_position_exit(symbol, current_price, timestamp)
                
                # Ищем новые сигналы (только если нет открытой позиции)
                if symbol not in self.positions and len(self.positions) < self.max_positions:
                    signal_data = self.analyze_signal(symbol, df, current_idx)
                    
                    signal = signal_data.get('signal', 'WAIT')
                    confidence = signal_data.get('confidence', 0)
                    
                    # Открываем позицию если сигнал сильный
                    if signal in ['LONG', 'SHORT'] and confidence >= 40:
                        self.open_position(symbol, signal, current_price, confidence, timestamp)
        
        # Закрываем все оставшиеся позиции
        for symbol in list(self.positions.keys()):
            last_price = float(historical_data[symbol].iloc[-1]['close'])
            self.close_position(symbol, last_price, timestamps[-1], "End of backtest")
        
        # Генерируем отчет
        self.generate_report()
    
    def generate_report(self):
        """Генерация отчета по результатам"""
        logger.info("\n" + "="*60)
        logger.info("📊 РЕЗУЛЬТАТЫ БЭКТЕСТИНГА")
        logger.info("="*60)
        
        if not self.trades:
            logger.info("❌ Сделок не было!")
            return
        
        # Основная статистика
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl'] > 0])
        losing_trades = len([t for t in self.trades if t['pnl'] < 0])
        
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_return = ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
        
        avg_win = np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]) if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss < 0 else float('inf')
        
        # Максимальная просадка
        balance_history = [self.initial_balance]
        for trade in self.trades:
            balance_history.append(balance_history[-1] + trade['pnl'])
        
        peak = balance_history[0]
        max_drawdown = 0
        for balance in balance_history:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Вывод результатов
        logger.info(f"💰 Стартовый капитал: ${self.initial_balance:.2f}")
        logger.info(f"💰 Финальный капитал: ${self.current_balance:.2f}")
        logger.info(f"📈 Общая прибыль: ${total_pnl:.2f} ({total_return:+.2f}%)")
        logger.info(f"📊 Всего сделок: {total_trades}")
        logger.info(f"✅ Прибыльных: {winning_trades} ({win_rate:.1f}%)")
        logger.info(f"❌ Убыточных: {losing_trades} ({100-win_rate:.1f}%)")
        logger.info(f"📊 Средняя прибыль: ${avg_win:.2f}")
        logger.info(f"📊 Средний убыток: ${avg_loss:.2f}")
        logger.info(f"🎯 Profit Factor: {profit_factor:.2f}")
        logger.info(f"📉 Максимальная просадка: {max_drawdown:.2f}%")
        
        # Статистика по парам
        logger.info("\n📊 СТАТИСТИКА ПО ПАРАМ:")
        symbol_stats = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_stats:
                symbol_stats[symbol] = {'trades': 0, 'pnl': 0, 'wins': 0}
            symbol_stats[symbol]['trades'] += 1
            symbol_stats[symbol]['pnl'] += trade['pnl']
            if trade['pnl'] > 0:
                symbol_stats[symbol]['wins'] += 1
        
        for symbol, stats in symbol_stats.items():
            win_rate_symbol = (stats['wins'] / stats['trades']) * 100
            logger.info(f"{symbol}: {stats['trades']} сделок, ${stats['pnl']:.2f}, WR: {win_rate_symbol:.1f}%")
        
        # Сохраняем детальный отчет
        self.save_detailed_report()
        
        logger.info("="*60)
        
    def save_detailed_report(self):
        """Сохранение детального отчета в файл"""
        try:
            # CSV с детализацией сделок
            df_trades = pd.DataFrame(self.trades)
            df_trades.to_csv('backtest_trades.csv', index=False)
            
            # JSON с общей статистикой
            report = {
                'period': {
                    'start': self.start_date.isoformat(),
                    'end': self.end_date.isoformat()
                },
                'capital': {
                    'initial': self.initial_balance,
                    'final': self.current_balance,
                    'total_return_pct': ((self.current_balance - self.initial_balance) / self.initial_balance) * 100
                },
                'trades': {
                    'total': len(self.trades),
                    'winning': len([t for t in self.trades if t['pnl'] > 0]),
                    'losing': len([t for t in self.trades if t['pnl'] < 0])
                },
                'generated_at': datetime.now().isoformat()
            }
            
            with open('backtest_summary.json', 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                
            logger.info("💾 Отчет сохранен: backtest_trades.csv, backtest_summary.json")
            
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения отчета: {e}")

def main():
    """Главная функция"""
    logger.info("🚀 ЗАПУСК БЭКТЕСТЕРА ПРИБЫЛЬНОСТИ")
    
    # Топ торговые пары для тестирования
    test_symbols = [
        'BTC/USDT',
        'ETH/USDT', 
        'ADA/USDT',
        'SOL/USDT',
        'XRP/USDT',
        'BNB/USDT',
        'DOGE/USDT',
        'AVAX/USDT',
        'DOT/USDT',
        'LINK/USDT'
    ]
    
    # Создаем и запускаем бэктестер
    backtester = ProfitabilityBacktester()
    backtester.run_backtest(test_symbols)

if __name__ == "__main__":
    main()

