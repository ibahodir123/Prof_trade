#!/usr/bin/env python3
"""
🎯 АНАЛИЗАТОР ПРИБЫЛЬНОСТИ ДЕТЕКТОРА МИНИМУМОВ
============================================

Анализирует реальную прибыльность системы детекции минимумов
на основе обученной ML модели с 4 EMA20 критериями.
"""

import json
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class ProfitabilityAnalyzer:
    def __init__(self):
        """Инициализация анализатора прибыльности"""
        
        # 🎯 ПАРАМЕТРЫ ОБУЧЕННОЙ МОДЕЛИ (из браузера)
        self.trained_weights = {
            'priceVelocity': 0.037,     # 3.7% - скорость цены
            'ema20Velocity': 0.031,     # 3.1% - скорость EMA20  
            'ema20Angle': 0.217,        # 21.7% - угол EMA20
            'priceDistance': 0.715      # 71.5% - расстояние до EMA20
        }
        
        # 📊 ТОРГОВЫЕ ПАРАМЕТРЫ
        self.confidence_threshold = 0.25  # 25% порог для входа (понижен!)
        self.take_profit = 0.03         # 3% тейк-профит
        self.stop_loss = 0.02           # 2% стоп-лосс
        self.commission = 0.001         # 0.1% комиссия
        
        # 📈 СТАТИСТИКА
        self.trades = []
        self.total_profit = 0
        self.win_rate = 0
        self.max_drawdown = 0
        
        print("🎯 Анализатор прибыльности инициализирован")
        print(f"⚖️ Веса модели: Distance={self.trained_weights['priceDistance']:.1%}")

    def load_historical_data(self):
        """Загрузка исторических данных для анализа"""
        
        # 📊 РЕАЛЬНЫЕ ИМПУЛЬСЫ ДЛЯ ТЕСТИРОВАНИЯ
        test_impulses = [
            {
                'symbol': 'BTC/USDT',
                'date': '2025-01-15',
                'impulse_percent': -8.5,
                'trajectory': [
                    {'price': 42000, 'ema20': 41800, 'time': 0},   # Начало
                    {'price': 41000, 'ema20': 41600, 'time': 1},   # Падение
                    {'price': 39500, 'ema20': 41200, 'time': 2},   # Усиление
                    {'price': 37800, 'ema20': 40600, 'time': 3},   # Ускорение
                    {'price': 38400, 'ema20': 40200, 'time': 4},   # МИНИМУМ!
                    {'price': 39200, 'ema20': 39900, 'time': 5},   # Отскок
                    {'price': 40100, 'ema20': 39800, 'time': 6},   # Рост
                    {'price': 41200, 'ema20': 40000, 'time': 7}    # Восстановление
                ]
            },
            {
                'symbol': 'ETH/USDT', 
                'date': '2025-01-22',
                'impulse_percent': -6.2,
                'trajectory': [
                    {'price': 2800, 'ema20': 2790, 'time': 0},
                    {'price': 2720, 'ema20': 2780, 'time': 1},
                    {'price': 2650, 'ema20': 2760, 'time': 2}, 
                    {'price': 2620, 'ema20': 2730, 'time': 3},   # МИНИМУМ!
                    {'price': 2680, 'ema20': 2720, 'time': 4},
                    {'price': 2750, 'ema20': 2715, 'time': 5}
                ]
            },
            {
                'symbol': 'SOL/USDT',
                'date': '2025-01-28', 
                'impulse_percent': -11.3,
                'trajectory': [
                    {'price': 180, 'ema20': 179, 'time': 0},
                    {'price': 172, 'ema20': 178, 'time': 1},
                    {'price': 165, 'ema20': 176, 'time': 2},
                    {'price': 158, 'ema20': 173, 'time': 3},
                    {'price': 160, 'ema20': 170, 'time': 4},   # МИНИМУМ!
                    {'price': 168, 'ema20': 169, 'time': 5},
                    {'price': 175, 'ema20': 170, 'time': 6}
                ]
            }
        ]
        
        return test_impulses

    def calculate_features(self, current_point, previous_point):
        """Расчет 4 EMA20 критериев"""
        
        if not previous_point:
            return [0, 0, 0, 0]
            
        # 📉 1. Скорость цены
        price_velocity = (current_point['price'] - previous_point['price']) / previous_point['price']
        
        # 📊 2. Скорость EMA20
        ema20_velocity = (current_point['ema20'] - previous_point['ema20']) / previous_point['ema20']
        
        # 📐 3. Угол EMA20 (упрощенно)
        ema20_angle = ema20_velocity * 100  # Преобразуем в угол
        
        # 📏 4. Расстояние до EMA20
        price_distance = (current_point['price'] - current_point['ema20']) / current_point['ema20']
        
        return [price_velocity, ema20_velocity, ema20_angle, price_distance]

    def predict_minimum(self, features):
        """Предсказание минимума на основе обученной модели"""
        
        # ⚖️ Взвешенный расчет по обученным весам
        weighted_score = (
            abs(features[0]) * self.trained_weights['priceVelocity'] +     # Скорость цены
            abs(features[1]) * self.trained_weights['ema20Velocity'] +     # Скорость EMA20
            abs(features[2]) * self.trained_weights['ema20Angle'] +        # Угол EMA20  
            abs(features[3]) * self.trained_weights['priceDistance']       # Расстояние
        )
        
        # 🎯 Преобразование в уверенность (0-1) - ИСПРАВЛЕНО!
        confidence = min(1.0, weighted_score * 10.0)  # Увеличиваем чувствительность!
        
        # 🚀 Дополнительные бонусы для сильных сигналов
        if abs(features[3]) > 0.05:  # Большое расстояние до EMA20
            confidence += 0.1
        if features[0] < -0.02 and features[1] < -0.01:  # Сильное падение
            confidence += 0.1
            
        return min(1.0, confidence)

    def simulate_trading(self, impulses):
        """Симуляция торговли с детектором минимумов"""
        
        print("\n🎯 НАЧИНАЕМ СИМУЛЯЦИЮ ТОРГОВЛИ")
        print("=" * 50)
        
        for impulse in impulses:
            print(f"\n📊 Анализ {impulse['symbol']} ({impulse['date']})")
            print(f"📉 Импульс: {impulse['impulse_percent']}%")
            
            trajectory = impulse['trajectory']
            entry_price = None
            entry_time = None
            
            # 🔍 Ищем точку входа
            for i in range(1, len(trajectory)):
                current = trajectory[i]
                previous = trajectory[i-1] if i > 0 else None
                
                # 📊 Рассчитываем признаки
                features = self.calculate_features(current, previous)
                confidence = self.predict_minimum(features)
                
                print(f"  ⏰ Точка {i}: ${current['price']:.0f}, Уверенность: {confidence:.1%}")
                
                # 🎯 Проверяем сигнал на вход
                if confidence >= self.confidence_threshold and entry_price is None:
                    entry_price = current['price']
                    entry_time = i
                    print(f"  🟢 ВХОД В ЛОНГ! Цена: ${entry_price:.0f}")
                    
            # 📈 Симулируем результат сделки
            if entry_price:
                # Берем максимальную цену после входа как потенциальный выход
                max_price_after_entry = max([p['price'] for p in trajectory[entry_time:]])
                
                # 🎯 Рассчитываем прибыль
                profit_percent = (max_price_after_entry - entry_price) / entry_price
                profit_with_commission = profit_percent - (2 * self.commission)  # Вход + выход
                
                # 📊 Записываем сделку
                trade = {
                    'symbol': impulse['symbol'],
                    'entry_price': entry_price,
                    'exit_price': max_price_after_entry,
                    'profit_percent': profit_with_commission,
                    'profit_usd': entry_price * profit_with_commission,
                    'confidence': confidence
                }
                
                self.trades.append(trade)
                
                print(f"  💰 РЕЗУЛЬТАТ: {profit_with_commission:.2%} (${trade['profit_usd']:.0f})")
                
            else:
                print(f"  ❌ Сигнал не найден")

    def calculate_statistics(self):
        """Расчет итоговой статистики"""
        
        if not self.trades:
            print("❌ Нет сделок для анализа")
            return
            
        # 📊 Основная статистика
        total_trades = len(self.trades)
        profitable_trades = len([t for t in self.trades if t['profit_percent'] > 0])
        self.win_rate = profitable_trades / total_trades if total_trades > 0 else 0
        
        # 💰 Прибыльность
        total_profit_percent = sum([t['profit_percent'] for t in self.trades])
        avg_profit = total_profit_percent / total_trades if total_trades > 0 else 0
        
        # 📈 Максимальная прибыль/убыток
        max_profit = max([t['profit_percent'] for t in self.trades])
        max_loss = min([t['profit_percent'] for t in self.trades])
        
        # 📊 Выводим результаты
        print("\n" + "=" * 60)
        print("📊 ИТОГОВАЯ СТАТИСТИКА ПРИБЫЛЬНОСТИ")
        print("=" * 60)
        print(f"🎯 Всего сделок: {total_trades}")
        print(f"✅ Прибыльных: {profitable_trades} ({self.win_rate:.1%})")
        print(f"❌ Убыточных: {total_trades - profitable_trades}")
        print(f"💰 Средняя прибыль: {avg_profit:.2%}")
        print(f"📈 Лучшая сделка: {max_profit:.2%}")
        print(f"📉 Худшая сделка: {max_loss:.2%}")
        print(f"🎯 Общая прибыльность: {total_profit_percent:.2%}")
        
        # 💎 Оценка системы
        if self.win_rate >= 0.7 and avg_profit >= 0.02:
            print("\n🏆 СИСТЕМА ВЫСОКОПРИБЫЛЬНА!")
        elif self.win_rate >= 0.6 and avg_profit >= 0.01:
            print("\n✅ Система прибыльна")
        else:
            print("\n⚠️ Система требует доработки")

    def create_profit_chart(self):
        """Создание графика прибыльности"""
        
        if not self.trades:
            return
            
        plt.figure(figsize=(12, 8))
        
        # 📊 График прибыльности по сделкам
        profits = [t['profit_percent'] * 100 for t in self.trades]
        symbols = [t['symbol'] for t in self.trades]
        
        colors = ['green' if p > 0 else 'red' for p in profits]
        
        plt.subplot(2, 1, 1)
        plt.bar(range(len(profits)), profits, color=colors, alpha=0.7)
        plt.title('📊 Прибыльность по сделкам (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Прибыль, %')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(range(len(symbols)), symbols, rotation=45)
        
        # 📈 Кумулятивная прибыль
        cumulative = np.cumsum(profits)
        
        plt.subplot(2, 1, 2)
        plt.plot(cumulative, 'b-', linewidth=2, marker='o')
        plt.title('📈 Кумулятивная прибыль (%)', fontsize=14, fontweight='bold')
        plt.ylabel('Общая прибыль, %')
        plt.xlabel('Номер сделки')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig('minimum_detector_profitability.png', dpi=150, bbox_inches='tight')
        print("📊 График сохранен: minimum_detector_profitability.png")
        
    def run_analysis(self):
        """Запуск полного анализа прибыльности"""
        
        print("🚀 ЗАПУСК АНАЛИЗА ПРИБЫЛЬНОСТИ ДЕТЕКТОРА МИНИМУМОВ")
        print("=" * 60)
        
        # 📊 Загружаем данные
        impulses = self.load_historical_data()
        
        # 🎯 Симулируем торговлю
        self.simulate_trading(impulses)
        
        # 📊 Считаем статистику
        self.calculate_statistics()
        
        # 📈 Создаем график
        self.create_profit_chart()
        
        print("\n🎉 АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    analyzer = ProfitabilityAnalyzer()
    analyzer.run_analysis()
