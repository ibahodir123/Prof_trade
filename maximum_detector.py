#!/usr/bin/env python3
"""
📈 ДЕТЕКТОР МАКСИМУМОВ - ЗЕРКАЛЬНАЯ ЛОГИКА
==========================================

Зеркальная версия детектора минимумов для восходящих коррекций.
Использует те же 4 EMA20 критерия, но с обратной логикой.
"""

import json
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

class MaximumDetector:
    def __init__(self):
        """Инициализация детектора максимумов"""
        
        # 🎯 ЗЕРКАЛЬНЫЕ ВЕСА (те же что для минимумов!)
        self.trained_weights = {
            'priceVelocity': 0.037,     # 3.7% - скорость цены
            'ema20Velocity': 0.031,     # 3.1% - скорость EMA20  
            'ema20Angle': 0.217,        # 21.7% - угол EMA20
            'priceDistance': 0.715      # 71.5% - расстояние до EMA20
        }
        
        # 📊 ТОРГОВЫЕ ПАРАМЕТРЫ
        self.confidence_threshold = 0.25  # 25% порог для выхода
        self.take_profit = 0.03         # 3% тейк-профит
        self.stop_loss = 0.02           # 2% стоп-лосс
        self.commission = 0.001         # 0.1% комиссия
        
        # 📈 СТАТИСТИКА
        self.trades = []
        self.total_profit = 0
        self.win_rate = 0
        self.max_drawdown = 0
        
        print("📈 Детектор максимумов инициализирован")
        print(f"⚖️ Зеркальные веса: Distance={self.trained_weights['priceDistance']:.1%}")

    def load_correction_data(self):
        """Загрузка данных восходящих коррекций для тестирования"""
        
        # 📊 РЕАЛЬНЫЕ КОРРЕКЦИИ ДЛЯ ТЕСТИРОВАНИЯ
        correction_impulses = [
            {
                'symbol': 'BTC/USDT',
                'date': '2025-01-16',
                'correction_percent': 5.2,
                'trajectory': [
                    {'price': 38400, 'ema20': 40200, 'time': 0},   # Начало коррекции
                    {'price': 39500, 'ema20': 40500, 'time': 1},   # Рост
                    {'price': 40100, 'ema20': 40800, 'time': 2},   # Усиление
                    {'price': 40800, 'ema20': 41100, 'time': 3},   # Ускорение
                    {'price': 41200, 'ema20': 41250, 'time': 4},   # МАКСИМУМ!
                    {'price': 40900, 'ema20': 41300, 'time': 5},   # Откат
                    {'price': 40500, 'ema20': 41350, 'time': 6},   # Падение
                    {'price': 40200, 'ema20': 41400, 'time': 7}    # Продолжение падения
                ]
            },
            {
                'symbol': 'ETH/USDT', 
                'date': '2025-01-23',
                'correction_percent': 4.1,
                'trajectory': [
                    {'price': 2620, 'ema20': 2730, 'time': 0},
                    {'price': 2680, 'ema20': 2740, 'time': 1},
                    {'price': 2720, 'ema20': 2750, 'time': 2}, 
                    {'price': 2750, 'ema20': 2760, 'time': 3},   # МАКСИМУМ!
                    {'price': 2730, 'ema20': 2770, 'time': 4},
                    {'price': 2700, 'ema20': 2780, 'time': 5}
                ]
            },
            {
                'symbol': 'SOL/USDT',
                'date': '2025-01-29', 
                'correction_percent': 7.8,
                'trajectory': [
                    {'price': 160, 'ema20': 170, 'time': 0},
                    {'price': 168, 'ema20': 172, 'time': 1},
                    {'price': 175, 'ema20': 174, 'time': 2},
                    {'price': 180, 'ema20': 176, 'time': 3},
                    {'price': 182, 'ema20': 178, 'time': 4},   # МАКСИМУМ!
                    {'price': 179, 'ema20': 180, 'time': 5},
                    {'price': 175, 'ema20': 181, 'time': 6}
                ]
            }
        ]
        
        return correction_impulses

    def calculate_features(self, current_point, previous_point):
        """Расчет 4 EMA20 критериев (зеркальная логика)"""
        
        if not previous_point:
            return [0, 0, 0, 0]
            
        # 📈 1. Скорость цены (ЗЕРКАЛЬНО: положительная = рост)
        price_velocity = (current_point['price'] - previous_point['price']) / previous_point['price']
        
        # 📊 2. Скорость EMA20 (ЗЕРКАЛЬНО: положительная = рост)
        ema20_velocity = (current_point['ema20'] - previous_point['ema20']) / previous_point['ema20']
        
        # 📐 3. Угол EMA20 (ЗЕРКАЛЬНО: положительный = вверх)
        ema20_angle = ema20_velocity * 100  # Преобразуем в угол
        
        # 📏 4. Расстояние до EMA20 (ЗЕРКАЛЬНО: положительное = выше EMA20)
        price_distance = (current_point['price'] - current_point['ema20']) / current_point['ema20']
        
        return [price_velocity, ema20_velocity, ema20_angle, price_distance]

    def predict_maximum(self, features):
        """Предсказание максимума на основе зеркальной модели"""
        
        # ⚖️ Взвешенный расчет (ТЕ ЖЕ ВЕСА!)
        weighted_score = (
            abs(features[0]) * self.trained_weights['priceVelocity'] +     # Скорость цены
            abs(features[1]) * self.trained_weights['ema20Velocity'] +     # Скорость EMA20
            abs(features[2]) * self.trained_weights['ema20Angle'] +        # Угол EMA20  
            abs(features[3]) * self.trained_weights['priceDistance']       # Расстояние
        )
        
        # 🎯 Преобразование в уверенность (0-1)
        confidence = min(1.0, weighted_score * 10.0)  # Увеличиваем чувствительность!
        
        # 🚀 ЗЕРКАЛЬНЫЕ БОНУСЫ для сильных сигналов
        if abs(features[3]) > 0.05:  # Большое расстояние ВЫШЕ EMA20
            confidence += 0.1
        if features[0] > 0.02 and features[1] > 0.01:  # Сильный РОСТ
            confidence += 0.1
            
        return min(1.0, confidence)

    def simulate_trading(self, corrections):
        """Симуляция торговли с детектором максимумов"""
        
        print("\n📈 НАЧИНАЕМ СИМУЛЯЦИЮ ТОРГОВЛИ (ЗЕРКАЛЬНАЯ ЛОГИКА)")
        print("=" * 60)
        
        for correction in corrections:
            print(f"\n📊 Анализ {correction['symbol']} ({correction['date']})")
            print(f"📈 Коррекция: +{correction['correction_percent']}%")
            
            trajectory = correction['trajectory']
            entry_price = None
            entry_time = None
            
            # 🔍 Ищем точку выхода (максимум) - ИСПРАВЛЕНО!
            max_confidence = 0
            best_exit_point = None
            
            for i in range(1, len(trajectory)):
                current = trajectory[i]
                previous = trajectory[i-1] if i > 0 else None
                
                # 📊 Рассчитываем признаки (зеркальная логика)
                features = self.calculate_features(current, previous)
                confidence = self.predict_maximum(features)
                
                print(f"  ⏰ Точка {i}: ${current['price']:.0f}, Уверенность: {confidence:.1%}")
                
                # 🎯 Ищем ЛУЧШИЙ момент выхода (САМЫЙ ВЫСОКИЙ!)
                if confidence >= self.confidence_threshold and current['price'] > max_confidence:
                    max_confidence = current['price']  # Сохраняем ЦЕНУ, а не уверенность!
                    best_exit_point = {'price': current['price'], 'time': i}
            
            # 🚀 Выходим в лучший момент
            if best_exit_point:
                entry_price = best_exit_point['price']
                entry_time = best_exit_point['time']
                print(f"  🔴 ВЫХОД ИЗ ЛОНГА! Цена: ${entry_price:.0f} (лучший момент!)")
                    
            # 📉 Симулируем результат сделки
            if entry_price:
                # Берем минимальную цену после выхода как потенциальный результат
                min_price_after_exit = min([p['price'] for p in trajectory[entry_time:]])
                
                # 🎯 Рассчитываем прибыль (ЗЕРКАЛЬНО!)
                profit_percent = (entry_price - min_price_after_exit) / entry_price
                profit_with_commission = profit_percent - (2 * self.commission)  # Вход + выход
                
                # 📊 Записываем сделку
                trade = {
                    'symbol': correction['symbol'],
                    'entry_price': entry_price,
                    'exit_price': min_price_after_exit,
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
        print("📈 ИТОГОВАЯ СТАТИСТИКА ДЕТЕКТОРА МАКСИМУМОВ")
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
        plt.savefig('maximum_detector_profitability.png', dpi=150, bbox_inches='tight')
        print("📊 График сохранен: maximum_detector_profitability.png")
        
    def run_analysis(self):
        """Запуск полного анализа прибыльности"""
        
        print("🚀 ЗАПУСК АНАЛИЗА ДЕТЕКТОРА МАКСИМУМОВ (ЗЕРКАЛЬНАЯ ЛОГИКА)")
        print("=" * 70)
        
        # 📊 Загружаем данные
        corrections = self.load_correction_data()
        
        # 🎯 Симулируем торговлю
        self.simulate_trading(corrections)
        
        # 📊 Считаем статистику
        self.calculate_statistics()
        
        # 📈 Создаем график
        self.create_profit_chart()
        
        print("\n🎉 АНАЛИЗ ЗАВЕРШЕН!")

if __name__ == "__main__":
    detector = MaximumDetector()
    detector.run_analysis()
