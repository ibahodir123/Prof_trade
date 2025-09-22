#!/usr/bin/env python3
"""
АНАЛИЗАТОР ЗАКОНОМЕРНОСТЕЙ
Ищет паттерны в 809 движениях min→max
"""

import json
import numpy as np
from typing import Dict, List

class FastPatternAnalyzer:
    def __init__(self):
        self.movements = []
        
    def load_data(self):
        """Быстрая загрузка данных"""
        try:
            with open('movements_database.json', 'r') as f:
                self.movements = json.load(f)
            print(f"📊 Загружено {len(self.movements)} движений")
            return True
        except Exception as e:
            print(f"❌ Ошибка: {e}")
            return False
    
    def analyze_fast(self):
        """Быстрый анализ паттернов"""
        print("\n🔍 БЫСТРЫЙ АНАЛИЗ ПАТТЕРНОВ")
        print("=" * 30)
        
        # Группы по прибыльности
        small = [m for m in self.movements if m['movement_percent'] < 3]
        medium = [m for m in self.movements if 3 <= m['movement_percent'] < 7]
        large = [m for m in self.movements if m['movement_percent'] >= 7]
        
        print(f"💰 Малые (1-3%): {len(small)}")
        print(f"📈 Средние (3-7%): {len(medium)}")
        print(f"🚀 Крупные (7%+): {len(large)}")
        
        # Анализ средней группы (самой стабильной)
        if medium:
            self._analyze_group(medium, "СРЕДНИЕ")
        
        # Анализ крупных движений
        if large:
            self._analyze_group(large, "КРУПНЫЕ")
    
    def _analyze_group(self, group, name):
        """Анализ группы движений"""
        print(f"\n🎯 ГРУППА {name} ({len(group)} движений):")
        
        # Собираем признаки MIN точек
        velocities = []
        angles_20 = []
        angles_50 = []
        sync = []
        
        for m in group:
            min_f = m['min_features']
            
            # Скорость цены
            if 'velocities' in min_f and 'price' in min_f['velocities']:
                velocities.append(min_f['velocities']['price'])
            
            # Углы EMA
            if 'angles' in min_f:
                if 'ema20' in min_f['angles']:
                    angles_20.append(min_f['angles']['ema20'])
                if 'ema50' in min_f['angles']:
                    angles_50.append(min_f['angles']['ema50'])
            
            # Синхронизация
            if 'synchronizations' in min_f and 'price_ema20' in min_f['synchronizations']:
                sync.append(min_f['synchronizations']['price_ema20'])
        
        # Выводим статистики
        if velocities:
            avg_vel = np.mean(velocities)
            std_vel = np.std(velocities)
            print(f"  ⚡ Скорость цены: {avg_vel:.6f} ± {std_vel:.6f}")
        
        if angles_20:
            avg_a20 = np.mean(angles_20)
            std_a20 = np.std(angles_20)
            print(f"  📐 Угол EMA20: {avg_a20:.1f}° ± {std_a20:.1f}°")
        
        if angles_50:
            avg_a50 = np.mean(angles_50)
            std_a50 = np.std(angles_50)
            print(f"  📐 Угол EMA50: {avg_a50:.1f}° ± {std_a50:.1f}°")
        
        if sync:
            avg_sync = np.mean(sync)
            std_sync = np.std(sync)
            print(f"  🔗 Синхронизация: {avg_sync:.3f} ± {std_sync:.3f}")
    
    def create_rules(self):
        """Создание торговых правил"""
        print("\n🎯 ТОРГОВЫЕ ПРАВИЛА:")
        print("-" * 25)
        
        # Берем средние движения как эталон
        medium = [m for m in self.movements if 3 <= m['movement_percent'] < 7]
        
        if not medium:
            print("❌ Нет данных для правил")
            return
        
        # Собираем признаки
        velocities = []
        angles_20 = []
        sync = []
        
        for m in medium:
            min_f = m['min_features']
            
            if 'velocities' in min_f and 'price' in min_f['velocities']:
                velocities.append(min_f['velocities']['price'])
            
            if 'angles' in min_f and 'ema20' in min_f['angles']:
                angles_20.append(min_f['angles']['ema20'])
            
            if 'synchronizations' in min_f and 'price_ema20' in min_f['synchronizations']:
                sync.append(min_f['synchronizations']['price_ema20'])
        
        print("📈 СИГНАЛЫ ВХОДА (для 3-7% прибыли):")
        
        if velocities:
            avg = np.mean(velocities)
            std = np.std(velocities)
            print(f"   • Скорость цены: {avg:.6f} ± {std:.6f}")
        
        if angles_20:
            avg = np.mean(angles_20)
            std = np.std(angles_20)
            print(f"   • Угол EMA20: {avg:.1f}° ± {std:.1f}°")
        
        if sync:
            avg = np.mean(sync)
            std = np.std(sync)
            print(f"   • Синхронизация: {avg:.3f} ± {std:.3f}")

if __name__ == "__main__":
    analyzer = FastPatternAnalyzer()
    
    if analyzer.load_data():
        analyzer.analyze_fast()
        analyzer.create_rules()
        print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
    else:
        print("❌ Ошибка загрузки данных")
