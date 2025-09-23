#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🧹 ОЧИСТКА ПРОЕКТА ОТ НЕНУЖНЫХ ФАЙЛОВ
Удаление временных файлов, дубликатов и результатов тестирования
"""

import os
import shutil
from datetime import datetime

class ProjectCleaner:
    def __init__(self):
        print("🧹 ОЧИСТКА ПРОЕКТА ОТ НЕНУЖНЫХ ФАЙЛОВ")
        print("=" * 45)
        
        # Файлы для удаления (временные, тестовые, дубликаты)
        self.files_to_delete = [
            # Временные файлы Python
            'python',
            '__pycache__',
            
            # Демо и тестовые файлы
            'demo_minimum_trainer.py',
            'demo_results.json',
            'demo_minimums_analysis.png',
            'demo_top_minimums.png',
            'DEMO_README.md',
            'run_demo.py',
            
            # Тестовые файлы
            'test_adaptive_learning.py',
            'test_contacts_button.py',
            'test_ml_models.py',
            'test_multilang.py',
            'test_signal_analysis.py',
            'test_uzbek_translations.py',
            
            # Временные результаты анализа
            'drawdown_analyzer.py',
            'drawdown_optimization_demo.py',
            'real_optimized_tester.py',
            'timeframe_analyzer.py',
            
            # Старые версии ботов
            'ml_bot_binance_backup.py',
            'ml_bot_binance_backup_20250920_124640.py',
            'ml_bot_binance_fixed.py',
            
            # Временные модели (оставляем только финальные)
            'clean_maximum_features_20250923_054703.pkl',
            'clean_maximum_model_20250923_054703.pkl',
            'fast_maximum_features_20250923_062911.pkl',
            'fast_maximum_model_20250923_062911.pkl',
            'fast_maximum_scaler_20250923_062911.pkl',
            'fast_minimum_features_20250923_062911.pkl',
            'fast_minimum_model_20250923_062911.pkl',
            'fast_minimum_scaler_20250923_062911.pkl',
            'historical_maximum_features_20250923_062225.pkl',
            'historical_maximum_model_20250923_062225.pkl',
            'historical_maximum_scaler_20250923_062225.pkl',
            'historical_minimum_features_20250923_062225.pkl',
            'historical_minimum_model_20250923_062225.pkl',
            'historical_minimum_scaler_20250923_062225.pkl',
            'practical_features_20250923_053051.pkl',
            'practical_model_20250923_053051.pkl',
            'pure_ema_maximum_features_20250923_055329.pkl',
            'pure_ema_maximum_model_20250923_055329.pkl',
            'real_maximums_20250923_054159.json',
            'real_maximum_features_20250923_054159.pkl',
            'real_maximum_model_20250923_054159.pkl',
            
            # Временные метаданные
            'fast_training_metadata_20250923_062911.json',
            'historical_training_metadata_20250923_062225.json',
            'historical_training_metadata_20250923_062633.json',
            'practical_minimums_20250923_053051.json',
            
            # Временные изображения
            'demo_minimums_analysis.png',
            'demo_top_minimums.png',
            'maximum_detector_profitability.png',
            'minimum_detector_profitability.png',
            'real_market_analysis.png',
            'sideways_bollinger_analysis.png',
            'sideways_detector_analysis.png',
            'sideways_profitability_analysis.png',
            
            # Временные отчеты
            'backtest_2025_report_20250923_063851.md',
            'BACKTEST_2025_SUMMARY.md',
            'CLEAN_MAXIMUM_RESULTS.md',
            'DRAWDOWN_OPTIMIZATION_REPORT.md',
            'HISTORICAL_TRAINING_RESULTS.md',
            'MAXIMUM_ANALYSIS_RESULTS.md',
            'PURE_EMA_MAXIMUM_RESULTS.md',
            'REAL_ANALYSIS_RESULTS.md',
            'REAL_BACKTEST_SUMMARY.md',
            'SAVE_RESULTS.md',
            'VALIDATION_RESULTS.md',
            
            # Временные результаты
            'backtest_2025_results_20250923_063851.json',
            'real_backtest_results_20250923_064511.json',
            
            # Пустые файлы
            'backtest_results.log',
            
            # Документы на других языках (оставляем только основные)
            'Инструкция.docx',
            'Кулланма.docx',
        ]
        
        # Папки для удаления
        self.folders_to_delete = [
            '__pycache__',
        ]
        
        # Файлы для сохранения (основные)
        self.essential_files = [
            'ml_bot_binance.py',  # Основной бот
            'advanced_ema_analyzer.py',
            'advanced_ml_trainer.py',
            'shooting_star_predictor.py',
            'bot_config.json',
            'requirements.txt',
            'ML_SIGNALS_GUIDE.md',
            'historical_models/',  # Папка с финальными моделями
            'models/',  # Папка с моделями
            'data_batch_10.json',  # Данные
            'movements_database.json',
            'patterns_database.json',
        ]
    
    def analyze_files(self):
        """Анализ файлов для удаления"""
        print("\n📊 АНАЛИЗ ФАЙЛОВ:")
        print("-" * 20)
        
        total_size = 0
        files_found = 0
        
        for file_path in self.files_to_delete:
            if os.path.exists(file_path):
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    total_size += size
                    files_found += 1
                    print(f"🗑️  {file_path:<50} ({size:,} bytes)")
                elif os.path.isdir(file_path):
                    # Подсчитываем размер папки
                    folder_size = self.get_folder_size(file_path)
                    total_size += folder_size
                    files_found += 1
                    print(f"🗑️  {file_path:<50} (папка, {folder_size:,} bytes)")
        
        print(f"\n📊 ИТОГО НАЙДЕНО:")
        print(f"• Файлов/папок: {files_found}")
        print(f"• Общий размер: {total_size:,} bytes ({total_size / 1024 / 1024:.1f} MB)")
        
        return files_found, total_size
    
    def get_folder_size(self, folder_path):
        """Подсчет размера папки"""
        total_size = 0
        try:
            for dirpath, dirnames, filenames in os.walk(folder_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
        except Exception:
            pass
        return total_size
    
    def create_backup(self):
        """Создание резервной копии перед удалением"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_folder = f"backup_before_cleanup_{timestamp}"
        
        print(f"\n💾 СОЗДАНИЕ РЕЗЕРВНОЙ КОПИИ: {backup_folder}")
        
        try:
            os.makedirs(backup_folder, exist_ok=True)
            
            # Копируем важные файлы
            for file_path in self.essential_files:
                if os.path.exists(file_path):
                    if os.path.isfile(file_path):
                        shutil.copy2(file_path, backup_folder)
                    elif os.path.isdir(file_path):
                        shutil.copytree(file_path, os.path.join(backup_folder, file_path))
            
            print(f"✅ Резервная копия создана: {backup_folder}")
            return backup_folder
        except Exception as e:
            print(f"❌ Ошибка создания резервной копии: {e}")
            return None
    
    def clean_files(self, create_backup=True):
        """Очистка файлов"""
        if create_backup:
            backup_folder = self.create_backup()
            if not backup_folder:
                print("❌ Не удалось создать резервную копию. Очистка отменена.")
                return False
        
        print(f"\n🧹 НАЧИНАЕМ ОЧИСТКУ:")
        print("-" * 25)
        
        deleted_count = 0
        deleted_size = 0
        
        for file_path in self.files_to_delete:
            if os.path.exists(file_path):
                try:
                    if os.path.isfile(file_path):
                        size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_count += 1
                        deleted_size += size
                        print(f"✅ Удален файл: {file_path}")
                    elif os.path.isdir(file_path):
                        size = self.get_folder_size(file_path)
                        shutil.rmtree(file_path)
                        deleted_count += 1
                        deleted_size += size
                        print(f"✅ Удалена папка: {file_path}")
                except Exception as e:
                    print(f"❌ Ошибка удаления {file_path}: {e}")
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ОЧИСТКИ:")
        print(f"• Удалено файлов/папок: {deleted_count}")
        print(f"• Освобождено места: {deleted_size:,} bytes ({deleted_size / 1024 / 1024:.1f} MB)")
        
        return True
    
    def show_remaining_files(self):
        """Показать оставшиеся файлы"""
        print(f"\n📁 ОСТАВШИЕСЯ ФАЙЛЫ:")
        print("-" * 25)
        
        remaining_files = []
        for item in os.listdir('.'):
            if os.path.isfile(item):
                size = os.path.getsize(item)
                remaining_files.append((item, size))
            elif os.path.isdir(item) and not item.startswith('.'):
                size = self.get_folder_size(item)
                remaining_files.append((item, size))
        
        # Сортируем по размеру
        remaining_files.sort(key=lambda x: x[1], reverse=True)
        
        for file_name, size in remaining_files:
            if size > 1024 * 1024:  # Больше 1MB
                print(f"📄 {file_name:<40} ({size / 1024 / 1024:.1f} MB)")
            elif size > 1024:  # Больше 1KB
                print(f"📄 {file_name:<40} ({size / 1024:.1f} KB)")
            else:
                print(f"📄 {file_name:<40} ({size} bytes)")

def main():
    """Основная функция"""
    cleaner = ProjectCleaner()
    
    # Анализируем файлы
    files_count, total_size = cleaner.analyze_files()
    
    if files_count == 0:
        print("✅ Нет файлов для удаления!")
        return
    
    print(f"\n❓ Удалить {files_count} файлов/папок ({total_size / 1024 / 1024:.1f} MB)?")
    print("Нажмите Enter для продолжения или Ctrl+C для отмены...")
    
    try:
        input()
    except KeyboardInterrupt:
        print("\n❌ Очистка отменена пользователем")
        return
    
    # Очищаем файлы
    if cleaner.clean_files():
        cleaner.show_remaining_files()
        print(f"\n✅ ОЧИСТКА ЗАВЕРШЕНА!")
        print("🎯 Проект готов для интеграции в Telegram бот!")
    else:
        print("❌ Очистка не удалась")

if __name__ == "__main__":
    main()
