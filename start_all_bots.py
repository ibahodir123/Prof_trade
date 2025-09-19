#!/usr/bin/env python3
"""
Скрипт для запуска всех ботов
"""

import subprocess
import time
import os
import sys

def start_bot(script_name, log_file):
    """Запускает бота в фоновом режиме"""
    try:
        print(f"🚀 Запускаю {script_name}...")
        
        # Запускаем бота в фоне
        process = subprocess.Popen([
            sys.executable, script_name
        ], stdout=open(log_file, 'w'), stderr=subprocess.STDOUT)
        
        print(f"✅ {script_name} запущен (PID: {process.pid})")
        return process
        
    except Exception as e:
        print(f"❌ Ошибка запуска {script_name}: {e}")
        return None

def main():
    """Основная функция"""
    print("🤖 Запуск всех ботов...")
    
    # Останавливаем существующие процессы
    try:
        subprocess.run(["pkill", "-f", "python"], check=False)
        print("🛑 Остановлены существующие процессы")
    except:
        pass
    
    # Запускаем ботов
    processes = []
    
    # Основной бот
    main_bot = start_bot("ml_bot_binance.py", "main_bot.log")
    if main_bot:
        processes.append(main_bot)
    
    time.sleep(2)  # Пауза между запусками
    
    # Автосигналы
    auto_bot = start_bot("auto_signals_bot.py", "auto_signals.log")
    if auto_bot:
        processes.append(auto_bot)
    
    print(f"\n🎉 Запущено {len(processes)} ботов!")
    print("📊 Статус процессов:")
    
    for i, process in enumerate(processes, 1):
        if process and process.poll() is None:
            print(f"  {i}. PID {process.pid} - Активен")
        else:
            print(f"  {i}. Неактивен")
    
    print("\n📝 Логи:")
    print("  - main_bot.log - Основной бот")
    print("  - auto_signals.log - Автосигналы")
    
    print("\n🔍 Для проверки статуса:")
    print("  tail -f main_bot.log")
    print("  tail -f auto_signals.log")
    
    return processes

if __name__ == "__main__":
    processes = main()
    
    # Ждем завершения
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Остановка ботов...")
        for process in processes:
            if process:
                process.terminate()
        print("✅ Все боты остановлены")
