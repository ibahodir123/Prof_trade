import time
import psutil
import platform
from datetime import datetime

def test_cpu():
    print("🧮 ТЕСТИРОВАНИЕ CPU...")
    start_time = time.time()
    count = 0
    for i in range(1000000):
        count += i * i
    end_time = time.time()
    print(f"Вычисления: {end_time - start_time:.2f} сек")
    return end_time - start_time

def test_memory():
    print("🧠 ТЕСТИРОВАНИЕ ПАМЯТИ...")
    # Получаем информацию о памяти
    memory = psutil.virtual_memory()
    print(f"Общая память: {memory.total / (1024**3):.1f} GB")
    print(f"Доступно: {memory.available / (1024**3):.1f} GB")
    print(f"Использовано: {memory.percent:.1f}%")

def system_info():
    print("🖥️ ИНФОРМАЦИЯ О СИСТЕМЕ:")
    print(f"ОС: {platform.system()} {platform.version()}")
    print(f"Процессор: {platform.processor()}")
    print(f"Ядра CPU: {psutil.cpu_count()}")
    if psutil.cpu_freq():
        print(f"Частота CPU: {psutil.cpu_freq().current:.0f} MHz")
    else:
        print("Частота CPU: Неизвестна")

if __name__ == "__main__":
    print("🚀 ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ WINDOWS")
    print("=" * 50)
    
    system_info()
    print()
    
    cpu_time = test_cpu()
    print()
    
    test_memory()
    print()
    
    print("📊 РЕЗУЛЬТАТЫ:")
    print(f"• CPU производительность: {1000/cpu_time:.0f} операций/сек")
    if cpu_time < 5:
        print("• Система готова для разработки!")
    else:
        print("• CPU может быть медленным для тяжелых задач")