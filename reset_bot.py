#!/usr/bin/env python3
"""
Скрипт для сброса Telegram webhook и очистки конфликтов
"""
import requests
import json

# Токен бота из конфигурации
BOT_TOKEN = "8198379607:AAGnEX6zsp1T7mj_VYsm4AYyURu_TD-ivaM"

def reset_webhook():
    """Сброс webhook и очистка конфликтов"""
    try:
        print("🔄 Сбрасываю webhook...")
        
        # Сброс webhook
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/deleteWebhook"
        response = requests.post(url, data={"drop_pending_updates": True})
        
        if response.status_code == 200:
            print("✅ Webhook сброшен успешно")
        else:
            print(f"❌ Ошибка сброса webhook: {response.status_code}")
            print(f"Ответ: {response.text}")
        
        # Получение информации о боте
        print("📊 Получаю информацию о боте...")
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"
        response = requests.get(url)
        
        if response.status_code == 200:
            bot_info = response.json()
            print(f"✅ Бот: {bot_info['result']['first_name']} (@{bot_info['result']['username']})")
        else:
            print(f"❌ Ошибка получения информации о боте: {response.status_code}")
        
        # Получение обновлений (очистка очереди)
        print("🧹 Очищаю очередь обновлений...")
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
        response = requests.post(url, data={"offset": -1, "limit": 1})
        
        if response.status_code == 200:
            print("✅ Очередь обновлений очищена")
        else:
            print(f"❌ Ошибка очистки очереди: {response.status_code}")
        
        print("\n🚀 Теперь можно запускать бота!")
        
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")

if __name__ == "__main__":
    reset_webhook()




