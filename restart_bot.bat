@echo off
echo Остановка всех процессов Python...
taskkill /f /im python.exe 2>nul
echo Ожидание 5 секунд...
timeout /t 5 /nobreak >nul
echo Запуск ML Trading Bot...
cd /d D:\Best_trade
python ml_bot_auto.py
pause

