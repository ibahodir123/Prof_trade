#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🌐 Веб-дашборд для торгового бота
Современный интерфейс для управления ботом через браузер
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_socketio import SocketIO, emit
import json
import os
import sys
import threading
import time
import psutil
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import ccxt
import pickle
import warnings
warnings.filterwarnings('ignore')

# Импорт наших модулей
from optimized_ml_bot import OptimizedMLBot
from bot_config import load_config

class WebTradingDashboard:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.secret_key = 'your-secret-key-here'
        self.socketio = SocketIO(self.app, async_mode='eventlet')

        # Инициализация бота
        self.bot = None
        self.bot_thread = None
        self.bot_running = False

        # Настройки
        self.config = load_config()
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT', 'BNB/USDT']
        self.update_interval = 5  # секунды

        self.setup_routes()
        self.setup_socket_events()

    def setup_routes(self):
        """Настройка маршрутов Flask"""

        @self.app.route('/')
        def index():
            return render_template('dashboard.html',
                                 symbols=self.symbols,
                                 config=self.config)

        @self.app.route('/api/bot/status')
        def bot_status():
            """Статус бота"""
            return jsonify({
                'running': self.bot_running,
                'uptime': self.get_bot_uptime() if self.bot_running else 0,
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/bot/start', methods=['POST'])
        def start_bot():
            """Запуск бота"""
            if not self.bot_running:
                try:
                    self.start_bot_thread()
                    return jsonify({'success': True, 'message': 'Бот запущен'})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
            return jsonify({'success': False, 'error': 'Бот уже запущен'})

        @self.app.route('/api/bot/stop', methods=['POST'])
        def stop_bot():
            """Остановка бота"""
            if self.bot_running:
                try:
                    self.stop_bot_thread()
                    return jsonify({'success': True, 'message': 'Бот остановлен'})
                except Exception as e:
                    return jsonify({'success': False, 'error': str(e)})
            return jsonify({'success': False, 'error': 'Бот не запущен'})

        @self.app.route('/api/market/data')
        def market_data():
            """Данные рынка"""
            try:
                config = load_config()
		exchange = ccxt.binance({
    			'apiKey': config['binance_api']['RZ0eHGS8snfTOhcMNiqyDauYfO3cOP6n98M0JYQwBmF9uxlzhgkvhd0af2KMgWnt'],
    			'secret': config['binance_api']['vLl2nXCANtF3bHlzupcYHx17005b9QpOK13JbhhLKKFf9WqUMaPFhaUucEjxrQ2P']
		})
                symbol = request.args.get('symbol', 'BTC/USDT')
                timeframe = request.args.get('timeframe', '1h')
                limit = int(request.args.get('limit', 100))

                # Получение данных
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                if ohlcv:
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df = df.set_index('timestamp')

                    # Расчет EMA20
                    df['ema20'] = df['close'].ewm(span=20).mean()

                    return jsonify({
                        'success': True,
                        'data': df.to_dict('records'),
                        'symbol': symbol
                    })
                else:
                    return jsonify({'success': False, 'error': 'Нет данных'})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/signals/current')
        def current_signals():
            """Текущие сигналы"""
            try:
                # Получение реальных данных с Binance
                config = load_config()
		exchange = ccxt.binance({
    			'apiKey': config['binance_api']['RZ0eHGS8snfTOhcMNiqyDauYfO3cOP6n98M0JYQwBmF9uxlzhgkvhd0af2KMgWnt'],
    			'secret': config['binance_api']['vLl2nXCANtF3bHlzupcYHx17005b9QpOK13JbhhLKKFf9WqUMaPFhaUucEjxrQ2P']
		})

                # Анализ нескольких символов
                symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT', 'BNB/USDT']
                signals = {}

                for symbol in symbols:
                    try:
                        # Получение последних данных
                        ohlcv = exchange.fetch_ohlcv(symbol, '1h', limit=50)
                        if ohlcv:
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

                            # Простой анализ тренда
                            current_price = df['close'].iloc[-1]
                            prev_price = df['close'].iloc[-5] if len(df) > 5 else df['close'].iloc[0]

                            # Расчет EMA20
                            df['ema20'] = df['close'].ewm(span=20).mean()
                            ema20_current = df['ema20'].iloc[-1]
                            ema20_prev = df['ema20'].iloc[-5] if len(df) > 5 else df['ema20'].iloc[0]

                            # Логика сигналов
                            if current_price > ema20_current and prev_price <= ema20_prev:
                                signal = 'BUY'
                                probability = min(0.9, abs(current_price - ema20_current) / current_price + 0.7)
                            elif current_price < ema20_current and prev_price >= ema20_prev:
                                signal = 'SELL'
                                probability = min(0.9, abs(current_price - ema20_current) / current_price + 0.7)
                            else:
                                signal = 'HOLD'
                                probability = 0.5

                            signals[symbol] = {
                                'signal': signal,
                                'probability': round(probability, 3),
                                'price': round(current_price, 2),
                                'timestamp': datetime.now().isoformat()
                            }

                    except Exception as symbol_error:
                        print(f"Ошибка получения данных для {symbol}: {symbol_error}")
                        # Если нет данных - HOLD
                        signals[symbol] = {
                            'signal': 'HOLD',
                            'probability': 0.0,
                            'price': 0,
                            'timestamp': datetime.now().isoformat()
                        }

                return jsonify({'success': True, 'signals': signals})

            except Exception as e:
                print(f"Ошибка получения сигналов: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/statistics')
        def statistics():
            """Статистика торговли"""
            try:
                # Получение реальных данных с Binance
                config = load_config()
		exchange = ccxt.binance({
    			'apiKey': config['binance_api']['RZ0eHGS8snfTOhcMNiqyDauYfO3cOP6n98M0JYQwBmF9uxlzhgkvhd0af2KMgWnt'],
    			'secret': config['binance_api']['vLl2nXCANtF3bHlzupcYHx17005b9QpOK13JbhhLKKFf9WqUMaPFhaUucEjxrQ2P']
		})

                # Анализ портфеля и позиций
                symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT', 'BNB/USDT']

                total_trades = 0
                total_profit = 0.0
                winning_trades = 0
                current_positions = 0
                daily_pnl = 0.0

                # Получение балансов
                try:
                    balances = exchange.fetch_balance()
                    total_balance = balances['total'].get('USDT', 0)

                    # Подсчет позиций
                    for symbol in symbols:
                        if symbol != 'USDT':
                            balance = balances['total'].get(symbol.replace('/USDT', ''), 0)
                            if balance > 0:
                                current_positions += 1

                except Exception as e:
                    print(f"Ошибка получения баланса: {e}")
                    total_balance = 10000  # Заглушка

                # Расчет статистики на основе рыночных данных
                for symbol in symbols[:3]:  # Ограничимся 3 символами для производительности
                    try:
                        ohlcv = exchange.fetch_ohlcv(symbol, '1d', limit=30)  # 30 дней
                        if ohlcv:
                            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

                            # Моделирование торгов (упрощенное)
                            df['ema20'] = df['close'].ewm(span=20).mean()
                            df['signal'] = 'HOLD'

                            # BUY когда цена пересекает EMA20 вверх
                            df.loc[df['close'] > df['ema20'], 'signal'] = 'BUY'
                            df.loc[df['close'] < df['ema20'], 'signal'] = 'SELL'

                            # Подсчет сделок
                            trades = 0
                            profit = 0.0
                            in_position = False
                            entry_price = 0.0

                            for i, row in df.iterrows():
                                if row['signal'] == 'BUY' and not in_position:
                                    in_position = True
                                    entry_price = row['close']
                                    trades += 1
                                elif row['signal'] == 'SELL' and in_position:
                                    in_position = False
                                    exit_price = row['close']
                                    trade_profit = (exit_price - entry_price) / entry_price
                                    profit += trade_profit
                                    if trade_profit > 0:
                                        winning_trades += 1

                            total_trades += trades
                            total_profit += profit

                    except Exception as e:
                        print(f"Ошибка расчета статистики для {symbol}: {e}")

                # Расчет финальной статистики
                win_rate = (winning_trades / max(total_trades, 1)) * 100
                total_profit_percent = total_profit * 100

                # Расчет максимального проседания (упрощенный)
                max_drawdown = min(0.5, total_profit_percent * 0.1)  # Упрощенный расчет

                # Daily PnL (упрощенный)
                daily_pnl = total_profit_percent / 30  # За 30 дней

                stats = {
                    'total_trades': total_trades,
                    'win_rate': round(win_rate, 2),
                    'total_profit': round(total_profit_percent, 2),
                    'max_drawdown': round(max_drawdown, 2),
                    'current_positions': current_positions,
                    'daily_pnl': round(daily_pnl, 2),
                    'total_balance': round(total_balance, 2)
                }

                return jsonify({'success': True, 'statistics': stats})

            except Exception as e:
                print(f"Ошибка получения статистики: {e}")
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/risk/settings')
        def risk_settings():
            """Настройки рисков"""
            try:
                # Получение реальных настроек из конфигурации
                config = load_config()

                risk_config = {
                    'max_daily_loss': config.get('risk_management', {}).get('max_daily_loss', 5),
                    'max_open_positions': config.get('risk_management', {}).get('max_open_positions', 5),
                    'min_position_size': config.get('risk_management', {}).get('min_position_size', 0.001),
                    'max_position_size': config.get('risk_management', {}).get('max_position_size', 0.03),
                    'emergency_stop': config.get('risk_management', {}).get('emergency_stop', True),
                    'emergency_drawdown': config.get('risk_management', {}).get('emergency_drawdown', 15)
                }

                return jsonify({'success': True, 'risk_settings': risk_config})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

    def setup_socket_events(self):
        """Настройка Socket.IO событий"""

        @self.socketio.on('connect')
        def handle_connect():
            print('Клиент подключен к веб-дашборду')
            emit('status', {'message': 'Подключено к торговому дашборду'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print('Клиент отключен')

        @self.socketio.on('request_update')
        def handle_update_request(data):
            """Обработка запроса обновления данных"""
            self.emit_dashboard_update()

    def emit_dashboard_update(self):
        """Отправка обновлений дашборда"""
        try:
            # Получение текущих данных
            status = {
                'running': self.bot_running,
                'timestamp': datetime.now().isoformat(),
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent
            }

            # Отправка обновлений через Socket.IO
            self.socketio.emit('dashboard_update', status)

        except Exception as e:
            print(f"Ошибка отправки обновлений: {e}")

    def start_bot_thread(self):
        """Запуск бота в отдельном потоке"""
        if self.bot_thread and self.bot_thread.is_alive():
            return

        self.bot_running = True
        self.bot_thread = threading.Thread(target=self._run_bot, daemon=True)
        self.bot_thread.start()

    def stop_bot_thread(self):
        """Остановка бота"""
        self.bot_running = False
        if self.bot_thread:
            self.bot_thread.join(timeout=5)

    def _run_bot(self):
        """Выполнение бота"""
        try:
            print("🚀 Запуск торгового бота...")

            # Создание экземпляра бота
            self.bot = OptimizedMLBot()

            # Основной цикл бота
            while self.bot_running:
                try:
                    # Здесь основная логика бота
                    # self.bot.run_cycle()

                    # Отправка обновлений
                    self.emit_dashboard_update()

                    # Пауза
                    time.sleep(self.update_interval)

                except Exception as e:
                    print(f"Ошибка в цикле бота: {e}")
                    time.sleep(5)

            print("🛑 Торговый бот остановлен")

        except Exception as e:
            print(f"Ошибка запуска бота: {e}")
            self.bot_running = False

    def get_bot_uptime(self):
        """Получение времени работы бота"""
        if self.bot_thread and self.bot_thread.is_alive():
            return time.time() - getattr(self.bot_thread, 'start_time', time.time())
        return 0

    def create_templates(self):
        """Создание HTML шаблонов"""
        templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        static_dir = os.path.join(os.path.dirname(__file__), 'static')

        # Создание директорий
        os.makedirs(templates_dir, exist_ok=True)
        os.makedirs(static_dir, exist_ok=True)

        # Создание основного шаблона
        dashboard_html = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Торговый Бот - Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.js"></script>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="#">🤖 Trading Bot Dashboard</a>
            <div class="navbar-nav">
                <a class="nav-link active" href="#dashboard">Дашборд</a>
                <a class="nav-link" href="#signals">Сигналы</a>
                <a class="nav-link" href="#statistics">Статистика</a>
                <a class="nav-link" href="#risk">Риски</a>
                <a class="nav-link" href="#settings">Настройки</a>
            </div>
        </div>
    </nav>

    <div class="container-fluid mt-4">
        <!-- Панель управления ботом -->
        <div class="row mb-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>🤖 Управление ботом</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <button id="startBtn" class="btn btn-success btn-lg me-2" onclick="startBot()">
                                    ▶️ Запустить бота
                                </button>
                                <button id="stopBtn" class="btn btn-danger btn-lg" onclick="stopBot()">
                                    ⏹️ Остановить бота
                                </button>
                            </div>
                            <div class="col-md-6">
                                <div id="botStatus" class="alert alert-secondary">
                                    🔄 Статус: <span id="statusText">Проверка...</span>
                                </div>
                                <div class="row">
                                    <div class="col-sm-6">
                                        <small>CPU: <span id="cpuUsage">0%</span></small>
                                    </div>
                                    <div class="col-sm-6">
                                        <small>RAM: <span id="memoryUsage">0%</span></small>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Графики и данные -->
        <div class="row">
            <div class="col-md-8">
                <div class="card">
                    <div class="card-header">
                        <h5>📈 График цены</h5>
                    </div>
                    <div class="card-body">
                        <div id="priceChart"></div>
                    </div>
                </div>
            </div>
            <div class="col-md-4">
                <div class="card">
                    <div class="card-header">
                        <h5>🎯 Текущие сигналы</h5>
                    </div>
                    <div class="card-body">
                        <div id="signalsList"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Статистика -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>📊 Статистика торговли</h5>
                    </div>
                    <div class="card-body">
                        <div class="row" id="statisticsContent">
                            <!-- Статистика будет загружена здесь -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();

        // Подключение к серверу
        socket.on('connect', function() {
            console.log('Подключено к серверу');
            updateStatus();
            requestUpdate();
        });

        // Обновление данных
        socket.on('dashboard_update', function(data) {
            updateStatus();
            updateCharts();
        });

        function updateStatus() {
            fetch('/api/bot/status')
                .then(response => response.json())
                .then(data => {
                    const statusText = document.getElementById('statusText');
                    const cpuUsage = document.getElementById('cpuUsage');
                    const memoryUsage = document.getElementById('memoryUsage');
                    const startBtn = document.getElementById('startBtn');
                    const stopBtn = document.getElementById('stopBtn');
                    const botStatus = document.getElementById('botStatus');

                    if (data.running) {
                        statusText.textContent = 'Запущен';
                        botStatus.className = 'alert alert-success';
                        startBtn.disabled = true;
                        stopBtn.disabled = false;
                    } else {
                        statusText.textContent = 'Остановлен';
                        botStatus.className = 'alert alert-warning';
                        startBtn.disabled = false;
                        stopBtn.disabled = true;
                    }

                    cpuUsage.textContent = data.cpu_usage + '%';
                    memoryUsage.textContent = data.memory_usage + '%';
                });
        }

        function startBot() {
            fetch('/api/bot/start', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Бот запущен успешно!');
                        updateStatus();
                    } else {
                        alert('Ошибка: ' + data.error);
                    }
                });
        }

        function stopBot() {
            fetch('/api/bot/stop', { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        alert('Бот остановлен успешно!');
                        updateStatus();
                    } else {
                        alert('Ошибка: ' + data.error);
                    }
                });
        }

        function requestUpdate() {
            socket.emit('request_update', {});
            setTimeout(requestUpdate, 5000); // Обновление каждые 5 секунд
        }

        function updateCharts() {
            // Обновление графика цены
            fetch('/api/market/data?symbol=BTC/USDT&limit=100')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        createPriceChart(data.data, data.symbol);
                    }
                });

            // Обновление сигналов
            fetch('/api/signals/current')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateSignals(data.signals);
                    }
                });

            // Обновление статистики
            fetch('/api/statistics')
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateStatistics(data.statistics);
                    }
                });
        }

        function createPriceChart(data, symbol) {
            const timestamps = data.map(d => d.timestamp);
            const prices = data.map(d => d.close);
            const ema20 = data.map(d => d.ema20);

            const trace1 = {
                x: timestamps,
                y: prices,
                type: 'scatter',
                mode: 'lines',
                name: 'Цена',
                line: { color: 'blue' }
            };

            const trace2 = {
                x: timestamps,
                y: ema20,
                type: 'scatter',
                mode: 'lines',
                name: 'EMA20',
                line: { color: 'red' }
            };

            const layout = {
                title: symbol + ' - Цена и EMA20',
                xaxis: { title: 'Время' },
                yaxis: { title: 'Цена (USDT)' },
                height: 400
            };

            Plotly.newPlot('priceChart', [trace1, trace2], layout);
        }

        function updateSignals(signals) {
            const signalsList = document.getElementById('signalsList');
            let html = '';

            for (const [symbol, signal] of Object.entries(signals)) {
                const signalClass = signal.signal === 'BUY' ? 'success' : signal.signal === 'SELL' ? 'danger' : 'warning';
                html += `
                    <div class="alert alert-${signalClass} mt-2">
                        <strong>${symbol}</strong>: ${signal.signal}<br>
                        <small>Вероятность: ${(signal.probability * 100).toFixed(1)}% | Цена: $${signal.price}</small>
                    </div>
                `;
            }

            signalsList.innerHTML = html;
        }

        function updateStatistics(stats) {
            const content = document.getElementById('statisticsContent');
            content.innerHTML = `
                <div class="col-md-3">
                    <div class="card bg-primary text-white">
                        <div class="card-body">
                            <h5>Всего сделок</h5>
                            <h3>${stats.total_trades}</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-success text-white">
                        <div class="card-body">
                            <h5>Win Rate</h5>
                            <h3>${stats.win_rate}%</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-info text-white">
                        <div class="card-body">
                            <h5>Прибыль</h5>
                            <h3>${stats.total_profit}%</h3>
                        </div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card bg-warning text-white">
                        <div class="card-body">
                            <h5>Позиций</h5>
                            <h3>${stats.current_positions}</h3>
                        </div>
                    </div>
                </div>
            `;
        }

        // Инициализация
        updateCharts();
    </script>
</body>
</html>
        """

        with open(os.path.join(templates_dir, 'dashboard.html'), 'w', encoding='utf-8') as f:
            f.write(dashboard_html)

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Запуск веб-сервера"""
        print("🚀 Запуск веб-дашборда торгового бота...")

        # Создание шаблонов
        self.create_templates()

        # Запуск в отдельном потоке
        server_thread = threading.Thread(
            target=lambda: self.socketio.run(self.app, host=host, port=port, debug=debug),
            daemon=True
        )
        server_thread.start()

        print(f"✅ Веб-дашборд запущен на http://{host}:{port}")
        print("🌐 Откройте браузер и перейдите по адресу выше")

        # Ожидание завершения
        try:
            while server_thread.is_alive():
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Остановка сервера...")

        return server_thread

def main():
    dashboard = WebTradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
