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
        self.socketio = SocketIO(self.app)

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
                exchange = ccxt.binance()
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
                # Здесь должна быть логика получения текущих сигналов
                # Пока заглушка
                signals = {
                    'BTC/USDT': {
                        'signal': 'BUY',
                        'probability': 0.87,
                        'price': 45000,
                        'timestamp': datetime.now().isoformat()
                    },
                    'ETH/USDT': {
                        'signal': 'HOLD',
                        'probability': 0.45,
                        'price': 2500,
                        'timestamp': datetime.now().isoformat()
                    }
                }

                return jsonify({'success': True, 'signals': signals})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/statistics')
        def statistics():
            """Статистика торговли"""
            try:
                # Заглушка для статистики
                stats = {
                    'total_trades': 1250,
                    'win_rate': 87.2,
                    'total_profit': 132.85,
                    'max_drawdown': 0.4,
                    'current_positions': 3,
                    'daily_pnl': 2.34
                }

                return jsonify({'success': True, 'statistics': stats})

            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})

        @self.app.route('/api/risk/settings')
        def risk_settings():
            """Настройки рисков"""
            try:
                risk_config = {
                    'max_drawdown': 20,
                    'max_position_size': 3,
                    'max_positions': 5,
                    'stop_loss_percent': 2,
                    'take_profit_percent': 5
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
            target=lambda: self.app.run(host=host, port=port, debug=debug),
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
