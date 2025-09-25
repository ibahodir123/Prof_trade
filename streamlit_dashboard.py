#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit Dashboard для визуализации работы детектора минимумов
Показывает графики, критерии и предсказания в реальном времени
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots
import ccxt
import json
import pickle
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Детектор Минимумов - Dashboard",
    page_icon="📊",
    layout="wide"
)

class MinimumDetectorDashboard:
    def __init__(self):
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'XRP/USDT', 'SOL/USDT']
        self.model = None
        self.feature_names = None
        self.feature_weights = None
        
    @st.cache_data
    def load_model(_self, model_filename: str = "minimum_detector_model.pkl"):
        """Загрузка модели с кешированием"""
        try:
            with open(model_filename, 'rb') as f:
                model = pickle.load(f)
            
            metadata_filename = model_filename.replace('.pkl', '_metadata.json')
            with open(metadata_filename, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            return model, metadata['feature_names'], metadata['feature_weights']
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки модели: {e}")
            return None, None, None
    
    @st.cache_data
    def get_data(_self, symbol: str, hours: int = 168) -> pd.DataFrame:
        """Получение данных с кешированием (на 5 минут)"""
        try:
            exchange = ccxt.binance()
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            since = int(start_time.timestamp() * 1000)
            
            ohlcv = exchange.fetch_ohlcv(symbol, '1h', since=since, limit=hours)
            
            if not ohlcv:
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            return df
            
        except Exception as e:
            st.error(f"❌ Ошибка загрузки данных {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_4_criteria(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет 4 критериев"""
        try:
            # EMA20
            df['ema20'] = df['close'].ewm(span=20).mean()
            
            # 4 критерия
            df['price_velocity'] = df['close'].pct_change() * 100
            df['ema20_velocity'] = df['ema20'].pct_change() * 100
            df['ema20_angle'] = ((df['ema20'] / df['ema20'].shift(10)) - 1) * 100
            df['distance_to_ema20'] = ((df['close'] - df['ema20']) / df['ema20']) * 100
            
            return df
            
        except Exception as e:
            st.error(f"❌ Ошибка расчета критериев: {e}")
            return df
    
    def predict_minimum(self, criteria_values: dict):
        """Предсказание минимума"""
        if not self.model:
            return None
        
        try:
            feature_vector = []
            for name in self.feature_names:
                value = criteria_values.get(name, 0.0)
                if np.isnan(value) or np.isinf(value):
                    value = 0.0
                feature_vector.append(value)
            
            feature_vector = np.array(feature_vector).reshape(1, -1)
            
            prediction = self.model.predict(feature_vector)[0]
            probabilities = self.model.predict_proba(feature_vector)[0]
            
            return {
                'is_minimum': bool(prediction),
                'probability': float(probabilities[1]),
                'criteria': criteria_values
            }
            
        except Exception as e:
            st.error(f"❌ Ошибка предсказания: {e}")
            return None
    
    def create_main_chart(self, df: pd.DataFrame, symbol: str):
        """Создание основного графика цены + EMA20"""
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(
                f'{symbol} - Цена и EMA20',
                'Объем',
                'Расстояние до EMA20'
            ),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # График свечей
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Цена',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1
        )
        
        # EMA20
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema20'],
                mode='lines',
                name='EMA20',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Объем
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(df['close'], df['open'])]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Объем',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Расстояние до EMA20
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['distance_to_ema20'],
                mode='lines',
                name='Расстояние до EMA20 (%)',
                line=dict(color='purple', width=2),
                fill='tonexty'
            ),
            row=3, col=1
        )
        
        # Линия нуля для расстояния
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)
        
        fig.update_layout(
            title=f"Анализ {symbol}",
            xaxis_title="Время",
            height=800,
            showlegend=True
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    def create_criteria_chart(self, df: pd.DataFrame):
        """График 4 критериев"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Скорость цены (%/час)',
                'Скорость EMA20 (%/час)', 
                'Угол EMA20 (%/10час)',
                'Расстояние до EMA20 (%)'
            )
        )
        
        # Скорость цены
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['price_velocity'],
                mode='lines',
                name='Скорость цены',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Скорость EMA20
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema20_velocity'],
                mode='lines',
                name='Скорость EMA20',
                line=dict(color='blue')
            ),
            row=1, col=2
        )
        
        # Угол EMA20
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ema20_angle'],
                mode='lines',
                name='Угол EMA20',
                line=dict(color='green')
            ),
            row=2, col=1
        )
        
        # Расстояние до EMA20
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['distance_to_ema20'],
                mode='lines',
                name='Расстояние',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
        
        # Добавляем линии нуля
        for row in [1, 2]:
            for col in [1, 2]:
                fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)
        
        fig.update_layout(
            title="4 Критерия детектора минимумов",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def run_dashboard(self):
        """Основная функция дашборда"""
        
        # Заголовок
        st.title("📊 Детектор Минимумов - Dashboard")
        st.markdown("Визуализация работы системы поиска минимумов для LONG позиций")
        
        # Загрузка модели
        if self.model is None:
            self.model, self.feature_names, self.feature_weights = self.load_model()
        
        if self.model is None:
            st.error("❌ Модель не загружена! Запустите сначала weighted_ml_trainer.py")
            return
        
        # Сайдбар с настройками
        st.sidebar.header("⚙️ Настройки")
        
        selected_symbol = st.sidebar.selectbox(
            "Выберите символ:",
            self.symbols,
            index=0
        )
        
        hours_back = st.sidebar.slider(
            "Часов назад:",
            min_value=24,
            max_value=720,
            value=168,
            step=24
        )
        
        auto_refresh = st.sidebar.checkbox("🔄 Автообновление (30 сек)", value=False)
        
        # Показываем веса критериев
        st.sidebar.header("⚖️ Веса критериев")
        if self.feature_weights:
            for feature, weight in self.feature_weights.items():
                st.sidebar.metric(
                    feature.replace('_', ' ').title(),
                    f"{weight:.1%}"
                )
        
        # Основная область
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Получение данных
            with st.spinner(f"📊 Загружаю данные {selected_symbol}..."):
                df = self.get_data(selected_symbol, hours_back)
            
            if df.empty:
                st.error("❌ Не удалось загрузить данные")
                return
            
            # Расчет критериев
            df = self.calculate_4_criteria(df)
            
            # Основной график
            main_chart = self.create_main_chart(df, selected_symbol)
            st.plotly_chart(main_chart, use_container_width=True)
            
            # График критериев
            criteria_chart = self.create_criteria_chart(df)
            st.plotly_chart(criteria_chart, use_container_width=True)
        
        with col2:
            st.header("🎯 Текущий анализ")
            
            # Текущие значения критериев
            current_data = df.iloc[-1]
            current_criteria = {
                'price_velocity': current_data['price_velocity'],
                'ema20_velocity': current_data['ema20_velocity'],
                'ema20_angle': current_data['ema20_angle'],
                'distance_to_ema20': current_data['distance_to_ema20']
            }
            
            # Проверяем на NaN
            valid_criteria = True
            for key, value in current_criteria.items():
                if pd.isna(value) or np.isinf(value):
                    current_criteria[key] = 0.0
                    valid_criteria = False
            
            # Предсказание
            if valid_criteria:
                prediction = self.predict_minimum(current_criteria)
                
                if prediction:
                    # Результат предсказания
                    if prediction['is_minimum']:
                        st.success("✅ МИНИМУМ ОБНАРУЖЕН!")
                        st.balloons()
                    else:
                        st.info("⏳ Минимум не обнаружен")
                    
                    # Вероятность
                    prob = prediction['probability']
                    st.metric(
                        "🎯 Вероятность минимума",
                        f"{prob:.1%}",
                        delta=f"{'Высокая' if prob > 0.7 else 'Средняя' if prob > 0.5 else 'Низкая'}"
                    )
                    
                    # Прогресс бар вероятности
                    st.progress(prob)
                    
                    # Торговая рекомендация
                    st.header("💡 Рекомендация")
                    if prob > 0.7 and current_criteria['distance_to_ema20'] < -1:
                        st.success("🚀 РЕКОМЕНДУЕТСЯ LONG ПОЗИЦИЯ")
                        st.markdown("**Условия выполнены:**")
                        st.markdown("- ✅ Высокая вероятность минимума")
                        st.markdown("- ✅ Цена ниже EMA20")
                    elif prob > 0.5:
                        st.warning("⚠️ ВОЗМОЖЕН МИНИМУМ - СЛЕДИТЕ")
                    else:
                        st.info("📊 ОЖИДАНИЕ")
            
            # Детали критериев
            st.header("📊 Текущие критерии")
            
            for feature in self.feature_names:
                value = current_criteria[feature]
                weight = self.feature_weights.get(feature, 0.25)
                
                # Цветовое кодирование
                if feature == 'distance_to_ema20':
                    color = "normal" if value > -2 else "inverse"
                else:
                    color = "normal"
                
                st.metric(
                    feature.replace('_', ' ').title(),
                    f"{value:.3f}",
                    delta=f"Вес: {weight:.1%}",
                    delta_color=color
                )
            
            # Информация о цене
            st.header("💰 Цена")
            current_price = current_data['close']
            ema20 = current_data['ema20']
            
            st.metric(
                "Текущая цена",
                f"${current_price:.4f}",
                delta=f"EMA20: ${ema20:.4f}"
            )
            
            # Время последнего обновления
            st.caption(f"Обновлено: {datetime.now().strftime('%H:%M:%S')}")
        
        # Автообновление
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()

def main():
    dashboard = MinimumDetectorDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()






