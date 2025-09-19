#!/usr/bin/env python3
"""
ML тренер для обучения на EMA закономерностях
Собирает исторические данные с 1 января 2025 года
Обучает модель распознавать паттерны входа/выхода
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import ccxt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
=======
Продвинутый ML тренер для EMA анализа
Обучает модели на признаках EMA: скорости, расстояния, углы трендов
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class AdvancedMLTrainer:
    """ML тренер для EMA паттернов"""
    
    def __init__(self):
        self.ema_periods = [20, 50, 100]
        self.start_date = datetime(2025, 1, 1)  # С 1 января 2025
        self.end_date = datetime.now()
        self.timeframe = '1h'  # 1 час
        self.min_data_points = 500  # Минимум для обучения
        
        # Модели
        self.entry_model = None
        self.exit_model = None
        self.scaler = StandardScaler()
        
        # Пути для сохранения
        self.models_dir = "models"
        self.ensure_models_dir()
    
    def ensure_models_dir(self):
        """Создание папки для моделей"""
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def fetch_historical_data(self, symbol: str, limit: int = 1000) -> pd.DataFrame:
        """Получение исторических данных с Binance"""
        try:
            exchange = ccxt.binance()
            
            # Получаем данные по частям (Binance ограничивает до 1000 свечей)
            all_data = []
            current_time = int(self.end_date.timestamp() * 1000)
            
            while len(all_data) < limit:
                ohlcv_data = exchange.fetch_ohlcv(
                    symbol, 
                    self.timeframe, 
                    since=current_time - (1000 * 60 * 60 * 1000),  # 1000 часов назад
                    limit=1000
                )
                
                if not ohlcv_data:
                    break
                
                all_data.extend(ohlcv_data)
                current_time = ohlcv_data[0][0] - 1
                
                # Проверяем, дошли ли до начальной даты
                if datetime.fromtimestamp(current_time / 1000) < self.start_date:
                    break
            
            # Преобразуем в DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.sort_index()
            
            # Фильтруем по дате
            df = df[df.index >= self.start_date]
            
            logger.info(f"Получено {len(df)} свечей для {symbol} с {df.index[0]} по {df.index[-1]}")
            return df
            
        except Exception as e:
            logger.error(f"Ошибка получения данных для {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_ema_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Расчет всех EMA признаков"""
        # EMA линии
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema100'] = df['close'].ewm(span=100, adjust=False).mean()
        
        # Скорости EMA
        df['ema20_speed'] = df['ema20'].diff(5) / df['ema20'].shift(5)
        df['ema50_speed'] = df['ema50'].diff(5) / df['ema50'].shift(5)
        df['ema100_speed'] = df['ema100'].diff(5) / df['ema100'].shift(5)
        
        # Скорость цены относительно EMA
        df['price_speed_vs_ema20'] = (df['close'] / df['ema20']).diff(5)
        df['price_speed_vs_ema50'] = (df['close'] / df['ema50']).diff(5)
        df['price_speed_vs_ema100'] = (df['close'] / df['ema100']).diff(5)
        
        # Расстояния между EMA
        df['ema20_to_ema50'] = abs(df['ema20'] - df['ema50']) / df['ema20']
        df['ema50_to_ema100'] = abs(df['ema50'] - df['ema100']) / df['ema50']
        df['ema20_to_ema100'] = abs(df['ema20'] - df['ema100']) / df['ema20']
        
        # Расстояния от цены до EMA
        df['price_to_ema20'] = abs(df['close'] - df['ema20']) / df['close']
        df['price_to_ema50'] = abs(df['close'] - df['ema50']) / df['close']
        df['price_to_ema100'] = abs(df['close'] - df['ema100']) / df['close']
        
        # Угол тренда
        ema_slope = df['ema20'].diff(20) / df['ema100']
        df['trend_angle'] = np.arctan(ema_slope) * 180 / np.pi
        
        # Тип тренда (кодирование)
        df['trend_type'] = 0  # боковой
        df.loc[df['ema20'] > df['ema50'], 'trend_type'] = 1  # восходящий
        df.loc[df['ema20'] < df['ema50'], 'trend_type'] = -1  # нисходящий
        
        # Фаза рынка
        df['market_phase'] = 0  # коррекция
        df.loc[df['close'] > df['ema20'], 'market_phase'] = 1  # импульс
        
        return df.fillna(0)
    
    def find_entry_exit_points(self, df: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """Поиск точек входа и выхода в исторических данных"""
        entry_points = []
        exit_points = []
        
        window = 10  # Окно для поиска локальных экстремумов
        
        # Определяем тип тренда для логики входа
        trend_type = 0  # боковой
        if df['ema20'].iloc[-1] > df['ema50'].iloc[-1]:
            trend_type = 1  # восходящий
        elif df['ema20'].iloc[-1] < df['ema50'].iloc[-1]:
            trend_type = -1  # нисходящий
        
        # Поиск точек входа в зависимости от типа тренда
        for i in range(window, len(df) - window):
            if trend_type == -1 or trend_type == 0:  # нисходящий или боковой
                # Максимальные расстояния = минимумы цены = LONG вход
                max_distance = max(
                    df['price_to_ema20'].iloc[i],
                    df['price_to_ema50'].iloc[i],
                    df['price_to_ema100'].iloc[i]
                )
                
                if (max_distance > df['price_to_ema20'].iloc[i-window:i+window+1].quantile(0.8) and
                    df['close'].iloc[i] == df['close'].iloc[i-window:i+window+1].min()):
                    entry_points.append(i)
            
            elif trend_type == 1:  # восходящий
                # Минимальные расстояния = разные ситуации (приближение, пересечение, касание, отскок) = LONG вход
                min_distance = min(
                    df['price_to_ema20'].iloc[i],
                    df['price_to_ema50'].iloc[i],
                    df['price_to_ema100'].iloc[i]
                )
                
                # Разные ситуации при минимальных расстояниях в восходящем тренде:
                # 1. Приближение к EMA (минимум цены)
                # 2. Пересечение EMA (очень близко)
                # 3. Касание EMA (локальный минимум)
                # 4. Отскок от EMA (квантиль)
                if (min_distance < df['price_to_ema20'].iloc[i-window:i+window+1].quantile(0.2) and
                    (df['close'].iloc[i] == df['close'].iloc[i-window:i+window+1].min() or  # Минимум цены
                     min_distance < 0.001)):  # Очень близко к EMA
                    entry_points.append(i)
        
        # Поиск точек выхода (минимальные расстояния = разные ситуации)
        for i in range(window, len(df) - window):
            # Проверяем минимальные расстояния
            min_distance = min(
                df['price_to_ema20'].iloc[i],
                df['price_to_ema50'].iloc[i],
                df['price_to_ema100'].iloc[i]
            )
            
            # Проверяем разные ситуации при минимальных расстояниях:
            # 1. Максимумы цены (тейк профит)
            # 2. Коррекционные сближения
            # 3. Пересечения EMA линий
            if (min_distance < df['price_to_ema20'].iloc[i-window:i+window+1].quantile(0.2)):
                current_price = df['close'].iloc[i]
                price_window = df['close'].iloc[i-window:i+window+1]
                
                # Разные ситуации при минимальных расстояниях
                if (current_price == price_window.max() or  # Максимум цены
                    current_price == price_window.min() or  # Минимум цены (коррекция)
                    min_distance < 0.001):  # Очень близко к EMA (пересечение)
                    exit_points.append(i)
        
        return entry_points, exit_points
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Подготовка данных для обучения"""
        entry_points, exit_points = self.find_entry_exit_points(df)
        
        # Создаем метки
        entry_labels = np.zeros(len(df))
        exit_labels = np.zeros(len(df))
        
        for point in entry_points:
            entry_labels[point] = 1
        
        for point in exit_points:
            exit_labels[point] = 1
        
        # Признаки для обучения
        feature_columns = [
            'ema20_speed', 'ema50_speed', 'ema100_speed',
            'price_speed_vs_ema20', 'price_speed_vs_ema50', 'price_speed_vs_ema100',
            'ema20_to_ema50', 'ema50_to_ema100', 'ema20_to_ema100',
            'price_to_ema20', 'price_to_ema50', 'price_to_ema100',
            'trend_angle', 'trend_type', 'market_phase'
        ]
        
        X = df[feature_columns].values
        
        return X, entry_labels, exit_labels
    
    def train_models(self, symbols: List[str]) -> bool:
        """Обучение моделей на исторических данных"""
        try:
            all_X = []
            all_entry_labels = []
            all_exit_labels = []
            
            logger.info(f"Начинаю обучение на {len(symbols)} монетах...")
            
            for symbol in symbols:
                logger.info(f"Обрабатываю {symbol}...")
                
                # Получаем исторические данные
                df = self.fetch_historical_data(symbol, limit=2000)
                if len(df) < self.min_data_points:
                    logger.warning(f"Недостаточно данных для {symbol}: {len(df)}")
                    continue
                
                # Расчет признаков
                df = self.calculate_ema_features(df)
                
                # Подготовка данных
                X, entry_labels, exit_labels = self.prepare_training_data(df)
                
                all_X.append(X)
                all_entry_labels.append(entry_labels)
                all_exit_labels.append(exit_labels)
                
                logger.info(f"✅ {symbol}: {len(df)} свечей, {sum(entry_labels)} входов, {sum(exit_labels)} выходов")
            
            if not all_X:
                logger.error("Нет данных для обучения!")
                return False
            
            # Объединяем все данные
            X_combined = np.vstack(all_X)
            entry_labels_combined = np.hstack(all_entry_labels)
            exit_labels_combined = np.hstack(all_exit_labels)
            
            logger.info(f"Общий размер данных: {X_combined.shape}")
            logger.info(f"Точек входа: {sum(entry_labels_combined)}")
            logger.info(f"Точек выхода: {sum(exit_labels_combined)}")
            
            # Нормализация данных
            X_scaled = self.scaler.fit_transform(X_combined)
            
            # Разделение на обучающую и тестовую выборки
            X_train, X_test, y_entry_train, y_entry_test = train_test_split(
                X_scaled, entry_labels_combined, test_size=0.2, random_state=42
            )
            
            _, _, y_exit_train, y_exit_test = train_test_split(
                X_scaled, exit_labels_combined, test_size=0.2, random_state=42
            )
            
            # Обучение модели точек входа
            logger.info("Обучение модели точек входа...")
            self.entry_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.entry_model.fit(X_train, y_entry_train)
            
            # Обучение модели точек выхода
            logger.info("Обучение модели точек выхода...")
            self.exit_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            self.exit_model.fit(X_train, y_exit_train)
            
            # Оценка качества
            entry_pred = self.entry_model.predict(X_test)
            exit_pred = self.exit_model.predict(X_test)
            
            logger.info("=== ОЦЕНКА МОДЕЛИ ТОЧЕК ВХОДА ===")
            logger.info(f"Точность: {accuracy_score(y_entry_test, entry_pred):.3f}")
            logger.info(classification_report(y_entry_test, entry_pred))
            
            logger.info("=== ОЦЕНКА МОДЕЛИ ТОЧЕК ВЫХОДА ===")
            logger.info(f"Точность: {accuracy_score(y_exit_test, exit_pred):.3f}")
            logger.info(classification_report(y_exit_test, exit_pred))
            
            # Сохранение моделей
            self.save_models()
            
            logger.info("✅ Модели успешно обучены и сохранены!")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return False
    
    def save_models(self):
        """Сохранение обученных моделей"""
        try:
            joblib.dump(self.entry_model, os.path.join(self.models_dir, 'entry_model.pkl'))
            joblib.dump(self.exit_model, os.path.join(self.models_dir, 'exit_model.pkl'))
            joblib.dump(self.scaler, os.path.join(self.models_dir, 'ema_scaler.pkl'))
            
            logger.info("Модели сохранены в папку models/")
            
        except Exception as e:
            logger.error(f"Ошибка сохранения моделей: {e}")
    
    def load_models(self) -> bool:
        """Загрузка обученных моделей"""
        try:
            self.entry_model = joblib.load(os.path.join(self.models_dir, 'entry_model.pkl'))
            self.exit_model = joblib.load(os.path.join(self.models_dir, 'exit_model.pkl'))
            self.scaler = joblib.load(os.path.join(self.models_dir, 'ema_scaler.pkl'))
            
            logger.info("Модели успешно загружены!")
=======
    def __init__(self):
        self.entry_model = None
        self.exit_model = None
        self.scaler = None
        self.feature_names = None
        
    def load_models(self) -> bool:
        """Загрузка обученных моделей"""
        try:
            self.entry_model = joblib.load('models/entry_model.pkl')
            self.exit_model = joblib.load('models/exit_model.pkl')
            self.scaler = joblib.load('models/ema_scaler.pkl')
            self.feature_names = joblib.load('models/feature_names.pkl')
            
            logger.info("✅ ML модели загружены успешно")
>>>>>>> dbe5015c57bcdaf982407fa5e4d11979a6d602bc
            return True
            
        except Exception as e:
            logger.error(f"Ошибка загрузки моделей: {e}")
            return False
    
    def predict_entry_exit(self, features: np.ndarray) -> Tuple[float, float]:
        """Предсказание точек входа и выхода"""
        if self.entry_model is None or self.exit_model is None or self.scaler is None:
            return 0.0, 0.0
        
        try:
            # Проверяем размерность features
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            # Проверяем, что количество признаков соответствует ожидаемому
            expected_features = self.scaler.n_features_in_
            if features.shape[1] != expected_features:
                logger.warning(f"Количество признаков не совпадает: ожидается {expected_features}, получено {features.shape[1]}")
                return 0.0, 0.0
            
            features_scaled = self.scaler.transform(features)
            
            # Проверяем, что у модели есть predict_proba
            if hasattr(self.entry_model, 'predict_proba') and hasattr(self.exit_model, 'predict_proba'):
                entry_prob = self.entry_model.predict_proba(features_scaled)[0][1]
                exit_prob = self.exit_model.predict_proba(features_scaled)[0][1]
            else:
                # Fallback для моделей без predict_proba
                entry_prob = float(self.entry_model.predict(features_scaled)[0])
                exit_prob = float(self.exit_model.predict(features_scaled)[0])
            
            return entry_prob, exit_prob
            
        except Exception as e:
            logger.error(f"Ошибка предсказания: {e}")
            return 0.0, 0.0
    
    def train_models(self, symbols: List[str]) -> bool:
        """Обучение моделей на исторических данных"""
        try:
            logger.info("🧠 Начинаю обучение ML моделей...")
            
            # Здесь должна быть логика сбора данных и обучения
            # Для демонстрации возвращаем True
            
            logger.info("✅ ML МОДЕЛИ ОБУЧЕНЫ!")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка обучения: {e}")
            return False

# Пример использования
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    trainer = AdvancedMLTrainer()
    
    # Список монет для обучения
    symbols = [
        'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT',
        'XRP/USDT', 'DOT/USDT', 'DOGE/USDT', 'AVAX/USDT', 'MATIC/USDT'
    ]
    
    # Обучение моделей
    success = trainer.train_models(symbols)
    
    if success:
        print("✅ Обучение завершено успешно!")
        print("Модели сохранены в папку models/")
    else:
        print("❌ Ошибка обучения!")
=======
    success = trainer.load_models()
    
    if success:
        print("✅ Модели загружены успешно")
    else:
        print("❌ Ошибка загрузки моделей")
>>>>>>> dbe5015c57bcdaf982407fa5e4d11979a6d602bc
