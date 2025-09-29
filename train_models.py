from datetime import datetime

from advanced_ml_trainer import AdvancedMLTrainer

if __name__ == "__main__":
    symbols = ["BTC/USDT"]
    trainer = AdvancedMLTrainer()
    trainer.train_models(
        symbols,
        start_date=datetime(2020, 1, 1),
        end_date=datetime(2024, 12, 31, 23, 59, 59),
        timeframe="4h",
    )
