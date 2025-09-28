from advanced_ml_trainer import AdvancedMLTrainer

if __name__ == "__main__":
    # Список пар можно менять под свои нужды
    symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT"]

    trainer = AdvancedMLTrainer()
    trainer.train_models(symbols)
