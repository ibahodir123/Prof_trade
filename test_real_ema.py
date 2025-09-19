#!/usr/bin/env python3
"""
Тестирование EMA анализа на реальных данных
"""

from ml_bot_binance import analyze_coin_signal_ema
import logging

def test_real_ema_analysis():
    """Тестирование EMA анализа на реальных данных"""
    
    # Настраиваем логирование
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Тестирование EMA анализа на реальных данных...")
    
    # Тестируем на популярных монетах
    test_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
    
    results = []
    
    for coin in test_coins:
        print(f"\n📊 Анализирую {coin}...")
        try:
            result = analyze_coin_signal_ema(coin)
            
            if result.get('error'):
                print(f"❌ Ошибка: {result['error']}")
                results.append({
                    'coin': coin,
                    'status': 'ERROR',
                    'error': result['error']
                })
            else:
                print(f"✅ Результат: {result['signal_type']}")
                print(f"📝 Обоснование: {result['strength_text']}")
                print(f"💰 Цена: ${result['entry_price']:.8f}")
                
                if result.get('take_profit'):
                    print(f"🎯 TP: ${result['take_profit']:.8f}")
                if result.get('stop_loss'):
                    print(f"🛡️ SL: ${result['stop_loss']:.8f}")
                    
                ema_analysis = result.get('ema_analysis', {})
                print(f"📈 Тренд: {ema_analysis.get('trend', 'Не определен')}")
                print(f"🔄 Фаза: {ema_analysis.get('phase', 'Не определена')}")
                print(f"🎯 Уверенность: {ema_analysis.get('confidence', 0)*100:.1f}%")
                
                results.append({
                    'coin': coin,
                    'status': 'SUCCESS',
                    'signal': result['signal_type'],
                    'trend': ema_analysis.get('trend', 'Не определен'),
                    'phase': ema_analysis.get('phase', 'Не определена'),
                    'confidence': ema_analysis.get('confidence', 0)*100
                })
                
        except Exception as e:
            print(f"❌ Критическая ошибка {coin}: {e}")
            results.append({
                'coin': coin,
                'status': 'CRITICAL_ERROR',
                'error': str(e)
            })
    
    # Итоговая статистика
    print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
    print(f"Всего проанализировано: {len(results)} монет")
    
    successful = [r for r in results if r['status'] == 'SUCCESS']
    errors = [r for r in results if r['status'] in ['ERROR', 'CRITICAL_ERROR']]
    
    print(f"Успешно: {len(successful)}")
    print(f"Ошибок: {len(errors)}")
    
    if successful:
        print(f"\n🎯 СИГНАЛЫ:")
        for result in successful:
            print(f"  {result['coin']}: {result['signal']} ({result['trend']}, {result['confidence']:.1f}%)")
    
    if errors:
        print(f"\n❌ ОШИБКИ:")
        for result in errors:
            print(f"  {result['coin']}: {result.get('error', 'Неизвестная ошибка')}")
    
    print(f"\n🎉 Тестирование завершено!")
    
    return results

if __name__ == "__main__":
    try:
        results = test_real_ema_analysis()
    except Exception as e:
        print(f"❌ Критическая ошибка тестирования: {e}")
        import traceback
        traceback.print_exc()




