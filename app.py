# app.py
from flask import Flask, render_template, request, jsonify
import asyncio
import threading
import pandas as pd
from datetime import datetime
from main import XeniaV2  # Your existing module

app = Flask(__name__)
trading_system = None
backtest_results = {}
current_signals = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_backtest', methods=['POST'])
def run_backtest():
    
    # Get form data
    symbols = request.form['symbols'].split(',')
    initial_balance = float(request.form['balance'])
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    transaction_cost = float(request.form['transaction_cost'])
    
    # Create trading system instance
    trading_system = XeniaV2(
        symbols=[s.strip().upper() for s in symbols],
        initial_balance=initial_balance,
        transaction_cost=transaction_cost
    )
    
    # Run backtest in background thread
    def backtest_task():
    
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(trading_system.run_backtest(
            start_date=start_date,
            end_date=end_date
        ))
        backtest_results = {
            'trades': trading_system.trades,
            'initial_balance': trading_system.initial_balance,
            'final_balance': trading_system.balance,
            'positions': trading_system.positions,
            'signals': trading_system.latest_signals
        }
        current_signals = trading_system.latest_signals

        loop.close()
    
    threading.Thread(target=backtest_task).start()
    
    return jsonify({'status': 'Backtest started'})

@app.route('/backtest_status')
def backtest_status():
    if trading_system and backtest_results:
        portfolio_value = backtest_results['final_balance']
        for symbol, shares in backtest_results['positions'].items():
            if shares > 0 and symbol in trading_system.data:
                current_price = trading_system.data[symbol]['Close'].iloc[-1]
                portfolio_value += shares * current_price

        print(bool(backtest_results, trading_system))
        return jsonify({
            'status': 'completed',
            'results': {
                'initial': backtest_results['initial_balance'],
                'final': portfolio_value,
                'return': ((portfolio_value - backtest_results['initial_balance']) / 
                          backtest_results['initial_balance']) * 100,
                'trades': len(backtest_results['trades'])
            }
        })
    return jsonify({'status': 'running'})

@app.route('/get_chart_data/<symbol>')
def get_chart_data(symbol):
    if not trading_system or symbol not in trading_system.data:
        print('No data to chart !!!')
        return jsonify({})
    
    df = trading_system.data[symbol]
    features = trading_system.create_features(df)
    
    # Prepare chart data
    chart_data = {
        'dates': df.index.strftime('%Y-%m-%d').tolist(),
        'close': df['Close'].values.tolist(),
        'sma20': features['sma_20'].values.tolist() if 'sma_20' in features else [],
        'rsi': features['rsi'].values.tolist() if 'rsi' in features else [],
        'macd': features['macd'].values.tolist() if 'macd' in features else [],
        'macd_signal': features['macd_signal'].values.tolist() if 'macd_signal' in features else [],
        'bb_upper': features['bb_upper'].values.tolist() if 'bb_upper' in features else [],
        'bb_lower': features['bb_lower'].values.tolist() if 'bb_lower' in features else [],
    }
    
    return jsonify(chart_data)

@app.route('/get_signals')
def get_signals():
    return jsonify(current_signals)

@app.route('/get_trades')
def get_trades():
    if not trading_system or not backtest_results.get('trades'):
        return jsonify([])
    
    # Process trades for display
    processed_trades = []
    for trade in backtest_results['trades']:
        t = {
            'symbol': trade['symbol'],
            'action': trade['action'],
            'date': trade['date'].strftime('%Y-%m-%d'),
            'price': f"${trade['price']:.2f}",
            'confidence': f"{trade['confidence']:.0%}",
        }
        
        if 'shares' in trade:
            t['shares'] = f"{trade['shares']:.2f}"
        if 'pnl_pct' in trade:
            t['pnl'] = f"{trade['pnl_pct']:.2f}%"
            t['pnl_class'] = 'profit' if trade['pnl_pct'] >= 0 else 'loss'
        if 'signal' in trade:
            t['signal'] = f"{trade['signal']:.3f}"
        
        processed_trades.append(t)
    
    return jsonify(processed_trades)

if __name__ == '__main__':
    app.run(debug=True)