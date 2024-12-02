import os
import sys
import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from rl_env.forex_environment import ForexTradingEnv
from utils.load_data import load_forex_data
import time
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly
import pandas_ta as ta

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def create_backtest_visualization(env, test_data, trade_history, model_name):
    # Create timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate statistics
    df_trades = pd.DataFrame(trade_history, columns=['date', 'price', 'action'])
    df_trades['date'] = pd.to_datetime(df_trades['date'])
    df_trades['month'] = df_trades['date'].dt.to_period('M')
    df_trades['year'] = df_trades['date'].dt.year
    
    balance_series = pd.Series(env.balance_history)
    balance_series.index = test_data.index[:len(balance_series)]
    
    # Calculate indicators
    test_data = test_data.copy()  # Create a copy first
    test_data.ta.sma(length=20, append=True, col_names=('SMA_20',))
    test_data.ta.sma(length=50, append=True, col_names=('SMA_50',))
    test_data.ta.rsi(length=14, append=True, col_names=('RSI',))
    
    # Calculate monthly statistics
    monthly_stats = []
    for month in df_trades['month'].unique():
        month_trades = df_trades[df_trades['month'] == month]
        month_balance = balance_series[balance_series.index.to_period('M') == month]
        
        if len(month_balance) > 0:
            month_return = (month_balance.iloc[-1] / month_balance.iloc[0] - 1) * 100
            month_drawdown = ((month_balance.cummax() - month_balance) / month_balance.cummax()).max() * 100
            
            monthly_stats.append({
                'Month': month.strftime('%Y-%m'),
                'Total Trades': len(month_trades),
                'Return (%)': f"{month_return:.2f}",
                'Max Drawdown (%)': f"{month_drawdown:.2f}",
                'Buy Trades': len(month_trades[month_trades['action'] == 1]),
                'Sell Trades': len(month_trades[month_trades['action'] == 2]),
                'Final Balance': f"${month_balance.iloc[-1]:,.2f}"
            })
    
    df_monthly = pd.DataFrame(monthly_stats)
    
    # Calculate yearly statistics
    yearly_stats = []
    for year in df_trades['year'].unique():
        year_trades = df_trades[df_trades['year'] == year]
        year_balance = balance_series[balance_series.index.year == year]
        
        if len(year_balance) > 0:
            year_return = (year_balance.iloc[-1] / year_balance.iloc[0] - 1) * 100
            year_drawdown = ((year_balance.cummax() - year_balance) / year_balance.cummax()).max() * 100
            
            yearly_stats.append({
                'Year': str(year),
                'Total Trades': len(year_trades),
                'Return (%)': f"{year_return:.2f}",
                'Max Drawdown (%)': f"{year_drawdown:.2f}",
                'Buy Trades': len(year_trades[year_trades['action'] == 1]),
                'Sell Trades': len(year_trades[year_trades['action'] == 2]),
                'Final Balance': f"${year_balance.iloc[-1]:,.2f}"
            })
    
    df_yearly = pd.DataFrame(yearly_stats)
    
    # Calculate trade P/L distribution
    trade_pnl = []
    current_position = None
    entry_price = 0
    entry_date = None
    trade_lines = []  # For storing trade lines
    
    for trade in trade_history:
        date, price, action = trade
        
        if action == 1:  # Buy
            if current_position is None:
                current_position = 'buy'
                entry_price = price
                entry_date = date
            elif current_position == 'sell':
                pnl = entry_price - price
                trade_pnl.append(pnl)
                # Add trade line
                trade_lines.append({
                    'x0': entry_date,
                    'x1': date,
                    'y0': entry_price,
                    'y1': price,
                    'type': 'sell'
                })
                current_position = None
        elif action == 2:  # Sell
            if current_position is None:
                current_position = 'sell'
                entry_price = price
                entry_date = date
            elif current_position == 'buy':
                pnl = price - entry_price
                trade_pnl.append(pnl)
                # Add trade line
                trade_lines.append({
                    'x0': entry_date,
                    'x1': date,
                    'y0': entry_price,
                    'y1': price,
                    'type': 'buy'
                })
                current_position = None
    
    # Create figures
    # 1. Price Chart with Trades and Indicators
    fig1 = make_subplots(rows=2, cols=1, 
                        row_heights=[0.7, 0.3],
                        vertical_spacing=0.05,
                        subplot_titles=('Price & Trades', 'Balance'))
    
    # Candlestick chart
    fig1.add_trace(
        go.Candlestick(
            x=test_data.index,
            open=test_data['open'],
            high=test_data['high'],
            low=test_data['low'],
            close=test_data['close'],
            name='USDJPY'
        ),
        row=1, col=1
    )
    
    # Add SMAs
    fig1.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data['SMA_20'],
            name='SMA 20',
            line=dict(color='orange', width=1)
        ),
        row=1, col=1
    )
    
    fig1.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data['SMA_50'],
            name='SMA 50',
            line=dict(color='blue', width=1)
        ),
        row=1, col=1
    )
    
    # Add RSI
    fig1.add_trace(
        go.Scatter(
            x=test_data.index,
            y=test_data['RSI'],
            name='RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    # Add trade points
    buy_points = [(date, price) for date, price, action in trade_history if action == 1]
    sell_points = [(date, price) for date, price, action in trade_history if action == 2]
    
    if buy_points:
        buy_dates, buy_prices = zip(*buy_points)
        fig1.add_trace(
            go.Scatter(
                x=buy_dates,
                y=buy_prices,
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=8, symbol='triangle-up'),
            ),
            row=1, col=1
        )
    
    if sell_points:
        sell_dates, sell_prices = zip(*sell_points)
        fig1.add_trace(
            go.Scatter(
                x=sell_dates,
                y=sell_prices,
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=8, symbol='triangle-down'),
            ),
            row=1, col=1
        )
    
    # Add trade lines
    for line in trade_lines:
        color = 'green' if line['type'] == 'buy' else 'red'
        fig1.add_shape(
            type="line",
            x0=line['x0'],
            y0=line['y0'],
            x1=line['x1'],
            y1=line['y1'],
            line=dict(color=color, width=1, dash="dot"),
            row=1, col=1
        )
    
    # Add balance chart
    fig1.add_trace(
        go.Scatter(
            x=balance_series.index,
            y=balance_series.values,
            name='Balance',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )
    
    fig1.update_layout(
        title='Price Chart with Trades',
        xaxis_title="Date",
        yaxis_title="Price",
        height=800,
        template='plotly_white',
        showlegend=True
    )
    
    # 2. Trade P/L Distribution
    fig2 = go.Figure()
    fig2.add_trace(
        go.Histogram(
            x=trade_pnl,
            name='Trade P/L',
            nbinsx=50,
            marker_color='blue'
        )
    )
    
    fig2.update_layout(
        title='Trade P/L Distribution',
        xaxis_title="Profit/Loss",
        yaxis_title="Frequency",
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    # 3. Monthly Statistics Table
    fig3 = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_monthly.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[df_monthly[col] for col in df_monthly.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig3.update_layout(
        title='Monthly Statistics',
        height=400 * (len(df_monthly) // 10 + 1)
    )
    
    # 4. Yearly Statistics Table
    fig4 = go.Figure(data=[go.Table(
        header=dict(
            values=list(df_yearly.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12)
        ),
        cells=dict(
            values=[df_yearly[col] for col in df_yearly.columns],
            fill_color='lavender',
            align='left',
            font=dict(size=11)
        )
    )])
    
    fig4.update_layout(
        title='Yearly Statistics',
        height=400 * (len(df_yearly) // 5 + 1)
    )
    
    # Save all figures to a single HTML file
    filename = f"backtest_results/backtest_summary_{timestamp}.html"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f'''
        <html>
        <head>
            <title>Backtest Results - {model_name}</title>
            <meta charset="utf-8">
        </head>
        <body>
            <div id="fig1">{fig1.to_html(full_html=False)}</div>
            <div id="fig2">{fig2.to_html(full_html=False)}</div>
            <div id="fig3">{fig3.to_html(full_html=False)}</div>
            <div id="fig4">{fig4.to_html(full_html=False)}</div>
        </body>
        </html>
        ''')
    
    print(f"Backtest visualization saved to: {filename}")

def test_agent(model_path=None):
    """
    Test trading agent with visualization
    
    Args:
        model_path: Path to the trained model
    """
    # Create directories if they don't exist
    os.makedirs("backtest_results", exist_ok=True)
    
    # If no model path provided or model doesn't exist, skip backtesting
    if model_path is None or not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Skipping backtesting.")
        return
    
    model_name = os.path.basename(os.path.dirname(model_path))
    
    # Load test data
    data_path = 'data/raw/USDJPY_5M_2024_data.csv'
    if not os.path.exists(data_path):
        raise ValueError(f"Test data not found at {data_path}")
    
    # Load and preprocess the data
    print("\nLoading test data...")
    test_data = load_forex_data(data_path)
    print(f"Loaded {len(test_data):,} data points")
    print(f"Date range: {test_data.index[0]} to {test_data.index[-1]}")
    
    # Load the environment and model
    print("\nInitializing environment and model...")
    env = ForexTradingEnv(test_data)
    model = PPO.load(model_path, env=env)
    
    # Initialize variables for tracking trades
    trade_history = []
    obs, _ = env.reset()
    step_count = 0
    total_steps = len(test_data)
    last_progress = 0
    start_time = time.time()
    max_retries = 3
    
    print("\nStarting backtest...")
    try:
        while step_count < total_steps:
            # Show progress every 5%
            progress = (step_count / total_steps) * 100
            if int(progress) > last_progress and int(progress) % 5 == 0:
                elapsed_time = time.time() - start_time
                steps_per_sec = step_count / elapsed_time if elapsed_time > 0 else 0
                eta = (total_steps - step_count) / steps_per_sec if steps_per_sec > 0 else 0
                print(f"Progress: {progress:.1f}% ({step_count:,}/{total_steps:,} steps) | "
                      f"Current Balance: {env.balance:.2f} | "
                      f"ETA: {eta/60:.1f} minutes")
                last_progress = int(progress)
            
            # Try to predict action with retries
            retry_count = 0
            while retry_count < max_retries:
                try:
                    action, _states = model.predict(obs)
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count == max_retries:
                        print(f"\nError predicting action after {max_retries} retries: {str(e)}")
                        raise
                    print(f"Retry {retry_count} after error: {str(e)}")
            
            # Execute action
            obs, reward, done, truncated, info = env.step(action)
            
            # Record trade if action is buy or sell
            if action in [1, 2]:  # 1 for buy, 2 for sell
                trade_history.append((test_data.index[step_count], test_data['close'].iloc[step_count], action))
            
            step_count += 1
            
            # Only break if we've reached the end of data
            if done and step_count >= total_steps - 1:
                break
        
        # Calculate trading statistics
        total_time = time.time() - start_time
        print(f"\nBacktest completed in {total_time/60:.1f} minutes")
        print(f"Total steps processed: {step_count:,}")
        
        if step_count < total_steps:
            print(f"\nWARNING: Backtest ended early at step {step_count} out of {total_steps}")
        
        # Create and save visualization
        create_backtest_visualization(env, test_data[:step_count], trade_history, model_name)
        
    except Exception as e:
        print(f"\nError during backtesting: {str(e)}")
        raise

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_agent(model_path=model_path)