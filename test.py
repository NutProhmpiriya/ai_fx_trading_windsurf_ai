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

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def create_backtest_visualization(env, test_data, trade_history, model_name):
    # Create timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create subplots
    fig = make_subplots(rows=3, cols=1,
                       subplot_titles=('Price and Trades', 'Account Balance', 'Trade Positions'),
                       row_heights=[0.5, 0.25, 0.25],
                       vertical_spacing=0.1)

    # Plot 1: Price movement and trade actions
    dates = test_data.index
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=dates,
            open=test_data['open'],
            high=test_data['high'],
            low=test_data['low'],
            close=test_data['close'],
            name='USDJPY'
        ),
        row=1, col=1
    )

    # Add buy/sell markers
    if trade_history:
        buy_points = [(date, price) for date, price, action in trade_history if action == 1]
        sell_points = [(date, price) for date, price, action in trade_history if action == 2]
        
        if buy_points:
            buy_dates, buy_prices = zip(*buy_points)
            fig.add_trace(
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
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode='markers',
                    name='Sell',
                    marker=dict(color='red', size=8, symbol='triangle-down'),
                ),
                row=1, col=1
            )

    # Plot 2: Account Balance
    balance_history = env.balance_history
    fig.add_trace(
        go.Scatter(
            x=dates[:len(balance_history)],
            y=balance_history,
            name='Balance',
            line=dict(color='blue'),
        ),
        row=2, col=1
    )

    # Plot 3: Position History
    position_history = env.position_history
    fig.add_trace(
        go.Scatter(
            x=dates[:len(position_history)],
            y=position_history,
            name='Position',
            line=dict(color='purple'),
        ),
        row=3, col=1
    )

    # Update layout
    fig.update_layout(
        title=f'Backtest Results - {model_name}',
        xaxis_rangeslider_visible=False,
        height=1200,
        showlegend=True
    )

    # Save the plot
    output_file = f"backtest_results/backtest_visualization_{timestamp}.html"
    fig.write_html(output_file)
    
    print(f"\nVisualization saved to: {output_file}")
    
    # Create summary statistics
    total_trades = len(trade_history)
    buy_trades = sum(1 for _, _, action in trade_history if action == 1)
    sell_trades = sum(1 for _, _, action in trade_history if action == 2)
    
    initial_balance = 10000
    final_balance = env.balance
    profit_loss = final_balance - initial_balance
    return_pct = (profit_loss / initial_balance) * 100
    
    print("\nBacktest Summary:")
    print(f"Total Trades: {total_trades}")
    print(f"Buy Trades: {buy_trades}")
    print(f"Sell Trades: {sell_trades}")
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Final Balance: ${final_balance:,.2f}")
    print(f"Profit/Loss: ${profit_loss:,.2f}")
    print(f"Return: {return_pct:.2f}%")

def test_agent(model_path=None):
    """
    Test trading agent with visualization
    
    Args:
        model_path: Path to the trained model
    """
    # Create directories if they don't exist
    os.makedirs("backtest_results", exist_ok=True)
    
    if model_path is None:
        model_path = "rl_models/USDJPY_5M_2023_model_20241202_163053/final_model.zip"
    
    model_name = os.path.basename(os.path.dirname(model_path))
    
    # Check if model exists
    if not os.path.exists(model_path):
        # Try to find the most recent model
        models_dir = "rl_models"
        if os.path.exists(models_dir):
            model_dirs = [d for d in os.listdir(models_dir) if d.startswith("USDJPY_5M_2023_model_")]
            if model_dirs:
                latest_model = max(model_dirs)
                model_path = os.path.join(models_dir, latest_model, "final_model.zip")
                if os.path.exists(model_path):
                    print(f"Using latest model: {model_path}")
                else:
                    raise ValueError(f"Model not found at {model_path}")
            else:
                raise ValueError("No model directories found")
        else:
            raise ValueError(f"Model not found at {model_path}")
    
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
    done = False
    truncated = False
    obs, _ = env.reset()  # Get only observation from reset
    step_count = 0
    total_steps = len(test_data)
    last_progress = 0
    start_time = time.time()
    
    print("\nStarting backtest...")
    while not (done or truncated) and step_count < total_steps:
        # Show progress every 5%
        progress = (step_count / total_steps) * 100
        if int(progress) > last_progress and int(progress) % 5 == 0:
            elapsed_time = time.time() - start_time
            steps_per_sec = step_count / elapsed_time
            eta = (total_steps - step_count) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"Progress: {progress:.1f}% ({step_count:,}/{total_steps:,} steps) | "
                  f"Current Balance: {env.balance:.2f} | "
                  f"ETA: {eta/60:.1f} minutes")
            last_progress = int(progress)
            
        action, _states = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)  # Handle new step return format
        
        # Record trade if action is buy or sell
        if action in [1, 2]:  # 1 for buy, 2 for sell
            trade_history.append((test_data.index[step_count], test_data['close'].iloc[step_count], action))
        
        step_count += 1
    
    # Calculate trading statistics
    total_time = time.time() - start_time
    print(f"\nBacktest completed in {total_time/60:.1f} minutes")
    print(f"Total steps processed: {step_count:,}")
    
    # Check if all steps were completed
    if step_count < total_steps:
        print(f"\nWARNING: Model testing failed - Incomplete steps!")
        print(f"Processed {step_count} steps out of {len(test_data)}")
        return
        
    print(f"Initial Balance: 10,000.00")
    print(f"Final Balance: {env.balance:.2f}")
    print(f"Total Return: {((env.balance/10000)-1)*100:.2f}%")
    
    # Create and save visualization
    create_backtest_visualization(env, test_data, trade_history, model_name)

if __name__ == "__main__":
    test_agent()