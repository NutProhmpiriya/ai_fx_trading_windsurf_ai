import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO
from environment.forex_environment import ForexTradingEnv
from utils.load_data import load_forex_data
import time

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def create_backtest_visualization(df, trades_info, model_name):
    """
    Create interactive visualization of backtest results
    """
    try:
        # Create figure with secondary y-axis
        fig = make_subplots(
            rows=2, 
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Trades', 'Account Balance'),
            row_heights=[0.7, 0.3]
        )

        # Add price line
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add buy trades
        buy_trades = trades_info[trades_info['action'] == 'buy']
        if not buy_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=buy_trades['time'],
                    y=buy_trades['price'],
                    mode='markers',
                    name='Buy',
                    marker=dict(
                        size=10,
                        color='green',
                        symbol='triangle-up'
                    )
                ),
                row=1, col=1
            )

        # Add sell trades
        sell_trades = trades_info[trades_info['action'] == 'sell']
        if not sell_trades.empty:
            fig.add_trace(
                go.Scatter(
                    x=sell_trades['time'],
                    y=sell_trades['price'],
                    mode='markers',
                    name='Sell',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='triangle-down'
                    )
                ),
                row=1, col=1
            )

        # Add balance
        fig.add_trace(
            go.Scatter(
                x=trades_info['time'],
                y=trades_info['balance'],
                mode='lines',
                name='Balance',
                line=dict(color='blue')
            ),
            row=2, col=1
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'Backtest Results - {model_name}',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Time',
            yaxis_title='Price',
            yaxis2_title='Balance',
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        # Update axes
        fig.update_xaxes(rangeslider_visible=False)
        
        # Save the plot
        output_path = f"backtest_results/backtest_{model_name}.html"
        fig.write_html(output_path)
        
    except Exception as e:
        print(f"Error in visualization: {str(e)}")
        print("trades_info shape:", trades_info.shape)
        print("trades_info columns:", trades_info.columns)
        print("First few rows of trades_info:")
        print(trades_info.head())

def test_agent(model_path=None):
    """
    Test trading agent with visualization
    
    Args:
        model_path: Path to the trained model
    """
    # Create directories if they don't exist
    os.makedirs("backtest_results", exist_ok=True)
    
    if model_path is None:
        model_path = "models/USDJPY_5M_2023_model_20241202_163053/final_model.zip"
    
    model_name = os.path.basename(model_path)
    
    # Check if model exists
    if not os.path.exists(model_path):
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
    trades = []
    trades_info = []
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
        
        trades_info.append({
            'time': env.data.index[env.current_step],
            'action': 'buy' if action == 1 else 'sell',
            'price': env.data['close'].iloc[env.current_step],
            'position': env.position,
            'balance': env.balance
        })
        
        if reward != 0:  # If we closed a position
            trades.append(reward)
            
        step_count += 1
    
    # Convert trades_info to DataFrame
    trades_df = pd.DataFrame(trades_info)
    
    # Calculate trading statistics
    total_time = time.time() - start_time
    print(f"\nBacktest completed in {total_time/60:.1f} minutes")
    print(f"Total steps processed: {step_count:,}")
    print(f"Initial Balance: 10,000.00")
    print(f"Final Balance: {env.balance:.2f}")
    print(f"Total Return: {((env.balance/10000)-1)*100:.2f}%")
    print(f"Number of Trades: {len(trades):,}")
    
    if trades:
        profitable_trades = len([t for t in trades if t > 0])
        print(f"Average Profit per Trade: {np.mean(trades):.5f}")
        print(f"Win Rate: {profitable_trades/len(trades):.2%}")
        print(f"Profit Factor: {sum([t for t in trades if t > 0])/abs(sum([t for t in trades if t < 0])):.2f}")
        print(f"Average Win: {np.mean([t for t in trades if t > 0]):.5f}")
        print(f"Average Loss: {np.mean([t for t in trades if t < 0]):.5f}")
        print(f"Largest Win: {max(trades):.5f}")
        print(f"Largest Loss: {min(trades):.5f}")
    
    # Create visualization
    print("\nGenerating visualization...")
    output_path = f"backtest_results/backtest_{model_name}.html"
    create_backtest_visualization(env.data, trades_df, model_name)
    print(f"Visualization saved to: {output_path}")

if __name__ == "__main__":
    test_agent()