from stable_baselines3 import PPO
from environment.forex_environment import ForexTradingEnv
import pandas as pd
from train import load_forex_data
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

def create_backtest_visualization(df, trades_info, name_model):
    """
    Create interactive visualization of backtest results
    """
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, 
        cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3]
    )

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='OHLC'
    ), row=1, col=1)

    # Add MA indicators
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['SMA_20'],
        name='SMA 20',
        line=dict(color='orange')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['SMA_50'],
        name='SMA 50',
        line=dict(color='blue')
    ), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(
        x=df['time'],
        y=df['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)

    # Add buy signals
    buy_points = trades_info[trades_info['action'] == 'buy']
    fig.add_trace(go.Scatter(
        x=buy_points['time'],
        y=buy_points['price'],
        name='Buy',
        mode='markers',
        marker=dict(
            symbol='triangle-up',
            size=15,
            color='green',
        )
    ), row=1, col=1)

    # Add sell signals
    sell_points = trades_info[trades_info['action'] == 'sell']
    fig.add_trace(go.Scatter(
        x=sell_points['time'],
        y=sell_points['price'],
        name='Sell',
        mode='markers',
        marker=dict(
            symbol='triangle-down',
            size=15,
            color='red',
        )
    ), row=1, col=1)

    # Update layout
    fig.update_layout(
        title=f'Backtest Results - {name_model}',
        yaxis_title='Price',
        yaxis2_title='RSI',
        xaxis_rangeslider_visible=False,
        height=1000
    )

    # Add RSI lines
    fig.add_hline(y=70, line_width=1, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_width=1, line_dash="dash", line_color="green", row=2, col=1)

    # Save the chart
    html_filename = f'src/backtest_results/{name_model}_backtest.html'
    fig.write_html(html_filename)
    print(f"Backtest visualization saved to {html_filename}")
    
    # Show the chart
    fig.show()

def test_agent():
    # Create directories if they don't exist
    os.makedirs("src/models", exist_ok=True)
    os.makedirs("src/backtest_results", exist_ok=True)
    
    model_path = "src/models/USDJPY_5M_2023_model_20241202_020332.zip"
    name_model = model_path.split('/')[-1].replace('.zip', '')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Load the same data used for training
    data_path = 'src/data/raw/USDJPY_5M_2024_data.csv'
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    forex_data = load_forex_data(data_path)
    print(f"Total data points: {len(forex_data)}")
    
    try:
        # Create environment with the same data
        env = ForexTradingEnv(data=forex_data)
        
        # Load the trained model
        model = PPO.load(model_path, env=env)
        
        # Run test episode
        obs, _ = env.reset()
        done = False
        total_reward = 0
        trades = []
        
        # Store trade information for visualization
        trades_info = {
            'time': [],
            'action': [],
            'price': [],
            'reward': [],
            'balance': []
        }
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            total_reward += reward
            
            # Log trading information
            if info['position'] != 0:  # If we have an open position
                print(f"Step: {env.current_step}")
                print(f"Action: {'Buy' if info['position'] == 1 else 'Sell'}")
                print(f"Price: {info['current_price']:.5f}")
                print(f"Reward: {reward:.5f}")
                print(f"Balance: {info['balance']:.2f}")
                print("-" * 50)
                
                # Store trade information for visualization
                trades_info['time'].append(env.data.iloc[env.current_step]['time'])
                trades_info['action'].append('buy' if info['position'] == 1 else 'sell')
                trades_info['price'].append(info['current_price'])
                trades_info['reward'].append(reward)
                trades_info['balance'].append(info['balance'])
            
            if reward != 0:  # If we closed a position
                trades.append(reward)
        
        # Convert trades_info to DataFrame
        trades_df = pd.DataFrame(trades_info)
        
        # Create visualization
        create_backtest_visualization(env.data, trades_df, name_model)
        
        # Print summary statistics
        print("\nTrading Summary:")
        print(f"Final Balance: {info['balance']:.2f}")
        print(f"Total Reward: {total_reward:.2f}")
        print(f"Number of Trades: {len(trades)}")
        if trades:
            print(f"Average Profit per Trade: {np.mean(trades):.5f}")
            print(f"Win Rate: {len([t for t in trades if t > 0]) / len(trades):.2%}")
            print(f"Best Trade: {max(trades):.5f}")
            print(f"Worst Trade: {min(trades):.5f}")
            
    except Exception as e:
        print(f"Error during testing: {str(e)}")

if __name__ == "__main__":
    test_agent()