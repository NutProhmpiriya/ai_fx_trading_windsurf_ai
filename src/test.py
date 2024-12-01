from stable_baselines3 import PPO
from environment.forex_environment import ForexTradingEnv
import pandas as pd
from train import load_forex_data
import numpy as np

def test_agent():
    model_path: str = "src\models\model_20241202_011115"
    # Load the same data used for training
    forex_data = load_forex_data('src/data/raw/USDJPY_16385_data.csv')
    
    # Create environment with the same data
    env = ForexTradingEnv(data=forex_data)
    
    # Load the trained model
    model = PPO.load(model_path, env=env)
    
    # Run test episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    trades = []
    
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
        
        if reward != 0:  # If we closed a position
            trades.append(reward)
    
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

if __name__ == "__main__":
    test_agent()