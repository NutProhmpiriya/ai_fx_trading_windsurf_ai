import os
import sys
import pandas as pd
import datetime
import time
from stable_baselines3 import PPO
from rl_env.forex_environment import ForexTradingEnv
from utils.load_data import load_forex_data
import plotly
from utils.visualization import analyze_trade_history, analyze_trade_patterns
from tqdm import tqdm

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def test_agent(model_path=None):
    """
    Test trading agent with visualization
    
    Args:
        model_path: Path to the trained model
    """
    print("\nLoading test data...", flush=True)
    test_data = load_forex_data('data/raw/USDJPY_5M_2023_data.csv')
    print(f"Loaded {len(test_data):,} data points", flush=True)
    
    print("Initializing environment...", flush=True)
    env = ForexTradingEnv(data=test_data)
    
    # Load model
    if model_path is None:
        print("No model path provided. Please provide path to trained model.", flush=True)
        return
        
    print("Loading model...", flush=True)
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully!", flush=True)
    except Exception as e:
        print(f"Error loading model: {str(e)}", flush=True)
        return
    model_name = os.path.basename(os.path.dirname(model_path))
    
    # Run test episode
    print("\nStarting backtesting...", flush=True)
    obs, _ = env.reset()
    done = False
    trade_history = []

    total_steps = len(test_data)
    print(f"Total steps to process: {total_steps}", flush=True)
    trades_count = 0
    start_time = datetime.datetime.now()
    last_progress_time = start_time
    
    while not done:
        try:
            # Add small delay to avoid rate limit
            time.sleep(0.001)  # 1ms delay
            
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            current_step = env.current_step  # Use environment's step counter
            
            if info.get('trade_executed', False):
                trade_history.append(info['trade_info'])
                trades_count += 1
            
            # Show progress every 5 seconds
            current_time = datetime.datetime.now()
            if (current_time - last_progress_time).total_seconds() >= 5:
                progress = (current_step / total_steps) * 100
                elapsed_time = (current_time - start_time).total_seconds()
                steps_per_second = current_step / elapsed_time if elapsed_time > 0 else 0
                
                print(f"\rProgress: {progress:.1f}% - Steps: {current_step}/{total_steps} - "
                      f"Trades: {trades_count} - Speed: {steps_per_second:.1f} steps/s", 
                      end="", flush=True)
                last_progress_time = current_time
                
        except Exception as e:
            print(f"\nError during backtesting: {str(e)}")
            break
    
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    print(f"\nBacktest completed in {elapsed_time:.1f} seconds!", flush=True)
    print(f"Total trades executed: {trades_count}", flush=True)
    
    # สร้าง DataFrame จาก trade_history
    print("\nProcessing trade data...", flush=True)
    trades_df = pd.DataFrame(trade_history)
    
    # Calculate statistics
    if not trades_df.empty:
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        cumulative_pnl = trades_df['pnl'].sum()
        
        # Calculate drawdown
        trades_df['cumulative_balance'] = env.initial_balance + trades_df['pnl'].cumsum()
        max_balance = trades_df['cumulative_balance'].expanding().max()
        drawdown = ((max_balance - trades_df['cumulative_balance']) / max_balance * 100)
        max_drawdown = drawdown.max()
        
        print("\nTrading Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL: {cumulative_pnl:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Save trades to CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_csv_path = os.path.join('backtest_results', f"trades_{model_name}_{timestamp}.csv")
    os.makedirs('backtest_results', exist_ok=True)
    trades_df.to_csv(trades_csv_path, index=False)
    print(f"\nTrade history saved to {trades_csv_path}")
    
    # Create visualization
    print("\nGenerating visualization...", flush=True)
    fig, stats = analyze_trade_history(trades_df, test_data)
    
    # Create and save analysis charts
    print("\nGenerating analysis charts...", flush=True)
    analysis_fig = analyze_trade_patterns(trades_df)
    
    # Save figures
    print("\nSaving results...", flush=True)
    fig.write_html(os.path.join('backtest_results', f"backtest_results_{timestamp}.html"))
    analysis_fig.write_html(os.path.join('backtest_results', f"backtest_analysis_{timestamp}.html"))
    print("Done!", flush=True)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_agent(model_path=model_path)