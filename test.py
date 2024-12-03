import os
import sys
import pandas as pd
import datetime
from stable_baselines3 import PPO
from rl_env.forex_environment import ForexTradingEnv
from utils.load_data import load_forex_data
import plotly
from utils.visualization import create_backtest_visualization
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
    model = PPO.load(model_path)
    model_name = os.path.basename(os.path.dirname(model_path))
    
    # Run test episode
    obs, _ = env.reset()
    done = False
    trade_history = []

    total_steps = len(test_data)
    last_step = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        # Record trade if action was taken
        if action > 0:
            current_step = env.current_step
            current_price = test_data.iloc[current_step]['close']
            trade_history.append((test_data.index[current_step], current_price, action))
        
        # Update progress
        progress = env.current_step - last_step
        if progress > 0:
            last_step = env.current_step
            progress_pct = (env.current_step / total_steps) * 100
            
            # Show current state
            print(f"\rProgress: {progress_pct:.1f}% | "
                  f"Step: {env.current_step}/{total_steps} | "
                  f"Balance: {env.balance:.2f} | "
                  f"Price: {test_data.iloc[env.current_step]['close']:.2f} | "
                  f"Action: {action}", end="", flush=True)
    
    print("Backtest completed!", flush=True)
    
    # สร้าง list สำหรับเก็บข้อมูลการเทรด
    print("\nProcessing trade data...", flush=True)
    trades_list = []
    
    # ตัวแปรสำหรับคำนวณผลการเทรด
    current_position = None
    entry_price = 0
    entry_date = None
    cumulative_pnl = 0.0
    max_balance = env.initial_balance
    max_drawdown = 0
    
    for date, price, action in trade_history:
        if action == 1:  # Buy
            if current_position is None:
                current_position = 'buy'
                entry_price = price
                entry_date = date
            elif current_position == 'sell':
                # ปิด Short Position
                pnl = entry_price - price
                cumulative_pnl += pnl
                
                # คำนวณ drawdown
                current_balance = env.initial_balance + cumulative_pnl
                if current_balance > max_balance:
                    max_balance = current_balance
                drawdown = (max_balance - current_balance) / max_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
                
                # บันทึกข้อมูลการเทรด
                trade_data = {
                    'entry_time': entry_date,
                    'exit_time': date,
                    'trade_type': 'sell',
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size': 1.0,  # หรือตามที่กำหนดใน environment
                    'pnl': pnl,
                    'cumulative_pnl': cumulative_pnl,
                    'return_pct': (pnl / entry_price) * 100,
                    'holding_period': (date - entry_date).total_seconds() / 60,  # เป็นนาที
                    'trade_result': 'win' if pnl > 0 else 'loss',
                    'drawdown': drawdown,
                    'max_drawdown': max_drawdown
                }
                trades_list.append(trade_data)
                current_position = None
                
        elif action == 2:  # Sell
            if current_position is None:
                current_position = 'sell'
                entry_price = price
                entry_date = date
            elif current_position == 'buy':
                # ปิด Long Position
                pnl = price - entry_price
                cumulative_pnl += pnl
                
                # คำนวณ drawdown
                current_balance = env.initial_balance + cumulative_pnl
                if current_balance > max_balance:
                    max_balance = current_balance
                drawdown = (max_balance - current_balance) / max_balance * 100
                max_drawdown = max(max_drawdown, drawdown)
                
                # บันทึกข้อมูลการเทรด
                trade_data = {
                    'entry_time': entry_date,
                    'exit_time': date,
                    'trade_type': 'buy',
                    'entry_price': entry_price,
                    'exit_price': price,
                    'position_size': 1.0,  # หรือตามที่กำหนดใน environment
                    'pnl': pnl,
                    'cumulative_pnl': cumulative_pnl,
                    'return_pct': (pnl / entry_price) * 100,
                    'holding_period': (date - entry_date).total_seconds() / 60,  # เป็นนาที
                    'trade_result': 'win' if pnl > 0 else 'loss',
                    'drawdown': drawdown,
                    'max_drawdown': max_drawdown
                }
                trades_list.append(trade_data)
                current_position = None
    
    # สร้าง DataFrame และคำนวณสถิติเพิ่มเติม
    trades_df = pd.DataFrame(trades_list)
    if not trades_df.empty:
        # คำนวณ Win Rate
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['trade_result'] == 'win'])
        win_rate = (winning_trades / total_trades) * 100
        
        # เพิ่มคอลัมน์ Win Rate
        trades_df['win_rate'] = win_rate
    
    # บันทึกข้อมูลลง CSV
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_csv_path = f"trades_{model_name}_{timestamp}.csv"
    trades_df.to_csv(trades_csv_path, index=False)
    print(f"Trade history saved to {trades_csv_path}", flush=True)
    
    if not trades_df.empty:
        print("\nTrading Statistics:")
        print(f"Total Trades: {total_trades}")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Total PnL: {cumulative_pnl:.2f}")
        print(f"Max Drawdown: {max_drawdown:.2f}%")
    
    # Create visualization
    print("\nGenerating visualization...", flush=True)
    fig, stats = create_backtest_visualization(test_data, trade_history, model_name)
    
    # Print statistics
    print("\nBacktest Statistics:", flush=True)
    for key, value in stats.items():
        print(f"{key}: {value}", flush=True)
    
    # Save figures
    print("\nSaving results...", flush=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fig.write_html(f"backtest_results_{timestamp}.html")
    print("Done!", flush=True)

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else None
    test_agent(model_path=model_path)