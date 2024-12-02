import sys
import os
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from environment.forex_environment import ForexTradingEnv
from test import test_agent
from utils.load_data import load_forex_data

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

def train_forex_model():
    # Create necessary directories
    os.makedirs("src/models", exist_ok=True)
    os.makedirs("src/logs", exist_ok=True)
    os.makedirs("src/data/raw", exist_ok=True)
    os.makedirs("src/backtest_results", exist_ok=True)
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f}MB")
        print(f"Cached: {torch.cuda.memory_reserved(0)/1024**2:.2f}MB")
    
    # Load and preprocess data
    name_data = "USDJPY_5M_2023_data.csv"
    data_path = 'src/data/raw/' + name_data
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    name_model = "USDJPY_5M_2023_model_" + now
    model_save_path = f"src/models/{name_model}"
    
    print("Loading data...")
    forex_data = load_forex_data(data_path)
    print(f"Data loaded. Shape: {forex_data.shape}")
    
    # Create and wrap the environment
    env = ForexTradingEnv(data=forex_data)
    env = Monitor(env, "src/logs")
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = ForexTradingEnv(data=forex_data)
    eval_env = Monitor(eval_env, "src/logs")
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path="src/logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    print("Creating model...")
    # Initialize the agent with GPU support
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="src/logs",
        verbose=1,
        device=device
    )
    
    num_bars = len(forex_data)
    print(f"Number of bars: {num_bars}")
    total_timesteps = num_bars 
    print(f"Total timesteps: {total_timesteps}")
    
    print("Starting training...")
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    final_model_path = f"{model_save_path}/final_model"
    model.save(final_model_path)
    print(f"Training completed. Model saved to {final_model_path}")
    return final_model_path

if __name__ == "__main__":
    model_path = train_forex_model()
    
    # Run backtesting
    print("\nStarting backtesting...")
    test_agent(model_path=model_path)
