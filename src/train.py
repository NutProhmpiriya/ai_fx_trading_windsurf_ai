from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
import pandas as pd
from environment.forex_environment import ForexTradingEnv
import torch

def load_forex_data(data_path: str) -> pd.DataFrame:
    """
    Load and preprocess forex data
    """
    # Read CSV file
    df = pd.read_csv(data_path)
    
    # Convert time column to datetime
    df['time'] = pd.to_datetime(df['time'])
    
    # Add volume if not present (common in Forex data)
    if 'volume' not in df.columns:
        df['volume'] = 1.0
    
    return df

def train_forex_model():
    # Create necessary directories
    os.makedirs("src/models", exist_ok=True)
    os.makedirs("src/logs", exist_ok=True)
    os.makedirs("src/data/raw", exist_ok=True)
    
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
    forex_data = load_forex_data(data_path)
    
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
        best_model_save_path=f"src/models/{name_model}",
        log_path="src/logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
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
        device=device  # Set device for training
    )
    
    num_bars = len(forex_data)
    print(f"Number of bars: {num_bars}")
    total_timesteps = num_bars
    print(f"Total timesteps: {total_timesteps}")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save(f"src/models/{name_model}")
    print(f"Training completed. Model saved to models/{name_model}")

if __name__ == "__main__":
    train_forex_model()