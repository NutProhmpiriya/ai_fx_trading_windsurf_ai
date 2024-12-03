import sys
import os
import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from rl_env.forex_environment import ForexTradingEnv
from test import test_agent
from utils.load_data import load_forex_data

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.append(src_dir)

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq: int = 1000, patience: int = 5, min_improvement: float = 0.01, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_mean_reward = -np.inf
        self.no_improvement_count = 0
        self.rewards_history = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Get the mean episode reward from the last 100 episodes
            if len(self.training_env.get_attr("rewards")) > 0:
                current_rewards = self.training_env.get_attr("rewards")[0][-100:]
                if len(current_rewards) > 0:
                    current_mean_reward = np.mean(current_rewards)
                    self.rewards_history.append(current_mean_reward)
                    
                    # Check if there's significant improvement
                    if current_mean_reward > self.best_mean_reward + self.min_improvement:
                        self.best_mean_reward = current_mean_reward
                        self.no_improvement_count = 0
                        if self.verbose > 0:
                            print(f"New best mean reward: {self.best_mean_reward:.3f}")
                    else:
                        self.no_improvement_count += 1
                        if self.verbose > 0:
                            print(f"No improvement for {self.no_improvement_count} checks")
                            print(f"Current mean reward: {current_mean_reward:.3f}")
                            print(f"Best mean reward: {self.best_mean_reward:.3f}")
                    
                    # Stop training if no improvement for too long
                    if self.no_improvement_count >= self.patience:
                        if self.verbose > 0:
                            print("Stopping training due to no improvement")
                        return False
                
        return True

def train_forex_model():
    # Create necessary directories
    os.makedirs("rl_models", exist_ok=True)
    os.makedirs("rl_model_logs", exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("backtest_results", exist_ok=True)
    
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
    data_path = 'data/raw/' + name_data
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return
        
    now = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    name_model = "USDJPY_5M_2023_model_" + now
    model_save_path = f"rl_models/{name_model}"
    
    print("Loading data...")
    forex_data = load_forex_data(data_path)
    print(f"Data loaded. Shape: {forex_data.shape}")
    
    # Create and wrap the environment
    env = ForexTradingEnv(data=forex_data)
    env = Monitor(env, "rl_model_logs")
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = ForexTradingEnv(data=forex_data)
    eval_env = Monitor(eval_env, "rl_model_logs")
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_save_path,
        log_path="rl_model_logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    print("Creating model...")
    # Initialize the agent with GPU support
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-3,  
        n_steps=2048,
        batch_size=128,      
        n_epochs=10,
        gamma=0.995,         
        gae_lambda=0.98,     
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.02,      
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log="rl_model_logs",
        verbose=1,
        device=device
    )
    
    num_bars = len(forex_data)
    print(f"Number of bars: {num_bars}")
    total_timesteps = num_bars 
    print(f"Total timesteps: {total_timesteps}")
    
    print("Starting training...")
    # Create early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        check_freq=5000,     
        patience=20,         
        min_improvement=0.0005  
    )
    
    # Train the agent
    try:
        # Train the model with early stopping
        model.learn(
            total_timesteps=total_timesteps,
            callback=early_stopping_callback,
            progress_bar=True
        )
        print("Training completed successfully")
        training_completed = True
    except KeyboardInterrupt:  # Handle manual interruption
        print("Training manually interrupted")
        training_completed = False
    except StopTrainingException:  # Handle early stopping
        print("Training completed early due to early stopping")
        training_completed = True
    except Exception as e:
        print(f"Training stopped due to error: {str(e)}")
        if "Stopping training due to no improvement" in str(e):
            print("This was due to early stopping - saving model")
            training_completed = True
        else:
            training_completed = False
    
    # Save the final model only if training completed or early stopped
    if training_completed:
        try:
            final_model_path = f"{model_save_path}/final_model.zip"
            os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
            model.save(final_model_path)
            print(f"Model saved to {final_model_path}")
            
            # Start backtesting immediately after saving
            print("\nStarting backtesting...")
            test_agent(model_path=final_model_path)
            
            return final_model_path, training_completed
        except Exception as e:
            print(f"Error saving model or running backtest: {str(e)}")
            return None, False
    else:
        print("Training did not complete. Model will not be saved.")
        return None, training_completed

if __name__ == "__main__":
    model_path, training_completed = train_forex_model()

        
