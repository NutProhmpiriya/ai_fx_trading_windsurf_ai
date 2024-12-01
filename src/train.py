from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from environment.forex_environment import ForexTradingEnv

def train_forex_model(
    data_path: str = "src/data/raw/USDJPY_16385_data.csv",
    total_timesteps: int = 100000,
    save_path: str = "models"
):
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs(save_path, exist_ok=True)
    
    # Create and wrap the environment
    env = ForexTradingEnv(data_path=data_path)
    env = Monitor(env, "logs")
    env = DummyVecEnv([lambda: env])
    
    # Create evaluation environment
    eval_env = ForexTradingEnv(data_path=data_path)
    eval_env = Monitor(eval_env, "logs")
    eval_env = DummyVecEnv([lambda: eval_env])
    
    # Create evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_path}/best_model",
        log_path="logs",
        eval_freq=10000,
        deterministic=True,
        render=False
    )
    
    # Initialize the agent
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
        tensorboard_log="logs",
        verbose=1
    )
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        progress_bar=True
    )
    
    # Save the final model
    model.save(f"{save_path}/final_model")
    print(f"Training completed. Model saved to {save_path}")

if __name__ == "__main__":
    train_forex_model()