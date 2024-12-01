from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import ForexTradingEnv

def train_agent():
    # Create and wrap the environment
    env = ForexTradingEnv()
    env = DummyVecEnv([lambda: env])
    
    # Initialize the agent
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1
    )
    
    # Train the agent
    model.learn(total_timesteps=100000)
    
    # Save the trained model
    model.save("forex_trading_model")

if __name__ == "__main__":
    train_agent()