from stable_baselines3 import PPO
from env import ForexTradingEnv

def test_agent():
    # Load the environment and model
    env = ForexTradingEnv()
    model = PPO.load("forex_trading_model")
    
    # Run test episode
    obs, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        print(f"Balance: {info['balance']:.2f}")
    
    print(f"Final balance: {info['balance']:.2f}")
    print(f"Total reward: {total_reward:.2f}")

if __name__ == "__main__":
    test_agent()