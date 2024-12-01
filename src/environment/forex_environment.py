import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data_processor import ForexDataProcessor

class ForexTradingEnv(gym.Env):
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M5, window_size=100):
        super(ForexTradingEnv, self).__init__()
        
        self.data_processor = ForexDataProcessor(symbol, timeframe)
        self.window_size = window_size
        self.commission = 0.0001  # 1 pip commission
        
        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space: OHLCV + indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        
        self.reset()
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.balance = 10000.0
        self.entry_price = 0.0
        
        # Get initial data
        self.data = self.data_processor.get_data(0, self.window_size + 100)
        self.data = self.data_processor.calculate_indicators(self.data)
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        current_data = self.data.iloc[self.current_step]
        
        obs = np.array([
            current_data['open'],
            current_data['high'],
            current_data['low'],
            current_data['close'],
            current_data['volume'],
            current_data['SMA_20'],
            current_data['SMA_50'],
            current_data['RSI']
        ], dtype=np.float32)
        
        return obs
    
    def step(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:
                reward = self.entry_price - current_price - self.commission
                self.position = 0
                
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:
                reward = current_price - self.entry_price - self.commission
                self.position = 0
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        # Update balance
        self.balance += reward
        
        return self._get_observation(), reward, done, False, {'balance': self.balance}