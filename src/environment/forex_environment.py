import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ForexTradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame):
        super(ForexTradingEnv, self).__init__()
        
        # Make a copy of the data to avoid modifying the original
        self.data = data.copy()
        if 'time' in self.data.columns:
            self.data.set_index('time', inplace=True)
            
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.position_price = 0
        self.transaction_fee = 0.0001  # 0.01% per trade
        self.rewards = []  # Track rewards for early stopping
        
        # Calculate indicators
        self.data['SMA20'] = self.data['close'].rolling(window=20).mean()
        self.data['SMA50'] = self.data['close'].rolling(window=50).mean()
        self.data['RSI'] = self._calculate_rsi(self.data['close'], periods=14)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell
        
        # Observation space: [position, normalized_price, normalized_volume, normalized_sma20, normalized_sma50, normalized_rsi]
        self.observation_space = spaces.Box(
            low=np.array([-1, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([1, np.inf, np.inf, np.inf, np.inf, 1], dtype=np.float32)
        )
        
    def _calculate_rsi(self, prices, periods=14):
        deltas = np.diff(prices)
        seed = deltas[:periods+1]
        up = seed[seed >= 0].sum()/periods
        down = -seed[seed < 0].sum()/periods
        rs = up/down
        rsi = np.zeros_like(prices)
        rsi[:periods] = 100. - 100./(1. + rs)
        
        for i in range(periods, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
                
            up = (up * (periods - 1) + upval) / periods
            down = (down * (periods - 1) + downval) / periods
            rs = up/down
            rsi[i] = 100. - 100./(1. + rs)
            
        return rsi
        
    def _get_observation(self):
        """
        Get current observation
        Returns normalized values of:
        - Current position
        - Price
        - Volume (or 1.0 if not available)
        - Technical indicators
        """
        current_price = self.data.iloc[self.current_step]['close']
        
        # Calculate technical indicators
        window_20 = self.data.iloc[max(0, self.current_step-19):self.current_step+1]['close']
        window_50 = self.data.iloc[max(0, self.current_step-49):self.current_step+1]['close']
        
        sma_20 = window_20.mean()
        sma_50 = window_50.mean()
        
        # Calculate RSI
        delta = self.data.iloc[max(0, self.current_step-14):self.current_step+1]['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs.iloc[-1]))
        
        # Use volume if available, otherwise use 1.0
        volume = 1.0
        if 'volume' in self.data.columns:
            volume = self.data.iloc[self.current_step]['volume'] / self.data.iloc[0]['volume']
        
        return np.array([
            self.position,  # Current position (-1: short, 0: neutral, 1: long)
            current_price / self.data.iloc[0]['close'],  # Normalized price
            volume,  # Normalized volume or 1.0
            sma_20 / self.data.iloc[0]['close'],  # Normalized SMA20
            sma_50 / self.data.iloc[0]['close'],  # Normalized SMA50
            rsi / 100.0  # Normalized RSI
        ], dtype=np.float32)
        
    def _calculate_reward(self, action):
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0
        
        # Penalize invalid actions
        if (action == 1 and self.position == 1) or (action == 2 and self.position == -1):
            return -0.1
            
        # Calculate profit/loss for position changes
        if action == 1:  # Buy
            if self.position == -1:  # Close short position
                profit = self.position_price - current_price
                reward = profit * 100  # Scale up the reward
            elif self.position == 0:  # Open long position
                reward = 0.1  # Small positive reward for opening position
            self.position = 1
            self.position_price = current_price
            
        elif action == 2:  # Sell
            if self.position == 1:  # Close long position
                profit = current_price - self.position_price
                reward = profit * 100  # Scale up the reward
            elif self.position == 0:  # Open short position
                reward = 0.1  # Small positive reward for opening position
            self.position = -1
            self.position_price = current_price
            
        # Add small negative reward for holding to encourage action
        elif action == 0 and self.position != 0:  # Holding a position
            price_change = abs(current_price - self.position_price)
            reward = -0.01 if price_change < 0.0001 else 0
            
        # Update balance based on current position value
        if self.position != 0:
            self.balance += reward
            
        return reward
        
    def step(self, action):
        # Get current price
        current_price = self.data.iloc[self.current_step]['close']
        
        # Execute action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.position_price = current_price
            elif self.position == -1:  # Close short position
                self.position = 0
                
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.position_price = current_price
            elif self.position == 1:  # Close long position
                self.position = 0
                
        # Calculate reward
        reward = self._calculate_reward(action)
        self.rewards.append(reward)  # Track reward
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        
        obs = self._get_observation()
        info = {
            'current_price': current_price,
            'position': self.position,
            'balance': self.balance
        }
        
        return obs, reward, done, False, info
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 50  # Start after indicators are calculated
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.rewards = []  # Reset rewards
        
        return self._get_observation(), {}
        
    def render(self):
        pass