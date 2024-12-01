import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

class ForexTradingEnv(gym.Env):
    def __init__(self, data: pd.DataFrame, window_size: int = 60):
        super(ForexTradingEnv, self).__init__()
        
        # Store the data
        self.data = data.copy()
        self.window_size = window_size
        
        # Calculate technical indicators
        self.data = self._calculate_indicators(self.data)
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell
        
        # 8 features: OHLCV + 3 technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(window_size, 8),
            dtype=np.float32
        )
        
        self.reset()
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # Add technical indicators
        df['SMA_20'] = df['close'].rolling(window=20).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        df['RSI'] = self._calculate_rsi(df['close'])
        
        # Forward fill NaN values
        df.fillna(method='ffill', inplace=True)
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _get_observation(self) -> np.ndarray:
        # Get the window of data
        start_idx = self.current_step - self.window_size + 1
        end_idx = self.current_step + 1
        window_data = self.data.iloc[start_idx:end_idx]
        
        # Create observation matrix
        obs = np.column_stack((
            window_data['open'].values,
            window_data['high'].values,
            window_data['low'].values,
            window_data['close'].values,
            window_data['volume'].values,
            window_data['SMA_20'].values,
            window_data['SMA_50'].values,
            window_data['RSI'].values
        ))
        
        # Normalize the data
        obs = (obs - obs.mean()) / (obs.std() + 1e-8)
        return obs.astype(np.float32)
    
    def reset(self, seed=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Start after the warmup period
        self.current_step = self.window_size + 50  # Allow for indicator calculation
        self.balance = 10000.0
        self.position = 0  # 0: no position, 1: long, -1: short
        self.entry_price = 0.0
        self.trades = []
        
        return self._get_observation(), {}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        current_price = self.data.iloc[self.current_step]['close']
        reward = 0.0
        
        # Execute trading action
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price
            elif self.position == -1:  # Close short position
                reward = self.entry_price - current_price
                self.position = 0
                self.trades.append(('short', reward))
                
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = current_price
            elif self.position == 1:  # Close long position
                reward = current_price - self.entry_price
                self.position = 0
                self.trades.append(('long', reward))
        
        # Apply transaction cost
        if reward != 0:
            transaction_cost = current_price * 0.0001  # 1 pip commission
            reward -= transaction_cost
            self.balance += reward
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'balance': self.balance,
            'position': self.position,
            'current_price': current_price,
            'trades': self.trades
        }
        
        return self._get_observation(), reward, done, False, info