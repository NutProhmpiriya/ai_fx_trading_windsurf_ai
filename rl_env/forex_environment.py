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
            
        # Calculate indicators first
        self.data['SMA20'] = self.data['close'].rolling(window=20).mean()
        self.data['SMA50'] = self.data['close'].rolling(window=50).mean()
        self.data['RSI'] = self._calculate_rsi(self.data['close'], periods=14)
        
        self.current_step = 0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance
        self.position = 0  # -1: short, 0: neutral, 1: long
        self.position_price = 0
        self.transaction_fee = 0.0001  # 0.01% per trade
        
        # Risk management parameters
        self.leverage = 100  # 1:100 leverage
        self.max_daily_loss_pct = 0.01  # 1% max daily loss
        self.max_daily_loss = self.initial_balance * self.max_daily_loss_pct
        
        # Tracking variables
        self.daily_profit_loss = 0
        self.daily_high_balance = self.initial_balance
        self.max_drawdown = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.last_trade_day = None
        self.balance_history = []
        self.position_history = []
        self.daily_high_balance_history = []
        self.daily_pl_history = []
        self.rewards = []  # Track rewards for early stopping
        
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
        
        # Get technical indicators with fallback values
        sma_20 = self.data.iloc[self.current_step]['SMA20']
        sma_50 = self.data.iloc[self.current_step]['SMA50']
        rsi = self.data.iloc[self.current_step]['RSI']
        
        # Handle NaN values
        if np.isnan(sma_20):
            sma_20 = current_price
        if np.isnan(sma_50):
            sma_50 = current_price
        if np.isnan(rsi):
            rsi = 50.0  # Neutral RSI value
        
        # Use volume if available, otherwise use 1.0
        volume = 1.0
        if 'volume' in self.data.columns:
            volume = self.data.iloc[self.current_step]['volume']
            if np.isnan(volume):
                volume = 1.0
            else:
                volume = volume / self.data.iloc[0]['volume']
        
        # Normalize values
        norm_price = current_price / self.data.iloc[0]['close']
        norm_sma20 = sma_20 / self.data.iloc[0]['close']
        norm_sma50 = sma_50 / self.data.iloc[0]['close']
        norm_rsi = rsi / 100.0  # RSI is already 0-100
        
        # Replace any remaining NaN values with 0
        obs = np.array([
            float(self.position),  # Current position (-1: short, 0: neutral, 1: long)
            float(norm_price),  # Normalized price
            float(volume),  # Normalized volume or 1.0
            float(norm_sma20),  # Normalized SMA20
            float(norm_sma50),  # Normalized SMA50
            float(norm_rsi)  # Normalized RSI
        ], dtype=np.float32)
        
        # Replace any remaining NaN values with 0
        obs = np.nan_to_num(obs, nan=0.0)
        
        return obs
        
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
        # Get current price and time
        current_price = self.data.iloc[self.current_step]['close']
        current_time = self.data.index[self.current_step]
        current_day = current_time.date()
        
        # Reset daily P/L if it's a new day
        if self.last_trade_day is None or current_day != self.last_trade_day:
            self.daily_profit_loss = 0
            self.daily_high_balance = self.balance
            self.last_trade_day = current_day
        
        # Check if daily loss limit is exceeded
        if self.daily_profit_loss < -self.max_daily_loss:
            # Force close any open position and reset daily stats
            if self.position != 0:
                self.position = 0
                self.position_price = 0
            # Reset daily tracking (start new trading day)
            self.daily_profit_loss = 0
            self.daily_high_balance = self.balance
            self.last_trade_day = current_day
            return self._get_observation(), -1, False, False, {'message': 'Daily loss limit exceeded - Starting new day'}
        
        # Check if it's Friday and near market close (after 21:00 UTC)
        if current_time.weekday() == 4 and current_time.hour >= 21 and self.position != 0:  # 4 = Friday
            # Calculate profit/loss for closing position
            if self.position == 1:  # Close long position
                trade_profit = (current_price - self.position_price) * self.leverage
            else:  # Close short position
                trade_profit = (self.position_price - current_price) * self.leverage
            
            # Update stats and close position
            self._update_trade_stats(trade_profit)
            self.balance += trade_profit
            self.daily_profit_loss += trade_profit
            self.position = 0
            self.position_price = 0
            info = {
                'message': 'Position closed for weekend (Friday market close)',
                'trade_executed': True,
                'trade_info': {
                    'action': 'weekend_close',
                    'price': current_price,
                    'pnl': trade_profit,
                    'balance': self.balance,
                    'timestamp': current_time
                }
            }
            return self._get_observation(), 0, False, False, info
            
        # Skip trading on weekends
        if current_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return self._get_observation(), 0, False, False, {'message': 'Market closed (Weekend)'}
            
        # Execute action with leverage
        reward = 0
        trade_profit = 0
        
        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.position_price = current_price
                self.total_trades += 1
            elif self.position == -1:  # Close short position
                trade_profit = (self.position_price - current_price) * self.leverage
                self.position = 0
                self.position_price = 0
                self._update_trade_stats(trade_profit)
                
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.position_price = current_price
                self.total_trades += 1
            elif self.position == 1:  # Close long position
                trade_profit = (current_price - self.position_price) * self.leverage
                self.position = 0
                self.position_price = 0
                self._update_trade_stats(trade_profit)
        
        # Update balance and track history
        self.balance += trade_profit
        self.daily_profit_loss += trade_profit
        self.balance_history.append(self.balance)
        self.position_history.append(self.position)
        self.daily_high_balance_history.append(self.daily_high_balance)
        self.daily_pl_history.append(self.daily_profit_loss)
        
        # Update drawdown
        if self.balance > self.daily_high_balance:
            self.daily_high_balance = self.balance
        current_drawdown = (self.daily_high_balance - self.balance) / self.daily_high_balance
        self.max_drawdown = max(self.max_drawdown, current_drawdown)
        
        # Calculate reward based on profit/loss
        reward = trade_profit
        
        # Move to next step
        self.current_step += 1
        
        # Check if we've reached the end of data
        done = False
        if self.current_step >= len(self.data) - 1:
            done = True
            # Close any open position at the last step
            if self.position != 0:
                final_trade_profit = 0
                if self.position == 1:
                    final_trade_profit = (current_price - self.position_price) * self.leverage
                else:
                    final_trade_profit = (self.position_price - current_price) * self.leverage
                self.balance += final_trade_profit
                self.daily_profit_loss += final_trade_profit
                self._update_trade_stats(final_trade_profit)
                self.position = 0
        
        # Get trading metrics for info
        info = {
            'current_price': current_price,
            'position': self.position,
            'balance': self.balance,
            'daily_profit_loss': self.daily_profit_loss,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.winning_trades / max(1, self.total_trades),
            'risk_reward_ratio': abs(self.total_profit / max(1e-5, self.total_loss)),
            'message': 'Trading completed' if done else 'Trading normally'
        }
        
        return self._get_observation(), reward, done, False, info

    def _update_trade_stats(self, trade_profit):
        """Update trading statistics"""
        if trade_profit > 0:
            self.winning_trades += 1
            self.total_profit += trade_profit
        else:
            self.losing_trades += 1
            self.total_loss += abs(trade_profit)
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0  # Start from the beginning of data
        self.balance = self.initial_balance
        self.position = 0
        self.position_price = 0
        self.daily_profit_loss = 0
        self.daily_high_balance = self.initial_balance
        self.max_drawdown = 0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0
        self.total_loss = 0
        self.last_trade_day = None
        self.balance_history = []
        self.position_history = []
        self.daily_high_balance_history = []
        self.daily_pl_history = []
        self.rewards = []
        
        return self._get_observation(), {}
        
    def render(self):
        pass