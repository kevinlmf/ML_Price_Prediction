#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trading Environment for Reinforcement Learning

A custom environment for training RL agents on stock trading tasks.
Compatible with Gym interface for easy integration with RL algorithms.
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class TradingEnvironment(gym.Env):
    """
    Custom trading environment for reinforcement learning.
    
    Action Space: Continuous [-1, 1] representing position size
    State Space: Technical indicators and market features
    Reward: Portfolio returns with transaction costs and risk penalties
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 sequence_length: int = 30,
                 initial_balance: float = 10000.0,
                 transaction_cost: float = 0.001,
                 risk_penalty: float = 0.1,
                 lookback_window: int = 252):
        
        super(TradingEnvironment, self).__init__()
        
        self.data = data.copy()
        self.sequence_length = sequence_length
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.risk_penalty = risk_penalty
        self.lookback_window = lookback_window
        
        # Prepare data
        self._prepare_data()
        
        # Define action and observation spaces
        # Action: continuous value between -1 and 1 (position size)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation: sequence of technical indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(self.sequence_length, self.n_features), 
            dtype=np.float32
        )
        
        # Initialize state
        self.reset()
    
    def _prepare_data(self):
        """Prepare and clean data for trading environment"""
        # Remove non-numeric columns for features
        feature_cols = [col for col in self.data.columns 
                       if col not in ['Date', 'symbol'] and 
                       self.data[col].dtype in ['float64', 'int64']]
        
        self.features = self.data[feature_cols].values
        self.prices = self.data['Close'].values
        self.returns = self.data['Close'].pct_change().fillna(0).values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
        self.n_features = self.features.shape[1]
        self.n_timesteps = len(self.features)
        
        print(f"Environment initialized with {self.n_timesteps} timesteps and {self.n_features} features")
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state"""
        self.current_step = self.sequence_length
        self.balance = self.initial_balance
        self.position = 0.0  # Current position (-1 to 1)
        self.portfolio_value = self.initial_balance
        self.total_return = 0.0
        self.max_portfolio_value = self.initial_balance
        self.transaction_costs = 0.0
        
        # Performance tracking
        self.episode_returns = []
        self.episode_positions = []
        self.episode_portfolio_values = [self.initial_balance]
        
        return self._get_observation()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one step in the environment"""
        # Ensure action is in valid range
        action = np.clip(action[0], -1.0, 1.0)
        
        # Calculate transaction cost
        position_change = abs(action - self.position)
        cost = position_change * self.transaction_cost * self.portfolio_value
        self.transaction_costs += cost
        
        # Update position
        old_position = self.position
        self.position = action
        
        # Calculate return for this step
        if self.current_step < len(self.returns):
            period_return = self.returns[self.current_step]
            
            # Portfolio return = position * stock return
            portfolio_return = self.position * period_return
            
            # Update portfolio value
            self.portfolio_value = self.portfolio_value * (1 + portfolio_return) - cost
            
            # Track maximum portfolio value for drawdown calculation
            if self.portfolio_value > self.max_portfolio_value:
                self.max_portfolio_value = self.portfolio_value
            
            # Calculate reward
            reward = self._calculate_reward(portfolio_return, position_change, cost)
            
            # Store performance data
            self.episode_returns.append(portfolio_return)
            self.episode_positions.append(self.position)
            self.episode_portfolio_values.append(self.portfolio_value)
        else:
            reward = 0.0
        
        # Move to next timestep
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.n_timesteps - 1
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'position': self.position,
            'transaction_cost': cost,
            'total_return': (self.portfolio_value / self.initial_balance) - 1,
            'step': self.current_step
        }
        
        return self._get_observation(), reward, done, info
    
    def _calculate_reward(self, portfolio_return: float, position_change: float, cost: float) -> float:
        """Calculate reward for current action"""
        # Base reward: portfolio return
        reward = portfolio_return
        
        # Transaction cost penalty
        reward -= (cost / self.portfolio_value)
        
        # Risk penalty: penalize large positions and frequent changes
        risk_penalty = self.risk_penalty * (abs(self.position) ** 2 + position_change ** 2)
        reward -= risk_penalty
        
        # Drawdown penalty
        if self.max_portfolio_value > 0:
            drawdown = (self.max_portfolio_value - self.portfolio_value) / self.max_portfolio_value
            reward -= drawdown * 0.5
        
        # Volatility penalty (encourage stable returns)
        if len(self.episode_returns) > 10:
            recent_returns = np.array(self.episode_returns[-10:])
            volatility = np.std(recent_returns)
            reward -= volatility * 0.1
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation (sequence of features)"""
        start_idx = max(0, self.current_step - self.sequence_length)
        end_idx = self.current_step
        
        # Get feature sequence
        obs = self.features[start_idx:end_idx]
        
        # Pad if necessary
        if len(obs) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(obs), self.n_features))
            obs = np.vstack([padding, obs])
        
        return obs.astype(np.float32)
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            total_return = (self.portfolio_value / self.initial_balance - 1) * 100
            print(f"Step: {self.current_step}, Portfolio Value: ${self.portfolio_value:.2f}, "
                  f"Position: {self.position:.3f}, Total Return: {total_return:.2f}%")
    
    def get_episode_stats(self) -> Dict[str, Any]:
        """Get statistics for the completed episode"""
        if len(self.episode_returns) == 0:
            return {}
        
        returns = np.array(self.episode_returns)
        portfolio_values = np.array(self.episode_portfolio_values)
        
        # Calculate performance metrics
        total_return = (self.portfolio_value / self.initial_balance) - 1
        
        # Sharpe ratio (annualized)
        if np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        # Win rate
        win_rate = np.mean(returns > 0) if len(returns) > 0 else 0.0
        
        # Number of trades (position changes)
        positions = np.array(self.episode_positions)
        position_changes = np.abs(np.diff(positions))
        num_trades = np.sum(position_changes > 0.01)  # Threshold for meaningful position change
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'final_portfolio_value': self.portfolio_value,
            'total_transaction_costs': self.transaction_costs,
            'average_position': np.mean(np.abs(positions)) if len(positions) > 0 else 0.0
        }


class MultiStockTradingEnvironment(TradingEnvironment):
    """
    Extended trading environment for multiple stocks
    """
    
    def __init__(self, data: pd.DataFrame, **kwargs):
        # Group data by symbol for multi-stock trading
        self.symbols = data['symbol'].unique()
        self.stock_data = {symbol: data[data['symbol'] == symbol].copy() 
                          for symbol in self.symbols}
        
        # Use combined data for now (can be extended for true multi-asset trading)
        super().__init__(data, **kwargs)
    
    def get_symbol_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance breakdown by symbol"""
        # This is a simplified version - can be extended for actual multi-stock trading
        return {symbol: self.get_episode_stats() for symbol in self.symbols}


if __name__ == "__main__":
    # Test the environment
    import pandas as pd
    
    # Load sample data
    try:
        data = pd.read_csv("data/processed/technical_indicators_data.csv")
        data['Date'] = pd.to_datetime(data['Date'])
        
        # Test single stock environment
        stock_data = data[data['symbol'] == 'AAPL'].copy()
        env = TradingEnvironment(stock_data)
        
        print("Testing Trading Environment...")
        
        # Random agent test
        obs = env.reset()
        done = False
        step_count = 0
        
        while not done and step_count < 100:
            action = env.action_space.sample()  # Random action
            obs, reward, done, info = env.step(action)
            
            if step_count % 20 == 0:
                env.render()
            
            step_count += 1
        
        # Get final statistics
        stats = env.get_episode_stats()
        print("\nFinal Episode Statistics:")
        for key, value in stats.items():
            print(f"{key}: {value:.4f}")
            
    except FileNotFoundError:
        print("Please run data collection and technical indicators first:")
        print("python data/collect_all.py")
        print("python data/technical_indicators.py")