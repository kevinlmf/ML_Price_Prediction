#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAC Training Script for Trading

Train a Soft Actor-Critic agent on stock trading tasks.
Includes training loop, evaluation, and performance tracking.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import os
import sys
sys.path.append('.')

from rl.trading_environment import TradingEnvironment
from rl.sac_agent import SACAgent
import warnings
warnings.filterwarnings('ignore')


class SACTrainer:
    """SAC training manager for trading tasks"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 sequence_length: int = 30,
                 train_split: float = 0.8,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.data = data
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.device = device
        
        # Split data
        self._split_data()
        
        # Initialize environment and agent
        self._setup_environment()
        self._setup_agent()
        
        # Training tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_stats = []
        self.evaluation_results = []
    
    def _split_data(self):
        """Split data into train and test sets"""
        n_samples = len(self.data)
        split_idx = int(n_samples * self.train_split)
        
        self.train_data = self.data.iloc[:split_idx].copy()
        self.test_data = self.data.iloc[split_idx:].copy()
        
        print(f"Training data: {len(self.train_data)} samples")
        print(f"Testing data: {len(self.test_data)} samples")
    
    def _setup_environment(self):
        """Setup training and testing environments"""
        self.train_env = TradingEnvironment(
            self.train_data,
            sequence_length=self.sequence_length,
            initial_balance=10000.0,
            transaction_cost=0.001,
            risk_penalty=0.1
        )
        
        self.test_env = TradingEnvironment(
            self.test_data,
            sequence_length=self.sequence_length,
            initial_balance=10000.0,
            transaction_cost=0.001,
            risk_penalty=0.1
        )
    
    def _setup_agent(self):
        """Setup SAC agent"""
        state_dim = self.train_env.n_features
        action_dim = 1
        
        self.agent = SACAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            sequence_length=self.sequence_length,
            hidden_dim=256,
            lr=3e-4,
            gamma=0.99,
            tau=0.005,
            alpha=0.2,
            buffer_size=100000,
            batch_size=256,
            device=self.device
        )
        
        print(f"SAC agent initialized with {state_dim} state features")
    
    def train(self, 
              episodes: int = 200,
              eval_frequency: int = 25,
              save_frequency: int = 50,
              model_path: str = "sac_trading_model.pt") -> Dict[str, Any]:
        """Train SAC agent"""
        
        print(f"Starting SAC training for {episodes} episodes...")
        print(f"Device: {self.device}")
        
        best_eval_return = -np.inf
        
        for episode in range(episodes):
            # Training episode
            episode_reward, episode_length = self._run_episode(self.train_env, training=True)
            
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Get training stats
            if episode % 10 == 0:
                stats = self.agent.get_training_stats()
                self.training_stats.append(stats)
                
                print(f"Episode {episode}: Reward={episode_reward:.4f}, "
                      f"Length={episode_length}, Alpha={stats.get('alpha', 0):.3f}")
            
            # Evaluation
            if episode % eval_frequency == 0 and episode > 0:
                eval_stats = self._evaluate()
                self.evaluation_results.append(eval_stats)
                
                print(f"Evaluation - Return: {eval_stats['total_return']:.4f}, "
                      f"Sharpe: {eval_stats['sharpe_ratio']:.3f}")
                
                # Save best model
                if eval_stats['total_return'] > best_eval_return:
                    best_eval_return = eval_stats['total_return']
                    self.agent.save(f"best_{model_path}")
        
        print("Training completed!")
        
        # Final evaluation
        final_stats = self._evaluate()
        print("\nFinal Evaluation Results:")
        for key, value in final_stats.items():
            print(f"{key}: {value:.4f}")
        
        return {
            'final_evaluation': final_stats
        }
    
    def _run_episode(self, env: TradingEnvironment, training: bool = True) -> Tuple[float, int]:
        """Run a single episode"""
        state = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action = self.agent.select_action(state, deterministic=not training)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store experience for training
            if training:
                self.agent.store_experience(state, action, reward, next_state, done)
                
                # Update agent
                if len(self.agent.replay_buffer) > self.agent.batch_size:
                    self.agent.update()
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        return episode_reward, episode_length
    
    def _evaluate(self, num_episodes: int = 3) -> Dict[str, Any]:
        """Evaluate agent performance"""
        eval_rewards = []
        eval_stats = []
        
        for _ in range(num_episodes):
            episode_reward, _ = self._run_episode(self.test_env, training=False)
            eval_rewards.append(episode_reward)
            
            # Get episode statistics
            stats = self.test_env.get_episode_stats()
            eval_stats.append(stats)
        
        # Aggregate statistics
        avg_stats = {}
        if eval_stats:
            stat_keys = eval_stats[0].keys()
            for key in stat_keys:
                values = [stats[key] for stats in eval_stats if key in stats]
                avg_stats[key] = np.mean(values) if values else 0.0
        
        return avg_stats
    
    def backtest_agent(self, model_path: str = "best_sac_trading_model.pt") -> Dict[str, Any]:
        """Backtest trained agent"""
        # Load best model if exists
        if os.path.exists(model_path):
            self.agent.load(model_path)
            print(f"Loaded model from {model_path}")
        
        # Run backtest on test data
        episode_reward, episode_length = self._run_episode(self.test_env, training=False)
        backtest_stats = self.test_env.get_episode_stats()
        
        return backtest_stats


def run_sac_experiment(df: pd.DataFrame) -> Dict[str, Any]:
    """Run SAC trading experiment"""
    print("=== SAC Trading Agent Experiment ===")
    
    # Initialize trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = SACTrainer(df, sequence_length=30, device=device)
    
    # Train agent
    results = trainer.train(episodes=100, eval_frequency=25, save_frequency=50)
    
    # Final backtest
    backtest_results = trainer.backtest_agent()
    
    return backtest_results


if __name__ == "__main__":
    # Load data
    data_path = "data/processed/technical_indicators_data.csv"
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        run_sac_experiment(df)
    else:
        print("Please run data collection first: python data/technical_indicators.py")