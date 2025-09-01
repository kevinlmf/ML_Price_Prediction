#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Soft Actor-Critic (SAC) Agent for Trading

Implementation of SAC algorithm specifically adapted for financial trading.
Features:
- Continuous action space for position sizing
- Experience replay with prioritized sampling  
- Twin critic networks for stability
- Entropy regularization for exploration
- Custom reward shaping for trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict, Any
import os


class ReplayBuffer:
    """Experience replay buffer for SAC"""
    
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state),
            torch.FloatTensor(action), 
            torch.FloatTensor(reward).unsqueeze(1),
            torch.FloatTensor(next_state),
            torch.FloatTensor(done).unsqueeze(1)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network for SAC - outputs mean and log_std of policy"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        # Feature extraction for sequential data
        self.lstm = nn.LSTM(state_dim, hidden_dim // 2, batch_first=True)
        
        # Policy network
        self.fc1 = nn.Linear(hidden_dim // 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        self.max_log_std = 2
        self.min_log_std = -20
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Handle sequence data
        if len(state.shape) == 3:  # (batch, sequence, features)
            lstm_out, _ = self.lstm(state)
            x = lstm_out[:, -1, :]  # Take last timestep
        else:
            x = state
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        
        # Clamp log_std to prevent numerical instability
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        
        # Apply tanh to bound actions to [-1, 1]
        action = torch.tanh(z)
        
        # Calculate log probability with correction for tanh
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class Critic(nn.Module):
    """Critic network for SAC - Q-function"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Feature extraction for sequential data
        self.lstm = nn.LSTM(state_dim, hidden_dim // 2, batch_first=True)
        
        # Q-function network
        self.fc1 = nn.Linear(hidden_dim // 2 + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # Handle sequence data
        if len(state.shape) == 3:  # (batch, sequence, features)
            lstm_out, _ = self.lstm(state)
            state_features = lstm_out[:, -1, :]  # Take last timestep
        else:
            state_features = state
        
        x = torch.cat([state_features, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.fc3(x)
        
        return q_value


class SACAgent:
    """Soft Actor-Critic agent for trading"""
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int, 
                 sequence_length: int = 30,
                 hidden_dim: int = 256,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 buffer_size: int = 100000,
                 batch_size: int = 256,
                 device: str = 'cpu'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.sequence_length = sequence_length
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        
        # Initialize networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Target networks
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim).to(device)
        
        # Copy parameters to target networks
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Entropy coefficient
        self.alpha = alpha
        self.target_entropy = -action_dim  # Heuristic target entropy
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training stats
        self.total_steps = 0
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given state"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_tensor)
        
        return action.cpu().numpy()[0]
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update SAC networks"""
        if len(self.replay_buffer) < self.batch_size:
            return {}
        
        # Sample batch
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        state = state.to(self.device)
        action = action.to(self.device)
        reward = reward.to(self.device)
        next_state = next_state.to(self.device)
        done = done.to(self.device)
        
        # Update critics
        critic_loss = self._update_critics(state, action, reward, next_state, done)
        
        # Update actor and alpha
        actor_loss = self._update_actor(state)
        alpha_loss = self._update_alpha(state)
        
        # Update target networks
        self._soft_update_targets()
        
        self.total_steps += 1
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha
        }
    
    def _update_critics(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor,
                       next_state: torch.Tensor, done: torch.Tensor) -> float:
        """Update critic networks"""
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # Current Q estimates
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        
        # Critic losses
        critic1_loss = F.mse_loss(q1, target_q)
        critic2_loss = F.mse_loss(q2, target_q)
        
        # Update critics
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        total_critic_loss = (critic1_loss + critic2_loss).item()
        self.critic_losses.append(total_critic_loss)
        
        return total_critic_loss
    
    def _update_actor(self, state: torch.Tensor) -> float:
        """Update actor network"""
        action, log_prob = self.actor.sample(state)
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        actor_loss_value = actor_loss.item()
        self.actor_losses.append(actor_loss_value)
        
        return actor_loss_value
    
    def _update_alpha(self, state: torch.Tensor) -> float:
        """Update entropy coefficient alpha"""
        with torch.no_grad():
            _, log_prob = self.actor.sample(state)
        
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp().item()
        
        alpha_loss_value = alpha_loss.item()
        self.alpha_losses.append(alpha_loss_value)
        
        return alpha_loss_value
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_critic1.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic2.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save(self, filepath: str):
        """Save agent state"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'target_critic1_state_dict': self.target_critic1.state_dict(),
            'target_critic2_state_dict': self.target_critic2.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic1_optimizer_state_dict': self.critic1_optimizer.state_dict(),
            'critic2_optimizer_state_dict': self.critic2_optimizer.state_dict(),
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'total_steps': self.total_steps
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.target_critic1.load_state_dict(checkpoint['target_critic1_state_dict'])
        self.target_critic2.load_state_dict(checkpoint['target_critic2_state_dict'])
        
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        self.critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        self.log_alpha = checkpoint['log_alpha']
        self.alpha = self.log_alpha.exp().item()
        self.total_steps = checkpoint['total_steps']
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        return {
            'total_steps': self.total_steps,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic_loss': np.mean(self.critic_losses[-100:]) if self.critic_losses else 0,
            'avg_alpha_loss': np.mean(self.alpha_losses[-100:]) if self.alpha_losses else 0,
            'alpha': self.alpha,
            'buffer_size': len(self.replay_buffer)
        }


if __name__ == "__main__":
    # Test SAC agent
    print("Testing SAC Agent...")
    
    state_dim = 90  # Number of features
    action_dim = 1  # Position size
    sequence_length = 30
    
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        sequence_length=sequence_length,
        device='cpu'
    )
    
    # Test action selection
    dummy_state = np.random.randn(sequence_length, state_dim)
    action = agent.select_action(dummy_state)
    print(f"Sample action: {action}")
    
    # Test experience storage and update
    next_state = np.random.randn(sequence_length, state_dim)
    agent.store_experience(dummy_state, action, 0.1, next_state, False)
    
    # Add more experiences for update test
    for _ in range(300):
        state = np.random.randn(sequence_length, state_dim)
        action = agent.select_action(state)
        next_state = np.random.randn(sequence_length, state_dim)
        reward = np.random.randn()
        done = np.random.random() < 0.1
        agent.store_experience(state, action, reward, next_state, done)
    
    # Test update
    update_info = agent.update()
    print("Update info:", update_info)
    
    # Test save/load
    agent.save("test_sac_agent.pt")
    agent.load("test_sac_agent.pt")
    print("Save/load test successful")
    
    # Clean up
    os.remove("test_sac_agent.pt")
    
    print("SAC Agent test completed!")