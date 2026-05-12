# rmp2_rl_policy.py
# =============================================================================
# RMP2-RL Policy Network and PPO Agent
# =============================================================================
# PyTorch implementation of the MLP policy for predicting RMP2 gains
# Plus PPO agent for training
#
# Architecture follows rmp2_bridge_rl.py MLPPolicy but in PyTorch
# for differentiable training

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import copy


# ==============================================================================
# Policy Network (Actor)
# ==============================================================================

class MLPPolicyNet(nn.Module):
    """MLP policy network predicting RMP2 gain parameters.
    
    Input: 26-dimensional state (from RMP2TrainingEnv)
    Output: 11-dimensional action (gain parameters)
    Architecture: input -> 64 -> 32 -> output, ReLU activations
    """
    
    def __init__(self, input_dim=23, hidden_dims=[64, 32], output_dim=11):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Policy network
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.mean = nn.Linear(hidden_dims[1], output_dim)
        
        # Log standard deviation (learnable)
        self.log_std = nn.Parameter(torch.zeros(output_dim))
        
        # Initialize weights (Xavier/Glorot)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in [self.fc1, self.fc2, self.mean]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass returning mean and std dev."""
        # Ensure input is float32
        x = x.float()
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        mean = self.mean(h2)
        
        # Scale mean to [0, 1] then map to gain bounds
        # Action dim 11 maps to gain bounds
        mean = torch.sigmoid(mean)  # [0, 1]
        
        std = torch.exp(self.log_std)
        
        return mean, std
        
        return mean, std
    
    def get_action(self, x, deterministic=False):
        """Get action from observation.
        
        Args:
            x: observation tensor (batch_size, 26) or (26,)
            deterministic: if True, return mean; else sample
        
        Returns:
            action: sampled or mean action
            log_prob: log probability of action under current policy
            entropy: entropy of the policy distribution
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        mean, std = self.forward(x)
        
        if deterministic:
            return mean.squeeze(0).detach(), torch.zeros(1), torch.zeros(1)

        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action.squeeze(0), log_prob.squeeze(0), entropy.squeeze(0)

    def evaluate_actions(self, x, actions):
        """Evaluate log probability and entropy for given actions."""
        mean, std = self.forward(x)
        dist = Normal(mean, std)
        
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy


# ==============================================================================
# Value Network (Critic)
# ==============================================================================

class CriticNet(nn.Module):
    """Value network for state value estimation.
    
    Input: 26-dimensional state
    Output: 1-dimensional value (V(s))
    """
    
    def __init__(self, input_dim=26, hidden_dims=[64, 32]):
        super().__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.value = nn.Linear(hidden_dims[1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2, self.value]:
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass returning state value."""
        x = x.float()
        
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        value = self.value(h2)
        
        return value.squeeze(-1)


# ==============================================================================
# PPO Agent
# ==============================================================================

class PPOAgent:
    """Proximal Policy Optimization agent for RMP2 gain learning.
    
    Handles:
    - Action selection
    - Experience collection
    - PPO updates with GAE advantages
    - Checkpoint saving/loading
    """
    
    def __init__(
        self,
        state_dim=26,
        action_dim=11,
        hidden_dims=[64, 32],
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        lr=3e-4,
        batch_size=64,
        n_epochs=8,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        # Networks
        self.policy = MLPPolicyNet(state_dim, hidden_dims, action_dim)
        self.value = CriticNet(state_dim, hidden_dims)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=lr
        )
        
        # Training statistics
        self.training_step = 0
        self.episode_count = 0
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value.to(self.device)

        # Observation normalization (on agent's device)
        self.obs_mean = torch.zeros(state_dim, device=self.device)
        self.obs_std = torch.ones(state_dim, device=self.device)
        self.obs_count = 0
        self.running_mean = deque(maxlen=10000)
    
    def to(self, device):
        """Move agent to specified device."""
        self.device = device
        self.policy.to(device)
        self.value.to(device)
        return self
    
    def get_action(self, state, deterministic=False):
        """Select action given state.
        
        Args:
            state: numpy array (state_dim,) or torch tensor
            deterministic: if True, return mean (for evaluation)
        
        Returns:
            action: numpy array (action_dim,)
            log_prob: float
            value: float
        """
        with torch.no_grad():
            # Convert to tensor
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float()
            
            # Normalize
            state = self._normalize_obs(state)
            
            state = state.to(self.device)
            
            action, log_prob, entropy = self.policy.get_action(state, deterministic)
            
            value = self.value(state)
            
            return action.cpu().numpy(), log_prob.item(), value.item()
    
    def _normalize_obs(self, obs):
        """Normalize observations using running mean/std (for data collection)."""
        # Update running statistics (only during collection)
        if obs.dim() == 1:
            obs_np = obs.cpu().numpy()
            self.running_mean.append(obs_np)
            self.obs_count += 1
            
            if self.obs_count > 1:
                mean_np = np.mean(self.running_mean, axis=0)
                std_np = np.std(self.running_mean, axis=0) + 1e-8
                self.obs_mean = torch.from_numpy(mean_np).float().to(self.device)
                self.obs_std = torch.from_numpy(std_np).float().to(self.device)
        
        # Move obs to agent's device and normalize
        if obs.device != self.device:
            obs = obs.to(self.device)
        obs_norm = (obs - self.obs_mean) / self.obs_std
        return obs_norm

    def _normalize_obs_static(self, obs):
        """Normalize observations using precomputed mean/std (for PPO update)."""
        if obs.device != self.obs_mean.device:
            obs_mean = self.obs_mean.to(obs.device)
            obs_std = self.obs_std.to(obs.device)
        else:
            obs_mean = self.obs_mean
            obs_std = self.obs_std
        obs_norm = (obs - obs_mean) / obs_std
        return obs_norm
    
    def update(self, rollout):
        """Update policy using PPO on collected rollout data.
        
        Args:
            rollout: RolloutStorage object containing transitions
        
        Returns:
            dict: Training metrics
        """
        # Compute returns and advantages
        returns, advantages = self._compute_gae(rollout)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten rollout data
        obs = rollout.obs
        actions = rollout.actions
        old_log_probs = rollout.log_probs
        values = rollout.values
        
        # PPO update
        losses = {
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'total_loss': [],
        }
        
        for epoch in range(self.n_epochs):
            # Shuffle indices
            indices = torch.randperm(len(obs))
            
            for start in range(0, len(obs), self.batch_size):
                end = min(start + self.batch_size, len(obs))
                batch_idx = indices[start:end]
                
                # Get batch
                batch_obs = obs[batch_idx].to(self.device)
                batch_actions = actions[batch_idx].to(self.device)
                batch_old_log_probs = old_log_probs[batch_idx].to(self.device)
                batch_returns = returns[batch_idx].to(self.device)
                batch_advantages = advantages[batch_idx].to(self.device)
                
                # Evaluate actions and value (use precomputed normalization from collection)
                batch_obs_norm = self._normalize_obs_static(batch_obs)
                new_log_probs, entropy = self.policy.evaluate_actions(batch_obs_norm, batch_actions)
                new_values = self.value(batch_obs_norm)
                
                # PPO policy loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(
                    ratio,
                    1 - self.clip_epsilon,
                    1 + self.clip_epsilon
                ) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(new_values, batch_returns)
                
                # Entropy loss (negative, we want to maximize entropy)
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.vf_coef * value_loss +
                    self.ent_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()
                
                # Record losses
                losses['policy_loss'].append(policy_loss.item())
                losses['value_loss'].append(value_loss.item())
                losses['entropy_loss'].append(entropy_loss.item())
                losses['total_loss'].append(total_loss.item())
        
        self.training_step += 1
        
        # Average losses
        metrics = {
            'policy_loss': np.mean(losses['policy_loss']),
            'value_loss': np.mean(losses['value_loss']),
            'entropy_loss': np.mean(losses['entropy_loss']),
            'total_loss': np.mean(losses['total_loss']),
            'learning_rate': self.optimizer.param_groups[0]['lr'],
        }
        
        return metrics
    
    def _compute_gae(self, rollout):
        """Compute Generalized Advantage Estimation.
        
        Args:
            rollout: RolloutStorage object
        
        Returns:
            returns: tensor of returns
            advantages: tensor of advantages
        """
        rewards = rollout.rewards
        values = rollout.values
        dones = rollout.dones
        
        # Convert to tensors
        rewards = rewards.to(self.device)
        values = values.to(self.device)
        dones = dones.to(self.device)
        
        # GAE computation
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        # Reverse iteration (from last step to first)
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Bootstrap from value network if not terminated, else 0
                next_value = 0.0 if dones[t] > 0.5 else values[t]
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[t])
            advantages[t] = gae
        
        returns = advantages + values
        
        return returns, advantages
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_state_dict': self.value.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'obs_mean': self.obs_mean,
            'obs_std': self.obs_std,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value.load_state_dict(checkpoint['value_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        self.obs_mean = checkpoint['obs_mean']
        self.obs_std = checkpoint['obs_std']
        print(f"Model loaded from {path}")


# ==============================================================================
# Rollout Storage
# ==============================================================================

class RolloutStorage:
    """Storage for collecting rollout data during training.
    
    Stores: observations, actions, rewards, log_probs, values, dones
    """
    
    def __init__(self, capacity, state_dim, action_dim, device='cpu'):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # Pre-allocate tensors on the specified device
        self.obs = torch.zeros(capacity, state_dim, dtype=torch.float32, device=device)
        self.actions = torch.zeros(capacity, action_dim, dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.values = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.float32, device=device)
        
        self.position = 0
        self.size = 0
    
    def add(self, obs, action, reward, log_prob, value, done):
        """Add a transition to storage."""
        self.obs[self.position] = torch.from_numpy(obs)
        self.actions[self.position] = torch.from_numpy(action)
        self.rewards[self.position] = reward
        self.log_probs[self.position] = log_prob
        self.values[self.position] = value
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def reset(self):
        """Reset storage."""
        self.obs.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.log_probs.zero_()
        self.values.zero_()
        self.dones.zero_()
        self.position = 0
        self.size = 0
    
    def get_all(self):
        """Get all stored data."""
        return RolloutStorageBatch(
            self.obs[:self.size],
            self.actions[:self.size],
            self.rewards[:self.size],
            self.log_probs[:self.size],
            self.values[:self.size],
            self.dones[:self.size],
        )


class RolloutStorageBatch:
    """Batch container for rollout data."""
    
    def __init__(self, obs, actions, rewards, log_probs, values, dones):
        self.obs = obs
        self.actions = actions
        self.rewards = rewards
        self.log_probs = log_probs
        self.values = values
        self.dones = dones


# ==============================================================================
# Testing
# ==============================================================================

if __name__ == "__main__":
    print("Testing RMP2 RL Policy...")
    
    # Test agent creation
    agent = PPOAgent(state_dim=23, action_dim=11)
    print(f"Device: {agent.device}")
    
    # Test action selection
    state = np.random.randn(26).astype(np.float32)
    action, log_prob, value = agent.get_action(state)
    print(f"Action shape: {action.shape}")
    print(f"Log prob: {log_prob}, Value: {value}")
    
    # Test update
    print("\nTesting PPO update...")
    
    # Create dummy rollout
    storage = RolloutStorage(capacity=128, state_dim=23, action_dim=11)
    
    for i in range(128):
        state = np.random.randn(26).astype(np.float32)
        action = np.random.randn(11).astype(np.float32)
        reward = np.random.randn()
        log_prob = -1.0
        value = 0.0
        done = 0.0
        
        storage.add(state, action, reward, log_prob, value, done)
    
    rollout = storage.get_all()
    metrics = agent.update(rollout)
    print(f"Training metrics: {metrics}")
    
    print("\nTest complete!")