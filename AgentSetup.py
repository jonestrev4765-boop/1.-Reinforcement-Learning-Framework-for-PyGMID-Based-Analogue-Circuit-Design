import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical, Normal
import numpy as np

class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO with continuous actions (Gaussian policy).
    Actor outputs mean for each action dimension + a learned log_std.
    Critic outputs a scalar state value.
    """

    def __init__(self, state_dim, action_dim=2, hidden_dim=128):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),          # Tanh preferred over ReLU for PPO
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )

        # Actor head (Gaussian mean)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.log_std    = nn.Parameter(torch.zeros(action_dim))

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.shared(state)

    def act(self, state):
        """
        Returns:
            action: sampled continuous action (tensor)
            log_prob: log probability of sampled action
        """
        features = self.forward(state)
    
        mean = self.actor_mean(features)
        std  = self.log_std.exp()
    
        dist = Normal(mean, std)
    
        # Sample continuous action
        raw_action = dist.rsample()
    
        # Bounds the action to [-1, 1]
        action = torch.tanh(raw_action) 

        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
    
        return action, log_prob

    def evaluate(self, state, action):
        features = self.forward(state)

        # Means
        mean = self.actor_mean(features)
        std = self.log_std.exp()

        # Create same distribution
        dist = Normal(mean, std)

        action_clamped = torch.clamp(action, -0.999, 0.999)
        raw_action = torch.atanh(action_clamped)
        
        # Compute log prob of the raw action
        log_prob = dist.log_prob(raw_action).sum(dim=-1)
        
        # Apply tanh correction
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        # Entropy for exploration
        entropy = dist.entropy().sum(dim=-1)

        # Critic value estimate
        value = self.critic(features).squeeze(-1)

        return log_prob, value, entropy

class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.05,
        max_grad_norm=0.5,
        epochs=4,
        batch_size=128,
        hidden_dim=128,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Initialize network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        # Storage for training
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
    def select_action(self, state, training=True):
        """Select action from policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob = self.policy.act(state)
            value = self.policy.critic(self.policy.forward(state)).squeeze()
        
        if training:
            self.states.append(state.squeeze(0).cpu().numpy())  # Remove batch dimension
            self.actions.append(action.squeeze(0).cpu().numpy())
            self.log_probs.append(log_prob.item())  # Store as scalar
            self.values.append(value.item())  # Store as scalar
        
        return action.squeeze(0).cpu().numpy()

    
    def store_transition(self, reward, done):
        """Store reward and done flag"""
        # Ensure reward is a scalar
        if isinstance(reward, np.ndarray):
            reward = reward.item()
        if isinstance(done, np.ndarray):
            done = done.item()
            
        self.rewards.append(reward)
        self.dones.append(done)
    
    def compute_gae(self, next_value):
        """Compute Generalized Advantage Estimation"""
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)  # Now already scalars
        
        advantages = np.zeros_like(rewards)
        last_gae = 0
        
        # Compute advantages backwards
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value_t = next_value
            else:
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        
        return advantages, returns
    
    def update(self, next_state):
        """Update policy using PPO"""
        print(f"UPDATE CALLED - Data counts: states={len(self.states)}, rewards={len(self.rewards)}, values={len(self.values)}, dones={len(self.dones)}")
        
        if len(self.states) == 0:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy_loss': 0}
        
        # Ensure all data has matching lengths
        min_len = min(len(self.states), len(self.rewards), len(self.values), len(self.dones))
        if not (len(self.states) == len(self.rewards) == len(self.values) == len(self.dones)):
            print(f"WARNING: Mismatched data lengths! Truncating to {min_len}")
            self.states = self.states[:min_len]
            self.rewards = self.rewards[:min_len]
            self.values = self.values[:min_len]
            self.dones = self.dones[:min_len]
        
        # Compute next state value
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.policy.critic(self.policy.forward(next_state)).cpu().numpy()[0, 0]
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)  # Stack as array, not concatenate
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        old_values = torch.FloatTensor(np.array(self.values)).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # PPO update for multiple epochs
        dataset_size = states.shape[0]
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for _ in range(self.epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = min(start + self.batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, new_values, entropy = self.policy.evaluate(batch_states, batch_actions)
                
                # Flatten all tensors to 1D to ensure matching dimensions
                log_probs = log_probs.flatten()
                new_values = new_values.flatten()
                entropy = entropy.flatten()
                batch_old_log_probs = batch_old_log_probs.flatten()
                batch_advantages = batch_advantages.flatten()
                batch_returns = batch_returns.flatten()
                
                # Compute ratio and clipped surrogate objective
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                
                # Compute losses
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = 0.5 * (batch_returns - new_values).pow(2).mean()
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
        
        # Clear storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
