import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
import warnings
import matplotlib.pyplot as plt
from datetime import datetime
import os

warnings.filterwarnings('ignore')

# ========== SETUP AND CONFIGURATION ==========
def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# ========== TRAINING METRICS TRACKER ==========
class TrainingMetrics:
    """Track and visualize training progress"""
    
    def __init__(self, save_dir: str = "training_logs"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics storage
        self.episode_rewards = []
        self.episode_portfolios = []
        self.episode_returns = []
        self.episode_trades = []
        self.episode_win_rates = []
        self.episode_durations = []
        
        # Loss tracking
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.total_losses = []
        
        # Action distribution
        self.action_counts = {0: [], 1: [], 2: []}
        
        # Running averages
        self.running_reward = 0
        self.best_reward = -float('inf')
        self.best_portfolio = 0
        
        # Timing
        self.start_time = datetime.now()
        
    def update(self, episode: int, reward: float, portfolio_value: float, 
               total_return: float, trades: int, win_rate: float, 
               duration: float, actions: List[int], losses: Dict):
        """Update all metrics"""
        
        self.episode_rewards.append(reward)
        self.episode_portfolios.append(portfolio_value)
        self.episode_returns.append(total_return)
        self.episode_trades.append(trades)
        self.episode_win_rates.append(win_rate)
        self.episode_durations.append(duration)
        
        # Loss tracking
        if losses:
            self.actor_losses.append(losses.get('actor_loss', 0))
            self.critic_losses.append(losses.get('critic_loss', 0))
            self.entropy_losses.append(losses.get('entropy', 0))
            self.total_losses.append(losses.get('total_loss', 0))
        
        # Action distribution
        unique, counts = np.unique(actions, return_counts=True)
        action_dict = dict(zip(unique, counts))
        for action in [0, 1, 2]:
            self.action_counts[action].append(action_dict.get(action, 0))
        
        # Update running metrics
        window = min(100, len(self.episode_rewards))
        self.running_reward = np.mean(self.episode_rewards[-window:])
        
        if portfolio_value > self.best_portfolio:
            self.best_portfolio = portfolio_value
            self.best_reward = reward
    
    def plot_progress(self, episode: int):
        """Generate training plots"""
        if episode % 50 != 0 and episode > 0:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=14)
        
        # 1. Episode Rewards
        axes[0, 0].plot(self.episode_rewards, alpha=0.6, label='Episode Reward')
        if len(self.episode_rewards) > 10:
            moving_avg = np.convolve(self.episode_rewards, np.ones(10)/10, mode='valid')
            axes[0, 0].plot(range(9, len(self.episode_rewards)), moving_avg, 'r-', label='Moving Avg (10)')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Portfolio Value
        axes[0, 1].plot(self.episode_portfolios, 'g-', alpha=0.6, label='Portfolio Value')
        axes[0, 1].axhline(y=10000, color='r', linestyle='--', label='Initial Capital')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Portfolio Value ($)')
        axes[0, 1].set_title('Portfolio Performance')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Total Return (%)
        axes[0, 2].plot(np.array(self.episode_returns) * 100, 'b-', alpha=0.6, label='Return %')
        axes[0, 2].axhline(y=0, color='r', linestyle='--', label='Break Even')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Return (%)')
        axes[0, 2].set_title('Total Return %')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Win Rate & Trades
        ax1 = axes[1, 0]
        ax1.plot(self.episode_win_rates, 'purple', alpha=0.6, label='Win Rate')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Win Rate (%)')
        ax1.set_title('Win Rate')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax2 = ax1.twinx()
        ax2.plot(self.episode_trades, 'orange', alpha=0.6, label='Trades')
        ax2.set_ylabel('Number of Trades')
        ax2.legend(loc='upper right')
        
        # 5. Losses
        if self.total_losses:
            axes[1, 1].plot(self.total_losses, 'r-', alpha=0.6, label='Total Loss')
            axes[1, 1].plot(self.actor_losses, 'b-', alpha=0.6, label='Actor Loss')
            axes[1, 1].plot(self.critic_losses, 'g-', alpha=0.6, label='Critic Loss')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Action Distribution
        if self.action_counts[0]:
            episodes_range = range(len(self.action_counts[0]))
            axes[1, 2].stackplot(episodes_range, 
                                 self.action_counts[0], 
                                 self.action_counts[1], 
                                 self.action_counts[2],
                                 labels=['Hold', 'Buy', 'Sell'],
                                 colors=['gray', 'green', 'red'],
                                 alpha=0.7)
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Action Count')
            axes[1, 2].set_title('Action Distribution Over Time')
            axes[1, 2].legend(loc='upper left')
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(f"{self.save_dir}/training_progress_ep{episode}.png", dpi=100, bbox_inches='tight')
        plt.close()
    
    def print_summary(self, episode: int, episode_reward: float, portfolio_value: float,
                      total_return: float, trades: int, win_rate: float, 
                      action_dist: Dict, elapsed_time: float):
        """Print formatted training summary"""
        
        # Create progress bar
        progress = episode / 300
        bar_length = 30
        filled = int(bar_length * progress)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f"\n{'='*80}")
        print(f"📊 EPISODE {episode} SUMMARY")
        print(f"{'='*80}")
        print(f"Progress: [{bar}] {progress*100:.1f}%")
        print(f"{'='*80}")
        
        print(f"\n📈 PERFORMANCE METRICS:")
        print(f"   ├─ Episode Reward:     {episode_reward:10.2f}")
        print(f"   ├─ Running Avg Reward: {self.running_reward:10.2f} (last 100 eps)")
        print(f"   ├─ Portfolio Value:    ${portfolio_value:10.2f}")
        print(f"   ├─ Total Return:       {total_return*100:9.2f}%")
        print(f"   └─ Best Portfolio:     ${self.best_portfolio:10.2f}")
        
        print(f"\n🎯 TRADING STATISTICS:")
        print(f"   ├─ Total Trades:       {trades:10d}")
        print(f"   ├─ Win Rate:           {win_rate:9.1f}%")
        print(f"   └─ Avg Trades/Ep:      {np.mean(self.episode_trades[-10:]):10.1f}")
        
        print(f"\n🎬 ACTION DISTRIBUTION (this episode):")
        total_actions = sum(action_dist.values())
        for action, count in action_dist.items():
            action_name = {0: "Hold", 1: "Buy", 2: "Sell"}[action]
            pct = count / total_actions * 100 if total_actions > 0 else 0
            print(f"   ├─ {action_name}:        {count:4d} ({pct:5.1f}%)")
        
        print(f"\n⚡ TRAINING SPEED:")
        print(f"   ├─ Episode Duration:   {elapsed_time:6.2f} seconds")
        print(f"   ├─ Total Time:         {datetime.now() - self.start_time}")
        print(f"   └─ Episodes/Sec:       {episode / (datetime.now() - self.start_time).total_seconds():6.2f}")
        
        if portfolio_value == self.best_portfolio:
            print(f"\n🏆 NEW BEST PORTFOLIO VALUE! 🏆")
        
        print(f"{'='*80}\n")
    
    def save_metrics(self):
        """Save metrics to CSV"""
        import pandas as pd
        
        df = pd.DataFrame({
            'episode': range(1, len(self.episode_rewards) + 1),
            'reward': self.episode_rewards,
            'portfolio_value': self.episode_portfolios,
            'return_pct': np.array(self.episode_returns) * 100,
            'trades': self.episode_trades,
            'win_rate': self.episode_win_rates,
            'duration': self.episode_durations
        })
        
        df.to_csv(f"{self.save_dir}/training_metrics.csv", index=False)
        print(f"✅ Metrics saved to {self.save_dir}/training_metrics.csv")


# ========== NEURAL NETWORK ARCHITECTURE ==========
class PPONetwork(nn.Module):
    """Actor-Critic Network for PPO with Discrete Actions"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256], action_dim: int = 3):
        super(PPONetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        body_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            body_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.body = nn.Sequential(*body_layers)
        
        # Actor head
        self.actor_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # Critic head
        self.critic_head = nn.Sequential(
            nn.Linear(prev_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass"""
        features = self.body(x)
        action_probs = self.actor_head(features)
        state_value = self.critic_head(features)
        return action_probs, state_value
    
    def get_action(self, state: np.ndarray, action_mask: Optional[List[int]] = None,        # This is perfect if only we remove action_mask
                   device: torch.device = torch.device('cpu')) -> Tuple[int, float]:
        """Sample action from policy"""
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
        else:
            state = state.unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            action_probs = action_probs.squeeze(0)
        
        if action_mask is not None:
            action_mask = torch.FloatTensor(action_mask).to(device)
            action_probs = action_probs * action_mask
            action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        # Check for NaN
        if torch.isnan(action_probs).any():
            action_probs = torch.ones_like(action_probs) / action_probs.shape[0]
            if action_mask is not None:
                action_probs = action_probs * action_mask
                action_probs = action_probs / (action_probs.sum() + 1e-8)
        
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item()
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training"""
        action_probs, state_values = self.forward(states)
        dist = Categorical(action_probs)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, state_values.squeeze(), entropy


# ========== TRAJECTORY BUFFER ==========
class Trajectory:
    """Stores episode experience"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            done: bool, log_prob: float, value: float):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
    
    def __len__(self):
        return len(self.rewards)


# ========== PPO AGENT ==========
class PPOAgent:
    """Proximal Policy Optimization Agent for Discrete Actions"""
    
    def __init__(self,
                 input_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 critic_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 10,
                 mini_batch_size: int = 64,
                 device: str = 'auto'):
        
        # Device configuration
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Networks
        self.network = PPONetwork(input_dim).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.critic_coef = critic_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        
        self.trajectory = Trajectory()
        self.training_stats = []
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                 done: bool, log_prob: float, value: float):
        self.trajectory.add(state, action, reward, done, log_prob, value)
    
    def compute_returns_and_advantages(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and GAE advantages
        
        Args:
            next_value: Value estimate of the state after the last action (bootstrapping)
        
        Returns:
            returns: Discounted rewards (target for critic)
            advantages: GAE advantages (for policy update)
        """
        rewards = self.trajectory.rewards
        values = self.trajectory.values
        dones = self.trajectory.dones
        
        # Clip rewards for stability
        rewards = np.clip(rewards, -10, 10)                                 # Hard-coded. Dependent on the overall scale of rewards.
        
        # ========== 1. Compute Returns (Discounted Rewards) ==========
        returns = np.zeros(len(rewards))
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0  # Reset at episode boundary
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # ========== 2. Compute Advantages using GAE ==========
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            # Calculate TD error (delta)
            if t == len(rewards) - 1:
                # Last take_action: use bootstrapped next_value
                # If done, next_value should be 0 (already handled by caller)
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                # Middle steps: use actual next value
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            # Update GAE
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return returns, advantages
    
    def learn(self, next_state: Optional[np.ndarray], done: bool) -> Dict:
        """Update policy using PPO"""
        # Handle next value
        if next_state is None or done:
            next_value = 0
        else:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_value = self.network(next_state_tensor)
                next_value = next_value.item()
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.trajectory.states)).to(self.device)
        actions = torch.LongTensor(np.array(self.trajectory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.trajectory.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        if len(advantages) > 1 and advantages.std() > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Store losses for this update
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'total_loss': 0}
        update_count = 0
        
        # PPO update
        dataset_size = len(states)
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate current policy
                log_probs, state_values, entropy = self.network.evaluate(batch_states, batch_actions)
                
                # Calculate ratio
                ratios = torch.exp(log_probs - batch_old_log_probs)
                ratios = torch.clamp(ratios, 0.1, 10)
                
                # PPO loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss
                critic_loss = nn.SmoothL1Loss()(state_values, batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Accumulate losses
                losses['actor_loss'] += actor_loss.item()
                losses['critic_loss'] += critic_loss.item()
                losses['entropy'] += entropy.mean().item()
                losses['total_loss'] += total_loss.item()
                update_count += 1
                
                if torch.isnan(total_loss):
                    continue
                
                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.take_action()
        
        # Average losses
        if update_count > 0:
            for key in losses:
                losses[key] /= update_count
        
        self.trajectory.clear()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return losses
    
    def get_action(self, state: np.ndarray, action_mask: Optional[List[int]] = None) -> Tuple[int, float]:
        return self.network.get_action(state, action_mask, self.device)
    
    def save(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
        }, path)
        print(f"✅ Model saved to {path}")


# ========== TRADING ENVIRONMENT ==========
class TradingEnvironment:
    """Trading environment with discrete actions"""
    
    def __init__(self, price_data: np.ndarray, lookback_window: int = 20,
                 initial_balance: float = 10000, transaction_cost: float = 0.001):
        
        self.price_data = price_data
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        # State tracking
        self.current_step = lookback_window
        self.balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        
        # Rolling window
        self.price_history = deque(maxlen=lookback_window + 50)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0
        self.entry_price = 0
        self.total_trades = 0
        self.profitable_trades = 0
        self.price_history.clear()
        
        start_idx = max(0, self.current_step - self.lookback_window - 20)
        for i in range(start_idx, self.current_step + 1):
            if i < len(self.price_data):
                self.price_history.append(self.price_data[i, 0])
        
        return self._get_state()
    
    def _calculate_technical_indicators(self, prices: np.ndarray) -> dict:
        """Calculate technical indicators from price history"""
        indicators = {}
        
        if len(prices) < 2:
            return {
                'ret_1': 0, 'ret_5': 0, 'ret_10': 0, 'ret_20': 0,
                'price_vs_sma5': 0, 'price_vs_sma20': 0, 'rsi': 50,
                'bb_position': 0.5, 'macd': 0, 'volatility': 0.02,
                'momentum': 0, 'trend': 0
            }
        
        # Returns
        indicators['ret_1'] = (prices[-1] - prices[-2]) / (abs(prices[-2]) + 1e-8)
        indicators['ret_5'] = (prices[-1] - prices[-6]) / (abs(prices[-6]) + 1e-8) if len(prices) >= 6 else 0
        indicators['ret_10'] = (prices[-1] - prices[-11]) / (abs(prices[-11]) + 1e-8) if len(prices) >= 11 else 0
        indicators['ret_20'] = (prices[-1] - prices[0]) / (abs(prices[0]) + 1e-8) if len(prices) >= 20 else 0
        
        for key in ['ret_1', 'ret_5', 'ret_10', 'ret_20']:
            indicators[key] = np.clip(indicators[key], -0.5, 0.5)
        
        # Moving averages
        if len(prices) >= 5:
            indicators['sma_5'] = np.mean(prices[-5:])
            indicators['price_vs_sma5'] = (prices[-1] - indicators['sma_5']) / (abs(indicators['sma_5']) + 1e-8)
        else:
            indicators['price_vs_sma5'] = 0
        
        if len(prices) >= 20:
            indicators['sma_20'] = np.mean(prices[-20:])
            indicators['price_vs_sma20'] = (prices[-1] - indicators['sma_20']) / (abs(indicators['sma_20']) + 1e-8)
        else:
            indicators['price_vs_sma20'] = 0
        
        indicators['price_vs_sma5'] = np.clip(indicators['price_vs_sma5'], -0.3, 0.3)
        indicators['price_vs_sma20'] = np.clip(indicators['price_vs_sma20'], -0.3, 0.3)
        
        # RSI
        if len(prices) >= 15:
            deltas = np.diff(prices[-15:])
            gains = deltas[deltas > 0].sum() / 14
            losses = -deltas[deltas < 0].sum() / 14
            if losses == 0:
                indicators['rsi'] = 100
            else:
                rs = gains / losses
                indicators['rsi'] = 100 - (100 / (1 + rs))
        else:
            indicators['rsi'] = 50
        
        # Bollinger Bands
        if len(prices) >= 20:
            std = np.std(prices[-20:])
            bb_lower = indicators['sma_20'] - 2 * std
            bb_upper = indicators['sma_20'] + 2 * std
            indicators['bb_position'] = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-8)
            indicators['bb_position'] = np.clip(indicators['bb_position'], 0, 1)
        else:
            indicators['bb_position'] = 0.5
        
        # MACD
        if len(prices) >= 26:
            ema_12 = np.mean(prices[-12:])
            ema_26 = np.mean(prices[-26:])
            indicators['macd'] = (ema_12 - ema_26) / (abs(ema_26) + 1e-8)
            indicators['macd'] = np.clip(indicators['macd'], -0.1, 0.1)
        else:
            indicators['macd'] = 0
        
        # Volatility
        if len(prices) >= 20:
            log_returns = np.diff(np.log(np.abs(prices[-20:]) + 1e-8))
            indicators['volatility'] = np.std(log_returns) * np.sqrt(252)
            indicators['volatility'] = np.clip(indicators['volatility'], 0, 1)
        else:
            indicators['volatility'] = 0.02
        
        # Momentum
        if len(prices) >= 14:
            indicators['momentum'] = (prices[-1] - prices[-14]) / (abs(prices[-14]) + 1e-8)
            indicators['momentum'] = np.clip(indicators['momentum'], -0.3, 0.3)
        else:
            indicators['momentum'] = 0
        
        # Trend (FIXED TYPO HERE)
        if len(prices) >= 10:
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            indicators['trend'] = slope / (abs(prices[0]) + 1e-8)
            indicators['trend'] = np.clip(indicators['trend'], -0.1, 0.1)  # FIXED: indicators not indicants
        else:
            indicators['trend'] = 0
        
        return indicators
    
    def _get_state(self) -> np.ndarray:                     #----------- Great, this is the only place to decice the format and contents of state.
        """Construct normalized state vector"""
        if self.current_step >= len(self.price_data):
            return None
        
        current_price = self.price_data[self.current_step, 0]
        price_window = np.array(list(self.price_history))
        
        indicators = self._calculate_technical_indicators(price_window)
        
        features = [
            indicators['ret_1'] * 100,
            indicators['ret_5'] * 100,
            indicators['ret_10'] * 100,
            indicators['ret_20'] * 100,
            indicators['price_vs_sma5'] * 100,
            indicators['price_vs_sma20'] * 100,
            (indicators['rsi'] - 50) / 50,
            indicators['bb_position'] * 2 - 1,
            indicators['macd'] * 100,
            indicators['volatility'] * 100,
            indicators['momentum'] * 100,
            indicators['trend'] * 100,
            self.position / 1000,
        ]
        
        state = np.array(features, dtype=np.float32)
        state = np.clip(state, -10, 10)
        
        self.price_history.append(current_price)
        
        return state
    
    def _get_action_mask(self) -> List[int]:
        """Get valid actions: [Hold, Buy, Sell]"""
        if self.position == 0:
            return [1, 1, 0]
        else:
            return [1, 0, 1]
    
    def take_action(self, action: int) -> Tuple[Optional[np.ndarray], float, bool]:
        """Execute action"""
        current_price = self.price_data[self.current_step, 0]
        reward = 0
        done = False
        
        portfolio_value_before = self.balance + self.position * current_price
        
        if action == 1:  # BUY
            if self.position == 0:
                shares = (self.balance * (1 - self.transaction_cost)) / (current_price + 1e-8)
                self.position = shares
                cost = shares * current_price
                self.balance -= cost
                self.entry_price = current_price
                self.total_trades += 1
                reward -= self.transaction_cost * cost
        
        elif action == 2:  # SELL
            if self.position > 0:
                sale_value = self.position * current_price * (1 - self.transaction_cost)
                self.balance = sale_value
                profit_pct = (current_price - self.entry_price) / (self.entry_price + 1e-8)
                
                if profit_pct > 0:
                    self.profitable_trades += 1
                    reward = profit_pct * 10
                else:
                    reward = profit_pct * 5
                
                self.position = 0
                self.entry_price = 0
                self.total_trades += 1
        
        portfolio_value_after = self.balance + self.position * current_price
        portfolio_return = (portfolio_value_after - portfolio_value_before) / (portfolio_value_before + 1e-8)
        reward += portfolio_return * 100
        
        self.current_step += 1
        
        if self.current_step >= len(self.price_data) - 1:
            done = True
            if self.position > 0:
                final_value = self.position * current_price * (1 - self.transaction_cost)
                self.balance += final_value
                self.position = 0
            
            total_return = (self.balance - self.initial_balance) / (self.initial_balance + 1e-8)
            reward += total_return * 100
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done
    
    def get_portfolio_value(self) -> float:                 #------------- We need an independent portfolio management.
        if self.current_step >= len(self.price_data):
            return self.balance
        
        current_price = self.price_data[min(self.current_step, len(self.price_data)-1), 0]
        return self.balance + self.position * current_price


# ========== TRAINING FUNCTION ==========
def train_ppo(env: TradingEnvironment, agent: PPOAgent, 
              episodes: int = 300, print_interval: int = 20):
    """Train PPO agent with progress tracking"""
    
    metrics = TrainingMetrics()
    
    print("\n" + "="*80)
    print("🚀 STARTING PPO TRAINING")
    print("="*80)
    print(f"Total Episodes: {episodes}")
    print(f"Print Interval: Every {print_interval} episodes")
    print("="*80 + "\n")
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_actions = []
        episode_start_time = datetime.now()
        
        while not done:
            action_mask = env._get_action_mask()
            action, log_prob = agent.get_action(state, action_mask)
            episode_actions.append(action)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                _, value = agent.network(state_tensor)
                value = value.item()
            
            next_state, reward, done = env.take_action(action)
            
            agent.remember(state, action, reward, done, log_prob, value)
            episode_reward += reward
            state = next_state
            
            if done:
                losses = agent.learn(next_state, done)
        
        episode_duration = (datetime.now() - episode_start_time).total_seconds()
        portfolio_value = env.get_portfolio_value()
        total_return = (portfolio_value - env.initial_balance) / env.initial_balance
        win_rate = (env.profitable_trades / max(env.total_trades, 1)) * 100
        
        action_dist = {0: episode_actions.count(0), 
                      1: episode_actions.count(1), 
                      2: episode_actions.count(2)}
        
        metrics.update(episode + 1, episode_reward, portfolio_value, total_return,
                      env.total_trades, win_rate, episode_duration, 
                      episode_actions, losses)
        
        if (episode + 1) % print_interval == 0:
            metrics.print_summary(episode + 1, episode_reward, portfolio_value,
                                 total_return, env.total_trades, win_rate,
                                 action_dist, episode_duration)
            metrics.plot_progress(episode + 1)
    
    metrics.save_metrics()
    metrics.plot_progress(episodes)
    
    return metrics


# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("="*80)
    print("📈 DISCRETE ACTIONS PPO TRADING SYSTEM")
    print("="*80)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 2000
    
    returns = np.random.randn(n_days) * 0.02
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Normalize
    price_data = (prices - prices.mean()) / (prices.std() + 1e-8)
    price_data = price_data.reshape(-1, 1)
    
    # Split
    train_size = 1500
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]
    
    print(f"\n📈 Configuration:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Test samples: {len(test_data)}")
    print(f"   Feature dimension: 13")
    
    env = TradingEnvironment(train_data, lookback_window=20, 
                             initial_balance=10000, transaction_cost=0.001)
    agent = PPOAgent(input_dim=13, device='auto')
    
    print(f"   Transaction cost: {env.transaction_cost:.1%}")
    print(f"   Mini-batch size: {agent.mini_batch_size}")
    print(f"   PPO Epochs: {agent.ppo_epochs}")
    
    # Train
    metrics = train_ppo(env, agent, episodes=300, print_interval=50)
    
    # Save
    agent.save("ppo_discrete_actions_final.pth")
    
    # Test
    print("\n🧪 Testing on unseen data...")
    test_env = TradingEnvironment(test_data, lookback_window=20, 
                                   initial_balance=10000, transaction_cost=0.001)
    
    state = test_env.reset()
    done = False
    test_actions = []
    
    while not done:
        action_mask = test_env._get_action_mask()
        action, _ = agent.get_action(state, action_mask)
        state, reward, done = test_env.take_action(action)
        test_actions.append(action)
    
    final_value = test_env.get_portfolio_value()
    total_return = (final_value - 10000) / 10000 * 100
    
    print(f"\n{'='*80}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*80}")
    print(f"Initial Portfolio: $10000.00")
    print(f"Final Portfolio:   ${final_value:.2f}")
    print(f"Total Return:      {total_return:7.2f}%")
    print(f"Total Trades:      {test_env.total_trades:4d}")
    print(f"Win Rate:          {test_env.profitable_trades/max(test_env.total_trades,1)*100:5.1f}%")
    
    print("\n✅ Complete!")