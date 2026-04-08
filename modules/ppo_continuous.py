import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# ========== SETUP ==========
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

set_seed(42)

# ========== NEURAL NETWORK (Stable Version) ==========
class PPONetwork(nn.Module):
    """
    Stable network for continuous actions with proper initialization
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128], max_position: float = 1.0):
        super(PPONetwork, self).__init__()
        
        self.max_position = max_position
        
        # Actor network (with smaller initialization)
        actor_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.actor = nn.Sequential(*actor_layers)
        self.actor_mean = nn.Linear(prev_dim, 1)
        self.actor_log_std = nn.Parameter(torch.zeros(1))
        
        # Critic network
        critic_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
            ])
            prev_dim = hidden_dim
        
        self.critic = nn.Sequential(*critic_layers)
        self.critic_value = nn.Linear(prev_dim, 1)
        
        # Initialize weights with small values
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization to prevent NaN"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small initialization for stability
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
        
        # Initialize mean layer even smaller
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        # Initialize critic value layer
        nn.init.orthogonal_(self.critic_value.weight, gain=0.01)
        nn.init.constant_(self.critic_value.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns: action_mean, action_std, state_value"""
        # Ensure input is valid
        x = torch.clamp(x, -10, 10)
        
        # Actor forward
        actor_features = self.actor(x)
        action_mean = torch.tanh(self.actor_mean(actor_features)) * self.max_position
        # Clamp mean to prevent extreme values
        action_mean = torch.clamp(action_mean, -self.max_position, self.max_position)
        
        # Ensure std is positive and stable
        action_std = torch.exp(self.actor_log_std).clamp(0.01, 1.0)
        
        # Critic forward
        critic_features = self.critic(x)
        state_value = self.critic_value(critic_features)
        # Clamp value to prevent explosion
        state_value = torch.clamp(state_value, -100, 100)
        
        return action_mean, action_std, state_value
    
    def get_action(self, state: np.ndarray, device: torch.device) -> Tuple[float, float, float]:
        """Sample continuous action safely"""
        # Ensure state is valid
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = np.clip(state, -10, 10)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            mean, std, _ = self.forward(state_tensor)
            
            # Create distribution and sample
            dist = Normal(mean, std)
            action = dist.sample()
            
            # Clamp action to valid range
            action = torch.clamp(action, -self.max_position, self.max_position)
            
            log_prob = dist.log_prob(action).sum(-1)
            
        return action.item(), log_prob.item(), mean.item()
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for training with numerical stability"""
        # Ensure inputs are valid
        states = torch.clamp(states, -10, 10)
        actions = torch.clamp(actions, -self.max_position, self.max_position)
        
        mean, std, state_values = self.forward(states)
        
        # Clamp std to prevent numerical issues
        std = torch.clamp(std, 0.01, 1.0)
        
        # Create distribution
        dist = Normal(mean, std)
        
        # Calculate log probs with safe clipping
        log_probs = dist.log_prob(actions).sum(-1)
        log_probs = torch.clamp(log_probs, -100, 100)
        
        # Calculate entropy
        entropy = dist.entropy().sum(-1)
        entropy = torch.clamp(entropy, 0, 10)
        
        return log_probs, state_values.squeeze(), entropy


# ========== TRAJECTORY BUFFER ==========
class Trajectory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, state, action, reward, done, log_prob, value):
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
    def __init__(self,
                 input_dim: int,
                 max_position: float = 1.0,
                 learning_rate: float = 1e-4,  # Smaller learning rate
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 critic_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 5,  # Fewer epochs initially
                 mini_batch_size: int = 32,  # Smaller batch size
                 device: str = 'auto'):
        
        # Device setup
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Using device: {self.device}")
        
        # Networks
        self.network = PPONetwork(input_dim, max_position=max_position).to(self.device)
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
    
    def remember(self, state, action, reward, done, log_prob, value):
        self.trajectory.add(state, action, reward, done, log_prob, value)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Stable GAE computation"""
        rewards = np.array(self.trajectory.rewards, dtype=np.float32)
        values = np.array(self.trajectory.values, dtype=np.float32)
        dones = np.array(self.trajectory.dones, dtype=np.float32)
        
        # Clip rewards to prevent explosion
        rewards = np.clip(rewards, -10, 10)
        
        # Compute returns
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + self.gamma * running_return
            returns[t] = running_return
        
        # Compute advantages
        advantages = np.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            
            # Clamp delta to prevent explosion
            delta = np.clip(delta, -100, 100)
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        return returns, advantages
    
    def learn(self, next_state: Optional[np.ndarray], done: bool):
        """Update policy using PPO with stability measures"""
        if len(self.trajectory) == 0:
            return
        
        # Get next value
        if next_state is None or done:
            next_value = 0.0
        else:
            # Ensure next_state is valid
            next_state = np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0)
            next_state = np.clip(next_state, -10, 10)
            
            with torch.no_grad():
                state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, _, value = self.network(state_tensor)
                next_value = value.item()
                next_value = np.clip(next_value, -100, 100)
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(next_value)
        
        # Normalize advantages safely
        if len(advantages) > 1 and np.std(advantages) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        else:
            advantages = np.zeros_like(advantages)
        
        # Clip advantages
        advantages = np.clip(advantages, -10, 10)
        
        # Convert to tensors
        states = np.array([np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0) for s in self.trajectory.states])
        states = np.clip(states, -10, 10)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(np.array(self.trajectory.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.trajectory.log_probs)).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO updates
        dataset_size = len(states)
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                batch_states = states[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Evaluate current policy
                log_probs, state_values, entropy = self.network.evaluate(batch_states, batch_actions)
                
                # PPO loss with clipping
                ratios = torch.exp(log_probs - batch_old_log_probs)
                ratios = torch.clamp(ratios, 0.1, 10)  # Prevent extreme ratios
                
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (Huber loss is more stable)
                critic_loss = nn.SmoothL1Loss()(state_values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Check for NaN
                if torch.isnan(total_loss):
                    print("Warning: NaN loss detected, skipping update")
                    continue
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
        
        self.trajectory.clear()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def get_action(self, state: np.ndarray) -> Tuple[float, float, float]:
        """Get continuous action safely"""
        return self.network.get_action(state, self.device)
    
    def save(self, path: str):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


# ========== TRADING ENVIRONMENT ==========
class TradingEnvironment:
    """
    Stable trading environment with numerical safety
    """
    
    def __init__(self, 
                 price_data: np.ndarray,
                 lookback_window: int = 20,
                 initial_capital: float = 10000,
                 max_position_pct: float = 1.0,
                 transaction_cost_pct: float = 0.001):
        
        self.price_data = price_data
        self.lookback_window = lookback_window
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.transaction_cost_pct = transaction_cost_pct
        
        # State tracking
        self.current_step = lookback_window
        self.capital = initial_capital
        self.position_pct = 0.0
        self.last_price = 0.0
        self.total_trades = 0
        
        # Price history
        self.price_history = deque(maxlen=lookback_window + 50)
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = self.lookback_window
        self.capital = self.initial_capital
        self.position_pct = 0.0
        self.total_trades = 0
        self.price_history.clear()
        
        # Initialize price history
        start_idx = max(0, self.current_step - self.lookback_window - 10)
        for i in range(start_idx, self.current_step + 1):
            if i < len(self.price_data):
                self.price_history.append(self.price_data[i, 0])
        
        # Ensure we have enough history
        while len(self.price_history) < self.lookback_window + 1 and self.current_step < len(self.price_data) - 1:
            self.current_step += 1
            self.price_history.append(self.price_data[self.current_step, 0])
        
        self.last_price = self.price_data[self.current_step, 0] if self.current_step < len(self.price_data) else 100
        
        return self._get_state()
    
    def _calculate_features(self, prices: np.ndarray) -> Dict[str, float]:
        """Calculate features safely"""
        features = {}
        
        if len(prices) < 2:
            return {
                'returns_1': 0.0, 'returns_5': 0.0, 'returns_20': 0.0,
                'volatility': 0.02, 'momentum': 0.0, 'mean_reversion': 0.0,
                'trend': 0.0, 'rsi': 50.0
            }
        
        # Returns (safe division)
        current_price = prices[-1]
        prev_price = prices[-2] if len(prices) >= 2 else current_price
        
        features['returns_1'] = (current_price - prev_price) / (abs(prev_price) + 1e-8)
        features['returns_1'] = np.clip(features['returns_1'], -0.1, 0.1)
        
        if len(prices) >= 6:
            price_5d_ago = prices[-6]
            features['returns_5'] = (current_price - price_5d_ago) / (abs(price_5d_ago) + 1e-8)
            features['returns_5'] = np.clip(features['returns_5'], -0.2, 0.2)
        else:
            features['returns_5'] = 0.0
        
        if len(prices) >= 20:
            price_20d_ago = prices[0]
            features['returns_20'] = (current_price - price_20d_ago) / (abs(price_20d_ago) + 1e-8)
            features['returns_20'] = np.clip(features['returns_20'], -0.3, 0.3)
        else:
            features['returns_20'] = 0.0
        
        # Volatility
        if len(prices) >= 20:
            log_returns = np.diff(np.log(np.abs(prices[-20:]) + 1e-8))
            features['volatility'] = np.std(log_returns) * np.sqrt(252)
            features['volatility'] = np.clip(features['volatility'], 0.01, 1.0)
        else:
            features['volatility'] = 0.02
        
        # Momentum
        if len(prices) >= 10:
            price_10d_ago = prices[-10]
            features['momentum'] = (current_price - price_10d_ago) / (abs(price_10d_ago) + 1e-8)
            features['momentum'] = np.clip(features['momentum'], -0.2, 0.2)
        else:
            features['momentum'] = 0.0
        
        # Mean reversion
        if len(prices) >= 20:
            sma = np.mean(prices[-20:])
            features['mean_reversion'] = (sma - current_price) / (abs(sma) + 1e-8)
            features['mean_reversion'] = np.clip(features['mean_reversion'], -0.1, 0.1)
        else:
            features['mean_reversion'] = 0.0
        
        # Trend strength
        if len(prices) >= 20:
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices, 1)[0]
            features['trend'] = slope / (abs(prices[0]) + 1e-8)
            features['trend'] = np.clip(features['trend'], -0.05, 0.05)
        else:
            features['trend'] = 0.0
        
        # RSI
        if len(prices) >= 15:
            deltas = np.diff(prices[-15:])
            gains = np.sum(np.maximum(deltas, 0)) / 14
            losses = -np.sum(np.minimum(deltas, 0)) / 14
            if losses < 1e-8:
                features['rsi'] = 100.0
            else:
                rs = gains / losses
                features['rsi'] = 100 - (100 / (1 + rs))
            features['rsi'] = np.clip(features['rsi'], 0, 100)
        else:
            features['rsi'] = 50.0
        
        return features
    
    def _get_state(self) -> np.ndarray:
        """Construct state vector"""
        if self.current_step >= len(self.price_data):
            return None
        
        current_price = self.price_data[self.current_step, 0]
        price_window = np.array(list(self.price_history))
        
        features = self._calculate_features(price_window)
        
        # Build state vector (all features normalized)
        state = np.array([
            features['returns_1'] * 100,
            features['returns_5'] * 100,
            features['returns_20'] * 100,
            features['volatility'] * 100,
            features['momentum'] * 100,
            features['mean_reversion'] * 100,
            features['trend'] * 100,
            (features['rsi'] - 50) / 50,
            self.position_pct,
            self.capital / self.initial_capital,
        ], dtype=np.float32)
        
        # Ensure no NaN or inf
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = np.clip(state, -5, 5)
        
        # Update history
        self.price_history.append(current_price)
        self.last_price = current_price
        
        return state
    
    def step(self, action: float) -> Tuple[Optional[np.ndarray], float, bool]:
        """Execute action safely"""
        current_price = self.price_data[self.current_step, 0]
        
        # Clip action
        action = np.clip(action, -self.max_position_pct, self.max_position_pct)
        
        # Calculate trade
        target_position = action
        trade_size = target_position - self.position_pct
        
        # Apply transaction costs
        cost = abs(trade_size) * self.capital * self.transaction_cost_pct
        self.capital -= cost
        
        # Update position
        self.position_pct = target_position
        
        # Calculate reward (log return)
        if self.last_price > 0 and current_price > 0:
            price_return = (current_price - self.last_price) / self.last_price
            pnl = self.position_pct * price_return
        else:
            pnl = 0
        
        # Reward based on capital change and P&L
        reward = pnl - abs(trade_size) * self.transaction_cost_pct * 10
        
        # Small penalty for holding large positions
        reward -= abs(self.position_pct) * 0.001
        
        # Clip reward
        reward = np.clip(reward, -1, 1)
        
        # Track trades
        if abs(trade_size) > 0.01:
            self.total_trades += 1
        
        # Move to next step
        self.current_step += 1
        done = self.current_step >= len(self.price_data) - 1
        
        if done:
            # Final reward based on total return
            total_return = (self.capital - self.initial_capital) / self.initial_capital
            reward += np.clip(total_return, -1, 1) * 10
        
        next_state = self._get_state() if not done else None
        
        return next_state, reward, done
    
    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        return self.capital


# ========== TRAINING LOOP ==========
def train_ppo(env: TradingEnvironment, agent: PPOAgent, 
              episodes: int = 500, print_interval: int = 10):
    """Train PPO agent"""
    
    episode_rewards = []
    episode_returns = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Get action
            action, log_prob, mean = agent.get_action(state)
            
            # Get value
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                _, _, value = agent.network(state_tensor)
                value = value.item()
            
            # Take step
            next_state, reward, done = env.step(action)
            
            # Store
            agent.remember(state, action, reward, done, log_prob, value)
            
            episode_reward += reward
            state = next_state
            
            if done:
                agent.learn(next_state, done)
        
        episode_rewards.append(episode_reward)
        final_value = env.get_portfolio_value()
        total_return = (final_value - env.initial_capital) / env.initial_capital
        episode_returns.append(total_return)
        
        if (episode + 1) % print_interval == 0:
            avg_reward = np.mean(episode_rewards[-print_interval:])
            avg_return = np.mean(episode_returns[-print_interval:]) * 100
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {avg_reward:6.3f} | "
                  f"Return: {avg_return:6.2f}% | "
                  f"Trades: {env.total_trades:4d} | "
                  f"Pos: {env.position_pct:5.2f}")
    
    return episode_rewards, episode_returns


# ========== MAIN ==========
if __name__ == "__main__":
    print("="*60)
    print("STABLE PPO TRADING SYSTEM")
    print("="*60)
    
    # Generate realistic data
    np.random.seed(42)
    n_days = 2000
    
    # Simple random walk
    returns = np.random.randn(n_days) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Normalize
    price_data = (prices - prices.mean()) / (prices.std() + 1e-8)
    price_data = price_data.reshape(-1, 1)
    
    # Split
    train_size = 1500
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]
    
    # Create environment and agent
    print(f"\n📈 Configuration:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Feature dimension: 10")
    
    env = TradingEnvironment(train_data, lookback_window=20, initial_capital=10000)
    agent = PPOAgent(input_dim=10, max_position=1.0, device='auto')
    
    print(f"   Transaction cost: {env.transaction_cost_pct:.1%}")
    print(f"   Learning rate: 1e-4")
    
    # Train
    print("\n🏋️ Training...")
    print("="*60)
    rewards, returns = train_ppo(env, agent, episodes=200, print_interval=20)
    
    # Save
    agent.save("ppo_stable.pth")
    
    # Test
    print("\n🧪 Testing...")
    test_env = TradingEnvironment(test_data, lookback_window=20, initial_capital=10000)
    
    state = test_env.reset()
    done = False
    values = []
    
    while not done:
        action, _, _ = agent.get_action(state)
        state, reward, done = test_env.step(action)
        values.append(test_env.get_portfolio_value())
    
    final_return = (values[-1] - 10000) / 10000 * 100
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"Final Portfolio: ${values[-1]:.2f}")
    print(f"Total Return: {final_return:.2f}%")
    print(f"Total Trades: {test_env.total_trades}")
    print(f"{'='*60}")
    
    print("\n✅ Complete!")