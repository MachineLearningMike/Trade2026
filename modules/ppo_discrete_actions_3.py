import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import deque
import random
from typing import List, Tuple, Dict, Optional
from enum import Enum
import warnings

from RL_Market_History import Market_History
from Legacy.Binance_new import Binance # Get_Price_Data_By_Time

warnings.filterwarnings('ignore')

# ========== ACTION ENUM ==========
class TradingAction(Enum):
    HOLD = 0
    BUY_1 = 1
    BUY_2 = 2
    BUY_3 = 3
    SELL_1 = 4
    SELL_2 = 5
    SELL_3 = 6
    SELL_ALL = 7
    
    @property
    def units(self) -> int:
        sizes = {self.HOLD: 0, 
                 self.BUY_1: 1, self.BUY_2: 2, self.BUY_3: 4, 
                 self.SELL_1: 1, self.SELL_2: 2, self.SELL_3: 4, self.SELL_ALL: 1}  # any non-zero for self.SELL_ALL
        return sizes[self]
    
    @property
    def name(self) -> str:
        names = {self.HOLD: "HOLD", self.BUY_1: "BUY_1", self.BUY_2: "BUY_2", self.BUY_3: "BUY_3", 
                 self.SELL_1: "SELL_1", self.SELL_2: "SELL_2", self.SELL_3: "SELL_3", self.SELL_ALL: "SELL_ALL", }
        return names[self]


# ========== STABLE NEURAL NETWORK ==========
class StablePPONetwork(nn.Module):
    """Numerically stable PPO Network"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128], action_dim: int = 5):
        super(StablePPONetwork, self).__init__()
        
        self.input_dim = input_dim
        self.action_dim = action_dim
        
        # Smaller initialization for stability
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Tanh(),  # Tanh is more stable than ReLU for financial data
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*layers)
        
        # Actor head with logits (no softmax here)
        self.actor = nn.Linear(prev_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(prev_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Conservative weight initialization to prevent NaNs"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small initialization for stability
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor):
        """Forward pass with numerical stability"""
        # Clip input to prevent extreme values
        x = torch.clamp(x, -10, 10)
        
        features = self.shared(x)
        
        # Actor logits (raw scores)
        logits = self.actor(features)
        
        # Stable softmax with temperature for numerical stability
        logits = torch.clamp(logits, -20, 20)  # Prevent exp explosion
        action_probs = torch.softmax(logits, dim=-1)
        
        # Critic value
        state_value = self.critic(features)
        state_value = torch.clamp(state_value, -100, 100)
        
        return action_probs, state_value
    
    def get_action(self, state: np.ndarray, device: torch.device) -> Tuple[int, float, torch.Tensor]:
        """Get action with NaN protection"""
        # Ensure state is valid
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = np.clip(state, -10, 10)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs, _ = self.forward(state_tensor)
            action_probs = action_probs.squeeze(0)
            
            # Final NaN check and fallback
            if torch.isnan(action_probs).any():
                # Fallback to uniform distribution
                action_probs = torch.ones(self.action_dim, device=device) / self.action_dim
            
            dist = Categorical(action_probs)
            action = dist.sample()              # If it isn't greedy, it explores always with shifting probabilty(?)
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), action_probs


# ========== STABLE PPO AGENT ==========
class StablePPOAgent:
    def __init__(self,
                 input_dim: int,
                 action_dim: int = 5,
                 learning_rate: float = 1e-4,  # Lower learning rate
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 critic_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 ppo_epochs: int = 5,  # Fewer epochs
                 mini_batch_size: int = 32,  # Smaller batch
                 device: str = 'auto'):
        
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🚀 Using device: {self.device}")
        
        self.network = StablePPONetwork(input_dim, action_dim=action_dim).to(self.device)
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
    
    def compute_returns_and_advantages(self, next_value: float):
        # action, log_prob = actor(state), value = critic(state), reward, done = env(action)
        # Discrepancy between actor and critic is represented.
        
        rewards = np.array(self.trajectory.rewards, dtype=np.float32)
        values = np.array(self.trajectory.values, dtype=np.float32)
        dones = np.array(self.trajectory.dones, dtype=np.float32)
        
        # Clip rewards aggressively
        rewards = np.clip(rewards, -5, 5)
        
        # Compute returns
        returns = np.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:                # so running_return will be realistic only if the step t has done == True in here or future.
                running_return = 0      # Particularly, dones[-1] == True is required for healthy running_return.
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
                
            # delta reflects the discrepancy between the actor and critic.
            # If values are consistent and the actor elects correct actions and they are optial, then delta is near zero.
            # If delta is far from zero, then either values are not consistent, actor elects incorrect actions, or they are not optimal.
            # Whatt actually happen: rewards[t] + self.gamma * values[t+1] * (1 - dones[t])
            # What is expected: values[t]
            
            # Clip delta
            delta = np.clip(delta, -10, 10)
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Normalize advantages
        if len(advantages) > 1 and np.std(advantages) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def learn(self, next_state, done):
        if len(self.trajectory) == 0:
            return {}
        
        # Get next value
        if next_state is None or done:
            next_value = 0.0
        else:
            # Ensure next_state is valid
            next_state = np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0)
            next_state = np.clip(next_state, -10, 10)   #----------------------------
            
            with torch.no_grad():
                next_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                _, next_val = self.network(next_tensor)
                next_value = next_val.item()
                next_value = np.clip(next_value, -10, 10)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)
        # advantages = advantages.detach()  # this is a numpy object and has no string to policy parameters.
        
        # Convert to tensors with NaN protection
        states = np.array([np.nan_to_num(s, nan=0.0, posinf=1.0, neginf=-1.0) for s in self.trajectory.states])
        states = np.clip(states, -10, 10)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(self.trajectory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.trajectory.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO updates
        dataset_size = len(states)
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'total_loss': 0}
        update_count = 0
        
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
                
                # Get current policy
                action_probs, state_values = self.network(batch_states)
                dist = Categorical(action_probs)
                
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy()
                
                # Calculate ratios with clipping
                ratios = torch.exp(log_probs - batch_old_log_probs)
                ratios = torch.clamp(ratios, 0.1, 10)
                
                # PPO loss or actor loss. Only ratio is dependent on policy parameters. batch_advantages is a constant.
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Critic loss (Huber loss is more stable). Dependent on state-value parameters. batch_returns is a constant.
                critic_loss = nn.SmoothL1Loss()(state_values.squeeze(), batch_returns)
                
                # Entropy bonus. Dependent on policy parameters.
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = actor_loss + self.critic_coef * critic_loss + self.entropy_coef * entropy_loss
                
                # Check for NaN
                if torch.isnan(total_loss):
                    print(f"Warning: NaN loss at batch {start}")
                    continue
                
                # Update
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate losses
                losses['actor_loss'] += actor_loss.item()
                losses['critic_loss'] += critic_loss.item()
                losses['entropy'] += entropy.mean().item()
                losses['total_loss'] += total_loss.item()
                update_count += 1
        
        if update_count > 0:
            for key in losses:
                losses[key] /= update_count
        
        self.trajectory.clear()
        
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return losses
    
    def get_action(self, state):
        action, log_prob, _ = self.network.get_action(state, self.device)
        return action, log_prob
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✅ Model saved to {path}")


# ========== TRAJECTORY ==========
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


# ========== SIMPLIFIED TRADING ENVIRONMENT ==========
class TradingEnvironment:
    def __init__(self, price_data, lookback_window=20,
                 initial_balance=10000, transaction_fee_rate=0.001,
                 unit_size=100, max_units=5, dim_compact_history=300, indicator_window=30):
        
        self.exchange = Binance.instantiate(connect=False)
        self.market_hisroty = Market_History(dim_compact_history=dim_compact_history, indicator_window=indicator_window)
                    
        self.price_data = price_data
        self.lookback_window = lookback_window
        self.initial_balance = initial_balance
        self.transaction_fee_rate = transaction_fee_rate
        self.unit_size = unit_size
        self.max_units = max_units
        
        self.current_step = lookback_window
        self.balance = initial_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
        self.price_history = deque(maxlen=lookback_window + 10)
        
        self.acc_cost = 0
        self.total_actions_taken = 0
        self.high_balance = - initial_balance * 10
        self.low_balance = initial_balance * 10
        
        # Normalization statistics
        self.price_mean = 0
        self.price_std = 1
        
        self.reset()
            
    def reset(self):
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
        self.price_history.clear()
        
        start_idx = max(0, self.current_step - self.lookback_window)
        for i in range(start_idx, self.current_step + 1):
            if i < len(self.price_data):
                self.price_history.append(self.price_data[i, 0])
                
        self.acc_cost = 0
        self.total_actions_taken = 0
        self.high_balance = - initial_balance * 10
        self.low_balance = initial_balance * 10
        
        # Calculate statistics from training data
        if self.price_mean == 0 and len(self.price_data) > 0:
            train_prices = self.price_data[:min(1000, len(self.price_data)), 0]
            self.price_mean = np.mean(train_prices)
            self.price_std = np.std(train_prices) + 1e-8
        
        return self._get_state()
    
    def _get_state(self):
        """Simplified state to avoid NaNs"""
        if self.current_step >= len(self.price_data):
            return None
        
        price_window = np.array(list(self.price_history))
        current_price = price_window[-1] if len(price_window) > 0 else 100
        
        # Simple returns (no complex indicators)
        ret_1 = 0
        ret_5 = 0
        ret_20 = 0
        
        if len(price_window) >= 2:
            ret_1 = (price_window[-1] - price_window[-2]) / (price_window[-2] + 1e-8)
        if len(price_window) >= 6:
            ret_5 = (price_window[-1] - price_window[-6]) / (price_window[-6] + 1e-8)
        if len(price_window) >= 21:
            ret_20 = (price_window[-1] - price_window[-21]) / (price_window[-21] + 1e-8)
        
        # Normalize returns
        ret_1 = np.clip(ret_1, -0.2, 0.2) * 100
        ret_5 = np.clip(ret_5, -0.4, 0.4) * 100
        ret_20 = np.clip(ret_20, -0.8, 0.8) * 100
        
        # Normalized price
        norm_price = (current_price - self.price_mean) / self.price_std
        norm_price = np.clip(norm_price, -3, 3)
        
        # Position features
        can_buy_1 = self.balance >= self.unit_size * (1 + self.transaction_fee_rate)
        can_buy_3 = self.balance >= self.unit_size * 3 * (1 + self.transaction_fee_rate)
        can_sell_1 = self.units_held >= 1
        can_sell_3 = self.units_held >= 3
        
        # Build state
        state = np.array([
            ret_1,
            ret_5,
            ret_20,
            norm_price,
            self.balance / self.initial_balance,
            self.units_held / self.max_units,
            float(can_buy_1),
            float(can_buy_3),
            float(can_sell_1),
            float(can_sell_3),
        ], dtype=np.float32)
        
        # Final NaN protection
        state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        state = np.clip(state, -10, 10)
        
        # Update history
        self.price_history.append(current_price)
        self.current_step += 1
        
        return state
    
    def step(self, action):
        current_price = self.price_data[self.current_step - 1, 0]  # Use the price we just processed
        reward = 0
        done = False
        sold = False
        
        if action == TradingAction.HOLD.value:
            pass
        elif action in [TradingAction.BUY_1.value, TradingAction.BUY_2.value, TradingAction.BUY_3.value]:
            units = TradingAction(action).units
            principal = self.unit_size * units * current_price
            fees = principal * self.transaction_fee_rate
            cash_out_flow = principal + fees
            if self.balance >= cash_out_flow and self.balance - cash_out_flow > self.initial_balance * 0.7:   # ask portfolio for a decision later
                self.balance -= cash_out_flow
                self.acc_cost += cash_out_flow
                self.units_held += units
                self.total_trades += 1
                reward -= fees
            
        elif action in [TradingAction.SELL_1.value, TradingAction.SELL_2.value, TradingAction.SELL_3.value, TradingAction.SELL_ALL.value]:
            units = TradingAction(action).units if action != TradingAction.SELL_ALL.value else self.units_held
            # print(f"units {units} action {action}")
            if self.units_held >= units and units > 0:
                avg_price = self.acc_cost / self.units_held     # as we want to recover the acc_cost with units_held.
                principal =  self.unit_size * units * current_price
                fees = principal * self.transaction_fee_rate
                cash_in_flow = principal - fees
                self.balance += cash_in_flow
                self.acc_cost -= cash_in_flow
                self.units_held -= units
                self.total_trades += 1
                profit = self.unit_size * units * (current_price - avg_price) - fees
                reward += profit / self.unit_size
                sold = True

        # Holding incentive
        if self.units_held > 0:
            reward += 0.01
 
        self.total_actions_taken += 1
        
        if (action == TradingAction.SELL_ALL.value and sold is True) or self.current_step >= len(self.price_data):
            done = True
        
        next_state = None if done else self._get_state()
        
        if self.high_balance < self.balance: self.high_balance = self.balance
        if self.low_balance > self.balance: self.low_balance = self.balance
        
        return next_state, reward, done
    
    def _get_position_value(self, current_price):
        if self.units_held == 0:
            return 0
        total = 0
        for entry_price in self.entry_prices:
            total += self.unit_size * (current_price / entry_price)
        return total
    
    def get_portfolio_value(self):
        if self.current_step >= len(self.price_data):
            return self.balance
        current_price = self.price_data[self.current_step - 1, 0] if self.current_step > 0 else self.price_data[0, 0]
        return self.balance + self._get_position_value(current_price)


# ========== TRAINING FUNCTION ==========
def train_ppo(env, agent, episodes=300, print_interval=50):
    print("\n" + "="*60)
    print("🚀 STARTING PPO TRAINING")
    print("="*60)
    
    profitabilities = deque(maxlen=int(episodes/10))
    
    for episode in range(episodes):

        state = env.reset()

        done = False
        episode_reward = 0
        step_count = 0
        
        while not done:
            action, log_prob = agent.get_action(state)      # action, log_prob = policy(state) = actor(state)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            with torch.no_grad():
                _, value = agent.network(state_tensor)      # value = state_value(state) = critic(state)
                value = value.item()
            
            next_state, reward, done = env.step(action)     # next_state, reward, done = env(action)
            
            # action, log_prob = actor(state), value = critic(state), reward, done = env(action)
            # Discrepancy between actor and critic is represented.
            agent.remember(state, action, reward, done, log_prob, value)
            episode_reward += reward
            state = next_state
            step_count += 1
            
            if done:
                losses = agent.learn(next_state, done)      # next_state is None if done.
        
        if (episode + 1) % print_interval == 0:
            win_rate = (env.profitable_trades / max(env.total_trades, 1)) * 100
            profitable = env.balance-env.initial_balance > 0
            profitabilities.append(1 if profitable else 0)
            p_frequency = sum(profitabilities) / len(profitabilities)
            print(f"Episode {episode+1:4d} | "
                  f"Reward: {episode_reward:.2f} | "
                  f"Balance: ${env.balance:.0f} ${env.low_balance:.0f} ${env.high_balance:.0f} | "
                  f"Trades: {env.total_trades:3d} | "
                  f"Trades/Actions: {env.total_trades/env.total_actions_taken:.2f} | "
                  f"Profit/Trades: {(env.balance-env.initial_balance)/env.total_trades:.3f} | "
                  f"p_freq: {p_frequency:.2f} | "
                  f"{'----------- ' if profitable else ''}"
                  )
    
    return env, agent


# ========== MAIN ==========
if __name__ == "__main__":
    print("="*60)
    print("STABLE PPO TRADING SYSTEM")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_days = 5000
    returns = np.random.randn(n_days) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    price_data = prices.reshape(-1, 1)
    
    # Split
    train_size = int(n_days * 0.8)
    train_data = price_data[:train_size]
    test_data = price_data[train_size:]
    
    print(f"\n📈 Configuration:")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Feature dimension: 10")

    mean_price = price_data.mean()
    unit_size = 100 / mean_price    # 1 unit will be of around 100 dollar value.
    initial_balance = 10_000        #
    max_units = 50
    action_dim = len(TradingAction)
    
    # Create environment and agent
    env = TradingEnvironment(train_data, lookback_window=20, 
                             initial_balance=initial_balance, unit_size=unit_size, max_units=max_units)
    agent = StablePPOAgent(input_dim=10, action_dim=action_dim, device='auto')
        
    # Train
    env, agent = train_ppo(env, agent, episodes=5000, print_interval=1)
    
    # Save
    agent.save("stable_ppo_model.pth")
    
    # Test
    print("\n🧪 Testing...")
    test_env = TradingEnvironment(test_data, lookback_window=20, 
                                   initial_balance=initial_balance, unit_size=unit_size, max_units=max_units)
    
    state = test_env.reset()
    done = False
    test_actions = []
    
    while not done:
        action, _ = agent.get_action(state)
        state, reward, done = test_env.step(action)
        test_actions.append(action)
    
    final_value = test_env.balance
    total_return = (final_value - initial_balance) / initial_balance * unit_size
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"Final Portfolio: ${final_value:.2f}")
    print(f"Total Return: {total_return:.2f}%")
    print(f"Total Trades: {test_env.total_trades}")
    print(f"Win Rate: {test_env.profitable_trades/max(test_env.total_trades,1)*100:.1f}%")
    
    print("\n✅ Complete!")