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
from datetime import datetime, timedelta

from Legacy.Binance_new import Binance # Get_Price_Data_By_Time
from RL_Market_History import Market_History

from RL_Indicators import normalize, get_level, get_change, get_volatility
from indicator_super import calculate_supertrend_close_only
from indicator_aggregated import calculate_best_indicators
# from DSRL_planning_with_fees_penalty_4 import find_strategy_with_fees_and_penalty_vectorized
# from DSRL_planning_with_fees_penalty_5 import find_strategy_with_fees_and_penalty_vectorized
from DSRL_planning_with_fees_penalty_7 import optimal_trading_strategy, skip_buys
# from DSRL_rewards_with_fees_penalty_3 import find_strategic_rewards_with_fees_and_penalty
from DSRL_rewards_with_fees_penalty_5 import find_strategic_rewards_with_fees_and_penalty

from ppo_lstm_networks import BiLSTM, BiLSTMWithAttention, BiLSTMWithLastTimestep
from RL_Trade_POC_Util import plot_arrays, interval_to_seconds

warnings.filterwarnings('ignore')

# ========== ACTION ENUM ==========
# class TradingAction(Enum):
#     HOLD = 0
#     BUY = 1
#     BUY_2 = 2
#     BUY_3 = 3
#     SELL_1 = 4
#     SELL_2 = 5
#     SELL_3 = 6
#     SELLOUT = 7
    
#     @property
#     def units(self) -> int:
#         sizes = {self.HOLD: 0, 
#                  self.BUY: 1, self.BUY_2: 2, self.BUY_3: 4, 
#                  self.SELL_1: 1, self.SELL_2: 2, self.SELL_3: 4, self.SELLOUT: 1}  # any non-zero for self.SELLOUT
#         return sizes[self]
    
#     @property
#     def name(self) -> str:
#         names = {self.HOLD: "HOLD", self.BUY: "BUY", self.BUY_2: "BUY_2", self.BUY_3: "BUY_3", 
#                  self.SELL_1: "SELL_1", self.SELL_2: "SELL_2", self.SELL_3: "SELL_3", self.SELLOUT: "SELLOUT", }
#         return names[self]


# ========== ACTION ENUM ==========
class TradingAction(Enum):
    BUY = 0
    HOLD = 1
    # BUY_2 = 2
    # BUY_3 = 3
    # SELL_1 = 4
    # SELL_2 = 5
    # SELL_3 = 6
    SELLOUT = 2
    
    @property
    def units(self) -> int:
        sizes = {
            self.BUY: 1,
            self.HOLD: 0,
            self.SELLOUT: -1
        }  # any non-zero for self.SELLOUT
        return sizes[self]
    
    @property
    def name(self) -> str:
        names = {
            self.BUY: "BUY",
            self.HOLD: "HOLD",
            self.SELLOUT: "SELLOUT", 
        }
        return names[self]

# ========== STABLE NEURAL NETWORK ==========
class PPO_Actor_Critic(nn.Module):

    def __init__(self, state_params: dict, history_process_params: dict, state_process_params: dict, action_dim: int):
        super(PPO_Actor_Critic, self).__init__()
        
        history_dim = state_params['history_dim']
        indicator_dim = state_params['indicator_dim']
        
        hp_infeature_dim = history_process_params['infeature_dim']
        assert hp_infeature_dim == history_dim
        hp_outfeature_dim = history_process_params['outfeature_dim']
        state_dim = hp_outfeature_dim + indicator_dim + history_process_params['max_episode_len']
        hp_num_layers = history_process_params['num_layers']
        hp_hidden_dim = history_process_params['hidden_dim']
        hp_dropout = history_process_params['dropout']
        
        sp_shared_hidden_dims = state_process_params['shared_hidden_dims']
        sp_actor_hidden_dims = state_process_params['actor_hidden_dims']
        sp_critic_hidden_dims = state_process_params['critic_hidden_dims']
        
        self.temperature = state_process_params['temperature']
        
        self.action_dim = action_dim
        
        # self.history_process = BiLSTM(
        # self.history_process = BiLSTMWithLastTimestep(
        self.history_process = BiLSTMWithAttention(  
            nInFeatures=hp_infeature_dim,   # input features per timestep
            nOutFeatures=hp_outfeature_dim, # output features
            hidden_size=hp_hidden_dim,      # LSTM hidden size
            num_layers=hp_num_layers,       # LSTM layers
            dropout=hp_dropout              # dropout
        )
        
        def create_mlp(starting_dim, hidden_dims):
            layers = []
            prev_dim = starting_dim
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.Tanh(),  # Tanh is more stable than ReLU for financial data
                    nn.Dropout(0)
                ])
                prev_dim = hidden_dim            
            mlp, output_dim = nn.Sequential(*layers), hidden_dims[-1]
            return mlp, output_dim
        
        self.shared, shared_dim = create_mlp(state_dim, sp_shared_hidden_dims)

        self.actor, output_dim = create_mlp(shared_dim, sp_actor_hidden_dims)
        self.action_preference = nn.Linear(output_dim, action_dim)

        self.critic, output_dim = create_mlp(shared_dim, sp_critic_hidden_dims)
        self.critic_head = nn.Linear(output_dim, 1)
        
        # self._init_weights()      #-------- NNNNNNNNNNNOOOOOOOOO, no gain=0.01
    
    def _init_weights(self):
        """Conservative weight initialization to prevent NaNs"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Small initialization for stability
                nn.init.orthogonal_(module.weight, gain=0.01)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state_tensor_tuples: tuple, device):
        # state_tensor_tuples elem shape: (dim_compact_history, 2), (nIndicators,)
        """Forward pass with numerical stability"""
        # # Clip input to prevent extreme values
        # x = torch.clamp(x, -10, 10)
       
        historical_state = torch.stack([tuple[0] for tuple in state_tensor_tuples], dim=0).to(device)  # (batch, dim_compact_history, 2)
        indicator_state = torch.stack([tuple[1] for tuple in state_tensor_tuples], dim=0).to(device)   # (batch, nIndicators)

        historical_state = self.history_process(historical_state)      # (batch, nFeatures)
        learned_state = torch.concat([historical_state, indicator_state], dim=1)      # (batch, nFeatures + nIndicators)
                       
        features = self.shared(learned_state)
        
        preference = self.action_preference(self.actor(features))        
        preference = torch.clamp(preference, -20, 20)  # Prevent exp explosion    # clamp prevents optimality or deterministrics.
        action_probs = torch.softmax(preference, dim=-1)
        
        state_value = self.critic_head(self.critic(features))
        state_value = torch.clamp(state_value, -100, 100)
        
        return action_probs, state_value
    
    def eval_get_action_probs(self, state_tuple: tuple, device: torch.device) -> torch.Tensor:
        """Get action with NaN protection"""
        # Ensure state is valid
        # handle-state
        # state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        # state = np.clip(state, -10, 10)
        
        # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        self.eval()
        
        (historical_state, indicator_state) = state_tuple
        # state_tensor_tuple = (torch.FloatTensor(historical_state).unsqueeze(0).to(device), torch.FloatTensor(indicator_state).unsqueeze(0).to(device))
        state_tensor_tuples = [(torch.FloatTensor(historical_state), torch.FloatTensor(indicator_state))] # elem shape: (dim_compact_history, 2), (nIndicators,)
        
        with torch.no_grad():
            # action_probs, _ = self.forward(state_tensor)
            action_probs, _ = self(state_tensor_tuples, device)
            action_probs = action_probs.squeeze(0)
            
            # Final NaN check and fallback
            if torch.isnan(action_probs).any():
                # Fallback to uniform distribution
                action_probs = torch.ones(self.action_dim, device=device) / self.action_dim
                
        return action_probs

    
    def eval_sample_action(self, state_tuple: tuple, device: torch.device) -> Tuple[int, float, torch.Tensor]:

        action_probs = self.eval_get_action_probs(state_tuple, device)
            
        dist = Categorical(action_probs / self.temperature)
        action = dist.sample()              # differentiable exploration that provides stochastic continuity
        log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), action_probs
    
    
    def eval_get_value(self, state_tuple, device):
        self.eval()
        with torch.no_grad():
            (historical_state, indicator_state) = state_tuple
            state_tensor_tuples = [(torch.FloatTensor(historical_state), torch.FloatTensor(indicator_state))] # elem shape: (dim_compact_history, 2), (nIndicators,)
            _, value = self(state_tensor_tuples, device)      # values_t = critic_t(states_t)
            value = value.item()
        return value

# ========== STABLE PPO AGENT ==========

class RewardNormalizer:
    def __init__(self, warmup_steps=10000, update_rate=0.005):
        self.warmup_steps = warmup_steps
        self.alpha = update_rate
        self.mean = 0.0
        self.std = 1.0
        self.step_count = 0
    
    def __call__(self, rewards):
        batch_mean = rewards.mean()
        batch_std = rewards.std()
        
        # Update running statistics (always)
        self.mean = (1 - self.alpha) * self.mean + self.alpha * batch_mean
        self.std = (1 - self.alpha) * self.std + self.alpha * batch_std
        self.step_count += len(rewards)
        
        # Avoid division by zero
        batch_std = max(batch_std, 1e-8)
        running_std = max(self.std, 1e-8)
        
        # Compute both normalizations
        per_batch = (rewards - batch_mean) / batch_std
        running = (rewards - self.mean) / running_std
        
        # Smooth transition
        if self.step_count >= self.warmup_steps:
            return running
        else:
            # Linear fade from per-batch to running
            alpha = 1.0 - (self.step_count / self.warmup_steps)
            return alpha * per_batch + (1.0 - alpha) * running


class StablePPOAgent:
        
    def __init__(self, state_params: dict, history_process_params: dict, state_process_params: dict, action_dim: int, agent_params: dict):
        
        device = agent_params['device']
        
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
                
        self.actor_critic = PPO_Actor_Critic(state_params, history_process_params, state_process_params, action_dim).to(self.device)
        
        learning_rate = agent_params['learning_rate']
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = agent_params['gamma']
        self.gae_lambda = agent_params['gae_lambda']
        self.clip_epsilon = agent_params['clip_epsilon']
        self.critic_coef = agent_params['critic_coef']
        self.entropy_coef = agent_params['entropy_coef']
        self.max_grad_norm = agent_params['max_grad_norm']
        self.ppo_epochs = agent_params['ppo_epochs']
        self.mini_batch_size = agent_params['mini_batch_size']       
        
        self.trajectory = Trajectory()
        self.training_stats = []
        
        self.reward_normalizer = RewardNormalizer(warmup_steps=1000, update_rate=0.005)
    
    def remember(self, state, action, reward, done, log_prob, value):
        self.trajectory.add(state, action, reward, done, log_prob, value)
        
    def get_remember_len(self):
        return self.trajectory.__len__()
    
    def compute_returns_and_advantages(self, next_value: float):
        # action, log_prob = actor(state), value = critic(state), reward, done = env(action)
        # Discrepancy between actor and critic is represented.
        
        rewards = np.array(self.trajectory.rewards, dtype=np.float32)
        rewards =  self.reward_normalizer(rewards)
        
        values = np.array(self.trajectory.values, dtype=np.float32)
        dones = np.array(self.trajectory.dones, dtype=np.float32)
        
        # Clip rewards aggressively
        # rewards = np.clip(rewards, -5, 5)
        
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


        # Standard GAE Formula
        # δ_t = r_t + γ * V(s_{t+1}) * (1 - done_{t+1}) - V(s_t)
        # A_t = δ_t + γ * λ * (1 - done_{t+1}) * A_{t+1}
        # However, dones[t] = done_{t+1} in my implementation, 
        # where actions_t, values_t, rewards_t and dones_t all comes from state_t.  
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
                gae = delta
            else:
                delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            
            # Clip delta
            # delta = np.clip(delta, -10, 10)
            
            advantages[t] = gae
        
        # # Normalize advantages
        # if len(advantages) > 1 and np.std(advantages) > 1e-6:
        #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8) 
               
        # if len(advantages) > 1 and torch.std(advantages) > 1e-6:
        #     advantages = advantages / (torch.sqrt(torch.sum(advantages**2)) + 1e-8)

        advantages = advantages / (np.max(np.abs(advantages)) + 1e-8)
        
        return returns, advantages
    
    
    def learn(self, next_state_tuple, done):
        if len(self.trajectory) == 0:
            return {}
               
        # rewards_t, dones_t, log_prob_t, values_t come from (actor_t, critic_t, states_t) in self.trajectory
        
        # =========================================================================
        # SECTION 1: BOOTSTRAP VALUE COMPUTATION (GRADIENTS DISABLED)
        # =========================================================================
        if next_state_tuple is None or done:
            next_value = 0.0
        else:
            # The eposide finished with an action that is not SELLOUT.
            # Our practical policy is to choose SELLOUT action as an additional closing action,
            # while the policy network might choose a different action based on the next_state_tuple.
            # What should the agent learn from? the practical policy or the policy network?
            # Practical! because the agent should learn an optimal policy that fits the practice.
            
            assert False
            
            with torch.no_grad():
                (historical_state, indicator_state) = next_state_tuple
                historical_tensor = torch.FloatTensor(historical_state).to(self.device)
                indicator_tensor = torch.FloatTensor(indicator_state).to(self.device)
                historical_tensor.requires_grad_(False)
                indicator_tensor.requires_grad_(False)
                state_tensor_tuples = [(historical_tensor, indicator_tensor)]
                _, next_val = self.actor_critic(state_tensor_tuples, self.device)
                next_value = next_val.item()
        
        # =========================================================================
        # SECTION 2: RETURNS AND ADVANTAGES (NUMPY, NO TENSORS)
        # =========================================================================
        returns, advantages = self.compute_returns_and_advantages(next_value)
        
        # =========================================================================
        # SECTION 3: TRAJECTORY TO TENSOR CONVERSION (GRADIENTS DISABLED)
        # =========================================================================
        with torch.no_grad():
            state_tensor_tuples = []
            for historical, indicator in self.trajectory.states:
                historical_tensor = torch.FloatTensor(historical)
                indicator_tensor = torch.FloatTensor(indicator)
                historical_tensor.requires_grad_(False)
                indicator_tensor.requires_grad_(False)
                state_tensor_tuples.append((historical_tensor, indicator_tensor))
            
            # Primary tensors
            actions = torch.LongTensor(self.trajectory.actions).to(self.device)
            actions.requires_grad_(False)
            
            old_log_probs = torch.FloatTensor(self.trajectory.log_probs).to(self.device)
            old_log_probs.requires_grad_(False)
            
            returns_tensor = torch.FloatTensor(returns).to(self.device)
            returns_tensor.requires_grad_(False)
            
            advantages_tensor = torch.FloatTensor(advantages).to(self.device)
            advantages_tensor.requires_grad_(False)
        
        # =========================================================================
        # SECTION 4: PPO TRAINING UPDATES (GRADIENTS EXPLICITLY ENABLED)
        # =========================================================================
        dataset_size = len(state_tensor_tuples)
        losses = {
            'actor_loss': 0.0,
            'critic_loss': 0.0,
            'entropy': 0.0,
            'total_loss': 0.0
        }
        update_count = 0
        
        self.actor_critic.train()
        
        with torch.enable_grad():
            for epoch in range(self.ppo_epochs):
                indices = np.random.permutation(dataset_size)
                for start in range(0, dataset_size, self.mini_batch_size):
                    end = min(start + self.mini_batch_size, dataset_size)
                    batch_idx = indices[start:end]
                    
                    # =============================================================
                    # BATCH SLICING - Creates NEW tensors (views/slices)
                    # Each inherits requires_grad=False from parent
                    # =============================================================
                    
                    # Slicing creates new tensors - explicitly set requires_grad=False
                    batch_state_tensor_tuples = [state_tensor_tuples[i] for i in batch_idx]
                    # Each tuple contains tensors with requires_grad=False
                    
                    batch_actions = actions[batch_idx]  # NEW tensor
                    batch_actions.requires_grad_(False)  # Explicit
                    
                    batch_old_log_probs = old_log_probs[batch_idx]  # NEW tensor
                    batch_old_log_probs.requires_grad_(False)  # Explicit
                    
                    batch_advantages = advantages_tensor[batch_idx]  # NEW tensor
                    batch_advantages.requires_grad_(False)  # Explicit
                    
                    batch_returns = returns_tensor[batch_idx]  # NEW tensor
                    batch_returns.requires_grad_(False)  # Explicit
                    
                    # Network forward pass (gradients enabled)
                    action_probs, state_values = self.actor_critic(batch_state_tensor_tuples, self.device)
                    
                    dist = Categorical(action_probs)
                    log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                    
                    ratios = torch.exp(log_probs - batch_old_log_probs)
                    
                    # surr1 = ratios * batch_advantages
                    # surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                    # actor_loss = - torch.min(surr1, surr2).mean()                                        
                    # critic_loss = nn.SmoothL1Loss()(state_values.squeeze(), batch_returns)
                    # entropy_loss = -entropy.mean()
                    # total_loss = actor_loss + self.critic_coef * critic_loss  + self.entropy_coef * entropy_loss      # comeback
                    
                    surr1 = ratios * batch_returns
                    surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_returns
                    actor_loss = - torch.min(surr1, surr2).mean()                                        
                    # critic_loss = nn.SmoothL1Loss()(state_values.squeeze(), batch_returns)
                    entropy_loss = -entropy.mean()
                    total_loss = actor_loss + self.entropy_coef * entropy_loss      # comeback

                    if torch.isnan(total_loss): 
                        print(f"Warning: NaN loss at batch {start}")
                        continue
                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                    
                    self.optimizer.step()     # comeback
                    
                    losses['actor_loss'] += actor_loss.item()
                    # losses['critic_loss'] += critic_loss.item()
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
    
    
    def eval_sample_action(self, state_tuple):
        action, log_prob, _ = self.actor_critic.eval_sample_action(state_tuple, self.device)
        return action, log_prob
    
    def eval_get_log_prob(self, state_tuple, action):
        action_probs =  self.actor_critic.eval_get_action_probs(state_tuple, self.device)
        log_prob = torch.log(action_probs[action]).item()
        return log_prob
    
    def eval_get_value(self, state_tuple):
        value = self.actor_critic.eval_get_value(state_tuple, self.device)
        return value
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✅ Model saved to {path}")
        

    def load(self, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        self.actor_critic.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"✅ Model loaded from {path}")
        return checkpoint
    

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
         
    def __init__(self, market_history, portfolio_param,
        dim_compact_history=300, indicator_window=30, max_eposode_len=1_000):
        
        self.exchange = Binance.instantiate(connect=False)
        self.historian = Market_History(dim_compact_history=dim_compact_history, indicator_window=indicator_window)
        self.dim_compact_history = dim_compact_history
        self.indicator_window = indicator_window
        self.max_episode_len = max_eposode_len
                    
        # They are references and can be changed from outside
        self.raw_price, self.raw_volume, self.raw_mark, self.price_mean, self.volue_mean = market_history
        self.price_mean = self.raw_price.mean()
        self.price_std = self.raw_price.std()
        self.history_len = len(self.raw_price)
        self.market_history = market_history
        self.coming_price = None
        self.coming_volume = None

        self.init_balance = portfolio_param['init_balance']
        self.mock_portion = portfolio_param['mock_portion']
        self.tx_cost = portfolio_param['tx_cost']
        self.slippage = portfolio_param['slippage']
        self.unit_size = portfolio_param['unit_size']
        self.max_units = portfolio_param['max_units']
        self.length_penalty = portfolio_param['length_penalty']
        
        self.step_anchor = 0
        self.compact_history = None
        self.balance = init_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
        
        self.acc_fees_and_slippage = 0
        self.liquidation_revenue = 0
        self.liquidation_loss = 0
        self.inventory_costs = 0
        self.total_actions = 0
        self.high_balance = - init_balance * 10
        self.low_balance = init_balance * 10
        
        self.tx_history = deque(maxlen=max_episode_len)
        
        self.codes = ''
        
        # self.reset()
            
    def reset(self, episode_anchor):
        
        self.coming_price = None
        self.coming_volume = None

        self.balance = self.init_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
                
        self.acc_fees_and_slippage = 0
        self.liquidation_revenue = 0
        self.liquidation_loss = 0
        self.inventory_costs = 0
        self.total_actions = 0
        self.high_balance = - init_balance * 10
        self.low_balance = init_balance * 10
        
        self.tx_history = deque(maxlen=max_episode_len)

        # get it dynamically
        history_len = len(self.raw_price)
        assert len(self.raw_volume) == history_len
        assert len(self.raw_price) == history_len

        true_data_size = history_len - self.max_episode_len - self.historian.lookback_horizon
        assert true_data_size > 0
        # print(f"=========== data_size {true_data_size} ===============")
            
        self.step_anchor = episode_anchor      # step_anchor starts from episode_anchor and increases by one after taking action.
        arange = np.arange(self.step_anchor, self.step_anchor + self.max_episode_len)
        self.coming_price = self.raw_price[arange]
        self.coming_volume = self.raw_volume[arange]
        self.compact_history, _, _ = self.historian.compress_history(
            self.market_history, self.step_anchor, use_backside=True, faithful_backward=True
        )
                
        next_state_tuple, reward, done = self.take_action(action=None)
        
        self.codes = ''
        
        return next_state_tuple
    
    indicator_dim = 11 + 21       #----------------- for consistency in the number of indicators generated below.
        
    def _form_state(self, compact_history, tx_history):
        """Simplified state to avoid NaNs"""
        
        compact_price, compact_volume, raw_price, raw_volume, raw_mark, price_mean, volue_mean = compact_history
        current_price = raw_price[-1]
        
        def normalize(array):
            mean, std = array.mean(), array.std()
            normal = (array - mean) / (std + 1e-9)
            return normal, mean, std

        normal_price, local_price_mean, _ = normalize(compact_price)
        normal_volume, local_volume_mean, _ = normalize(compact_volume)
        
        # normal_price, local_price_mean, _ = normalize(raw_price[-self.dim_compact_history:])    #------------------------- temporary
        # normal_volume, local_volume_mean, _ = normalize(raw_volume[-self.dim_compact_history:])
        
        historical_state = np.stack([normal_price, normal_volume], axis=1)  # (dim_compact_history, 2)
        # assert historical_state.shape == (self.dim_compact_history, 2)

        indicator_state = []   
        indicator_state.append(get_level(raw_price, len(raw_price)-1, self.indicator_window, axis=0))
        indicator_state.append(get_change(raw_price, len(raw_price)-1, self.indicator_window, axis=0))
        indicator_state.append(get_level(raw_volume, len(raw_volume)-1, self.indicator_window, axis=0))
        indicator_state.append(get_change(raw_volume, len(raw_volume)-1, self.indicator_window, axis=0))
        indicator_state.append(get_volatility(raw_price, len(raw_price)-1, self.indicator_window, span=20))
        supertrend, trend = calculate_supertrend_close_only(raw_price[-self.indicator_window:], period=10, multiplier=3)
        indicator_state.append(trend[-1])
    
        # I hope this will compensate for the loss of meaning incurred by normalizing compact_price and compact_volume.
        indicator_state.append(local_price_mean / price_mean)
        indicator_state.append(local_volume_mean / volue_mean)
        
        cash_in_flow, profit, profit_rate, fees, inventory_price = self.estimate_unrealized_pnl(self.units_held, current_price) if self.units_held > 0 else (0, 0, 0, 0, 0)
        indicator_state.append(profit_rate)
        
        indicator_state.append(self.balance / self.init_balance)
        indicator_state.append(self.units_held/self.max_units)
        
        
        dict = calculate_best_indicators(raw_price[-self.indicator_window:], raw_volume[-self.indicator_window:])
        indicators = [v for v in dict.values()]
        indicator_state = indicator_state + indicators
        
        assert len(indicator_state) == TradingEnvironment.indicator_dim
        indicator_state = np.array(indicator_state)     # (nIndicators,)
        # indicator_state = stablize(indicator_state, -10, 10)

        pre_txs = list(tx_history)[::-1]
        tx_state = pre_txs + [0] * (self.max_episode_len - len(tx_history))
        tx_state = np.array(tx_state)
        
        indicator_state = np.concatenate([indicator_state, tx_state])
        
        # I believe learned bi-diractional LSTM of historical state can be the best indicator.
        # This is not the place to run LSTM, which should be a part of actor's actor_critic that gets actions from states.
        state_tuple = (historical_state, indicator_state)
        
        return state_tuple
    
    def estimate_unrealized_pnl(self, units_to_sell, sell_price):  # assuming Binance, which reduces fees and slippage from what you receive.
        # assert self.units_held > 0 and units > 0
        inventory_price = self.inventory_costs / (self.unit_size * self.units_held) if self.units_held > 0 else 0   # price
        sell_cost = self.unit_size * units_to_sell * inventory_price    # dollar
        principal =  self.unit_size * units_to_sell * sell_price        # dollar
        revenue_loss = principal * (self.tx_cost + self.slippage)       # dollar
        cash_in_flow = principal - revenue_loss                         # dollar
        sell_profit = cash_in_flow - sell_cost                          # dollar
        profit_rate = sell_profit / sell_cost                           # rate
        return cash_in_flow, sell_profit, profit_rate, revenue_loss, inventory_price    # d, d, d, p

    def take_action(self, action):
        
        reward, done, sold, tx = 0, False, False, None

        c_price, c_volume, r_price, r_volume, r_mark, price_mean, volue_mean = self.compact_history
        current_price = r_price[-1]
        
        incentive_for_length = 10
        length_penalty = self.length_penalty / self.max_episode_len
                
        if action in [TradingAction.HOLD.value,]:
            tx = 0
            self.codes += '.'
            reward = 0
 
        elif action in [TradingAction.BUY.value,]:

            units = min(self.max_units - self.units_held, TradingAction(action).units)  # units
            principal = self.unit_size * units * current_price  # dollar
            cash_out_flow = principal                           # dollar
 
            mock_balance = self.init_balance * self.mock_portion
            if self.balance - cash_out_flow > mock_balance:   # ask portfolio for a decision later
                
                # execute buy, assuming Binance, which reduces fees and slippage from what you receive.
                self.balance -= cash_out_flow   # self.balance is not affected by fees.
                units_loss = units * (self.slippage + self.tx_cost)     # btc
                self.units_held += (units - units_loss)     # units_loss will reduce sell revenue, so future self.balance, by units_loss * sell price
                self.inventory_costs += cash_out_flow       # dollar
                self.acc_fees_and_slippage += self.unit_size * units_loss * current_price    # units_loss adds to fees by units_loss * buy_price # dollar
                self.total_trades += 1

                tx = 1 # cash_out_flow / self.init_balance / (1 - self.mock_portion)     # rate
                self.codes += '-'
                reward -= length_penalty # incentive_for_length / self.max_episode_len # self.unit_size * incentive_for_length * current_price * self.tx_cost

            else:
                tx = 0
                self.codes += ';'
                reward -= length_penalty # incentive_for_length / self.max_episode_len # self.unit_size * incentive_for_length * current_price * self.tx_cost
            
        elif action in [TradingAction.SELLOUT.value]:

            units = min(self.units_held, self.units_held if action == TradingAction.SELLOUT.value else TradingAction(action).units)            

            if units > 0:
                cash_in_flow, profit, profit_rate, revenue_loss, inventory_price = self.estimate_unrealized_pnl(units, current_price)  # d, d, r, p  
 
                # execute sell
                self.balance += cash_in_flow            # self.balance is reduced by revenue_loss   # d
                self.units_held -= units                # units
                self.inventory_costs = self.unit_size * self.units_held * inventory_price   # dollar
                self.acc_fees_and_slippage += revenue_loss  # fees are increased by revenue_loss. dollar
                self.total_trades += 1

                # tx = - cash_in_flow / self.init_balance / (1 - self.mock_portion)     # NO, this is the last action and this tx has no effect.

                # Let the actor know the steps it has taken.
                tx = -1
                
                self.codes += ('↑' if profit > 0 else '↓')
                reward += 1 if profit > 0 else -1 # profit / self.init_balance / (1 - self.mock_portion) * 100    # rate  1 if profit > 0 else -1
                reward -= length_penalty
                sold = True
                
            else:
                tx = 0
                self.codes += ':'
                reward -= length_penalty # reward += self.unit_size * incentive_for_length * current_price * self.tx_cost
                            
        elif action is None:
            tx = None
            pass    # takes no action and just returns self._form_state()
        
        self.total_actions += (1 if action is not None else 0)
        
        if action == TradingAction.SELLOUT.value: done = True
        
        if tx is not None: self.tx_history.append(tx)
        
        self.step_anchor += 1
        arange = np.arange(self.step_anchor, self.step_anchor + self.max_episode_len)
        self.coming_price = self.raw_price[arange]
        self.coming_volume = self.raw_volume[arange]
        self.compact_history, _, _ = self.historian.compress_history(
            self.market_history, self.step_anchor, use_backside=True, faithful_backward=True
        )
        next_state_tuple = None if done else self._form_state(self.compact_history, self.tx_history)    # no slippage?
        
        
        if self.high_balance < self.balance: self.high_balance = self.balance
        if self.low_balance > self.balance: self.low_balance = self.balance
        
        if not done:
            # Update in every step. In training, an episode may be terminated, when self.units_held should be liquidated.
            # The updated values should be used only if the final action is not SELLOUT.
            cash_in_flow, profit, profit_rate, revenue_loss, inventory_price = self.estimate_unrealized_pnl(self.units_held, current_price)
            self.liquidation_revenue = cash_in_flow
            self.liquidation_loss = revenue_loss
        
        return next_state_tuple, reward, done
    
    def get_valid_episode_anchor(self, search_start):

        assert self.historian.lookback_horizon <= search_start        
        episode_anchor = search_start
        while episode_anchor < self.history_len - self.max_episode_len:
            # # episode_anchor is the inclusive end index into the market_history 
            # # and is used to indicate the right-most, current candle of the history, from which the current episode begins.
            # episode_anchor = random.randint(self.historian.lookback_horizon, self.history_len - self.max_episode_len)     # check for empty range error
            
            # data_complexity = 20
            # data_complexity = min(data_complexity, history_len - self.max_episode_len - self.historian.lookback_horizon)  # Choose rightward interval
            # episode_anchor = random.randint(history_len - self.max_episode_len - data_complexity, history_len - self.max_episode_len)     # comeback
            
            # # raw_prices, raw_volumes, marks, price_mean, volue_mean = self.market_history
            # # # episode_anchor = history_len - self.max_episode_len - data_complexity            
            # # arrays = [raw_prices[episode_anchor : episode_anchor + self.max_episode_len]]
            # # plot_arrays(arrays, labels=['price'], title="Price Volume", xlabel="Time", ylabel="Price", figsize=(18, 2), flip=False, save_path=None, lw=1)
            
            # # episode_anchor = history_len - self.max_episode_len
            # # arrays = [raw_prices[episode_anchor : episode_anchor + self.max_episode_len]]
            # # plot_arrays(arrays, labels=['price'], title="Price Volume", xlabel="Time", ylabel="Price", figsize=(18, 2), flip=False, save_path=None, lw=1)
            
            candlespace = range(episode_anchor - self.historian.lookback_horizon, episode_anchor + self.max_episode_len)
            if (self.raw_mark[candlespace]==0).all(): break     # no forged candles. See Binance.Get_Price_Data_By_Time
            print(f"Found forged candles. picking others..........................................")
            episode_anchor += (len(candlespace) + np.argmax(self.raw_mark[candlespace][::-1] != 0))
        
        return episode_anchor if episode_anchor < self.history_len - self.max_episode_len else -1

# ========== TRAINING FUNCTION ==========

from collections import deque
import math
          
def train_ppo(env, agent, epochs=300, print_interval=50, training=True, online_test = False, interval_sec=60, device='cuda'):
    print("\n" + "="*60)
    print(f"🚀 STARTING PPO {'TRAINING' if training else 'TEST'}")
    print("="*60)
            
    pm = PerformanceMeasure()
    pm_1 = PerformanceMeasure()
    reward_dist = DistributionTracker()
    
    for epoch in range(epochs):
        episode_anchor = env.historian.lookback_horizon + 1
   
        episodes = -1
        delta = env.max_episode_len
        null_strategies_to_come = 0
        while episode_anchor < env.history_len - env.max_episode_len - delta:
            episode_anchor = env.get_valid_episode_anchor(search_start=episode_anchor)
            if episode_anchor >= env.history_len - env.max_episode_len - delta: continue
            episodes += 1
            
            state_tuple = env.reset(episode_anchor)

            #------------------------------ find target_actions / target_rewards -----------------
            max_buy_cost = env.init_balance * (1 - env.mock_portion)
            full_actions, best_profit, first_holds, trimmed = optimal_trading_strategy(env.coming_price, r=(env.tx_cost + env.slippage) * 100, unit_size=env.unit_size, max_buy_cost=max_buy_cost)
            pm.net_profit = best_profit

            target_actions = []
            for a in full_actions:
                if a == 'B': target_actions.append(TradingAction.BUY)
                elif a == 'H': target_actions.append(TradingAction.HOLD)
                elif a == 'S': 
                    target_actions.append(TradingAction.SELLOUT)
                    break
            
            target_rewards = find_strategic_rewards_with_fees_and_penalty(
                env.coming_price[:len(target_actions)], 
                r=(env.tx_cost + env.slippage) * 100, 
                unit_size=env.unit_size, 
                actions=full_actions[:len(target_actions)], 
                moving_price_mean=env.price_mean, 
                moving_price_std=env.price_std, 
                net_profit=best_profit
            )
            for r in target_rewards: reward_dist.observe(r)
                
            target_rewards = [ (r - reward_dist.mean) / (reward_dist.var ** 0.5 + 1e-9) for r in target_rewards]
            
            assert len(target_actions) == len(target_rewards)

            # if null_strategies_to_come > 0:
            #     null_strategies_to_come -= 1
            #     pm.buys = 0; pm.holds = 1
            #     target_actions = [TradingAction.SELLOUT] * 1
            # else:         
            #     full_actions, best_profit, first_holds, trimmed = optimal_trading_strategy(env.coming_price, r=(env.tx_cost + env.slippage) * 100, p=0, unit_size=env.unit_size)
            #     if first_holds > 0:
            #         null_strategies_to_come = first_holds - 1
            #         pm.buys = 0; pm.holds = 1
            #         target_actions = [TradingAction.SELLOUT] * 1
            #     else:
            #         pm.buys = len([a for a in trimmed if a is 'B'])
            #         pm.holds = len([a for a in trimmed if a == 'H']) + 1    # + 1: SELLOUT is a kind of HOLD.
            #         target_actions = [TradingAction.BUY] * pm.buys + [TradingAction.HOLD] * (pm.holds-1) + [TradingAction.SELLOUT] * 1
            #         target_rewards = find_strategic_rewards_with_fees_and_penalty(env.coming_price, r=(env.tx_cost + env.slippage) * 100, p=0, buys=pm.buys, holds=pm.holds, unit_size=env.unit_size)
            # assert len(target_actions) == len(target_rewards)

            #------------------------------ Teach agent with target_actions / target_rewards -----------------
            done, episode_return, step_count = False, 0, 0
            while not done: # and step_count < len(target_actions): # SELLOUT is guaranteed to be the last actoin.

                action = target_actions[step_count].value
                log_prob = agent.eval_get_log_prob(state_tuple, action)
                value = agent.eval_get_value(state_tuple)
                
                next_state_tuple, _reward, done = env.take_action(action)
                reward = target_rewards[step_count]
                episode_return += reward
                                
                if training:
                    agent.remember(state_tuple, action, reward, done, log_prob, value)  # rewards_t, dones_t, log_prob_t, values_t come from (actor_t, critic_t, states_t)
                    # print(f"action {action}, reward {reward}, log_prob {log_prob}, value {value}")
                    
                state_tuple = next_state_tuple
                step_count += 1
                
            pm.episode_return = episode_return
                            
            if training and agent.get_remember_len() > 128:
                agent.learn(next_state_tuple, done)

            #---------------------------------------------------------------------------------------------------------
            
            PA = display_performance("EE", pm, env, epoch, episodes, episode_anchor, done)

            if training and PA > pm.PA_old:
                agent.save("stable_ppo_model_D06.pth")
                pm.PA_old = PA
                
            #------------------------------ Test agent ------------------------------------------------------------

            if online_test:
                state_tuple = env.reset(episode_anchor)

                done_1, episode_return_1, step_count_1 = False, 0, 0
                while not done_1 and step_count_1 < env.max_episode_len:
                    if step_count_1 < env.max_episode_len - 1:
                        action, log_prob = agent.eval_sample_action(state_tuple)
                    else:
                        action = TradingAction.SELLOUT.value

                    next_state_tuple, reward, done_1 = env.take_action(action)

                    step_count_1 += 1

                PA = display_performance("  ", pm_1, env, epoch, episodes, episode_anchor, done_1)
                                        
            #------------------------------- Prepare for the next episode --------------------------------------------------
           
            assert step_count == len(target_rewards)
            episode_anchor += len(target_rewards)
    
    return env, agent

class PerformanceMeasure():
    def __init__(self):
        smooth = 1000
        self.profitables = deque(maxlen=smooth)
        self.TxActRates = deque(maxlen=smooth)
        self.ProTraRates = deque(maxlen=smooth)
        self.acc_profit = 0
        self.acc_RF = 0

        PA_len = smooth
        self.profits = deque(maxlen=PA_len)
        self.total_actions = deque(maxlen=PA_len)
        self.PA_old = float('-inf')
        
        self.buys = 0
        self.holds = 0
        self.net_profit = 0
        self.episode_return = 0

def display_performance(flag, pm, env, epoch, episodes, episode_anchor, done):
    liquidated = not done # last_action != TradingAction.SELLOUT.value
    win_rate = (env.profitable_trades / max(env.total_trades, 1)) * 100
    profit = (env.balance-env.init_balance) if not liquidated else (env.balance - env.init_balance + env.liquidation_revenue)
    fees = env.acc_fees_and_slippage if not liquidated else env.acc_fees_and_slippage + env.liquidation_loss
    pm.acc_profit += profit
    pm.profitables.append(1 if profit > 0 else 0)
    p_frequency = sum(pm.profitables) / len(pm.profitables)
    
    TxActR = env.total_trades/env.total_actions
    pm.TxActRates.append(TxActR)
    
    pm.profits.append(profit)
    pm.total_actions.append(env.total_actions)
    
    PA = sum(pm.profits)/sum(pm.total_actions)
    RF = profit + fees
    pm.acc_RF += RF
    
    print(f"{flag} {epoch:3d} {episodes:3d} a {episode_anchor:3d} np {pm.net_profit: 4.3f} ({pm.buys:2d} {pm.holds:2d}) R {pm.episode_return:4.2f} | "
        f"B {env.balance:4.0f} {env.low_balance:4.0f} {env.high_balance:4.0f} | "
        f"T/A {env.total_trades:2d} {env.total_actions:2d} {sum(pm.TxActRates)/len(pm.TxActRates):5.2f} {sum(pm.total_actions)/len(pm.total_actions):5.2f} | "
        f"Pfr {p_frequency:5.3f} AccP {pm.acc_profit: 5.0f} P/A {PA: 7.3f} {(PA/interval_sec*3600*24): 5.3f} | "
        f"P {profit: 4.2f} {profit/env.total_actions: 3.2f} F {fees:3.2f} PF {RF: .2f} acc {pm.acc_RF: 4.2f} | "
        f"{flag} {env.codes} "
        #   f"{rewards} "
        f"{f'{profit:3.2f}' if profit > 0 else ''}"
        )
    return PA

class DistributionTracker():
    
    def __init__(self):
        self.mean = 0
        self.abs_var = 0
        self.var = 0
        self.n = 0
    def observe(self, value):
        new_n = self.n + 1
        self.mean = (self.mean * self.n + value) / new_n
        self.var = (self.var * self.n + (value - self.mean) ** 2) / new_n
        self.abs_var = (self.abs_var * self.n + value ** 2) / new_n
        self.n = self.n + 1
    

# ========== MAIN ==========
if __name__ == "__main__":
    print("="*60)
    print("STABLE PPO TRADING SYSTEM")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(325)
    
    #--------------------------------------- Get market history -----------------------------------
    bn = Binance.instantiate(connect=False)
    # Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
    datatype = "klines" # "aggTrades" #, "klines"
    symbol = "BTCUSDT"      # Any pair listed on Binance.
    start = datetime(2025, 2, 27) + timedelta(minutes=0)
    end = datetime(2026, 3, 1) # + timedelta(minutes=100)     # Any datetime
    # end = start + timedelta(minutes=4808) # gives true_data_size === 2 when max_episode_len = 60 and dim_compact_history = 500

    interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
    interval_sec = interval_to_seconds(interval)
    candles, nFills, successful, _, reason = bn.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
    # "Returning 10 columns: O H L C V, Quote V, nTrades, Buy base V, Buy quote V, Mark (0: original not forged)"
    open_idx, high_idx, low_idx, close_idx, volume_idx, mark_idx = 0, 1, 2, 3, 4, 9
    if successful:          # price is a numpy array.
        print("\n Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nForged = {}".format(candles.shape, nFills))
    else:
        print("\n Failed to Get_Price_Data_By_Time", reason)
        exit(0)
    
    #------------------------------------- Get train/test data ------------------------------------
    n_candles = len(candles)
    train_size = int(n_candles * 0.8)
    
    def split_array(raw_price, train_size):
        raw_price_train = raw_price[:train_size]
        raw_price_test = raw_price[train_size:]
        return raw_price_train, raw_price_test
    
    # The agent can make 100 dollars every 5 minutes if we use volume in place of price !!!!!!!!!!!!!!!!!!!!!!!!!!!   
    raw_price_train, raw_price_test = split_array(candles[:, close_idx], train_size)  #----------------- temporary
    raw_volume_train, raw_volume_test = split_array(candles[:, volume_idx], train_size)  #----------------- temporary
    
    arrays = [raw_price_train]
    # plot_arrays(arrays, labels=['price'], title="Price Volume Candle", xlabel="Time", ylabel="Price", figsize=(18, 2), flip=False, save_path=None, lw=1)
    
    
    # raw_price_train = raw_price_train / 10000  #----------------------------------------- temporary
    # raw_price_test = raw_price_test / 10000
    
    # plot_arrays([raw_price_train, raw_volume_train], labels=['price', 'volume'], title="price volume", xlabel="time", ylabel="", figsize=(18, 2), flip=False, save_path=None)
    
    raw_mark_train, raw_mark_test = split_array(candles[:, mark_idx], train_size)
    price_mean, volue_mean = raw_price_train.mean(), raw_volume_train.mean()
    
    market_history_train = (raw_price_train, raw_volume_train, raw_mark_train, price_mean, volue_mean)
    market_history_test = (raw_price_test, raw_volume_test, raw_mark_test, price_mean, volue_mean)

    print(f"\n📈 Configuration:")
    print(f"   Training samples: {train_size}")

    #------------------------------------ Setup env/agent -----------------------------------------
    mean_price = raw_price_train.mean()
    init_balance = 10_000
    max_units = 80
    unit_size = 0.005 # init_balance / max_units / 2 / mean_price    # so max_units will worth a 2nd of init_balance.
    action_dim = len(TradingAction)
    
    print(f"mean_price {mean_price}, unit_size {unit_size}")
    
    #------------------------ scale ----------------
    max_episode_len = 30
    dim_compact_history = 500
    dim_history_indicator = 50      # sensitive
    indicator_window = 10
    #-----------------------------------------------
    
    history_dim = len(['price', 'volume'])
    tx_cost = 0.001
    slippage = 0.0001
   
    state_params = { 
        'history_dim': history_dim, 
        'indicator_dim': TradingEnvironment.indicator_dim
    }
    
    history_process_params = {
        'infeature_dim': history_dim,
        'outfeature_dim': dim_history_indicator,
        'num_layers': 6,
        'hidden_dim': dim_history_indicator * 2,
        'max_episode_len': max_episode_len,
        'dropout': 0,
    }
       
    state_dim = dim_history_indicator + TradingEnvironment.indicator_dim + max_episode_len
    state_process_params = {
        'shared_hidden_dims': [state_dim * 2] * 6,
        'actor_hidden_dims': [state_dim * 2] * 4,
        'critic_hidden_dims': [state_dim * 2] * 4,
        'temperature' : 1.0     # sllightly toward uniform not one-hot distribution
    }
    
    print(state_process_params)
    
    agent_params = {
        'learning_rate': 1e-5,          # : 1e-5
        'gamma': 1,          # 0.99     # 16: 0.99
        'gae_lambda': 1,     # 0.95     # 16  0.95
        'clip_epsilon': 0.5,    # 0.2
        'critic_coef': 0.5,     # 0.5
        'entropy_coef': 0.20,   # 0.01
        'max_grad_norm': 0.2,   # 0.5
        'ppo_epochs': 10,
        'mini_batch_size': 64,
        'device': 'auto'
    }
    
    portfolio_param = {
        'init_balance': init_balance,
        'mock_portion': 0.5,
        'tx_cost': tx_cost,
        'slippage': slippage,
        'unit_size': unit_size,
        'max_units': max_units,
        'length_penalty': 0.5,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    the_agent = StablePPOAgent(state_params, history_process_params, state_process_params, action_dim, agent_params)

    the_agent.load("stable_ppo_model_D06.pth")
        
    #------------------------------------ test pre-train -----------------------------------------
    test_env = TradingEnvironment(market_history_test, portfolio_param, 
        dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    # test(test_env, the_agent, max_episode_len)
    # test_env, the_agent = train_ppo(test_env, the_agent, max_episode_len, episodes=1_00, print_interval=1, training=False, device=device) 
    
    #------------------------------------- Train agent --------------------------------------------
    train_env = TradingEnvironment(market_history_train, portfolio_param, 
        dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    train_env, the_agent = train_ppo(train_env, the_agent, epochs=500_000, print_interval=1, training=True, online_test=False, interval_sec=interval_sec, device=device)
    
    #------------------------------------- Test post-train ---------------------------------------------
    the_agent.load("stable_ppo_model_D06.pth")
    
    print("\n🧪 Testing post-train...")
    test_env = TradingEnvironment(market_history_test, portfolio_param, 
        dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    # test(test_env, the_agent, max_episode_len)
    test_env, the_agent = train_ppo(test_env, the_agent, max_episode_len, episodes=1_00, print_interval=1, training=False, device=device)    
        
    print("\n✅ Complete!")