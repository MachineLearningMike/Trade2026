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
from ppo_lstm_networks import BiLSTM, BiLSTMWithAttention, BiLSTMWithLastTimestep
from RL_Trade_POC_Util import plot_arrays

warnings.filterwarnings('ignore')

# ========== ACTION ENUM ==========
# class TradingAction(Enum):
#     HOLD = 0
#     BUY_1 = 1
#     BUY_2 = 2
#     BUY_3 = 3
#     SELL_1 = 4
#     SELL_2 = 5
#     SELL_3 = 6
#     SELL_ALL = 7
    
#     @property
#     def units(self) -> int:
#         sizes = {self.HOLD: 0, 
#                  self.BUY_1: 1, self.BUY_2: 2, self.BUY_3: 4, 
#                  self.SELL_1: 1, self.SELL_2: 2, self.SELL_3: 4, self.SELL_ALL: 1}  # any non-zero for self.SELL_ALL
#         return sizes[self]
    
#     @property
#     def name(self) -> str:
#         names = {self.HOLD: "HOLD", self.BUY_1: "BUY_1", self.BUY_2: "BUY_2", self.BUY_3: "BUY_3", 
#                  self.SELL_1: "SELL_1", self.SELL_2: "SELL_2", self.SELL_3: "SELL_3", self.SELL_ALL: "SELL_ALL", }
#         return names[self]


# ========== ACTION ENUM ==========
class TradingAction(Enum):
    HOLD = 0
    BUY_1 = 1
    # BUY_2 = 2
    # BUY_3 = 3
    # SELL_1 = 4
    # SELL_2 = 5
    # SELL_3 = 6
    SELL_ALL = 2
    
    @property
    def units(self) -> int:
        sizes = {
            self.HOLD: 0, 
            self.BUY_1: 1,  
            self.SELL_ALL: 1
        }  # any non-zero for self.SELL_ALL
        return sizes[self]
    
    @property
    def name(self) -> str:
        names = {
            self.HOLD: "HOLD", 
            self.BUY_1: "BUY_1",  
            self.SELL_ALL: "SELL_ALL", 
        }
        return names[self]

# ========== STABLE NEURAL NETWORK ==========
class StablePPONetwork(nn.Module):
    """Numerically stable PPO Network"""
    
    # def __init__(self, input_dim: int, hidden_dims: List[int] = [128, 128], action_dim: int = 5):
    #     super(StablePPONetwork, self).__init__()
        
    # def __init__(self, hisorical_shape: tuple, indicator_shape: tuple, num_layers: int = 2, hidden_dims: List[int] = [128, 128], action_dim: int = 5):

    def __init__(self, state_params: dict, history_process_params: dict, state_process_params: dict, action_dim: int):
        super(StablePPONetwork, self).__init__()
        
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
        
        self.action_dim = action_dim
        
        # self.history_process = BiLSTM(
        self.history_process = BiLSTMWithLastTimestep(
        # self.history_process = BiLSTMWithAttention(  
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
                    nn.Dropout(0.1)
                ])
                prev_dim = hidden_dim            
            mlp, output_dim = nn.Sequential(*layers), hidden_dims[-1]
            return mlp, output_dim
        
        self.shared, shared_dim = create_mlp(state_dim, sp_shared_hidden_dims)

        self.actor, output_dim = create_mlp(shared_dim, sp_actor_hidden_dims)
        self.actor_logits = nn.Linear(output_dim, action_dim)

        self.critic, output_dim = create_mlp(shared_dim, sp_critic_hidden_dims)
        self.critic_head = nn.Linear(output_dim, 1)
        
        self._init_weights()
    
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
        
        logits = self.actor_logits(self.actor(features))        
        logits = torch.clamp(logits, -20, 20)  # Prevent exp explosion
        action_probs = torch.softmax(logits, dim=-1)
        
        state_value = self.critic_head(self.critic(features))
        state_value = torch.clamp(state_value, -100, 100)
        
        return action_probs, state_value
    
    def get_action(self, state_tuple: tuple, device: torch.device) -> Tuple[int, float, torch.Tensor]:
        """Get action with NaN protection"""
        # Ensure state is valid
        # handle-state
        # state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)
        # state = np.clip(state, -10, 10)
        
        # state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
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
            
            dist = Categorical(action_probs)
            action = dist.sample()              # If it isn't greedy, it explores always with shifting probabilty(?)
            log_prob = dist.log_prob(action)
            
        return action.item(), log_prob.item(), action_probs


# ========== STABLE PPO AGENT ==========
class StablePPOAgent:
        
    def __init__(self, state_params: dict, history_process_params: dict, state_process_params: dict, action_dim: int, agent_params: dict):
        
        device = agent_params['device']
        
        if device == 'auto':
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
                
        self.network = StablePPONetwork(state_params, history_process_params, state_process_params, action_dim).to(self.device)
        
        learning_rate = agent_params['learning_rate']
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
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
    
    def remember(self, state, action, reward, done, log_prob, value):
        self.trajectory.add(state, action, reward, done, log_prob, value)
    
    def compute_returns_and_advantages(self, next_value: float):
        # action, log_prob = actor(state), value = critic(state), reward, done = env(action)
        # Discrepancy between actor and critic is represented.
        
        rewards = np.array(self.trajectory.rewards, dtype=np.float32)
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
            # delta = np.clip(delta, -10, 10)
            
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        # Normalize advantages
        if len(advantages) > 1 and np.std(advantages) > 1e-6:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def learn(self, next_state_tuple, done):
        if len(self.trajectory) == 0:
            return {}
        
        # done == latest SELL_ALL and sold. next_state_tuple if done
        
        # Get next value
        if next_state_tuple is None or done:
            next_value = 0.0
        else:
            # Ensure next_state is valid
            # next_state = np.nan_to_num(next_state, nan=0.0, posinf=1.0, neginf=-1.0)
            # next_state = np.clip(next_state, -10, 10)   #----------------------------
            
            with torch.no_grad():
                (historical_state, indicator_state) = next_state_tuple
                state_tensor_tuples = [(torch.FloatTensor(historical_state), torch.FloatTensor(indicator_state))] # elem shape: (dim_compact_history, 2), (nIndicators,)                
                _, next_val = self.network(state_tensor_tuples, self.device)
                next_value = next_val.item()
                # next_value = np.clip(next_value, -10, 10)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns_and_advantages(next_value)
        # advantages = advantages.detach()  # this is a numpy object and has no string to policy parameters.
        
        # Convert to tensors with NaN protection
        # handle-state: state is now a tuple of np arrays. Remove this block.
        # raw_states = np.array([np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0) for state in self.trajectory.states])
        # states = np.clip(states, -10, 10)
        


        # historical_state = np.stack([s_tuple[0] for s_tuple in state_tuples], axis=0)
        # historical_state = torch.FloatTensor(historical_state).to(self.device)
        # indicator_state = np.stack([s_tuple[1] for s_tuple in state_tuples], axis=0)
        # indicator_state = torch.FloatTensor(indicator_state).to(self.device)
        
        # (historical_state, indicator_state) = state_tuple
        # state_tensor_tuple = (torch.FloatTensor(historical_state).to(self.device), torch.FloatTensor(indicator_state).to(self.device))

        # Postpone device assignment for state to the last moment to save GPU memory.
        state_tensor_tuples = [(torch.FloatTensor(historical), torch.FloatTensor(indicator)) \
            for (historical, indicator) in self.trajectory.states]  # elem shape: (dim_compact_history, 2), (nIndicators,)
        actions = torch.LongTensor(self.trajectory.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.trajectory.log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # PPO updates
        dataset_size = len(state_tensor_tuples)
        losses = {'actor_loss': 0, 'critic_loss': 0, 'entropy': 0, 'total_loss': 0}
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            indices = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, self.mini_batch_size):
                end = min(start + self.mini_batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                # batch_raw_states = state_tensor_tuples[batch_idx]        #
                batch_state_tensor_tuples = [state_tensor_tuples[i] for i in batch_idx]                
                batch_actions = actions[batch_idx]
                batch_old_log_probs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                
                # Get current policy
                action_probs, state_values = self.network(batch_state_tensor_tuples, self.device)
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
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
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
    
    def get_action(self, state_tuple):
        action, log_prob, _ = self.network.get_action(state_tuple, self.device)
        return action, log_prob
    
    def save(self, path):
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f"✅ Model saved to {path}")
        

    def load(self, path, device='cuda'):
        checkpoint = torch.load(path, map_location=device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
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
        self.max_eposode_len = max_eposode_len
                    
        # They are references and can be changed from outside
        self.raw_price, self.raw_volume, self.raw_mark, self.price_mean, self.volue_mean = market_history
        self.market_history = market_history

        self.init_balance = portfolio_param['init_balance']
        self.tx_cost = portfolio_param['tx_cost']
        self.slippage = portfolio_param['slippage']
        self.unit_size = portfolio_param['unit_size']
        self.max_units = portfolio_param['max_units']
        
        self.step_anchor = 0
        self.balance = init_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
        
        self.acc_cost = 0
        self.total_actions = 0
        self.high_balance = - init_balance * 10
        self.low_balance = init_balance * 10
        
        self.tx_history = deque(maxlen=max_episode_len)
        
        self.codes = ''
        
        self.reset()
            
    def reset(self):

        self.balance = self.init_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
                
        self.acc_cost = 0
        self.total_actions = 0
        self.high_balance = - init_balance * 10
        self.low_balance = init_balance * 10
        
        self.tx_history = deque(maxlen=max_episode_len)

        # get it dynamically
        history_len = len(self.raw_price)
        assert len(self.raw_volume) == history_len
        assert len(self.raw_price) == history_len

        while True:
            # episode_anchor is the inclusive end index into the market_history 
            # and is used to indicate the right-most, current candle of the history, from which the current episode begins.
            assert self.historian.len_to_lookback <= history_len - self.max_eposode_len, f"{self.historian.len_to_lookback} > {history_len - self.max_eposode_len}"
            episode_anchor = random.randint(self.historian.len_to_lookback, history_len - self.max_eposode_len)     # check for empty range error
            candlespace = range(episode_anchor - self.historian.len_to_lookback, episode_anchor + self.max_eposode_len)
            if (self.raw_mark[candlespace]==0).all(): break     # no forged candles. See Binance.Get_Price_Data_By_Time
            print(f"Found forged candles. picking others..........................................")
            
        self.step_anchor = episode_anchor # - 1      # step_anchor starts from episode_anchor - 1 and increases by one when taking action.
        
        next_state_tuple, reward, done = self.take_action(action=None)
        
        self.codes = ''
        
        return next_state_tuple
    
    indicator_dim = 10       #----------------- for consistency in the number of indicators generated below.
        
    def _form_state(self, c_history, tx_history):
        """Simplified state to avoid NaNs"""
        
        compact_price, compact_volume, raw_price, raw_volume, raw_mark, price_mean, volue_mean = c_history
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
        assert historical_state.shape == (self.dim_compact_history, 2)

        indicator_state = []   
        indicator_state.append(get_level(raw_price, len(raw_price)-1, self.indicator_window, axis=0))
        indicator_state.append(get_change(raw_price, len(raw_price)-1, self.indicator_window, axis=0))
        indicator_state.append(get_level(raw_volume, len(raw_volume)-1, self.indicator_window, axis=0))
        indicator_state.append(get_change(raw_volume, len(raw_volume)-1, self.indicator_window, axis=0))
        indicator_state.append(get_volatility(raw_price, len(raw_price)-1, self.indicator_window, span=20))
        
        # I hope this will compensate for the loss of meaning incurred by normalizing compact_price and compact_volume.
        indicator_state.append(local_price_mean / price_mean)
        indicator_state.append(local_volume_mean / volue_mean)
        
        cash_in_flow, profit, profit_rate = self.get_expected_reward(self.units_held, current_price) if self.units_held > 0 else (0, 0, 0)
        indicator_state.append(profit_rate)
        
        indicator_state.append(self.balance / self.init_balance)
        indicator_state.append(self.units_held/self.max_units)
        
        assert len(indicator_state) == TradingEnvironment.indicator_dim
        indicator_state = np.array(indicator_state)     # (nIndicators,)
        # indicator_state = stablize(indicator_state, -10, 10)

        pre_txs = list(tx_history)[::-1]
        tx_state = pre_txs + [0] * (max_episode_len - len(tx_history))
        tx_state = np.array(tx_state)
        
        indicator_state = np.concat([indicator_state, tx_state])
        
        # I believe learned bi-diractional LSTM of historical state can be the best indicator.
        # This is not the place to run LSTM, which should be a part of actor's network that gets actions from states.
        state_tuple = (historical_state, indicator_state)
        
        return state_tuple
    
    def get_expected_reward(self, units, current_price):
        # assert self.units_held > 0 and units > 0
        avg_price = self.acc_cost / (self.unit_size * self.units_held)     # as we want to recover the acc_cost with units_held.             
        principal =  self.unit_size * units * current_price
        fees = principal * self.tx_cost
        cash_in_flow = principal - fees
        cost = self.unit_size * units * avg_price
        profit = cash_in_flow - cost
        profit_rate = profit / cost
        return cash_in_flow, profit, profit_rate

    def take_action(self, action):
        reward = 0
        done = False
        sold = False

        c_history, _, _ = self.historian.compress_history(
            self.market_history, self.step_anchor, use_backside=True, faithful_backward=True
        )
        c_price, c_volume, r_price, r_volume, r_mark, price_mean, volue_mean = c_history
        current_price = r_price[-1]
        tx = None
                
        if action == TradingAction.HOLD.value:
            self.codes += '.'
            tx = None
            reward -= self.unit_size * 1 * current_price * self.tx_cost / self.price_mean
        # elif action in [TradingAction.BUY_1.value, TradingAction.BUY_2.value, TradingAction.BUY_3.value]:
        elif action in [TradingAction.BUY_1.value,]:
            units = TradingAction(action).units
            units = min(self.max_units - self.units_held, units)
            principal = self.unit_size * units * current_price
            cash_out_flow = principal
            if self.balance >= cash_out_flow and self.balance - cash_out_flow > self.init_balance * 0.7:   # ask portfolio for a decision later
                
                tx = 1
                # execute buy
                self.balance -= cash_out_flow
                self.acc_cost += cash_out_flow
                self.units_held += units * (1 - self.slippage) * (1 - self.tx_cost)
                self.total_trades += 1
                
                reward -= self.unit_size * units * current_price * self.tx_cost / self.price_mean
                self.codes += '-'
                                
            else:
                tx = 0
                # reward -= self.unit_size * units * current_price * self.tx_cost / self.price_mean
                self.codes += ';'
            
        # elif action in [TradingAction.SELL_1.value, TradingAction.SELL_2.value, TradingAction.SELL_3.value, TradingAction.SELL_ALL.value]:
        elif action in [TradingAction.SELL_ALL.value]:
            units = TradingAction(action).units if action != TradingAction.SELL_ALL.value else self.units_held
            # print(f"units {units} action {action}")
            units = min(self.units_held, units)            
            if units > 0:
                tx = 0
                cash_in_flow, profit, profit_rate = self.get_expected_reward(units, current_price * (1 - self.slippage))
                
                # execute sell
                self.balance += cash_in_flow 
                self.acc_cost -= cash_in_flow
                self.units_held -= units
                self.total_trades += 1
                self.codes += ('↑' if profit > 0 else '↓')

                # reward += profit / self.price_mean * (2 if profit >= 0 and action == TradingAction.SELL_ALL else 1 if profit > 0 else 3)         
                reward = profit #  1 if profit > 0 else -1      #----------------------------- 
                sold = True
            else:
                tx = 0
                self.codes += ':'
                reward -= self.init_balance / self.price_mean * 0.1
                # reward = -1
                            
        elif action is None:
            tx = None
            pass    # takes no action and just returns self._form_state()
        
        self.total_actions += (1 if action is not None else 0)

        # reward -= (self.init_balance / self.price_mean * 0.05 if action is not None else 0)
        
        if (action == TradingAction.SELL_ALL.value and sold is True): done = True
        
        self.step_anchor += 1
        
        if tx is not None: self.tx_history.append(tx)
        next_state_tuple = None if done else self._form_state(c_history, self.tx_history)    # no slippage?
        
        if self.high_balance < self.balance: self.high_balance = self.balance
        if self.low_balance > self.balance: self.low_balance = self.balance
        
        return next_state_tuple, reward, done


# ========== TRAINING FUNCTION ==========

from collections import deque
import math

def train_ppo(env, agent, max_episode_len, episodes=300, print_interval=50, training=True, device='cuda'):
    print("\n" + "="*60)
    print(f"🚀 STARTING PPO {'TRAINING' if training else 'TEST'}")
    print("="*60)
        
    profitabilities = deque(maxlen=int(episodes/10))
    TxActRates = deque(maxlen=100)
    ProTraRates = deque(maxlen=100)
    
    acc_profit = 0
    
    PA_len = 100
    profits = deque(maxlen=PA_len)
    total_actions = deque(maxlen=PA_len)
    PA_old = -100000
    
    for episode in range(episodes):
        
        state_tuple = env.reset()

        done = False
        episode_return = 0
        step_count = 0
        
        learned = False
        while not done and step_count < max_episode_len:

            if training:
                action, log_prob = agent.get_action(state_tuple)      # action, log_prob = policy(state) = actor(state)
 
            else:
                if step_count < max_episode_len - 1:
                    action, log_prob = agent.get_action(state_tuple)      # action, log_prob = policy(state) = actor(state)
                else:
                    action, log_prob = TradingAction.SELL_ALL.value, math.log(0.1)    #------------------ 0.1 for a forced action

            with torch.no_grad():
                (historical_state, indicator_state) = state_tuple
                state_tensor_tuples = [(torch.FloatTensor(historical_state), torch.FloatTensor(indicator_state))] # elem shape: (dim_compact_history, 2), (nIndicators,)
                _, value = agent.network(state_tensor_tuples, device)      # value = state_value(state) = critic(state)
                value = value.item()
            
            # done == SELL_ALL and sold. next_state_tuple if done
            next_state_tuple, reward, done = env.take_action(action)     # next_state, reward, done = env(action)
            
            # action, log_prob = actor(state), value = critic(state), reward, done = env(action)
            # Discrepancy between actor and critic is represented.
            if training:
                agent.remember(state_tuple, action, reward, done, log_prob, value)
            episode_return += reward
            
            # if training and done:
            #     losses = agent.learn(next_state_tuple, done)      # next_state is None if done.
            #     learned = True
                
            state_tuple = next_state_tuple
            step_count += 1
        
        # if training and not learned: agent.learn(next_state_tuple, done=False)       #
        
        if training: agent.learn(next_state_tuple, done)
                
        if (episode + 1) % print_interval == 0:
            win_rate = (env.profitable_trades / max(env.total_trades, 1)) * 100
            # A training episode may end with non-sell giving big nagative profit, which doesn't happen in inference, 
            # resulting in unfair profit estimation. This why ...
            profit = (env.balance-env.init_balance) if action == TradingAction.SELL_ALL.value else 0
            acc_profit += profit 
            profitable = profit > 0
            profitabilities.append(1 if profitable else 0)
            p_frequency = sum(profitabilities) / len(profitabilities)
            
            TxActR = env.total_trades/env.total_actions
            TxActRates.append(TxActR)
            
            profits.append(profit)
            total_actions.append(env.total_actions)
            
            PA = sum(profits)/sum(total_actions)
            
            print(f"Epi: {episode+1 :5d} | "
                  f"R: {episode_return:5.4f} | "
                  f"B: {env.balance:6.1f} {env.low_balance:6.1f} {env.high_balance:6.1f} | "
                  f"Ts/As: {env.total_trades:3d} {env.total_actions:3d} {sum(TxActRates)/len(TxActRates):5.2f} | "
                  f"P_freq: {p_frequency:5.3f} | "
                  f"Acc_P: {acc_profit:6.0f} | "
                  f"P/epi: {acc_profit/(episode+1):4.2f} | "
                  f"P/A : {sum(profits)/sum(total_actions) : 5.4f} | "
                  f"{env.codes} "
                  f"{'=====' if profitable else ''}"
                  )
            
        if training and PA > PA_old:
            agent.save("stable_ppo_model.pth")
            PA_old = PA
    
    return env, agent


# ========== MAIN ==========
if __name__ == "__main__":
    print("="*60)
    print("STABLE PPO TRADING SYSTEM")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(321)
    
    #--------------------------------------- Get market history -----------------------------------
    bn = Binance.instantiate(connect=False)
    # Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
    datatype = "klines" # "aggTrades" #, "klines"
    symbol = "BTCUSDT"      # Any pair listed on Binance.
    start = datetime(2025, 1, 1)    # Any datetime. Note Binance started operation in July 2017.
    end = datetime(2026, 4, 10) # + timedelta(minutes=100)     # Any datetime
    interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
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
    unit_size = 0.001 # 1000 / mean_price
    init_balance = 10_000        #
    max_units = 100
    action_dim = len(TradingAction)
    
    #------------------------ scale ----------------
    max_episode_len = 30
    dim_compact_history = 500
    dim_history_indicator = 30      # Sensitive to training speed
    indicator_window = 15
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
        'num_layers': 3,
        'hidden_dim': dim_history_indicator * 2,
        'max_episode_len': max_episode_len,
        'dropout': 0,
    }
    
    state_dim = dim_history_indicator + TradingEnvironment.indicator_dim + max_episode_len
    state_process_params = {
        'shared_hidden_dims': [state_dim * 4] * 3,
        'actor_hidden_dims': [state_dim * 4] * 3,
        'critic_hidden_dims': [state_dim * 4] * 3,
    }
    
    agent_params = {
        'learning_rate': 1e-4,
        'gamma': 0.99,          # 0.99
        'gae_lambda': 0.95,     # 0.95
        'clip_epsilon': 0.2,    # 0.2
        'critic_coef': 0.5,     # DeepSeek
        'entropy_coef': 0.01,   # DeepSeek
        'max_grad_norm': 0.5,
        'ppo_epochs': 5,
        'mini_batch_size': 32,
        'device': 'auto'
    }
    
    portfolio_param = {
        'init_balance': init_balance,
        'tx_cost': tx_cost,
        'slippage': slippage,
        'unit_size': unit_size,
        'max_units': max_units,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    the_agent = StablePPOAgent(state_params, history_process_params, state_process_params, action_dim, agent_params)
        
    #------------------------------------ test pre-train -----------------------------------------
    test_env = TradingEnvironment(market_history_test, portfolio_param, 
        dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    # test(test_env, the_agent, max_episode_len)
    # test_env, the_agent = train_ppo(test_env, the_agent, max_episode_len, episodes=1_00, print_interval=1, training=False, device=device) 
    
    #------------------------------------- Train agent --------------------------------------------
    train_env = TradingEnvironment(market_history_train, portfolio_param, 
        dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    train_env, the_agent = train_ppo(train_env, the_agent, max_episode_len, episodes=4_000, print_interval=1, training=True, device=device)
    
    #------------------------------------- Test post-train ---------------------------------------------
    the_agent.load("stable_ppo_model.pth")
    
    print("\n🧪 Testing post-train...")
    test_env = TradingEnvironment(market_history_test, portfolio_param, 
        dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    # test(test_env, the_agent, max_episode_len)
    test_env, the_agent = train_ppo(test_env, the_agent, max_episode_len, episodes=1_00, print_interval=1, training=False, device=device)    
        
    print("\n✅ Complete!")