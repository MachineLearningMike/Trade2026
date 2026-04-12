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
    def __init__(self, market_history, initial_balance=10000, transaction_cost_rate=0.003,
                 unit_size=100, max_units=5, dim_compact_history=300, indicator_window=30, max_eposode_len=1_000):
        
        self.exchange = Binance.instantiate(connect=False)
        self.historian = Market_History(dim_compact_history=dim_compact_history, indicator_window=indicator_window)
        self.indicator_window = indicator_window
        self.max_eposode_len = max_eposode_len
                    
        # They are references and can be changed from outside
        self.normal_price, self.normal_volume, self.raw_price, self.raw_volume, self.raw_mark, self.price_mean, self.volue_mean = market_history
        self.market_history = market_history

        self.initial_balance = initial_balance
        self.transaction_cost_rate = transaction_cost_rate
        self.unit_size = unit_size
        self.max_units = max_units
        
        self.step_anchor = 0
        self.balance = initial_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
        
        self.acc_cost = 0
        self.total_actions_taken = 0
        self.high_balance = - initial_balance * 10
        self.low_balance = initial_balance * 10
        
        self.reset()
            
    def reset(self):

        self.balance = self.initial_balance
        self.units_held = 0
        self.entry_prices = []
        self.total_trades = 0
        self.profitable_trades = 0
                
        self.acc_cost = 0
        self.total_actions_taken = 0
        self.high_balance = - initial_balance * 10
        self.low_balance = initial_balance * 10

        # get it dynamically
        history_len = len(self.normal_price)
        assert len(self.normal_volume) == history_len
        assert len(self.raw_price) == history_len

        while True:
            # episode_anchor is the inclusive end index into the market_history 
            # and is used to indicate the right-most, current candle of the history, from which the current episode begins.
            episode_anchor = random.randint(self.historian.len_to_lookback, history_len - self.max_eposode_len)
            candlespace = range(episode_anchor - self.historian.len_to_lookback, episode_anchor + self.max_eposode_len)
            if (self.raw_mark[candlespace]==0).all(): break     # no forged candles. See Binance.Get_Price_Data_By_Time
            print(f"Found forged candles. picking another..........................................")
            
        self.step_anchor = episode_anchor - 1      # step_anchor starts from episode_anchor - 1 and increases by one when taking action.
        
        next_state, reward, done = self.take_action(action=None)
        
        return next_state
    
    def _get_state(self, c_history):
        """Simplified state to avoid NaNs"""
        
        def stablize(arr, a, b):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
            arr = np.clip(arr, a, b)
            return arr
        
        norm_price, norm_volume, raw_price, raw_volume, raw_mark, price_mean, volue_mean = c_history
        current_price = raw_price[-1]

        indicator_state = []   
        indicator_state.append(get_level(raw_price, len(raw_price)-1, self.indicator_window, axis=0))
        indicator_state.append(get_change(raw_price, len(raw_price)-1, self.indicator_window, axis=0))
        indicator_state.append(get_level(raw_volume, len(raw_volume)-1, self.indicator_window, axis=0))
        indicator_state.append(get_change(raw_volume, len(raw_volume)-1, self.indicator_window, axis=0))
        indicator_state.append(get_volatility(raw_price, len(raw_price)-1, self.indicator_window, span=20))
        indicator_state = np.array(indicator_state)
        indicator_state = stablize(indicator_state, -10, 10)

        portfolio_state = []
        cash_in_flow, profit, profit_rate = self.get_expected_reward(self.units_held, current_price) if self.units_held > 0 else (0, 0)
        portfolio_state.append(profit_rate)  # expected profit on selling all. hope the net to learn to choose sell actions on positive profit_rate.
        portfolio_state = np.array(portfolio_state)
        portfolio_state = stablize(portfolio_state, -10, 10)
                
        historical_state = np.stack([norm_price, norm_volume], axis=0)  #!!!!!!!!!! It may adversarilly generate vast space of states !!!!!!!!!!!!!!!
        
        state = None
        
        
        return state
    
    def get_expected_reward(self, units, current_price):
        assert self.units_held > 0 and units > 0
        avg_price = self.acc_cost / self.units_held     # as we want to recover the acc_cost with units_held.             
        principal =  self.unit_size * units * current_price
        fees = principal * self.transaction_cost_rate
        cash_in_flow = principal - fees               
        profit = cash_in_flow - self.unit_size * units * avg_price
        profit_rate = profit / (self.unit_size * units * avg_price)
        return cash_in_flow, profit, profit_rate

    def take_action(self, action):
        reward = 0
        done = False
        sold = False
        
        self.step_anchor += 1

        # c_price / c_volume : compressed price / volume of length self.dim_compact_history.
        # r_price : past raw price
        # current_price : current price
        # ALL looking back from the inclusive current time point self.step_anchor
        c_history, _, _ = self.historian.compress_history(
            self.market_history, self.step_anchor, use_backside=True, faithful_backward=True
        )
        c_price, c_volume, r_price, r_volume, r_mark, price_mean, volue_mean = c_history
        current_price = r_price[-1]
        
        if action == TradingAction.HOLD.value:
            pass
        elif action in [TradingAction.BUY_1.value, TradingAction.BUY_2.value, TradingAction.BUY_3.value]:
            units = TradingAction(action).units
            principal = self.unit_size * units * current_price
            fees = principal * self.transaction_cost_rate
            cash_out_flow = principal + fees
            if self.balance >= cash_out_flow and self.balance - cash_out_flow > self.initial_balance * 0.8:   # ask portfolio for a decision later
                
                # execute buy
                self.balance -= cash_out_flow
                self.acc_cost += cash_out_flow
                self.units_held += units
                self.total_trades += 1
                
                reward -= fees
            
        elif action in [TradingAction.SELL_1.value, TradingAction.SELL_2.value, TradingAction.SELL_3.value, TradingAction.SELL_ALL.value]:
            units = TradingAction(action).units if action != TradingAction.SELL_ALL.value else self.units_held
            # print(f"units {units} action {action}")
            if self.units_held >= units and units > 0:
                cash_in_flow, profit, profit_rate = self.get_expected_reward(units, current_price)
                
                # execute sell
                self.balance += cash_in_flow
                self.acc_cost -= cash_in_flow
                self.units_held -= units
                self.total_trades += 1

                reward += profit / self.unit_size
                sold = True
            
        elif action is None:
            pass    # takes no action and just returns self._get_state()

        # Holding incentive
        if self.units_held > 0:
            reward += 0.01
 
        reward -= self.initial_balance/1_000        #--------------- penalize all actions to save actions and time.
        self.total_actions_taken += 1
        
        if (action == TradingAction.SELL_ALL.value and sold is True):
            done = True
        
        next_state = None if done else self._get_state(c_history)
        
        if self.high_balance < self.balance: self.high_balance = self.balance
        if self.low_balance > self.balance: self.low_balance = self.balance
        
        return next_state, reward, done


# ========== TRAINING FUNCTION ==========

from collections import deque

def train_ppo(env, agent, max_episode_len, episodes=300, print_interval=50):
    print("\n" + "="*60)
    print("🚀 STARTING PPO TRAINING")
    print("="*60)
    
    profitabilities = deque(maxlen=int(episodes/10))
    acc_profit = 0
    
    for episode in range(episodes):

        state = env.reset()

        done = False
        episode_reward = 0
        step_count = 0
        
        learned = False
        while not done and step_count < max_episode_len:
        # while step_count < max_episode_len:
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
                learned = True
        
        if not learned:
            agent.learn(next_state, done=False)
        
        if (episode + 1) % print_interval == 0:
            win_rate = (env.profitable_trades / max(env.total_trades, 1)) * 100
            profit = env.balance-env.initial_balance
            acc_profit += profit 
            profitable = profit > 0
            profitabilities.append(1 if profitable else 0)
            p_frequency = sum(profitabilities) / len(profitabilities)
            print(f"E: {episode+1 :5d} | "
                  f"Rwd: {episode_reward:5.0f} | "
                  f"Blnc: {env.balance:5.0f} {env.low_balance:5.0f} {env.high_balance:5.0f} | "
                  f"Trades/Actions: {env.total_trades:3d}/{env.total_actions_taken:3d} | "
                  f"Profit/Trades: {(env.balance-env.initial_balance)/(env.total_trades + 1e-9):5.2f} | "
                  f"p_freq: {p_frequency:5.2f} | "
                  f"acc_profit: {acc_profit:6.0f} | "
                  f"{'!!!!!' if profitable else ''}"
                  )
    
    return env, agent

def test(test_env, agent):

    def play_episode(env, agent):
        state = env.reset()
        done = False
        test_actions = []
        while not done:
            action, _ = agent.get_action(state)
            state, reward, done = env.step(action)
            test_actions.append(action)
        return env
    
    loops, balances, actions, trades = 100, [], 0, 0
    for episode in range(loops):
        env = play_episode(test_env, agent)
        balances.append(env.balance)
        actions += env.total_actions_taken
        trades += env.total_trades
    avg_balance = sum(balances) / len(balances)
    actions /= loops
    trades /= loops
    
    print(f"\n{'='*60}")
    print(f"📊 TEST RESULTS")
    print(f"{'='*60}")
    print(f"average profit: ${avg_balance - initial_balance :.2f} avg trades/actions {trades:5.2f}/{actions:5.2f}")


# ========== MAIN ==========
if __name__ == "__main__":
    print("="*60)
    print("STABLE PPO TRADING SYSTEM")
    print("="*60)
    
    # Generate synthetic data
    np.random.seed(523)
    
    #--------------------------------------- Get market history -----------------------------------
    bn = Binance.instantiate(connect=False)
    # Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
    datatype = "klines" # "aggTrades" #, "klines"
    symbol = "BTCUSDT"      # Any pair listed on Binance.
    start = datetime(2025, 7, 12)    # Any datetime. Note Binance started operation in July 2017.
    end = datetime(2026, 2, 26) # + timedelta(minutes=100)     # Any datetime
    interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
    candles, nFills, successful, _, reason = bn.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
    # "Returning 10 columns: O H L C V, Quote V, nTrades, Buy base V, Buy quote V, Mark (0: original not forged)"
    open_idx, high_idx, low_idx, close_idx, volume_idx, mark_idx = 0, 1, 2, 3, 4, 9
    if successful:          # price is a numpy array.
        print("\n Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nCreated = {}".format(candles.shape, nFills))
    else:
        print("\n Failed to Get_Price_Data_By_Time", reason)
        exit(0)
    
    #------------------------------------- Get train/test data ------------------------------------
    n_candles = len(candles)
    train_size = int(n_candles * 0.8)
    
    def split_array(raw_price, train_size):
        raw_price_train = raw_price[:train_size]
        mean, std = raw_price_train.mean(), raw_price_train.std()
        normal_price_train = (raw_price_train - mean) / (std + 1e-9)
        raw_price_test = raw_price[train_size:]
        normal_price_test = (raw_price_test - mean) / (std + 1e-9)
        return raw_price_train, normal_price_train, raw_price_test, normal_price_test
    
    raw_price_train, normal_price_train, raw_price_test, normal_price_test = split_array(candles[:, close_idx], train_size)
    raw_volume_train, normal_volume_train, raw_volume_test, normal_volume_test = split_array(candles[:, volume_idx], train_size)
    raw_mark_train, _, raw_mark_test, _ = split_array(candles[:, mark_idx], train_size)
    price_mean, volue_mean = raw_price_train.mean(), raw_volume_train.mean()
    
    market_history_train = (normal_price_train, normal_volume_train, raw_price_train, raw_volume_train, raw_mark_train, price_mean, volue_mean)
    market_history_test = (normal_price_test, normal_volume_test, raw_price_test, raw_volume_test, raw_mark_test, price_mean, volue_mean)

    print(f"\n📈 Configuration:")
    print(f"   Training samples: {train_size}")

    #------------------------------------ Setup env/agent -----------------------------------------
    mean_price = raw_price_train.mean()
    unit_size = 100 / mean_price    # 1 unit will be of around 100 dollar value.
    initial_balance = 10_000        #
    max_units = 50
    action_dim = len(TradingAction)
    
    dim_compact_history = 300
    indicator_window = 30
    max_episode_len = 1_000
    transaction_cost_rate = 0.003
    
    
    # Create environment and agent

    agent = StablePPOAgent(input_dim=10, action_dim=action_dim, device='auto', learning_rate=1e-5)
        
    #------------------------------------ test pre-train -----------------------------------------
    test_env = TradingEnvironment(market_history_test, initial_balance, transaction_cost_rate=transaction_cost_rate, unit_size=unit_size, 
        max_units=max_units, dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    test(test_env, agent)
    
    #------------------------------------- Train agent --------------------------------------------
    train_env = TradingEnvironment(market_history_train, initial_balance, transaction_cost_rate=transaction_cost_rate, unit_size=unit_size, 
        max_units=max_units, dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    train_env, agent = train_ppo(train_env, agent, max_episode_len, episodes=3000, print_interval=1)
    agent.save("stable_ppo_model.pth")
    
    #------------------------------------- Test post-train ---------------------------------------------
    print("\n🧪 Testing post-train...")
    test_env = TradingEnvironment(market_history_test, initial_balance, transaction_cost_rate=transaction_cost_rate, unit_size=unit_size, 
        max_units=max_units, dim_compact_history=dim_compact_history, indicator_window=indicator_window, max_eposode_len=max_episode_len)
    
    test(test_env, agent)
        
        
    print("\n✅ Complete!")