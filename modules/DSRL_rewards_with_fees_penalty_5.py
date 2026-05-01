import numpy as np
import math

from DSRL_planning_with_fees_penalty_6 import optimal_trading_strategy

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_math(x):
    return 1 / (1 + math.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def softmax_math(values):
    """Softmax using only math module"""
    # Find max for numerical stability
    max_val = max(values)
    
    # Compute exponentials
    exp_values = [math.exp(v - max_val) for v in values]
    
    # Sum of all exponentials
    sum_exp = sum(exp_values)
    
    # Return probabilities
    return [e / sum_exp for e in exp_values]

def find_strategic_rewards_with_fees_and_penalty(prices, r=0, unit_size=1.0, actions=None, moving_price_mean=0, moving_price_std=0, net_profit=None):
    """
    Calculate reward for each action in the trimmed action sequence.
    
    Attribution principle:
    - Each BUY action gets credit for price appreciation from its purchase price to sell price
    - Each HOLD action gets credit for price appreciation during that period
    - The sum of all rewards equals final profit exactly
    
    Args:
        prices: numpy array of price candles
        r: fee percentage (0-100)
        p: penalty percentage (unused)
        actions: list of strings with 'B' (buy), 'H' (hold), 'S' (sell)
        unit_size: size of each BUY action (default 1.0)
    
    Returns:
        list of rewards for each action in order
    """
    if actions is None or len(actions) == 0:
        return []
    
    fee_rate = r / 100
    buy_multiplier = (1 - fee_rate)
    sell_multiplier = (1 - fee_rate)
    
    rewards = [0] * len(prices)
    sell_action_index = next((i for i, a in enumerate(actions) if a == 'S'), None)
    assert sell_action_index is not None
    sell_price = prices[sell_action_index]

    for idx in range(len(actions)):

        if idx <= sell_action_index:
            
            if actions[idx] == 'B':
                buy_price = prices[idx]
                buy_cost = buy_price * unit_size
                sell_revenue = sell_price * sell_multiplier * buy_multiplier * unit_size
                profit = sell_revenue - buy_cost
                buy_price_normal = (buy_price - moving_price_mean) / moving_price_std
                buy_risk = sigmoid_math(buy_price_normal)    # This buy takes as much risk as sigmoid(buy_price_normal)
                rewards[idx] += buy_risk * profit
                price_inc = sell_price - buy_price
                try:
                    for i in range(idx+1, sell_action_index+1): # include SELLOUT, which is the last HOLD
                        rewards[i] += (prices[i]-prices[i-1]) / price_inc * profit * (1-buy_risk)
                except:
                    print(f"Bomb explodes!!!")
                    pass
            else:
                assert actions[idx] == 'H' or actions[idx] == 'S'
        else:
            rewards[idx] = 0
        
    # if net_profit is not None:
    #     assert abs( (sum(rewards) - net_profit) / net_profit ) < 1e-5
        
    return rewards

def run_tests():
    """Test the corrected reward calculation"""
    
    test_cases = [
        {
            "prices": [10, 20],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "prices": [10, 15, 20],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "prices": [10, 12, 20],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "prices": [100, 85, 115, 95, 110, 90, 108],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "prices": [100, 90, 105],
            "r": 2,
            "unit_size": 1.0
        },
        {
            "prices": [100, 90, 80, 70, 80, 90, 100],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "prices": [100, 120, 95, 115, 110],
            "r": 0,
            "unit_size": 1.0
        }
    ]
    
    print("=" * 100)
    print("CORRECTED MULTI-PHASE REWARD CALCULATION (SHAPLEY VALUE)")
    print("=" * 100)
    
    passed = 0
    failed = 0
    
    for test in test_cases:
        prices = np.array(test["prices"])
        r = test["r"]
        unit_size = test["unit_size"]
        
        full_actions, best_profit, first_holds, trimmed = optimal_trading_strategy(prices, r, unit_size)
        
        # Calculate rewards
        rewards = find_strategic_rewards_with_fees_and_penalty(
            prices, r, unit_size, full_actions, prices.mean(), prices.std(), best_profit
        )
        returns = [sum(rewards[i:]) for i in range(len(rewards))]
        
        print(f"prices {prices}")
        print(f"actions {full_actions}")
        print(f"rewards {rewards}")
        print(f"returns {returns}")
        print(f"net_profit {best_profit}")
        print("-----------------")
        

if __name__ == "__main__":
    run_tests()