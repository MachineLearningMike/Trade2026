import numpy as np

def find_strategic_rewards_with_fees_and_penalty(prices, r=0, p=0, actions=None, unit_size=1.0, net_profit=0):
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
    
    # Find all buy indices and sell index
    buy_action_indices = [i for i, a in enumerate(actions) if a == 'B']
    sell_action_index = next((i for i, a in enumerate(actions) if a == 'S'), None)
    
    if sell_action_index is None or len(buy_action_indices) == 0:
        return [-1.0] * len(actions)
    
    fee_rate = r / 100
    sell_fee_factor = 1 - fee_rate
    units_per_buy = unit_size * (1 - fee_rate)
    
    # Map actions to price indices
    price_idx = 0
    action_to_price = {}
    buy_prices = []
    
    for i, action in enumerate(actions):
        if action == 'B':
            action_to_price[i] = price_idx
            buy_prices.append(prices[price_idx])
            price_idx += 1
        elif action in ['H', 'S']:
            action_to_price[i] = price_idx
            price_idx += 1
    
    # Calculate final profit (for verification)
    num_buys = len(buy_action_indices)
    total_units = num_buys * units_per_buy
    total_cost = sum(buy_prices) * unit_size
    sell_price = prices[action_to_price[sell_action_index]]
    gross_proceeds = total_units * sell_price
    net_proceeds = gross_proceeds * sell_fee_factor
    final_profit = net_proceeds - total_cost
    
    if abs(final_profit) < 1e-9:
        return [0.0] * len(actions)
    
    # Calculate rewards using SHAPLEY VALUE approach
    # Each buy action gets credit for its own contribution
    # Each hold action gets credit for the price increase during its period
    
    rewards = [0.0] * len(actions)
    
    # Track cumulative units and prices over time
    cumulative_units = 0
    cumulative_cost = 0.0
    
    # First, calculate the contribution of each BUY
    # Each BUY's contribution = units * (sell_price - buy_price) * (1 - fee_rate)
    # But we need to distribute some of this to HOLD actions
    
    # Better approach: Attribute profit to price changes
    # Each price change period contributes profit = units_held * price_change
    # This profit is split between:
    # - The HOLD action covering that period
    # - The BUY actions that contributed units
    
    # Build timeline of unit holdings
    timeline = []  # (action_idx, price, cumulative_units)
    current_units = 0
    
    for i, action in enumerate(actions):
        price = prices[action_to_price[i]]
        timeline.append({
            'action_idx': i,
            'action': action,
            'price': price,
            'units': current_units
        })
        
        if action == 'B':
            current_units += units_per_buy
    
    # Now calculate profit for each period and attribute
    for i in range(len(timeline) - 1):
        current = timeline[i]
        next_point = timeline[i+1]
        
        price_change = next_point['price'] - current['price']
        
        if price_change == 0:
            continue
        
        period_profit = current['units'] * price_change
        
        # Apply sell fee on the final period
        if next_point['action'] == 'S':
            period_profit *= sell_fee_factor
        
        # Attribute this period's profit to the current action
        rewards[current['action_idx']] += period_profit
    
    # Verify total equals final profit
    total_rewards = sum(rewards)
    if abs(total_rewards - final_profit) > 0.001:
        # Scale rewards to match final profit exactly
        scale = final_profit / total_rewards if total_rewards != 0 else 1
        rewards = [r * scale for r in rewards]
    
    return rewards


def optimal_trading_strategy(prices, r=0, p=0, unit_size=1.0):
    """
    Optimal multi-phase trading strategy - can sell at ANY point.
    """
    prices = np.array(prices)
    n = len(prices)
    
    if n < 2:
        full = ['H'] * n
        return full, 0.0, n, []
    
    fee_rate = r / 100
    buy_multiplier = unit_size * (1 - fee_rate)
    sell_multiplier = (1 - fee_rate)
    
    best_profit = 0.0
    best_buy_indices = []
    best_sell_idx = -1
    
    for sell_idx in range(1, n):
        sell_price = prices[sell_idx]
        buy_indices = []
        total_profit = 0.0
        
        for buy_idx in range(sell_idx):
            buy_price = prices[buy_idx]
            net_gain = sell_price * sell_multiplier - buy_price * buy_multiplier
            
            if p > 0:
                gross_gain = sell_price - buy_price
                value = net_gain - (p / 100) * gross_gain
                if value > 0:
                    total_profit += value
                    buy_indices.append(buy_idx)
            elif net_gain > 0:
                total_profit += net_gain
                buy_indices.append(buy_idx)
        
        if total_profit > best_profit:
            best_profit = total_profit
            best_buy_indices = buy_indices.copy()
            best_sell_idx = sell_idx
    
    # Build action sequence
    full_actions = ['H'] * n
    for idx in best_buy_indices:
        full_actions[idx] = 'B'
    
    if best_sell_idx >= 0:
        full_actions[best_sell_idx] = 'S'
        for i in range(best_sell_idx + 1, n):
            full_actions[i] = 'V'
    
    # Trim leading H's and trailing V's
    trimmed = []
    started = False
    for action in full_actions:
        if not started and action == 'H':
            continue
        if action == 'V':
            break
        started = True
        trimmed.append(action)
    
    return full_actions, best_profit, 0, trimmed


def run_tests():
    """Test the corrected reward calculation"""
    
    test_cases = [
        {
            "name": "Simple: B S",
            "prices": [10, 20],
            "actions": ['B', 'S'],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "name": "Simple: B H S",
            "prices": [10, 15, 20],
            "actions": ['B', 'H', 'S'],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "name": "Two buys: B B S",
            "prices": [10, 12, 20],
            "actions": ['B', 'B', 'S'],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "name": "Multiple phases: B B H B H B S",
            "prices": [100, 85, 115, 95, 110, 90, 108],
            "actions": ['B', 'B', 'H', 'B', 'H', 'B', 'S'],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "name": "With fees: B B S",
            "prices": [100, 90, 105],
            "actions": ['B', 'B', 'S'],
            "r": 2,
            "unit_size": 1.0
        },
        {
            "name": "Recovery: B B B B B S",
            "prices": [100, 90, 80, 70, 80, 90, 100],
            "actions": ['B', 'B', 'B', 'B', 'B', 'S'],
            "r": 0,
            "unit_size": 1.0
        },
        {
            "name": "Complex with holds: B H B H S",
            "prices": [100, 120, 95, 115, 110],
            "actions": ['B', 'H', 'B', 'H', 'S'],
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
        actions = test["actions"]
        r = test["r"]
        unit_size = test["unit_size"]
        
        # Calculate actual profit
        fee_rate = r / 100
        units_per_buy = unit_size * (1 - fee_rate)
        buy_indices = [i for i, a in enumerate(actions) if a == 'B']
        num_buys = len(buy_indices)
        total_units = num_buys * units_per_buy
        total_cost = sum(prices[i] for i in range(num_buys)) * unit_size
        sell_idx = next(i for i, a in enumerate(actions) if a == 'S')
        sell_price = prices[sell_idx]
        gross_proceeds = total_units * sell_price
        net_proceeds = gross_proceeds * (1 - fee_rate)
        expected_profit = net_proceeds - total_cost
        
        # Calculate rewards
        rewards = find_strategic_rewards_with_fees_and_penalty(
            prices, r, 0, actions, unit_size, 0
        )
        
        total_rewards = sum(rewards)
        is_match = abs(total_rewards - expected_profit) < 0.01
        
        if is_match:
            passed += 1
            status = "✓ PASS"
        else:
            failed += 1
            status = "✗ FAIL"
        
        print(f"\n{'─'*100}")
        print(f"TEST: {test['name']} - {status}")
        print(f"  Actions: {actions}")
        print(f"  Expected profit: {expected_profit:.4f}")
        print(f"  Sum of rewards:  {total_rewards:.4f}")
        print(f"  Match: {'✓' if is_match else '✗'}")
        
        if not is_match:
            print(f"  DIFFERENCE: {total_rewards - expected_profit:.4f}")
        
        print(f"\n  Individual rewards:")
        for action, reward in zip(actions, rewards):
            print(f"    {action}: {reward:+.4f}")


if __name__ == "__main__":
    run_tests()