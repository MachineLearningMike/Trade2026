import numpy as np
import copy

def optimal_trading_strategy(prices, r=0, unit_size=1.0, max_buy_cost=None):
    """
    Optimal multi-phase trading strategy - can sell at ANY point.
    
    Returns:
        full_actions: List with 'B', 'H', 'S', 'V' for all positions
        profit: Maximum net profit
        first_holds: Number of initial 'H' actions before first action
        trimmed_actions: Actions with leading 'H's and trailing 'V's removed
    """
    prices = np.array(prices)
    n = len(prices)
    
    if n < 2:
        full = ['S'] * n
        return full, 0.0, n, []
    
    fee_rate = r / 100
    buy_multiplier = (1 - fee_rate)
    sell_multiplier = (1 - fee_rate)
    
    best_profit = float('-inf')
    best_buy_indices = []
    best_sell_idx = -1
    
    # Try every possible sell point
    for sell_idx in range(0, n):        # SELLOUT is enabled as the first action, which means no actions at all.
        sell_price = prices[sell_idx]
        buy_indices = []
        total_profit = 0.0
        
        for buy_idx in range(sell_idx):
            buy_price = prices[buy_idx]
            net_gain = sell_price * sell_multiplier * unit_size * buy_multiplier - buy_price * unit_size
            
            if net_gain > 0:
                total_profit += net_gain
                buy_indices.append(buy_idx)
        
        if total_profit > best_profit:
            best_profit = total_profit
            best_buy_indices = buy_indices.copy()
            best_sell_idx = sell_idx
    
    best_buy_indices = skip_buys(prices, unit_size, best_buy_indices, max_buy_cost)
    
    # Build action sequence
    full_actions = ['H'] * n
    for idx in best_buy_indices:
        full_actions[idx] = 'B'
    
    if best_sell_idx >= 0:
        full_actions[best_sell_idx] = 'S'
        # Mark positions after sell as 'V' (void)
        for i in range(best_sell_idx + 1, n):
            full_actions[i] = 'V'
    
    # Count first HOLDs
    first_holds = 0
    for action in full_actions:
        if action == 'H':
            first_holds += 1
        else:
            break
    
    # Create trimmed actions (remove leading H's and trailing V's)
    trimmed = []
    started = False
    for action in full_actions:
        if not started and action == 'H':
            continue
        if action == 'V':
            break
        started = True
        trimmed.append(action)
    
    return full_actions, best_profit, first_holds, trimmed

def skip_buys(prices, unit_size=1.0, buy_indices: list = [], max_buy_cost=None) -> list:
    new_indices = buy_indices.copy()
    
    if max_buy_cost is not None:
        buy_prices = prices[new_indices]

        while buy_prices.sum() * unit_size > max_buy_cost:
            i = np.argmax(buy_prices)
            # print(new_indices, buy_prices, i)
            buy_prices = np.concatenate([buy_prices[:i], buy_prices[i+1:]])
            new_indices.pop(i)
            # print(new_indices, buy_prices)
            # print('----------')
    return new_indices

# Convenience function for simple use cases
def get_strategy(prices, r=0):
    """Simplified interface - returns trimmed actions and profit"""
    _, profit, first_holds, trimmed = optimal_trading_strategy(prices, r)
    return trimmed, profit, first_holds


# Example usage
if __name__ == "__main__":
    # Test cases
    test_cases = [
        ([100, 95, 110, 90, 85, 80, 75], "Peak crash"),
        ([100, 90, 80, 70, 80, 90, 100], "Recovery with wait"),
        ([110, 108, 105, 102, 100, 98, 95], "Downtrend"),
        ([100, 90, 105, 98, 102, 95, 100], "With fees"),
        ([100, 85, 115, 95, 110, 90, 108], "Multiple peaks"),
    ]
    
    print("="*80)
    print("OPTIMAL TRADING STRATEGY - SUMMARY")
    print("="*80)
    
    for prices, name in test_cases:
        full, profit, first_holds, trimmed = optimal_trading_strategy(prices)
        
        print(f"\n{name}:")
        print(f"  Prices:        {prices}")
        print(f"  Full actions:  {full}")
        print(f"  First HOLDs:   {first_holds}")
        print(f"  Trimmed:       {trimmed}")
        print(f"  Profit:        {profit:.2f}")
    
    # Simple interface example
    print("\n" + "="*80)
    print("SIMPLE INTERFACE EXAMPLE")
    print("="*80)
    
    prices = [100, 90, 80, 70, 80, 90, 100]
    actions, profit, holds = get_strategy(prices)
    print(f"Prices: {prices}")
    print(f"Strategy: {actions}")
    print(f"Initial HOLDs: {holds}")
    print(f"Profit: {profit:.2f}")