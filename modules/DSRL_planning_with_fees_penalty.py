import numpy as np

def find_strategy_with_fees_and_penalty(prices, r=0, p=0):
    """
    Find optimal b (consecutive buys) and h (consecutive holds)
    to maximize net profit after:
    - r% transaction fees on buy and sell volume
    - p% penalty on gross profit (only if positive)
    
    Fee Model:
    - Each BUY action: pays full price, receives (1 - r/100) units
    - Each SELLOUT action: sells all units, receives proceeds * (1 - r/100)
    
    Timeline:
    - Steps 0..b-1: Buy (b steps)
    - Steps b..b+h-1: Hold (h steps)
    - Step b+h: Sell all (1 step)
    
    Args:
        prices: numpy array of n consecutive price candles
        r: fee percentage (0-100)
        p: penalty percentage on gross profit
    
    Returns:
        tuple (b, h, net_profit)
    """
    n = len(prices)
    
    # Option 1: No trade (b = 0)
    max_profit = 0
    best_b = 0
    best_h = 0
    
    if n < 2:
        return (0, 0, 0)
    
    # Prefix sums: prefix[b] = sum of first b prices (prices[0..b-1])
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(prices)
    
    # Fee rate
    fee_rate = r / 100
    
    # Units received per buy after fee
    units_per_buy = 1 - fee_rate
    
    # Try all b ≥ 1
    for b in range(1, n):
        # h can be from 0 to n - b - 1
        for h in range(0, n - b):
            sell_idx = b + h
            sell_price = prices[sell_idx]
            buy_sum = prefix[b]
            
            # Total units accumulated after b buys
            total_units = b * units_per_buy
            
            # Total cost (pay full price for each buy)
            total_cost = buy_sum
            
            # Gross proceeds before sell fee
            gross_proceeds = total_units * sell_price
            
            # Apply sell fee (reduces proceeds)
            net_proceeds = gross_proceeds * (1 - fee_rate)
            
            # Gross profit for penalty calculation
            # Based on gross proceeds (before sell fee) minus cost
            gross_profit = gross_proceeds - total_cost
            
            # Net profit after fees
            profit_after_fees = net_proceeds - total_cost
            
            # Apply penalty on gross profit (only if positive)
            if gross_profit > 0:
                penalty = (p / 100) * gross_profit
                net_profit = profit_after_fees - penalty
            else:
                net_profit = profit_after_fees
            
            if net_profit > max_profit:
                max_profit = net_profit
                best_b = b
                best_h = h
    
    return best_b, best_h, max_profit


# Vectorized version for better performance
def find_strategy_with_fees_and_penalty_vectorized(prices, r=0, p=0):
    """
    Vectorized version for better performance with large arrays
    """
    n = len(prices)
    
    if n < 2:
        return (0, 0, 0)
    
    max_profit = 0
    best_b = 0
    best_h = 0
    
    # Prefix sums
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(prices)
    
    fee_rate = r / 100
    units_per_buy = 1 - fee_rate
    sell_fee_multiplier = 1 - fee_rate
    
    # For each possible b
    for b in range(1, n):
        max_h = n - b
        if max_h <= 0:
            continue
        
        # Prices for all possible sell indices
        sell_prices = prices[b:b+max_h]
        buy_sum = prefix[b]
        
        # Calculate for all h at once
        total_units = b * units_per_buy
        total_cost = buy_sum
        
        gross_proceeds = total_units * sell_prices
        net_proceeds = gross_proceeds * sell_fee_multiplier
        
        gross_profit = gross_proceeds - total_cost
        profit_after_fees = net_proceeds - total_cost
        
        # Apply penalty where gross_profit > 0
        penalty = np.where(gross_profit > 0, (p / 100) * gross_profit, 0)
        net_profits = profit_after_fees - penalty
        
        # Find best h for this b
        best_idx = np.argmax(net_profits)
        best_net_profit = net_profits[best_idx]
        
        if best_net_profit > max_profit:
            max_profit = best_net_profit
            best_b = b
            best_h = best_idx
    
    return best_b, best_h, max_profit


# Example usage
if __name__ == "__main__":
    test_cases = [
        {"name": "Profitable - b>0 best", "prices": np.array([10, 15, 20]), "r": 0, "p": 0},
        {"name": "All trades lose with fees", "prices": np.array([100, 95, 90, 85, 80]), "r": 1, "p": 0},
        {"name": "Breakeven - b=0 best", "prices": np.array([100, 100, 100]), "r": 0.5, "p": 0},
        {"name": "Slightly profitable with fees", "prices": np.array([100, 101, 102]), "r": 0.5, "p": 0},
        {"name": "With penalty kills profit", "prices": np.array([10, 15, 5, 20]), "r": 0.5, "p": 50},
    ]
    
    for test in test_cases:
        prices = test["prices"]
        r = test["r"]
        p = test["p"]
        n = len(prices)
        
        print("=" * 70)
        print(f"Test: {test['name']}")
        print(f"Prices: {prices}")
        print(f"n = {n}, r = {r}%, p = {p}%")
        print("-" * 70)
        
        b, h, net_profit = find_strategy_with_fees_and_penalty(prices, r, p)
        
        if b == 0:
            print(f"Optimal: b = 0 (no trade), profit = 0")
            print(f"  All trades would yield negative profit")
        else:
            sell_idx = b + h
            buy_sum = np.sum(prices[:b])
            sell_price = prices[sell_idx]
            
            fee_rate = r / 100
            units_per_buy = 1 - fee_rate
            total_units = b * units_per_buy
            total_cost = buy_sum
            
            gross_proceeds = total_units * sell_price
            gross_profit = gross_proceeds - total_cost
            net_proceeds = gross_proceeds * (1 - fee_rate)
            profit_after_fees = net_proceeds - total_cost
            
            print(f"Optimal: b = {b}, h = {h}")
            print(f"  Timeline:")
            print(f"    - Buys at indices 0..{b-1}: {prices[:b]}")
            if h > 0:
                print(f"    - Hold at indices {b}..{b+h-1}: {prices[b:b+h]}")
            else:
                print(f"    - Hold: none")
            print(f"    - Sell at index {sell_idx}: price = {sell_price}")
            print(f"  Total steps used: {b + h + 1} of {n}")
            
            print(f"\n  Fee Model (fees on BUY and SELLOUT):")
            print(f"    BUY fee ({r}%): Each BUY receives {units_per_buy:.4f} units (instead of 1.0)")
            print(f"    SELLOUT fee ({r}%): Proceeds multiplied by {1 - fee_rate:.4f}")
            print(f"")
            print(f"  Calculations:")
            print(f"    Total units purchased: {b} × {units_per_buy:.4f} = {total_units:.4f} units")
            print(f"    Total cost (paid to buy): ${total_cost:.2f}")
            print(f"    Gross proceeds: {total_units:.4f} units × ${sell_price} = ${gross_proceeds:.4f}")
            print(f"    Sell fee ({r}%): ${gross_proceeds * fee_rate:.4f}")
            print(f"    Net proceeds after sell fee: ${net_proceeds:.4f}")
            print(f"    Profit after fees: ${profit_after_fees:.4f}")
            print(f"    Gross profit (for penalty): ${gross_profit:.4f}")
            
            if p > 0 and gross_profit > 0:
                penalty = (p / 100) * gross_profit
                print(f"    Penalty ({p}% of gross profit): ${penalty:.4f}")
                print(f"    Net profit after penalty: ${net_profit:.4f}")
            else:
                print(f"    Net profit: ${net_profit:.4f}")
        print()