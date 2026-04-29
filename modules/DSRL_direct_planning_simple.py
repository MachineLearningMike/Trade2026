def find_best_b_h(prices):
    """
    Find optimal b (consecutive buys) and h (consecutive holds)
    to maximize profit: b * sell_price - sum(buy_prices)
    
    Timeline:
    - Steps 0..b-1: Buy (b steps)
    - Steps b..b+h-1: Hold (h steps)
    - Step b+h: Sell all (1 step)
    
    Constraints:
    - b ≥ 0, h ≥ 0
    - If b = 0: no trade, profit = 0
    - If b ≥ 1: need b + h + 1 ≤ n
    
    Args:
        prices: list of n consecutive price candles
    
    Returns:
        tuple (b, h) that maximizes profit (0 if no profitable trade)
    """
    n = len(prices)
    
    # Option 1: No trade (b = 0)
    max_profit = 0
    best_b = 0
    best_h = 0
    
    if n < 2:
        return (0, 0)  # Need at least buy + sell for any trade
    
    # Prefix sums: prefix[b] = sum of first b prices (prices[0..b-1])
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + prices[i]
    
    # Try all b ≥ 1
    for b in range(1, n):  # b can go up to n-1 (need at least 1 sell step)
        # h can be from 0 to n - b - 1 (ensuring sell index exists)
        for h in range(0, n - b):
            sell_idx = b + h
            sell_price = prices[sell_idx]
            buy_sum = prefix[b]
            
            profit = b * sell_price - buy_sum
            
            if profit > max_profit:
                max_profit = profit
                best_b = b
                best_h = h
    
    return best_b, best_h


# Example usage
if __name__ == "__main__":
    # Example 1 - Profitable
    prices = [10, 15, 5, 20]
    b, h = find_best_b_h(prices)
    print(f"Prices: {prices}")
    print(f"Best b = {b}, h = {h}")
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        profit = b * prices[sell_idx] - buy_sum
        print(f"Profit = {profit}")
        print(f"  Buys at indices 0..{b-1}: {prices[:b]}")
        if h > 0:
            print(f"  Holds at indices {b}..{b+h-1}: {prices[b:b+h]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
    else:
        print("No profitable trade (b = 0)")
    print()
    
    # Example 2 - Peak in middle
    prices = [5, 10, 20, 15, 8]
    b, h = find_best_b_h(prices)
    print(f"Prices: {prices}")
    print(f"Best b = {b}, h = {h}")
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        profit = b * prices[sell_idx] - buy_sum
        print(f"Profit = {profit}")
        print(f"  Buys at indices 0..{b-1}: {prices[:b]}")
        if h > 0:
            print(f"  Holds at indices {b}..{b+h-1}: {prices[b:b+h]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
    else:
        print("No profitable trade (b = 0)")
    print()
    
    # Example 3 - Decreasing prices (no profit possible)
    prices = [20, 15, 10, 5]
    b, h = find_best_b_h(prices)
    print(f"Prices: {prices}")
    print(f"Best b = {b}, h = {h}")
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        profit = b * prices[sell_idx] - buy_sum
        print(f"Profit = {profit}")
        print(f"  Buys at indices 0..{b-1}: {prices[:b]}")
        if h > 0:
            print(f"  Holds at indices {b}..{b+h-1}: {prices[b:b+h]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
    else:
        print("No profitable trade (b = 0)")
    print()
    
    # Example 4 - Small increasing
    prices = [10, 20, 30]
    b, h = find_best_b_h(prices)
    print(f"Prices: {prices}")
    print(f"Best b = {b}, h = {h}")
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        profit = b * prices[sell_idx] - buy_sum
        print(f"Profit = {profit}")
        print(f"  Buys at indices 0..{b-1}: {prices[:b]}")
        if h > 0:
            print(f"  Holds at indices {b}..{b+h-1}: {prices[b:b+h]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
    else:
        print("No profitable trade (b = 0)")
    print()
    
    # Example 5 - Only 2 candles
    prices = [100, 90]
    b, h = find_best_b_h(prices)
    print(f"Prices: {prices}")
    print(f"Best b = {b}, h = {h}")
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        profit = b * prices[sell_idx] - buy_sum
        print(f"Profit = {profit}")
        print(f"  Buys at indices 0..{b-1}: {prices[:b]}")
        if h > 0:
            print(f"  Holds at indices {b}..{b+h-1}: {prices[b:b+h]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
    else:
        print("No profitable trade (b = 0)")