def find_best_b_h_with_fees(prices, r):
    """
    Find optimal b and h to maximize net profit after r% transaction fees.
    
    Timeline:
    - Steps 0..b-1: Buy (b steps)
    - Steps b..b+h-1: Hold (h steps)
    - Step b+h: Sell all (1 step)
    
    Net profit = sell_revenue * (1 - r/100) - buy_cost * (1 + r/100)
    
    Args:
        prices: list of n consecutive price candles
        r: fee percentage (0-100)
    
    Returns:
        tuple (b, h, net_profit) - b=0 means no trade
    """
    n = len(prices)
    
    # Option 1: No trade (b = 0)
    max_profit = 0
    best_b = 0
    best_h = 0
    
    if n < 2:
        return (0, 0, 0)  # Need at least buy + sell for any trade
    
    fee_buy = 1 + r / 100
    fee_sell = 1 - r / 100
    
    # Prefix sums
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + prices[i]
    
    # Try all b ≥ 1
    for b in range(1, n):  # b can go up to n-1 (need at least 1 sell step)
        for h in range(0, n - b):  # h from 0 to n-b-1
            sell_idx = b + h
            sell_price = prices[sell_idx]
            buy_sum = prefix[b]
            
            # Net profit after fees
            profit = b * sell_price * fee_sell - buy_sum * fee_buy
            
            if profit > max_profit:
                max_profit = profit
                best_b = b
                best_h = h
    
    return best_b, best_h, max_profit


# Example usage with verification
if __name__ == "__main__":
    # Test case 1
    prices = [10, 15, 5, 20]
    r = 0.5  # 0.5% fee
    
    print(f"Prices: {prices}")
    print(f"Fee rate: {r}%")
    print("=" * 60)
    
    b, h, profit = find_best_b_h_with_fees(prices, r)
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        sell_revenue = b * prices[sell_idx]
        
        # Calculate fees explicitly
        buy_fee = (r / 100) * buy_sum
        sell_fee = (r / 100) * sell_revenue
        total_fees = buy_fee + sell_fee
        
        gross_profit = sell_revenue - buy_sum
        net_profit = gross_profit - total_fees
        
        print(f"\nOptimal strategy: b = {b}, h = {h}")
        print(f"  Buys at indices 0..{b-1}: {prices[:b]}")
        if h > 0:
            print(f"  Holds at indices {b}..{b+h-1}: {prices[b:b+h]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
        print(f"  Total steps used: {b + h + 1} of {len(prices)}")
        print(f"  Total buy cost: {buy_sum}")
        print(f"  Buy fee ({r}%): {buy_fee:.2f}")
        print(f"  Sell revenue: {sell_revenue}")
        print(f"  Sell fee ({r}%): {sell_fee:.2f}")
        print(f"  Total fees: {total_fees:.2f}")
        print(f"  Gross profit: {gross_profit}")
        print(f"  Net profit: {net_profit:.2f}")
        print(f"  Function computed profit: {profit:.2f}")
        print(f"  ✓ Match" if abs(net_profit - profit) < 0.01 else "  ✗ Mismatch")
    else:
        print("No profitable trade found (b = 0)")
    
    print("\n" + "=" * 60)
    
    # Test case 2: Higher fees
    r = 5  # 5% fee
    print(f"\nPrices: {prices}")
    print(f"Fee rate: {r}%")
    print("-" * 60)
    
    b, h, profit = find_best_b_h_with_fees(prices, r)
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        sell_revenue = b * prices[sell_idx]
        
        buy_fee = (r / 100) * buy_sum
        sell_fee = (r / 100) * sell_revenue
        total_fees = buy_fee + sell_fee
        
        gross_profit = sell_revenue - buy_sum
        net_profit = gross_profit - total_fees
        
        print(f"\nOptimal strategy: b = {b}, h = {h}")
        print(f"  Buys: {prices[:b]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
        print(f"  Buy cost: {buy_sum}")
        print(f"  Buy fee: {buy_fee:.2f}")
        print(f"  Sell revenue: {sell_revenue}")
        print(f"  Sell fee: {sell_fee:.2f}")
        print(f"  Net profit: {net_profit:.2f}")
        print(f"  Function computed profit: {profit:.2f}")
    else:
        print("No profitable trade found (b = 0)")
    
    print("\n" + "=" * 60)
    
    # Test case 3: Decreasing prices (no profit)
    prices = [100, 95, 90, 85, 80]
    r = 1  # 1% fee
    print(f"\nPrices: {prices}")
    print(f"Fee rate: {r}%")
    print("-" * 60)
    
    b, h, profit = find_best_b_h_with_fees(prices, r)
    
    if b > 0:
        sell_idx = b + h
        print(f"\nOptimal strategy: b = {b}, h = {h}")
        print(f"  Profit: {profit:.2f}")
    else:
        print("No profitable trade found (b = 0)")
    
    print("\n" + "=" * 60)
    
    # Test case 4: Small increasing
    prices = [10, 20, 30]
    r = 0.5
    print(f"\nPrices: {prices}")
    print(f"Fee rate: {r}%")
    print("-" * 60)
    
    b, h, profit = find_best_b_h_with_fees(prices, r)
    
    if b > 0:
        sell_idx = b + h
        buy_sum = sum(prices[:b])
        sell_revenue = b * prices[sell_idx]
        
        buy_fee = (r / 100) * buy_sum
        sell_fee = (r / 100) * sell_revenue
        
        gross_profit = sell_revenue - buy_sum
        net_profit = gross_profit - buy_fee - sell_fee
        
        print(f"\nOptimal strategy: b = {b}, h = {h}")
        print(f"  Buys: {prices[:b]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
        print(f"  Gross profit: {gross_profit}")
        print(f"  Fees: {buy_fee + sell_fee:.2f}")
        print(f"  Net profit: {net_profit:.2f}")
        print(f"  Function computed profit: {profit:.2f}")
    else:
        print("No profitable trade found (b = 0)")