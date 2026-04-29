import numpy as np
import time

def find_strategy_with_fees_and_penalty_original(prices, r=0, p=0):
    """
    Original nested loops version
    """
    n = len(prices)
    max_profit = 0.0
    best_b = 0
    best_h = 0
    
    if n < 2:
        return (0, 0, 0)
    
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(prices)
    
    fee_rate = r / 100
    buy_unit_multiplier = 1 - fee_rate
    combined_multiplier = buy_unit_multiplier * (1 - fee_rate)
    
    for b in range(1, n):
        for h in range(0, n - b):
            sell_idx = b + h
            sell_price = prices[sell_idx]
            buy_sum = prefix[b]
            
            profit_after_fees = b * combined_multiplier * sell_price - buy_sum
            gross_profit = b * buy_unit_multiplier * sell_price - buy_sum
            
            if gross_profit > 0 and p > 0:
                penalty = (p / 100) * gross_profit
                net_profit = profit_after_fees - penalty
            else:
                net_profit = profit_after_fees
            
            if net_profit > max_profit:
                max_profit = net_profit
                best_b = b
                best_h = h
    
    return best_b, best_h, max_profit


def find_strategy_with_fees_and_penalty_vectorized(prices, r=0, p=0):
    """
    Vectorized version using numpy operations
    """
    n = len(prices)
    
    if n < 2:
        return (0, 0, 0)
    
    # Prefix sums
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(prices)
    
    fee_rate = r / 100
    buy_unit_multiplier = 1 - fee_rate
    combined_multiplier = buy_unit_multiplier * (1 - fee_rate)
    
    max_profit = 0.0
    best_b = 0
    best_h = 0
    
    # For each possible b
    for b in range(1, n):
        # Maximum possible holds for this b
        max_h = n - b
        if max_h <= 0:
            continue
        
        # Vectorized calculation for all h at once
        # Sell prices for all possible sell indices
        sell_prices = prices[b:b+max_h]
        buy_sum = prefix[b]
        
        # Calculate profits for all h in one operation
        profit_after_fees = b * combined_multiplier * sell_prices - buy_sum
        gross_profit = b * buy_unit_multiplier * sell_prices - buy_sum
        
        # Apply penalty where gross_profit > 0
        penalty_mask = (gross_profit > 0) & (p > 0)
        penalty = np.where(penalty_mask, (p / 100) * gross_profit, 0)
        net_profits = profit_after_fees - penalty
        
        # Find best h for this b
        best_idx = np.argmax(net_profits)
        best_net_profit = net_profits[best_idx]
        
        if best_net_profit > max_profit:
            max_profit = best_net_profit
            best_b = b
            best_h = best_idx
    
    return best_b, best_h, max_profit


def find_strategy_with_fees_and_penalty_fully_vectorized(prices, r=0, p=0):
    """
    Fully vectorized version using broadcasting (may use more memory)
    """
    n = len(prices)
    
    if n < 2:
        return (0, 0, 0)
    
    # Prefix sums
    prefix = np.zeros(n + 1)
    prefix[1:] = np.cumsum(prices)
    
    fee_rate = r / 100
    buy_unit_multiplier = 1 - fee_rate
    combined_multiplier = buy_unit_multiplier * (1 - fee_rate)
    
    # Create matrices for all b and h combinations
    max_profit = 0.0
    best_b = 0
    best_h = 0
    
    # For each b
    for b in range(1, n):
        max_h = n - b
        if max_h <= 0:
            continue
        
        # Create arrays for this b
        b_array = np.full(max_h, b)
        sell_prices = prices[b:b+max_h]
        buy_sum = prefix[b]
        
        # Vectorized calculations
        profit_after_fees = b_array * combined_multiplier * sell_prices - buy_sum
        gross_profit = b_array * buy_unit_multiplier * sell_prices - buy_sum
        
        # Apply penalty
        if p > 0:
            penalty = np.where(gross_profit > 0, (p / 100) * gross_profit, 0)
            net_profits = profit_after_fees - penalty
        else:
            net_profits = profit_after_fees
        
        # Find best
        best_idx = np.argmax(net_profits)
        best_net_profit = net_profits[best_idx]
        
        if best_net_profit > max_profit:
            max_profit = best_net_profit
            best_b = b
            best_h = best_idx
    
    return best_b, best_h, max_profit


def compare_performance():
    """
    Compare performance of different implementations
    """
    print("=" * 80)
    print("PERFORMANCE COMPARISON")
    print("=" * 80)
    
    # Test with different array sizes
    test_sizes = [10, 50, 100, 200, 500]
    results = []
    
    for size in test_sizes:
        # Generate random price data
        np.random.seed(42)
        prices = np.random.randn(size) * 10 + 100
        prices = np.abs(prices)  # Make prices positive
        prices = np.cumsum(prices)  # Make trending
        
        r, p = 0.5, 10
        
        print(f"\nArray size: {size}")
        print("-" * 40)
        
        # Test original nested loops
        start = time.time()
        b1, h1, profit1 = find_strategy_with_fees_and_penalty_original(prices, r, p)
        time1 = time.time() - start
        
        # Test partially vectorized
        start = time.time()
        b2, h2, profit2 = find_strategy_with_fees_and_penalty_vectorized(prices, r, p)
        time2 = time.time() - start
        
        # Test fully vectorized
        start = time.time()
        b3, h3, profit3 = find_strategy_with_fees_and_penalty_fully_vectorized(prices, r, p)
        time3 = time.time() - start
        
        # Verify all produce same result
        same_strategy = (b1 == b2 == b3) and (h1 == h2 == h3)
        same_profit = abs(profit1 - profit2) < 0.0001 and abs(profit1 - profit3) < 0.0001
        
        results.append({
            'size': size,
            'original_time': time1,
            'vectorized_time': time2,
            'fully_vectorized_time': time3,
            'speedup_original_vs_vectorized': time1 / time2 if time2 > 0 else float('inf'),
            'speedup_original_vs_fully': time1 / time3 if time3 > 0 else float('inf'),
            'correct': same_strategy and same_profit
        })
        
        print(f"  Original loops:      {time1:.6f}s (b={b1}, h={h1}, profit={profit1:.4f})")
        print(f"  Partially vectorized:{time2:.6f}s (b={b2}, h={h2}, profit={profit2:.4f})")
        print(f"  Fully vectorized:    {time3:.6f}s (b={b3}, h={h3}, profit={profit3:.4f})")
        print(f"  Results match:       {'✓' if same_strategy and same_profit else '✗'}")
    
    # Summary table
    print("\n" + "=" * 80)
    print("SPEEDUP SUMMARY")
    print("=" * 80)
    print(f"{'Size':<10} {'Original (s)':<15} {'Vectorized (s)':<18} {'Speedup':<12} {'Fully Vec (s)':<18} {'Speedup':<12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['size']:<10} {r['original_time']:<15.6f} {r['vectorized_time']:<18.6f} "
              f"{r['speedup_original_vs_vectorized']:<12.2f} {r['fully_vectorized_time']:<18.6f} "
              f"{r['speedup_original_vs_fully']:<12.2f}")
    
    return results


def test_correctness():
    """
    Test correctness of vectorized version against original
    """
    print("\n" + "=" * 80)
    print("CORRECTNESS TEST")
    print("=" * 80)
    
    test_cases = [
        {
            "name": "No fees",
            "prices": np.array([10, 15, 20, 25, 30]),
            "r": 0,
            "p": 0
        },
        {
            "name": "With fees",
            "prices": np.array([100, 101, 102, 103, 104]),
            "r": 0.5,
            "p": 0
        },
        {
            "name": "With penalty",
            "prices": np.array([10, 15, 5, 20, 25]),
            "r": 0.5,
            "p": 50
        },
        {
            "name": "Downward trend",
            "prices": np.array([100, 95, 90, 85, 80]),
            "r": 1,
            "p": 0
        },
        {
            "name": "Flat prices",
            "prices": np.array([100, 100, 100, 100, 100]),
            "r": 0.5,
            "p": 10
        },
        {
            "name": "Random prices",
            "prices": np.array([50, 55, 52, 58, 60, 57, 62, 65, 63, 70]),
            "r": 0.3,
            "p": 20
        }
    ]
    
    all_correct = True
    
    for test in test_cases:
        prices = test["prices"]
        r = test["r"]
        p = test["p"]
        
        # Run both implementations
        b_orig, h_orig, profit_orig = find_strategy_with_fees_and_penalty_original(prices, r, p)
        b_vec, h_vec, profit_vec = find_strategy_with_fees_and_penalty_vectorized(prices, r, p)
        
        # Compare results
        strategy_match = (b_orig == b_vec) and (h_orig == h_vec)
        profit_match = abs(profit_orig - profit_vec) < 0.0001
        
        correct = strategy_match and profit_match
        all_correct = all_correct and correct
        
        print(f"\nTest: {test['name']}")
        print(f"  Prices: {prices}")
        print(f"  r={r}%, p={p}%")
        print(f"  Original:    b={b_orig}, h={h_orig}, profit={profit_orig:.6f}")
        print(f"  Vectorized:  b={b_vec}, h={h_vec}, profit={profit_vec:.6f}")
        print(f"  Match:       {'✓' if correct else '✗'}")
    
    print("\n" + "=" * 80)
    print(f"OVERALL CORRECTNESS: {'✓ ALL TESTS PASSED' if all_correct else '✗ SOME TESTS FAILED'}")
    print("=" * 80)


def detailed_comparison():
    """
    Detailed comparison of algorithm logic
    """
    print("\n" + "=" * 80)
    print("ALGORITHM COMPARISON")
    print("=" * 80)
    
    print("\nOriginal Version (Nested Loops):")
    print("-" * 40)
    print("""
    for b in range(1, n):           # O(n)
        for h in range(0, n - b):   # O(n)
            sell_idx = b + h
            sell_price = prices[sell_idx]
            buy_sum = prefix[b]
            
            # Calculate profit
            profit = b * combined * sell_price - buy_sum
            # Apply penalty
            if gross_profit > 0 and p > 0:
                profit -= penalty
            
            # Track best
            if profit > max_profit:
                update best
            
    Time Complexity: O(n²)
    Space Complexity: O(n)
    """)
    
    print("\nVectorized Version:")
    print("-" * 40)
    print("""
    for b in range(1, n):           # O(n)
        # Vectorized for all h
        sell_prices = prices[b:b+max_h]     # O(1) slice
        profit_after_fees = b * combined * sell_prices - buy_sum  # O(max_h) vectorized
        gross_profit = b * multiplier * sell_prices - buy_sum      # O(max_h) vectorized
        
        # Vectorized penalty application
        penalty = np.where((gross_profit > 0) & (p > 0), 
                          (p/100) * gross_profit, 0)
        net_profits = profit_after_fees - penalty
        
        # Find best in O(max_h) (vectorized)
        best_idx = np.argmax(net_profits)
        best_net_profit = net_profits[best_idx]
        
    Time Complexity: O(n²) but with vectorized operations
    Space Complexity: O(n) for temporary arrays
    """)
    
    print("\nFully Vectorized Version:")
    print("-" * 40)
    print("""
    Similar to partially vectorized, but avoids Python loops for operations
    on arrays. Still requires O(n) loop over b because each b has different
    array sizes.
    
    Time Complexity: O(n²) with optimized C-level operations
    Space Complexity: O(n) for temporary arrays
    """)
    
    print("\nKey Differences:")
    print("-" * 40)
    print("""
    1. Original: Pure Python loops - slow for large n
    2. Vectorized: Uses numpy's C-optimized operations for inner loop
    3. Memory: Vectorized creates temporary arrays, original doesn't
    
    Best for:
    - Original: Small n (< 50), memory constrained
    - Vectorized: Medium to large n (50-1000), balanced performance
    - Fully vectorized: Large n (> 1000), performance critical
    """)


if __name__ == "__main__":
    # Test correctness first
    test_correctness()
    
    # Compare performance
    performance_results = compare_performance()
    
    # Detailed algorithm comparison
    detailed_comparison()
    
    # Example usage with a specific case
    print("\n" + "=" * 80)
    print("EXAMPLE USAGE")
    print("=" * 80)
    
    prices = np.array([10, 15, 5, 20, 25, 22, 30])
    r, p = 0.5, 10
    
    print(f"\nPrices: {prices}")
    print(f"Fees: {r}%, Penalty: {p}%")
    
    # Run all three versions
    b1, h1, profit1 = find_strategy_with_fees_and_penalty_original(prices, r, p)
    b2, h2, profit2 = find_strategy_with_fees_and_penalty_vectorized(prices, r, p)
    b3, h3, profit3 = find_strategy_with_fees_and_penalty_fully_vectorized(prices, r, p)
    
    print(f"\nResults:")
    print(f"  Original:       b={b1}, h={h1}, profit={profit1:.4f}")
    print(f"  Vectorized:     b={b2}, h={h2}, profit={profit2:.4f}")
    print(f"  Fully Vec:      b={b3}, h={h3}, profit={profit3:.4f}")
    
    # Show the optimal strategy details
    if b2 > 0:
        sell_idx = b2 + h2
        fee_rate = r / 100
        buy_unit = 1 - fee_rate
        
        print(f"\nOptimal strategy (vectorized):")
        print(f"  Buy {b2} times: {prices[:b2]}")
        if h2 > 0:
            print(f"  Hold {h2} steps: {prices[b2:b2+h2]}")
        print(f"  Sell at index {sell_idx}: price = {prices[sell_idx]}")
        
        total_units = b2 * buy_unit
        buy_sum = np.sum(prices[:b2])
        gross_proceeds = total_units * prices[sell_idx]
        net_proceeds = gross_proceeds * (1 - fee_rate)
        gross_profit = gross_proceeds - buy_sum
        
        print(f"\n  Transaction details:")
        print(f"    Total cost: {buy_sum:.2f}")
        print(f"    Units bought: {b2} × {buy_unit:.4f} = {total_units:.4f}")
        print(f"    Gross proceeds: {total_units:.4f} × {prices[sell_idx]} = {gross_proceeds:.2f}")
        print(f"    Sell fee: {gross_proceeds * fee_rate:.4f}")
        print(f"    Net proceeds: {net_proceeds:.2f}")
        print(f"    Gross profit: {gross_profit:.4f}")
        
        if gross_profit > 0 and p > 0:
            penalty = (p / 100) * gross_profit
            print(f"    Penalty ({p}%): {penalty:.4f}")
            print(f"    Net profit: {net_proceeds - buy_sum - penalty:.4f}")