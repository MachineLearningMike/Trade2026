import numpy as np
import time

def find_strategy_with_fees_and_penalty_vectorized(prices, r=0, p=0, unit_size=1.0):
    n = len(prices)
    fee_rate = r / 100
    buy_unit_multiplier = unit_size * (1 - fee_rate)
    combined_multiplier = buy_unit_multiplier * (1 - fee_rate)
    
    max_profit = float('-inf')
    best_holds1 = -1
    best_buys = -1
    best_holds2 = -1
    total_combinations = 0
    
    for holds1 in range(0, n - 1):
        for buys in range(1, n - holds1):
            for holds2 in range(1, n - holds1 - buys + 1):
                total_combinations += 1
                
                buy_start = holds1
                buy_end = holds1 + buys - 1
                sell_idx = holds1 + buys + holds2 - 1
                
                buy_prices = prices[buy_start:buy_end + 1]
                buy_sum = np.sum(buy_prices) * unit_size
                sell_price = prices[sell_idx]
                
                profit_after_fees = buys * combined_multiplier * sell_price - buy_sum
                gross_profit = buys * buy_unit_multiplier * sell_price - buy_sum
                
                if gross_profit > 0 and p > 0:
                    penalty = (p / 100) * gross_profit
                    net_profit = profit_after_fees - penalty
                else:
                    net_profit = profit_after_fees
                
                if net_profit > max_profit:
                    max_profit = net_profit
                    best_holds1 = holds1
                    best_buys = buys
                    best_holds2 = holds2
    
    return best_holds1, best_buys, best_holds2, max_profit, total_combinations


def find_cases_where_waiting_wins():
    """Find specific scenarios where waiting is optimal"""
    
    print("="*100)
    print("FINDING CASES WHERE WAITING BEATS BUYING FROM START")
    print("="*100)
    
    test_cases = [
        # Recovery doesn't reach early buy prices
        {
            "name": "Deep dip, partial recovery",
            "prices": np.array([100, 90, 80, 30, 35, 40, 45]),
            "r": 0, "p": 0
        },
        {
            "name": "Crash with small bounce",
            "prices": np.array([100, 95, 90, 85, 50, 52, 54, 56]),
            "r": 0, "p": 0
        },
        {
            "name": "Peak then crash (sell at loss if buy early)",
            "prices": np.array([120, 110, 100, 90, 80, 85, 90]),
            "r": 0, "p": 0
        },
        {
            "name": "Slow decline then small recovery",
            "prices": np.array([100, 95, 90, 85, 80, 75, 78, 81]),
            "r": 0, "p": 0
        },
        {
            "name": "Sharp drop, bounce to mid",
            "prices": np.array([100, 80, 60, 40, 20, 40, 60, 70]),
            "r": 0, "p": 0
        }
    ]
    
    results = []
    
    for test in test_cases:
        print(f"\n{'#'*100}")
        print(f"TEST: {test['name']}")
        print(f"{'#'*100}")
        print(f"Prices: {test['prices']}")
        
        # Run brute force
        h1, buys, h2, profit, _ = find_strategy_with_fees_and_penalty_vectorized(
            test['prices'],
            r=test['r'],
            p=test['p'],
            unit_size=1.0
        )
        
        # Calculate buy-from-start strategy
        buy_from_start_profit = 0
        if len(test['prices']) > 1:
            buy_all = test['prices'][:-1]
            sell_at_end = test['prices'][-1]
            buy_from_start_profit = len(buy_all) * sell_at_end - np.sum(buy_all)
        
        print(f"\nResults:")
        print(f"  Optimal strategy: holds1={h1}, buys={buys}, holds2={h2}, profit={profit:.2f}")
        print(f"  Buy from start profit: {buy_from_start_profit:.2f}")
        
        if h1 > 0:
            print(f"\n  ✓ WAITING IS OPTIMAL! (waits {h1} candles)")
            buy_start = h1
            buy_end = h1 + buys - 1
            print(f"    Buys at: {test['prices'][buy_start:buy_end+1]}")
            sell_idx = h1 + buys + h2 - 1
            print(f"    Sells at: {test['prices'][sell_idx]}")
        elif profit > buy_from_start_profit:
            print(f"  ⚠ Interesting: Optimal uses different sell point (not at end)")
            print(f"    Sell index: {h1+buys+h2-1}, price: {test['prices'][h1+buys+h2-1]}")
        else:
            print(f"  ✗ Buying from start is optimal")
        
        results.append({
            'name': test['name'],
            'optimal_waits': h1,
            'profit_optimal': profit,
            'profit_start': buy_from_start_profit
        })
    
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    for r in results:
        status = "✓ WAITS" if r['optimal_waits'] > 0 else "✗ BUYS FROM START"
        print(f"{status:20} | {r['name']:35} | profit: {r['profit_optimal']:.2f}")
    
    return results


def analyze_threshold():
    """Find the threshold where waiting becomes optimal"""
    
    print("\n" + "="*100)
    print("FINDING THE THRESHOLD WHERE WAITING BECOMES OPTIMAL")
    print("="*100)
    
    # Create a price pattern where we can vary the drop depth
    base_prices = np.array([100, 90, 80, 70, 60, 70, 80, 90, 100])
    
    print(f"\nBase pattern: {base_prices}")
    print("Varying the depth of the dip...")
    print()
    
    for drop_depth in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        # Create modified prices with deeper dip
        prices = base_prices.copy()
        dip_index = 4  # Center of dip
        prices[dip_index] = 100 - drop_depth
        
        # Adjust surrounding prices to make dip smooth
        if drop_depth > 10:
            prices[3] = 100 - drop_depth * 0.6
            prices[2] = 100 - drop_depth * 0.3
            prices[5] = 100 - drop_depth * 0.6
            prices[6] = 100 - drop_depth * 0.3
        
        h1, buys, h2, profit, _ = find_strategy_with_fees_and_penalty_vectorized(prices, r=0, p=0, unit_size=1.0)
        
        buy_from_start = len(prices[:-1]) * prices[-1] - np.sum(prices[:-1])
        
        waits = "WAITS" if h1 > 0 else "buys start"
        print(f"Drop to {prices[dip_index]:3.0f} ({drop_depth:2d}%): {waits:12} | optimal profit={profit:6.2f} | buy-start profit={buy_from_start:6.2f}")


if __name__ == "__main__":
    # Find cases where waiting wins
    find_cases_where_waiting_wins()
    
    # Analyze threshold
    analyze_threshold()