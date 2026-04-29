import numpy as np

def find_strategic_rewards_with_fees_and_penalty(prices, r=0, p=0, buys=0, holds=0, net_profit=0):
    """
    Calculate reward for each action.
    
    All actions (BUY and HOLD including SELLOUT) share the final profit.
    Rewards can be positive or negative based on contribution.
    holds INCLUDES SELLOUT as the final HOLD action.
    
    Attribution:
    - Each BUY action contributes: units * (p_after_buy - p_buy) + units * (p_sell - p_after_buy) * 0.5
    - Each HOLD action contributes: total_units * price_change * 0.5
    - Rewards = (contribution / total_contributions) * final_profit
    
    Fees:
    - Buy fee: reduces units (units_per_buy = 1 - r/100)
    - Sell fee: reduces final profit (already accounted)
    """
    n = len(prices)
    
    if buys == 0:
        return [0.0] * holds
    
    if buys + holds > n:
        raise ValueError(f"Not enough price data: need {buys + holds}, have {n}")
    
    fee_rate = r / 100
    units_per_buy = 1 - fee_rate
    sell_fee_factor = 1 - fee_rate
    
    # Key price points
    p_after_buy = prices[buys - 1]
    p_sell = prices[buys + holds - 1]
    
    total_units = buys * units_per_buy
    total_cost = sum(prices[:buys])
    
    # Net proceeds and final profit
    gross_proceeds = total_units * p_sell
    net_proceeds = gross_proceeds * sell_fee_factor
    final_profit = net_proceeds - total_cost
    
    # Calculate contributions
    contributions = []
    
    # BUY contributions
    for i in range(buys):
        p_buy = prices[i]
        units = units_per_buy
        
        buy_phase = units * (p_after_buy - p_buy)
        post_buy = units * (p_sell - p_after_buy) * 0.5
        total_contrib = buy_phase + post_buy
        contributions.append(total_contrib)
    
    # HOLD contributions
    current_price = p_after_buy
    for i in range(holds):
        if i < holds - 1:
            next_price = prices[buys + i]
        else:
            next_price = p_sell
        
        price_change = next_price - current_price
        hold_contrib = total_units * price_change * 0.5
        contributions.append(hold_contrib)
        current_price = next_price
    
    # Distribute final profit proportionally to contributions
    total_contributions = sum(contributions)
    
    if total_contributions != 0:
        rewards = [(contrib / total_contributions) * final_profit for contrib in contributions]
    else:
        # Equal distribution if no contributions
        rewards = [final_profit / len(contributions)] * len(contributions)
    
    return rewards


def run_comprehensive_tests():
    """Run comprehensive test cases"""
    
    test_cases = []
    
    # ============================================
    # BASIC NO-FEE CASES
    # ============================================
    
    # Test 1: Single buy, immediate sell (no holds)
    test_cases.append({
        "name": "1 BUY, 1 HOLD (SELLOUT) - Profit",
        "prices": np.array([10, 20]),
        "r": 0,
        "buys": 1,
        "holds": 1,
        "expected_profit": 10.0
    })
    
    # Test 2: Single buy, one hold, then sell
    test_cases.append({
        "name": "1 BUY, 2 HOLDS - Profit",
        "prices": np.array([10, 15, 20]),
        "r": 0,
        "buys": 1,
        "holds": 2,
        "expected_profit": 10.0
    })
    
    # Test 3: Two buys, immediate sell
    test_cases.append({
        "name": "2 BUYS, 1 HOLD (SELLOUT) - Profit",
        "prices": np.array([10, 12, 20]),
        "r": 0,
        "buys": 2,
        "holds": 1,
        "expected_profit": 18.0  # (20-10)+(20-12)=18
    })
    
    # Test 4: Two buys with holds
    test_cases.append({
        "name": "2 BUYS, 2 HOLDS - Profit",
        "prices": np.array([10, 12, 15, 20]),
        "r": 0,
        "buys": 2,
        "holds": 2,
        "expected_profit": 18.0
    })
    
    # Test 5: Losing trade
    test_cases.append({
        "name": "1 BUY, 2 HOLDS - Loss",
        "prices": np.array([20, 15, 10]),
        "r": 0,
        "buys": 1,
        "holds": 2,
        "expected_profit": -10.0
    })
    
    # ============================================
    # FEE CASES
    # ============================================
    
    # Test 6: With buy and sell fees
    test_cases.append({
        "name": "With 0.5% fees - Profit",
        "prices": np.array([100, 101, 102, 103]),
        "r": 0.5,
        "buys": 2,
        "holds": 2,
        "expected_profit": None  # Will calculate
    })
    
    # Test 7: Higher fees makes trade unprofitable
    test_cases.append({
        "name": "With 5% fees - Small loss",
        "prices": np.array([100, 101, 102]),
        "r": 5,
        "buys": 1,
        "holds": 2,
        "expected_profit": None
    })
    
    # ============================================
    # COMPLEX PRICE MOVEMENTS
    # ============================================
    
    # Test 8: Volatile with recovery
    test_cases.append({
        "name": "Volatile - Recovery to profit",
        "prices": np.array([100, 95, 90, 105, 110]),
        "r": 0,
        "buys": 2,
        "holds": 3,
        "expected_profit": None
    })
    
    # Test 9: Multiple buys at different prices
    test_cases.append({
        "name": "3 BUYS, 3 HOLDS - Staircase up",
        "prices": np.array([10, 12, 14, 16, 18, 20]),
        "r": 0,
        "buys": 3,
        "holds": 3,
        "expected_profit": 24.0
    })
    
    # Test 10: Peak then drop
    test_cases.append({
        "name": "Peak then drop - Loss",
        "prices": np.array([10, 12, 15, 20, 18, 15]),
        "r": 0,
        "buys": 2,
        "holds": 3,
        "expected_profit": None
    })
    
    # ============================================
    # EDGE CASES
    # ============================================
    
    # Test 11: Flat prices
    test_cases.append({
        "name": "Flat prices - No profit/loss",
        "prices": np.array([100, 100, 100, 100]),
        "r": 0,
        "buys": 2,
        "holds": 2,
        "expected_profit": 0.0
    })
    
    # Test 12: Single buy, no holds, flat price
    test_cases.append({
        "name": "Flat price - No profit/loss",
        "prices": np.array([100, 100]),
        "r": 0,
        "buys": 1,
        "holds": 1,
        "expected_profit": 0.0
    })
    
    # Test 13: No buys edge case
    test_cases.append({
        "name": "No buys - Only holds",
        "prices": np.array([10, 15, 20]),
        "r": 0,
        "buys": 0,
        "holds": 2,
        "expected_profit": 0.0
    })
    
    # Test 14: Very small price movement with fees
    test_cases.append({
        "name": "Small profit eaten by fees",
        "prices": np.array([100, 100.5, 101]),
        "r": 1,
        "buys": 1,
        "holds": 2,
        "expected_profit": None
    })
    
    # Test 15: Large price movement with small fees
    test_cases.append({
        "name": "Large profit with small fees",
        "prices": np.array([10, 15, 20, 25, 30, 35, 40]),
        "r": 0.1,
        "buys": 3,
        "holds": 4,
        "expected_profit": None
    })
    
    # ============================================
    # RUN TESTS
    # ============================================
    
    print("\n" + "=" * 100)
    print("COMPREHENSIVE TEST SUITE")
    print("=" * 100)
    
    passed = 0
    failed = 0
    
    for i, test in enumerate(test_cases, 1):
        prices = test["prices"]
        r = test["r"]
        buys = test["buys"]
        holds = test["holds"]
        
        # Calculate expected profit if not provided
        if test["expected_profit"] is None:
            fee_rate = r / 100
            units_per_buy = 1 - fee_rate
            total_units = buys * units_per_buy
            total_cost = sum(prices[:buys]) if buys > 0 else 0
            p_sell = prices[buys + holds - 1] if buys + holds <= len(prices) and buys + holds - 1 >= 0 else 0
            gross_proceeds = total_units * p_sell
            expected_profit = gross_proceeds * (1 - fee_rate) - total_cost
        else:
            expected_profit = test["expected_profit"]
        
        # Calculate rewards
        rewards = find_strategic_rewards_with_fees_and_penalty(prices, r, 0, buys, holds, 0)
        total_rewards = sum(rewards)
        
        # Determine action names for display
        actions = ['BUY'] * buys + ['HOLD'] * (holds - 1) + ['SELLOUT'] if holds > 0 else []
        
        # Check if matches expected
        is_match = abs(total_rewards - expected_profit) < 0.0001
        if is_match and expected_profit == 0 and abs(total_rewards) < 0.0001:
            is_match = True
        
        if is_match:
            passed += 1
            status = "✓ PASS"
        else:
            failed += 1
            status = "✗ FAIL"
        
        print(f"\n{'='*100}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*100}")
        print(f"  Prices: {prices}")
        print(f"  Strategy: {buys} BUY(s), {holds} HOLD(s) (last is SELLOUT)")
        print(f"  Fee: {r}%")
        print(f"  Expected profit: {expected_profit:.6f}")
        print(f"  Total rewards:   {total_rewards:.6f}")
        print(f"  Status: {status}")
        
        if not is_match:
            print(f"\n  WARNING: Rewards do not sum to expected profit!")
            print(f"  Difference: {total_rewards - expected_profit:.6f}")
        
        # Show individual rewards for interesting cases
        if i in [1, 2, 3, 6, 8, 9, 15] or not is_match:
            print(f"\n  Individual rewards:")
            for j, (action, reward) in enumerate(zip(actions, rewards)):
                print(f"    {action}: {reward:+.6f}")
            
            # Show contribution breakdown for failed cases
            if not is_match:
                print(f"\n  Debug info:")
                fee_rate = r / 100
                units_per_buy = 1 - fee_rate
                total_units = buys * units_per_buy
                total_cost = sum(prices[:buys]) if buys > 0 else 0
                p_after_buy = prices[buys - 1] if buys > 0 else 0
                p_sell = prices[buys + holds - 1] if buys + holds <= len(prices) and buys + holds - 1 >= 0 else 0
                print(f"    p_after_buy: {p_after_buy:.2f}, p_sell: {p_sell:.2f}")
                print(f"    total_units: {total_units:.4f}, total_cost: {total_cost:.2f}")
                print(f"    gross_proceeds: {total_units * p_sell:.2f}")
                print(f"    net_proceeds: {total_units * p_sell * (1 - fee_rate):.2f}")
                print(f"    final_profit: {total_units * p_sell * (1 - fee_rate) - total_cost:.6f}")
    
    # Summary
    print("\n" + "=" * 100)
    print("TEST SUMMARY")
    print("=" * 100)
    print(f"  Total tests: {passed + failed}")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {passed/(passed+failed)*100:.1f}%")
    print("=" * 100)
    
    return passed, failed


def run_custom_test(prices, r, buys, holds):
    """Run a single custom test with detailed output"""
    print("\n" + "=" * 80)
    print("CUSTOM TEST")
    print("=" * 80)
    print(f"  Prices: {prices}")
    print(f"  Strategy: {buys} BUY(s), {holds} HOLD(s) (last is SELLOUT)")
    print(f"  Fee: {r}%")
    
    rewards = find_strategic_rewards_with_fees_and_penalty(prices, r, 0, buys, holds, 0)
    
    fee_rate = r / 100
    units_per_buy = 1 - fee_rate
    total_units = buys * units_per_buy
    total_cost = sum(prices[:buys]) if buys > 0 else 0
    p_sell = prices[buys + holds - 1]
    net_proceeds = total_units * p_sell * (1 - fee_rate)
    final_profit = net_proceeds - total_cost
    
    actions = ['BUY'] * buys + ['HOLD'] * (holds - 1) + ['SELLOUT']
    
    print(f"\n  Transaction Summary:")
    print(f"    Total units: {total_units:.4f}")
    print(f"    Total cost: {total_cost:.2f}")
    print(f"    Net proceeds: {net_proceeds:.2f}")
    print(f"    Final profit: {final_profit:.4f}")
    
    print(f"\n  Individual Rewards:")
    for action, reward in zip(actions, rewards):
        print(f"    {action}: {reward:+.6f}")
    
    print(f"\n  Total: {sum(rewards):+.6f}")
    print(f"  Expected: {final_profit:+.6f}")
    print(f"  Match: {'✓' if abs(sum(rewards) - final_profit) < 0.0001 else '✗'}")
    
    return rewards


if __name__ == "__main__":
    # Run comprehensive tests
    passed, failed = run_comprehensive_tests()
    
    # Run a few custom tests
    print("\n" + "=" * 100)
    print("ADDITIONAL CUSTOM TESTS")
    print("=" * 100)
    
    # Test: DCA (Dollar Cost Averaging) during dip
    run_custom_test(np.array([100, 80, 90, 120]), 0, 3, 1)
    
    # Test: Buy at top, hold through dip, recover partially
    run_custom_test(np.array([100, 90, 80, 70, 85]), 0, 2, 3)
    
    # Test: High frequency buys and holds with fees
    run_custom_test(np.array([50, 52, 51, 53, 55, 54, 56, 58, 60]), 0.2, 4, 5)
    
    # Test: Extreme fees making everything negative
    run_custom_test(np.array([100, 101, 102]), 10, 2, 1)