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


def find_strategic_rewards_with_fees_and_penalty_detailed(prices, r=0, p=0, buys=0, holds=0, net_profit=0):
    """Detailed version with full explanation"""
    n = len(prices)
    
    if buys == 0:
        actions = ['SELLOUT'] if holds == 1 else ['HOLD'] * (holds - 1) + ['SELLOUT']
        return {'rewards': [0.0] * len(actions), 'actions': actions, 'total_profit': 0.0}
    
    if buys + holds > n:
        raise ValueError(f"Not enough price data: need {buys + holds}, have {n}")
    
    fee_rate = r / 100
    units_per_buy = 1 - fee_rate
    sell_fee_factor = 1 - fee_rate
    
    p_after_buy = prices[buys - 1]
    p_sell = prices[buys + holds - 1]
    
    total_units = buys * units_per_buy
    total_cost = sum(prices[:buys])
    
    gross_proceeds = total_units * p_sell
    net_proceeds = gross_proceeds * sell_fee_factor
    final_profit = net_proceeds - total_cost
    
    print("\n" + "=" * 80)
    print("STRATEGIC REWARDS (All actions share profit - positive or negative)")
    print("=" * 80)
    
    print(f"\n📊 INPUT PARAMETERS:")
    print(f"  Prices: {prices}")
    print(f"  Buy fee: {r}% (each BUY gets {units_per_buy:.4f} units)")
    print(f"  Sell fee: {r}% (multiplier: {sell_fee_factor:.4f})")
    print(f"  Strategy: {buys} BUY(s), {holds} HOLD(s) (last HOLD is SELLOUT)")
    
    print(f"\n📊 PRICE POINTS:")
    print(f"  After buys (index {buys-1}): p1 = {p_after_buy:.2f}")
    for i in range(holds - 1):
        hold_idx = buys + i
        print(f"  After HOLD {i+1} (index {hold_idx}): {prices[hold_idx]:.2f}")
    print(f"  SELLOUT price (index {buys+holds-1}): p_sell = {p_sell:.2f}")
    
    print(f"\n💰 TRANSACTION SUMMARY:")
    print(f"  Total units purchased: {buys} × {units_per_buy:.4f} = {total_units:.4f}")
    print(f"  Total cost: {total_cost:.2f}")
    print(f"  Gross proceeds: {total_units:.4f} × {p_sell} = {gross_proceeds:.2f}")
    if fee_rate > 0:
        print(f"  Sell fee ({r}%): {gross_proceeds * fee_rate:.4f}")
        print(f"  Net proceeds: {net_proceeds:.2f}")
    print(f"  Final profit: {final_profit:.4f}")
    
    print(f"\n📈 CONTRIBUTION CALCULATION:")
    print("-" * 60)
    
    contributions = []
    
    # BUY contributions
    print("\n  BUY actions (100% of p1-p_buy + 50% of p_sell-p1):")
    for i in range(buys):
        p_buy = prices[i]
        units = units_per_buy
        
        buy_phase = units * (p_after_buy - p_buy)
        post_buy = units * (p_sell - p_after_buy) * 0.5
        total_contrib = buy_phase + post_buy
        contributions.append(total_contrib)
        
        print(f"    BUY {i+1}: {buy_phase:+.4f} + {post_buy:+.4f} = {total_contrib:+.4f}")
    
    # HOLD contributions
    print("\n  HOLD actions (50% of period's appreciation):")
    current_price = p_after_buy
    for i in range(holds):
        if i < holds - 1:
            next_price = prices[buys + i]
            label = f"HOLD {i+1}"
        else:
            next_price = p_sell
            label = "SELLOUT"
        
        price_change = next_price - current_price
        hold_contrib = total_units * price_change * 0.5
        contributions.append(hold_contrib)
        
        print(f"    {label}: {price_change:+.4f} × {total_units:.4f} × 0.5 = {hold_contrib:+.4f}")
        current_price = next_price
    
    total_contributions = sum(contributions)
    
    print(f"\n📊 CONTRIBUTION SUMMARY:")
    print(f"  Total contributions: {total_contributions:+.4f}")
    print(f"  Final profit: {final_profit:+.4f}")
    
    # Calculate rewards
    if total_contributions != 0:
        rewards = [(contrib / total_contributions) * final_profit for contrib in contributions]
    else:
        rewards = [final_profit / len(contributions)] * len(contributions)
    
    print(f"\n💰 REWARDS CALCULATION (contribution / total_contributions × final_profit):")
    print("-" * 60)
    
    # Display rewards
    reward_idx = 0
    for i in range(buys):
        print(f"  BUY {i+1}: ({contributions[i]:+.4f} / {total_contributions:+.4f}) × {final_profit:.4f} = {rewards[reward_idx]:+.4f}")
        reward_idx += 1
    
    for i in range(holds):
        if i < holds - 1:
            label = f"HOLD {i+1}"
        else:
            label = "SELLOUT"
        print(f"  {label}: ({contributions[reward_idx]:+.4f} / {total_contributions:+.4f}) × {final_profit:.4f} = {rewards[reward_idx]:+.4f}")
        reward_idx += 1
    
    # Summary
    total_rewards = sum(rewards)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("-" * 60)
    print(f"  Total rewards: {total_rewards:+.6f}")
    print(f"  Final profit:  {final_profit:+.6f}")
    print(f"  Match:         {'✓' if abs(total_rewards - final_profit) < 1e-6 else '✗'}")
    
    actions = ['BUY'] * buys + ['HOLD'] * (holds - 1) + ['SELLOUT']
    
    print(f"\n{'='*60}")
    print("FINAL REWARDS PER ACTION")
    print("-" * 60)
    for action, reward in zip(actions, rewards):
        print(f"  {action:8s}: {reward:+.6f}")
    
    return {
        'rewards': rewards,
        'actions': actions,
        'total_profit': final_profit
    }


if __name__ == "__main__":
    # Example 1: Simple profitable case
    prices1 = np.array([10, 15, 20])
    r1, p1 = 0, 0
    buys1, holds1 = 1, 2  # 1 BUY, HOLD1, SELLOUT
    
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Profitable - 1 BUY, 2 HOLDS (No fees)")
    print("=" * 80)
    result1 = find_strategic_rewards_with_fees_and_penalty_detailed(prices1, r1, p1, buys1, holds1)
    
    # Example 2: With fees
    prices2 = np.array([100, 101, 102, 103])
    r2, p2 = 0.5, 0
    buys2, holds2 = 2, 2  # 2 BUYS, HOLD1, SELLOUT
    
    print("\n" + "=" * 80)
    print("EXAMPLE 2: With fees (0.5%) - 2 BUYS, 2 HOLDS")
    print("=" * 80)
    result2 = find_strategic_rewards_with_fees_and_penalty_detailed(prices2, r2, p2, buys2, holds2)
    
    # Example 3: Losing trade (negative profit)
    prices3 = np.array([20, 15, 10])
    r3, p3 = 0, 0
    buys3, holds3 = 1, 2  # 1 BUY, HOLD1, SELLOUT
    
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Losing trade - 1 BUY, 2 HOLDS (No fees)")
    print("=" * 80)
    result3 = find_strategic_rewards_with_fees_and_penalty_detailed(prices3, r3, p3, buys3, holds3)