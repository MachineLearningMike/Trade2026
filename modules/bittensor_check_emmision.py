import modules.bittensor_check_emmision as bt

# 1. Initialize the subtensor connection (defaults to the Finney mainnet)
subtensor = bt.subtensor(network="finney")

# 2. Retrieve information for all subnets on the network
all_subnets = subtensor.get_all_subnets_info()

# 3. Print the results
print(f"{'NETUID':<8} {'EMISSION (%)':<15} {'OWNER'}")
print("-" * 60)

for subnet in all_subnets:
    # Convert emission from raw value to a readable percentage
    emission_pct = float(subnet.emission) / 1e9 * 100 
    print(f"{subnet.netuid:<8} {emission_pct:<15.2f} {subnet.owner_ss58}")


# Sync the metagraph for Subnet 18
metagraph = subtensor.metagraph(netuid=18)

# Count neurons that are currently active miners 
# (Total neurons minus those with significant validator stake/permit)
miner_count = len([n for n in metagraph.neurons if not n.validator_permit])

print(f"Total active miners on Subnet 18: {miner_count}")


import bittensor as bt

# 1. Setup connection
subtensor = bt.subtensor(network="finney")
netuid = 18

# 2. Get Subnet emission data
subnet_info = subtensor.get_subnet_info(netuid)
# Total daily TAO for the whole network is 3,600
# Calculate Subnet 18's total daily TAO
total_daily_tao = 3600 * (float(subnet_info.emission) / 1e9 / 1) # Normalizing raw emission
# Miners get 41% of that
total_miner_daily_tao = total_daily_tao * 0.41

# 3. Get Metagraph to see individual UID performance
metagraph = subtensor.metagraph(netuid)

# 4. Filter for miners (those without a validator permit) and sort by incentive
miners = [n for n in metagraph.neurons if not n.validator_permit]
top_5_miners = sorted(miners, key=lambda x: x.incentive, reverse=True)[:5]

print(f"--- Subnet 18 Top 5 Miner Daily Earnings ---")
print(f"Subnet Daily Miner Pool: {total_miner_daily_tao:.4f} TAO\n")
print(f"{'UID':<10} {'Incentive (%)':<20} {'Daily TAO'}")
print("-" * 45)

for miner in top_5_miners:
    # Incentive is a float representing the fraction of the miner pool
    daily_earnings = miner.incentive * total_miner_daily_tao
    print(f"{miner.uid:<10} {miner.incentive * 100:<20.2f} {daily_earnings:.4f} TAO")

import bittensor as bt

subtensor = bt.subtensor(network="finney")
netuid = 18

# Fetch current registration cost in TAO
burn_cost = subtensor.get_burn_cost(netuid=netuid)
print(f"Current Registration Burn for Subnet 18: {burn_cost / 1e9:.4f} TAO")


# Create a Balance object for 1 Alpha to check its value
one_alpha = bt.Balance.from_tao(1.0).set_unit(netuid)

# Fetch exchange rate (with slippage accounted for)
# Note: You can also use subtensor.get_alpha_price(netuid) in newer SDK versions
subnet = subtensor.subnet(netuid=netuid)
tao_value = subnet.alpha_to_tao_with_slippage(one_alpha)

print(f"Current Exchange Rate: 1 Alpha = {tao_value.tao:.6f} TAO")


import bittensor as bt

# Connect to the Finney mainnet
subtensor = bt.subtensor(network="finney")
netuid = 18

# Get real-time subnet info
info = subtensor.get_subnet_info(netuid)

# 1. Check Total Liquidity (Sum of all past flows)
print(f"Total TAO Reserve (Liquidity): {info.tao_in / 1e9:.2f} TAO")

# 2. Check the Resulting Emission Share
# This is the % of the 3,600 daily TAO currently directed here by the flow
emission_share = (float(info.emission) / 1e9) / 0.5 * 100 # Normalized to block rewards
print(f"Current Daily Emission Share: {emission_share:.2f}%")

# 3. Check for 'Negative Flow' Risk
# If this number drops significantly over 24 hours, the subnet is losing momentum
print(f"Current alpha price (market confidence): {info.price:.6f} TAO")


import bittensor as bt

subtensor = bt.subtensor(network="finney")
netuid = 18

# 1. Get current pool reserves
subnet = subtensor.get_subnet_info(netuid)
tao_reserve = float(subnet.tao_in) / 1e9
alpha_reserve = float(subnet.alpha_in) / 1e9

# 2. Current Price (TAO per Alpha)
current_price = tao_reserve / alpha_reserve

# 3. Calculate Dilution (Assuming 0 TAO emission and 7,200 Alpha injected daily)
# The protocol injects 1 Alpha per block into the pool (7,200/day)
new_alpha_reserve = alpha_reserve + 7200
new_price = tao_reserve / new_alpha_reserve

dilution_percentage = ((current_price - new_price) / current_price) * 100

print(f"--- Subnet 18 Dilution Analysis ---")
print(f"Current Price: {current_price:.6f} TAO")
print(f"Projected Price (24h): {new_price:.6f} TAO")
print(f"Daily Dilution Rate: {dilution_percentage:.2f}%")


import bittensor as bt
import time

subtensor = bt.subtensor(network="finney")
netuid = 18

def get_pool_status():
    # Sync with the latest block state
    s_info = subtensor.get_subnet_info(netuid)
    return {
        "alpha_in": float(s_info.alpha_in) / 1e9,
        "total_supply": float(s_info.total_supply) / 1e9,
        "emission": float(s_info.emission) / 1e9
    }

# 1. Capture initial state
print("Checking status... (Waiting 15 seconds for a block)")
initial = get_pool_status()

# 2. Wait for one block to be processed
time.sleep(15)

# 3. Capture new state
current = get_pool_status()

# 4. Compare results
alpha_in_delta = current["alpha_in"] - initial["alpha_in"]
supply_delta = current["total_supply"] - initial["total_supply"]

print(f"\n--- Subnet 18 Protocol Injection Check ---")
print(f"Current TAO Emission: {current['emission']:.6f} TAO/block")
print(f"Change in Pool Alpha (alpha_in): {alpha_in_delta:.9f} α")
print(f"Change in Total Alpha Supply:    {supply_delta:.9f} α")

if current['emission'] == 0 and alpha_in_delta == 0:
    print("\n✅ CONFIRMED: Protocol has HALTED pool injections to preserve price.")
    if supply_delta > 0:
        print("💡 NOTE: New Alpha was minted to participants, but none was added to the pool.")


import bittensor as bt

# 1. Connect to the Finney mainnet
subtensor = bt.subtensor(network="finney")
netuid = 18

# 2. Get Metagraph for Subnet 18
metagraph = subtensor.metagraph(netuid=netuid)

# 3. Daily Alpha miner pool (41% of 7,200 daily participant mint)
DAILY_MINER_ALPHA = 7200 * 0.41

# 4. Sort neurons by incentive and filter for top 6 winners
# Incentive is the fraction (0.0 to 1.0) of the miner pool
top_winners = sorted(metagraph.neurons, key=lambda x: x.incentive, reverse=True)[:6]

print(f"--- Subnet 18: Top 6 Winners Analysis ---")
print(f"{'UID':<8} {'Incentive (%)':<15} {'Daily Alpha':<15} {'Trust'}")
print("-" * 55)

for miner in top_winners:
    daily_alpha = miner.incentive * DAILY_MINER_ALPHA
    print(f"{miner.uid:<8} {miner.incentive * 100:<15.2f} {daily_alpha:<15.2f} {miner.trust:.4f}")


import bittensor as bt

subtensor = bt.subtensor(network="finney")
metagraph = subtensor.metagraph(netuid=18)

# Sort all miners by their Incentive score
# In the winner-takes-all model, UIDs 7+ should have near-zero incentive
top_miners = sorted(metagraph.neurons, key=lambda x: x.incentive, reverse=True)[:6]

print(f"{'UID':<10} {'Incentive (%)':<15} {'Trust Score'}")
print("-" * 40)
for miner in top_miners:
    print(f"{miner.uid:<10} {miner.incentive * 100:<15.2f} {miner.trust:.4f}")
