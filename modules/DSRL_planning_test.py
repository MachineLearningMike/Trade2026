import numpy as np

prices = [1,2,3,4,5]
prices = np.array(prices)
indexes = [1,2,3]

def skip_buys(prices, buy_indices: list, max_buy_cost=None) -> list:
    new_indices = buy_indices.copy()
    
    if max_buy_cost is not None:
        buy_prices = prices[new_indices]

        while buy_prices.sum() > max_buy_cost:   
            i = np.argmax(buy_prices)
            print(new_indices, buy_prices, i)
            buy_prices = np.concatenate([buy_prices[:i], buy_prices[i+1:]])
            new_indices.pop(i)
            print(new_indices, buy_prices)
            print('----------')
    return new_indices
        
new_indices = skip_buys(prices, indexes, max_buy_cost=5)

print(new_indices)