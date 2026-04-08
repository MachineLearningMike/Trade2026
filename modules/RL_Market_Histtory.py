import math
import numpy as np


class Market_Histtory():
    def __self__(self, dim_compact_history=300, indicator_window=30):

        nodes = [ ((k) * math.log(k) - k) / math.log(2) for k in range(1, dim_compact_history + 4) ]
        compress_kernel = [((nodes[i-1]+nodes[i])/2, (nodes[i]+nodes[i+1])/2) for i in range(1, dim_compact_history + 2)]
        assert len(compress_kernel) == dim_compact_history + 1
        min_len_to_compress = math.ceil(compress_kernel[-1][1]) + 1

        indicator_window = 30           # we look lookback as far as 'lookback'
        min_len_to_lookback = max(3 * indicator_window - 2, min_len_to_compress)     # dim_price_state: tight
    
        self.compress_kernel = compress_kernel
        self.min_len_to_compress = min_len_to_compress
        self.min_len_to_lookback = min_len_to_lookback


    def compress_sequence(self, array, bidir=False): # bidir = False is better.
        compressed = [] 
        for (a, b) in self.compress_kernel:
            assert a < b
            if a < 0 and b < 0: continue
            if a < 0: a = -0.99999
            assert a < b
            
            if math.ceil(b) >= array.shape[0]: break
            
            s = 0
            assert math.floor(a) == math.ceil(a) - 1
            s = s + array[math.ceil(a)] * (math.ceil(a)-a)      # Lose (a - math.floor(a)) part of array[math.ceil(a)]
            s = s + array[math.ceil(b)] * (b-math.floor(b))     # Gain (b - math.floor(a)) part of array[math.ceil(b)]
            for i in range(math.ceil(a) + 1, math.floor(b)+1): s += array[i]
                
            if bidir:
                s = s + array[math.floor(a)] * (math.ceil(a)-a)      # Lose (a - math.floor(a)) part of array[math.floor(a)]
                s = s + array[math.floor(b)] * (b-math.floor(b))     # Gain (b - math.floor(a)) part of array[math.floor(b)]
                for i in range(math.ceil(a), math.floor(b)): s += array[i]
                    
            if bidir: s /= 2
                            
            compressed.append(s/(b-a))
            
        return np.array(compressed)


    def compress_historical_sequence(self, arr):
        assert len(arr) >= self.min_len_to_compress, f"len(arr) must not be less that {math.ceil(self.compress_kernel[-1][1]) + 1}"
        arr = arr[:self.min_len_to_compress]
        # arr_reversed = arr[::-1]
        arr_reversed_compressed = self.compress_sequence(arr, bidir=False)
        # print(f"len(arr_reversed_compressed) {len(arr_reversed_compressed)}, len(compress_kernel) {len(compress_kernel)}")
        assert len(arr_reversed_compressed) == len(self.compress_kernel) - 1
        # arr_compressed = arr_reversed_compressed[::-1]
        return arr_reversed_compressed
    

    def compress_hitorical_sequence_preserve_last(self, arr):
        last = arr[-1]
        arr_compressed = self.compress_historical_sequence(arr)
        arr_compressed[-1] = last
        return arr_compressed


    def compress_history(self, normal_prices, normal_volumes, true_prices, running_anchor):
        prices = normal_prices[running_anchor - self.min_len_to_lookback : running_anchor + 1]
        prices = self.compress_hitorical_sequence_preserve_last(prices)
        volumes = normal_volumes[running_anchor - self.min_len_to_lookback : running_anchor + 1]
        volumes = self.compress_hitorical_sequence_preserve_last(volumes)
        price_curr = true_prices[running_anchor]
        return prices, volumes, price_curr