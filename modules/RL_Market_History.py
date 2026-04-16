import math
import numpy as np
import sys


class Market_History:
      
    def __init__(self, dim_compact_history=300, indicator_window=30):

        nodes = [ ((k) * math.log(k) - k) / math.log(2) for k in range(1, dim_compact_history + 4) ]
        compress_kernel = [((nodes[i-1]+nodes[i])/2, (nodes[i]+nodes[i+1])/2) for i in range(1, dim_compact_history + 2)]
        assert len(compress_kernel) == dim_compact_history + 1
        len_to_compress = math.ceil(compress_kernel[-1][1]) + 1

        indicator_window = 30           # we look lookback as far as 'lookback'
        len_to_lookback = max(3 * indicator_window - 2, len_to_compress)     # dim_price_state: tight
    
        self.compress_kernel = compress_kernel
        self._len_to_compress = len_to_compress
        self._len_to_lookback = len_to_lookback
        
    @property
    def len_to_compress(self):
        return self._len_to_compress
    
    @property
    def len_to_lookback(self):
        return self._len_to_lookback

    def abs_normalize(self, arr):
        return arr / (arr.std() + 1e-9)

    def compress_sequence(self, array, bidir=False): # bidir = False is better. Starts from zero idx.
        compressed, start, end = [], int(sys.maxsize), -int(sys.maxsize)
        for (a, b) in self.compress_kernel: # Starts from zero idx, aligning zero-th kernel with zero-th array element.
            assert a < b
            if a < 0 and b < 0: continue
            if a < 0: a = -0.99999
            assert a < b
            
            if math.ceil(b) >= array.shape[0]: break
            
            s = 0
            assert math.floor(a) == math.ceil(a) - 1
            ceil_a = math.ceil(a); ceil_b = math.ceil(b)
            if start > ceil_a: start = ceil_a
            if end < ceil_b + 1: end = ceil_b + 1
            s = s + array[ceil_a] * (ceil_a-a)      # Lose (a - math.floor(a)) part of array[math.ceil(a)]
            s = s + array[ceil_b] * (b-math.floor(b))     # Gain (b - math.floor(a)) part of array[math.ceil(b)]
            for i in range(ceil_a + 1, math.floor(b)+1): s += array[i]
                
            if bidir:
                floor_a = math.floor(a); floor_b = math.floor(b)
                if start > floor_a: start = floor_a
                if end < floor_b: end = floor_b + 1
                s = s + array[floor_a] * (math.ceil(a)-a)      # Lose (a - math.floor(a)) part of array[math.floor(a)]
                s = s + array[floor_b] * (b-math.floor(b))     # Gain (b - math.floor(a)) part of array[math.floor(b)]
                for i in range(math.ceil(a), floor_b): s += array[i]
                    
            if bidir: s /= 2
                            
            compressed.append(s/(b-a))
            
        return np.array(compressed), start, end


    def compress_historical_sequence(self, arr, faithful_backward=True):
        assert len(arr) == self.len_to_compress
        
        arr_copy = arr[::-1] if faithful_backward else arr
        
        arr_copy, start, end = self.compress_sequence(arr_copy, bidir=False)

        assert len(arr_copy) == len(self.compress_kernel) - 1
        assert end - start == self.len_to_compress
            
        arr_copy = arr_copy[::-1] if faithful_backward else arr_copy

        if faithful_backward:
            start, end = len(arr) - end, len(arr) - start

        return arr_copy, start, end
    

    def compress_hitorical_sequence_preserve_last(self, arr, use_backside=True, faithful_backward=True):
        assert len(arr) >= self.len_to_compress     # check it

        arr_copy = arr[-self.len_to_compress:] if use_backside else arr[:self.len_to_compress]

        save = arr_copy[-1] if faithful_backward else arr_copy[0]
        arr_copy, start, end = self.compress_historical_sequence(arr_copy, faithful_backward=faithful_backward)
        if faithful_backward: arr_copy[-1] = save
        else: arr_copy[0] = save

        if use_backside:
            start, end = start + len(arr) - self.len_to_compress, end + len(arr) - self.len_to_compress

        return arr_copy, start, end


    # def compress_history(self, prices, volumes, raw_prices, anchor, use_backside=True, faithful_backward=True):
    #     prices = prices[anchor - self.len_to_lookback : anchor + 1]
    #     c_prices, start, end = self.compress_hitorical_sequence_preserve_last(prices, use_backside=use_backside, faithful_backward=faithful_backward)
    #     volumes = volumes[anchor - self.len_to_lookback : anchor + 1]
    #     c_volumes, start_v, end_v = self.compress_hitorical_sequence_preserve_last(volumes, use_backside=use_backside, faithful_backward=faithful_backward)
    #     assert start == start_v and end == end_v
    #     r_prices = raw_prices[anchor - self.len_to_lookback : anchor + 1]
    #     price_curr = raw_prices[anchor]
    #     return c_prices, c_volumes, r_prices, price_curr, start, end
    
    def compress_history(self, history, anchor, use_backside=True, faithful_backward=True):
        raw_prices, raw_volumes, marks, price_mean, volue_mean = history
        assert anchor >= self.len_to_lookback
        r_prices = raw_prices[anchor - self.len_to_lookback : anchor + 1]
        r_volumes = raw_volumes[anchor - self.len_to_lookback : anchor + 1]
        
        compact_prices, start, end = self.compress_hitorical_sequence_preserve_last(r_prices, use_backside=use_backside, faithful_backward=faithful_backward)
        compact_volumes, start_v, end_v = self.compress_hitorical_sequence_preserve_last(r_volumes, use_backside=use_backside, faithful_backward=faithful_backward)
        assert start == start_v and end == end_v
        
        r_marks = marks[anchor - self.len_to_lookback : anchor + 1]
        c_history = (compact_prices, compact_volumes, r_prices, r_volumes, r_marks, price_mean, volue_mean)
        
        return c_history, start, end
    
if __name__ == "__main__":
    mh = Market_History(dim_compact_history=300, indicator_window=30)
 