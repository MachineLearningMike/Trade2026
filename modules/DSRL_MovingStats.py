import numpy as np
import json

class MovingStats:
    """
    A class to track moving average and moving standard deviation of streaming price data.
    Data is fed one point at a time.
    
    Attributes:
        window (int): Size of the moving window
    """
    
    def __init__(self, window):
        """
        Initialize MovingStats with window size.
        
        Args:
            window (int): Size of the moving window
        """
        self.window = window
        self.reset()
        
    def reset(self):
        """
        Reset all internal state to initial values.
        """
        self.buffer = np.zeros(self.window)
        self.index = 0
        self.count = 0
        self.sum_val = 0.0
        self.sum_sq = 0.0
        
    def update(self, price):
        """
        Add a new price value and update moving statistics.
        
        Args:
            price (float): New price value
            
        Returns:
            tuple: (moving_average, moving_std) or (None, None) if insufficient data
        """
        # Remove oldest value if buffer is full
        if self.count == self.window:
            old_price = self.buffer[self.index]
            self.sum_val -= old_price
            self.sum_sq -= old_price ** 2
        else:
            self.count += 1
        
        # Add new price
        self.buffer[self.index] = price
        self.sum_val += price
        self.sum_sq += price ** 2
        
        # Update circular buffer index
        self.index = (self.index + 1) % self.window
        
        # Calculate and return stats
        return self.get_current_stats()
    
    def get_current_stats(self):
        """
        Get current moving average and standard deviation without updating.
        
        Returns:
            tuple: (moving_average, moving_std) or (None, None) if insufficient data
        """
        if self.count == 0:
            return None, None
        
        mean = self.sum_val / self.count
        
        if self.count >= 2:
            variance = (self.sum_sq / self.count) - (mean ** 2)
            std = np.sqrt(max(variance, 0))
        else:
            std = 0.0
            
        return mean, std
    
    def get_moving_average(self):
        """
        Get current moving average only.
        
        Returns:
            float or None: Current moving average or None if insufficient data
        """
        mean, _ = self.get_current_stats()
        return mean
    
    def get_moving_std(self):
        """
        Get current moving standard deviation only.
        
        Returns:
            float or None: Current moving standard deviation or None if insufficient data
        """
        _, std = self.get_current_stats()
        return std
    
    def get_bollinger_bands(self, num_std=2):
        """
        Calculate Bollinger Bands (moving average ± N * standard deviation).
        
        Args:
            num_std (float): Number of standard deviations for the bands
            
        Returns:
            tuple: (upper_band, lower_band) or (None, None) if insufficient data
        """
        mean, std = self.get_current_stats()
        
        if mean is None or std is None:
            return None, None
        
        upper_band = mean + (num_std * std)
        lower_band = mean - (num_std * std)
        
        return upper_band, lower_band
    
    def is_ready(self):
        """
        Check if enough data has been collected.
        
        Returns:
            bool: True if at least 2 data points have been added
        """
        return self.count >= 2
    
    def get_window_fill_percentage(self):
        """
        Get how full the current window is.
        
        Returns:
            float: Percentage of window filled (0.0 to 1.0)
        """
        return self.count / self.window
    
    def save_state(self, filepath):
        """
        Save the current state to a file.
        
        Args:
            filepath (str): Path to save the state file
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            state = {
                'window': self.window,
                'buffer': self.buffer.tolist(),
                'index': self.index,
                'count': self.count,
                'sum_val': float(self.sum_val),
                'sum_sq': float(self.sum_sq)
            }
            
            with open(filepath, 'w') as f:
                json.dump(state, f, indent=2)
            return True
            
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state(self, filepath):
        """
        Load a saved state from a file.
        
        Args:
            filepath (str): Path to the saved state file
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            # Validate window size matches
            if state['window'] != self.window:
                print(f"Warning: Loaded window size ({state['window']}) differs from current window ({self.window})")
                return False
            
            # Restore state
            self.buffer = np.array(state['buffer'])
            self.index = state['index']
            self.count = state['count']
            self.sum_val = state['sum_val']
            self.sum_sq = state['sum_sq']
            
            return True
            
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return False
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            return False
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def save_state_numpy(self, filepath):
        """
        Save the current state to a .npz file (NumPy format, more efficient).
        
        Args:
            filepath (str): Path to save the state file (should end with .npz)
            
        Returns:
            bool: True if save was successful, False otherwise
        """
        try:
            np.savez(filepath,
                     window=self.window,
                     buffer=self.buffer,
                     index=self.index,
                     count=self.count,
                     sum_val=self.sum_val,
                     sum_sq=self.sum_sq)
            return True
            
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
    
    def load_state_numpy(self, filepath):
        """
        Load a saved state from a .npz file (NumPy format).
        
        Args:
            filepath (str): Path to the saved state file
            
        Returns:
            bool: True if load was successful, False otherwise
        """
        try:
            with np.load(filepath) as data:
                # Validate window size matches
                if data['window'] != self.window:
                    print(f"Warning: Loaded window size ({data['window']}) differs from current window ({self.window})")
                    return False
                
                # Restore state
                self.buffer = data['buffer']
                self.index = int(data['index'])
                self.count = int(data['count'])
                self.sum_val = float(data['sum_val'])
                self.sum_sq = float(data['sum_sq'])
                
            return True
            
        except FileNotFoundError:
            print(f"Error: File {filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading state: {e}")
            return False
    
    def get_state_dict(self):
        """
        Get the current state as a dictionary (useful for manual save/export).
        
        Returns:
            dict: Current state
        """
        return {
            'window': self.window,
            'buffer': self.buffer.tolist(),
            'index': self.index,
            'count': self.count,
            'sum_val': float(self.sum_val),
            'sum_sq': float(self.sum_sq)
        }
    
    def set_state_dict(self, state):
        """
        Restore state from a dictionary.
        
        Args:
            state (dict): State dictionary (from get_state_dict)
            
        Returns:
            bool: True if restore was successful, False otherwise
        """
        try:
            if state['window'] != self.window:
                print(f"Warning: State window size ({state['window']}) differs from current window ({self.window})")
                return False
            
            self.buffer = np.array(state['buffer'])
            self.index = state['index']
            self.count = state['count']
            self.sum_val = state['sum_val']
            self.sum_sq = state['sum_sq']
            return True
            
        except Exception as e:
            print(f"Error restoring state from dict: {e}")
            return False


# Example usage
if __name__ == "__main__":
    # Create tracker with window size 3
    tracker = MovingStats(window=3)
    
    # Feed some data
    prices = [100, 101, 102, 101, 100]
    
    print("=== Tracking Data ===")
    for i, price in enumerate(prices):
        ma, std = tracker.update(price)
        if ma is not None:
            print(f"Step {i+1}: Price={price:3d} | MA={ma:.2f} | STD={std:.2f}")
        else:
            print(f"Step {i+1}: Price={price:3d} | Collecting... ({tracker.count}/{tracker.window})")
    
    # Save state to JSON
    tracker.save_state('moving_stats_state.json')
    print("\n✓ State saved to 'moving_stats_state.json'")
    
    # Save state to NumPy format
    tracker.save_state_numpy('moving_stats_state.npz')
    print("✓ State saved to 'moving_stats_state.npz'")
    
    # Get state as dict
    state_dict = tracker.get_state_dict()
    print(f"\nCurrent state dict: {state_dict}")
    
    # Demonstrate loading state
    print("\n=== Loading State ===")
    new_tracker = MovingStats(window=3)
    print("Before load:")
    print(f"  Count: {new_tracker.count}, Index: {new_tracker.index}")
    
    # Load from JSON
    new_tracker.load_state('moving_stats_state.json')
    print("\nAfter loading from JSON:")
    print(f"  Count: {new_tracker.count}, Index: {new_tracker.index}")
    print(f"  Buffer: {new_tracker.buffer}")
    
    # Continue feeding data after load
    more_prices = [99, 98]
    for price in more_prices:
        ma, std = new_tracker.update(price)
        if ma is not None:
            print(f"Continued: Price={price:3d} | MA={ma:.2f} | STD={std:.2f}")
    
    # Demonstrate reset
    print("\n=== After Reset ===")
    tracker.reset()
    print(f"After reset - Count: {tracker.count}, Buffer: {tracker.buffer}")
    
    # Feed new data after reset
    new_prices = [200, 201, 202]
    for price in new_prices:
        ma, std = tracker.update(price)
        if ma is not None:
            print(f"Price={price} | MA={ma:.2f} | STD={std:.2f}")
        else:
            print(f"Price={price} | Collecting... ({tracker.count}/{tracker.window})")