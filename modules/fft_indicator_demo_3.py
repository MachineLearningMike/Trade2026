import torch
import matplotlib.pyplot as plt
import numpy as np

from datetime import datetime
from Legacy.Binance_new import Binance # Get_Price_Data_By_Time

# Set random seed for reproducibility
torch.manual_seed(42)

#--------------------------------------- Get market history -----------------------------------
bn = Binance.instantiate(connect=False)
# Get_Price_Data_By_Time function seamlessly collects/creates price candles from Binance.
datatype = "klines" # "aggTrades" #, "klines"
symbol = "BTCUSDT"      # Any pair listed on Binance.
start = datetime(2025, 1, 1)    # Any datetime. Note Binance started operation in July 2017.
end = datetime(2026, 1, 1) # + timedelta(minutes=100)     # Any datetime
interval = '1m'    # in '1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d'.  (excludes: '3d', '1w')
candles, nFills, successful, _, reason = bn.Get_Price_Data_By_Time(datatype, symbol, interval, start, end)
# "Returning 10 columns: O H L C V, Quote V, nTrades, Buy base V, Buy quote V, Mark (0: original not forged)"
open_idx, high_idx, low_idx, close_idx, volume_idx, mark_idx = 0, 1, 2, 3, 4, 9
if successful:          # price is a numpy array.
    print("\n Successful !!!!!!!!!! (nCandles, nAttributes) = {}, nForged = {}".format(candles.shape, nFills))
else:
    print("\n Failed to Get_Price_Data_By_Time", reason)
    exit(0)
    

def generate_sample_price_data(n_bars=500):

    price = candles[:1000, close_idx]
    norm_price = (price - price.mean()) / price.std()
    signal = torch.from_numpy(norm_price).float()
    
    return signal[:n_bars]

class FFTIndicators:
    def __init__(self, price, lookback=200):
        # Convert to torch tensor if numpy array
        if isinstance(price, np.ndarray):
            price = torch.from_numpy(price).float()
        self.price = price
        self.lookback = lookback
        self.original_signal = None
        self.detrended = None
        self.freqs = None
        self.magnitude = None
        self.phase = None
        self.reconstructed = None
        self._compute_fft()
    
    def _compute_fft(self):
        """Compute FFT with proper detrending"""
        # Take last lookback bars
        signal = self.price[-self.lookback:]
        self.original_signal = signal.clone()
        
        # Detrend using linear regression (pure PyTorch)
        t = torch.arange(len(signal), dtype=torch.float32)
        t_mean = t.mean()
        signal_mean = signal.mean()
        
        a = ((t - t_mean) * (signal - signal_mean)).sum() / ((t - t_mean)**2).sum()
        b = signal_mean - a * t_mean
        trend = a * t + b
        self.detrended = signal - trend
        
        # Apply Hann window
        windowed = self.detrended * torch.hann_window(len(signal))
        
        # Compute FFT
        n = len(signal)
        fft_result = torch.fft.fft(windowed)
        
        # Get frequencies (cycles per bar)
        self.freqs = torch.fft.fftfreq(n, d=1.0)[:n//2]
        self.magnitude = torch.abs(fft_result[:n//2])
        self.phase = torch.angle(fft_result[:n//2])
        
        # Remove DC component
        self.magnitude[0] = 0
        
        # Normalize magnitude
        if self.magnitude.sum() > 0:
            self.magnitude = self.magnitude / self.magnitude.sum()
    
    def reconstruct_signal(self, n_components=None):
        """Reconstruct signal using top N frequency components"""
        if n_components is None:
            n_components = min(20, len(self.magnitude))
        
        # Get top frequency components
        top_indices = self.magnitude.argsort(descending=True)[:n_components]
        top_freqs = self.freqs[top_indices]
        top_mags = self.magnitude[top_indices]
        top_phases = self.phase[top_indices]
        
        t = torch.arange(len(self.detrended))
        reconstructed_detrended = torch.zeros(len(self.detrended))
        
        for freq, mag, phase in zip(top_freqs, top_mags, top_phases):
            reconstructed_detrended += mag * torch.cos(2 * torch.pi * freq * t + phase)
        
        # Scale to match original amplitude
        reconstructed_detrended = reconstructed_detrended * (self.detrended.std() / reconstructed_detrended.std())
        
        # Add back trend to get price reconstruction
        t_orig = torch.arange(len(self.original_signal), dtype=torch.float32)
        t_mean = t_orig.mean()
        signal_mean = self.original_signal.mean()
        a = ((t_orig - t_mean) * (self.original_signal - signal_mean)).sum() / ((t_orig - t_mean)**2).sum()
        b = signal_mean - a * t_mean
        trend = a * t_orig + b
        
        self.reconstructed = reconstructed_detrended + trend
        
        return self.reconstructed, reconstructed_detrended
    
    def plot_raw_vs_reconstructed(self, n_components_list=[5, 10, 20, 50]):
        """Plot raw price vs reconstructed with different numbers of components"""
        
        fig, axes = plt.subplots(len(n_components_list), 2, figsize=(16, 4*len(n_components_list)))
        
        for idx, n_comp in enumerate(n_components_list):
            reconstructed, _ = self.reconstruct_signal(n_comp)
            
            # Plot 1: Raw vs Reconstructed
            ax1 = axes[idx, 0] if len(n_components_list) > 1 else axes[0]
            ax1.plot(self.original_signal.numpy(), label='Raw Price', alpha=0.7, linewidth=1)
            ax1.plot(reconstructed.numpy(), label=f'Reconstructed ({n_comp} components)', alpha=0.7, linewidth=2)
            ax1.set_title(f'Raw vs Reconstructed Price ({n_comp} FFT Components)')
            ax1.set_xlabel('Bar')
            ax1.set_ylabel('Price')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Error
            ax2 = axes[idx, 1] if len(n_components_list) > 1 else axes[1]
            error = self.original_signal - reconstructed
            ax2.plot(error.numpy(), color='red', alpha=0.7)
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            ax2.fill_between(range(len(error)), 0, error.numpy(), alpha=0.3, color='red')
            ax2.set_title(f'Reconstruction Error (RMSE: {error.std():.3f})')
            ax2.set_xlabel('Bar')
            ax2.set_ylabel('Error')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_comprehensive_comparison(self):
        """Create a comprehensive comparison plot"""
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Raw Price
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.original_signal.numpy(), color='blue', linewidth=1)
        ax1.set_title('1. Raw Price Data')
        ax1.set_xlabel('Bar')
        ax1.set_ylabel('Price')
        ax1.grid(True, alpha=0.3)
        
        # 2. Detrended Signal
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.detrended.numpy(), color='green', linewidth=1)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_title('2. Detrended Signal (Cycles Only)')
        ax2.set_xlabel('Bar')
        ax2.set_ylabel('Amplitude')
        ax2.grid(True, alpha=0.3)
        
        # 3. FFT Magnitude Spectrum
        ax3 = plt.subplot(3, 2, 3)
        mask = self.freqs > 0
        periods = 1.0 / self.freqs[mask]
        magnitudes = self.magnitude[mask]
        ax3.bar(periods.numpy()[:50], magnitudes.numpy()[:50], width=2, alpha=0.7)
        ax3.set_title('3. FFT Magnitude Spectrum (Periodogram)')
        ax3.set_xlabel('Period (bars)')
        ax3.set_ylabel('Strength')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cumulative Energy
        ax4 = plt.subplot(3, 2, 4)
        cumulative_energy = torch.cumsum(self.magnitude, dim=0)
        cumulative_energy = cumulative_energy / cumulative_energy[-1]
        ax4.plot(cumulative_energy.numpy())
        ax4.axhline(y=0.95, color='r', linestyle='--', label='95% Energy')
        ax4.set_title('4. Cumulative Spectral Energy')
        ax4.set_xlabel('Frequency Component')
        ax4.set_ylabel('Cumulative Energy')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Reconstruction with 5, 10, 20 components
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(self.original_signal.numpy(), label='Raw', alpha=0.5, linewidth=1)
        
        colors = ['red', 'orange', 'green']
        for idx, n_comp in enumerate([5, 10, 20]):
            reconstructed, _ = self.reconstruct_signal(n_comp)
            ax5.plot(reconstructed.numpy(), label=f'{n_comp} components', 
                    color=colors[idx], alpha=0.7, linewidth=1.5)
        
        ax5.set_title('5. Raw vs Reconstructed Price')
        ax5.set_xlabel('Bar')
        ax5.set_ylabel('Price')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Reconstruction Error Comparison
        ax6 = plt.subplot(3, 2, 6)
        errors = []
        components_range = range(1, 51)
        
        for n_comp in components_range:
            reconstructed, _ = self.reconstruct_signal(n_comp)
            rmse = (self.original_signal - reconstructed).std().item()
            errors.append(rmse)
        
        ax6.plot(components_range, errors)
        ax6.set_title('6. Reconstruction Error vs Components Used')
        ax6.set_xlabel('Number of FFT Components')
        ax6.set_ylabel('RMSE')
        ax6.grid(True, alpha=0.3)
        
        # Mark optimal point (where error stops decreasing significantly)
        error_reduction = np.diff(errors)
        optimal_idx = np.where(error_reduction > -0.001)[0][0] if len(error_reduction) > 0 else 20
        ax6.axvline(x=optimal_idx, color='r', linestyle='--', 
                   label=f'Optimal (~{optimal_idx} components)')
        ax6.legend()
        
        plt.tight_layout()
        plt.show()
        
        return optimal_idx
    
    def indicator_1_magnitude_spectrum(self):
        """Indicator 1: FFT Magnitude Spectrum / Periodogram"""
        mask = self.freqs > 0
        periods = 1.0 / self.freqs[mask]
        magnitudes = self.magnitude[mask]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Periodogram
        axes[0].bar(periods.numpy(), magnitudes.numpy(), width=2, alpha=0.7)
        axes[0].set_xlabel('Period (bars)')
        axes[0].set_ylabel('Strength')
        axes[0].set_title('Cycle Strength Periodogram')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(magnitudes.mean().item(), color='r', linestyle='--', alpha=0.5, label='Mean strength')
        axes[0].legend()
        
        # Top cycles
        top_indices = magnitudes.argsort(descending=True)[:5]
        top_periods = periods[top_indices]
        top_strengths = magnitudes[top_indices]
        
        axes[1].barh(range(len(top_periods)), top_strengths.numpy(), alpha=0.7)
        axes[1].set_yticks(range(len(top_periods)))
        axes[1].set_yticklabels([f"{p:.1f} bars" for p in top_periods.numpy()])
        axes[1].set_xlabel('Strength')
        axes[1].set_title('Top 5 Dominant Cycles')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return dict(zip(top_periods.numpy(), top_strengths.numpy()))


# ============ MAIN DEMONSTRATION ============

def main():
    """Main function to demonstrate FFT reconstruction"""
    
    print("=" * 70)
    print("FFT PRICE RECONSTRUCTION DEMONSTRATION")
    print("=" * 70)
    
    # Generate sample price data
    print("\n1. Generating sample price data with 60, 30, and 15-day cycles...")
    price = generate_sample_price_data(500)
    
    # Plot raw price
    plt.figure(figsize=(14, 4))
    plt.plot(price)
    plt.title('Raw Price Data with Multiple Cycles')
    plt.xlabel('Bar')
    plt.ylabel('Price')
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Initialize FFT indicators
    print("\n2. Computing FFT and reconstructing price...")
    indicators = FFTIndicators(price, lookback=400)
    
    # Plot raw vs reconstructed with different component counts
    print("\n3. Comparing raw price with reconstructions...")
    indicators.plot_raw_vs_reconstructed(n_components_list=[5, 10, 20, 50])
    
    # Comprehensive comparison
    print("\n4. Generating comprehensive comparison...")
    optimal_components = indicators.plot_comprehensive_comparison()
    print(f"Optimal number of components: {optimal_components}")
    
    # Show magnitude spectrum
    print("\n5. Displaying FFT magnitude spectrum...")
    top_cycles = indicators.indicator_1_magnitude_spectrum()
    print(f"Top cycles detected: {top_cycles}")
    
    # Final reconstruction with optimal components
    print(f"\n6. Final reconstruction with {optimal_components} components...")
    reconstructed, detrended_reconstructed = indicators.reconstruct_signal(optimal_components)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Price comparison
    axes[0].plot(indicators.original_signal.numpy(), label='Raw Price', alpha=0.7, linewidth=1)
    axes[0].plot(reconstructed.numpy(), label=f'Reconstructed ({optimal_components} components)', 
                color='red', alpha=0.7, linewidth=2)
    axes[0].set_title(f'Final Reconstruction with {optimal_components} FFT Components')
    axes[0].set_xlabel('Bar')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error
    error = indicators.original_signal - reconstructed
    axes[1].plot(error.numpy(), color='red', alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    axes[1].fill_between(range(len(error)), 0, error.numpy(), alpha=0.3, color='red')
    axes[1].set_title(f'Reconstruction Error (RMSE: {error.std():.3f}, MAE: {error.abs().mean():.3f})')
    axes[1].set_xlabel('Bar')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Statistics
    print("\n" + "=" * 70)
    print("RECONSTRUCTION STATISTICS")
    print("=" * 70)
    print(f"Original price range: [{indicators.original_signal.min():.2f}, {indicators.original_signal.max():.2f}]")
    print(f"Reconstructed range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")
    print(f"Correlation: {torch.corrcoef(torch.stack([indicators.original_signal, reconstructed]))[0,1]:.4f}")
    print(f"RMSE: {error.std():.4f}")
    print(f"MAE: {error.abs().mean():.4f}")
    print(f"Max Error: {error.abs().max():.4f}")
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    
    return indicators

if __name__ == "__main__":
    indicators = main()