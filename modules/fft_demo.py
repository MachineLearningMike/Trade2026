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
    

# ============ DEMO 1: Basic FFT on Sine Waves ============
def demo_basic_fft():
    print("=== FFT on Financial Data with 1/60 Cycle ===")
    
    price = candles[:1000, close_idx]
    norm_price = (price - price.mean()) / price.std()
    signal = torch.from_numpy(norm_price).float()
    
    # Manual detrend using PyTorch only
    t = torch.arange(len(signal), dtype=torch.float32)
    # Linear regression: y = a*t + b
    t_mean = t.mean()
    signal_mean = signal.mean()
    a = ((t - t_mean) * (signal - signal_mean)).sum() / ((t - t_mean)**2).sum()
    b = signal_mean - a * t_mean
    trend = a * t + b
    detrended = signal - trend
    
    # Apply window
    windowed = detrended * torch.hann_window(len(signal))
    
    # Compute FFT
    fft_result = torch.fft.fft(windowed)
    n = len(signal)
    
    # Frequencies (cycles per sample)
    freqs = torch.fft.fftfreq(n, d=1.0)
    
    magnitude = torch.abs(fft_result)
    half_n = n // 2
    freqs_pos = freqs[:half_n]
    magnitude_pos = magnitude[:half_n]
    magnitude_pos[0] = 0  # Remove DC
    
    target_freq = 1/60
    
    # Plotting (same as above)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    axes[0, 0].plot(detrended[:500].numpy())
    axes[0, 0].set_title('Detrended Signal')
    axes[0, 0].set_xlabel('Candle Number')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(freqs.numpy(), magnitude.numpy())
    axes[0, 1].set_title('Full Magnitude Spectrum')
    axes[0, 1].set_xlabel('Frequency (cycles/sample)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(target_freq, color='r', linestyle='--', label='1/60 cycle')
    axes[0, 1].legend()
    
    axes[1, 0].plot(freqs_pos.numpy(), magnitude_pos.numpy())
    axes[1, 0].set_title('Positive Frequency Spectrum')
    axes[1, 0].set_xlabel('Frequency (cycles/sample)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(target_freq, color='r', linestyle='--', label='1/60 cycle')
    axes[1, 0].legend()
    
    axes[1, 1].plot(freqs_pos.numpy(), torch.angle(fft_result[:half_n]).numpy())
    axes[1, 1].set_title('Phase Spectrum')
    axes[1, 1].set_xlabel('Frequency (cycles/sample)')
    axes[1, 1].set_ylabel('Phase (radians)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(target_freq, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.show()
    
    # Find peak near 1/60
    window = 0.01
    mask = (freqs_pos > target_freq - window) & (freqs_pos < target_freq + window)
    if mask.any():
        nearby_mags = magnitude_pos[mask]
        if len(nearby_mags) > 0:
            peak_local_idx = nearby_mags.argmax()
            peak_global_idx = torch.where(mask)[0][peak_local_idx]
            detected_freq = freqs_pos[peak_global_idx].item()
            print(f"\n✓ Detected frequency: {detected_freq:.4f} cycles/candle")
            print(f"  Period: {1/detected_freq:.1f} candles")
    
    # Top frequencies
    top_indices = magnitude_pos.argsort(descending=True)[:5]
    print(f"\nTop 5 frequencies:")
    for i, idx in enumerate(top_indices):
        f = freqs_pos[idx].item()
        if f > 0:
            print(f"  {i+1}. {f:.4f} cycles/candle → period = {1/f:.1f} candles")

# ============ DEMO 2: FFT Properties and Convolution ============
def demo_fft_properties():
    print("=== Demo 2: FFT Properties and Convolution ===")
    
    # Create test signals
    fs = 100
    t = torch.arange(0, 1, 1/fs)
    signal1 = torch.sin(2 * torch.pi * 10 * t)  # 10 Hz
    signal2 = torch.cos(2 * torch.pi * 20 * t)  # 20 Hz
    
    # Property 1: Linearity
    fft_sum = torch.fft.fft(signal1 + signal2)
    fft_sum_indiv = torch.fft.fft(signal1) + torch.fft.fft(signal2)
    linearity_check = torch.allclose(fft_sum, fft_sum_indiv, atol=1e-6)
    
    # Property 2: Inverse FFT recovers original
    recovered = torch.fft.ifft(torch.fft.fft(signal1))
    inverse_check = torch.allclose(signal1, recovered.real, atol=1e-6)
    
    # Property 3: Convolution theorem
    kernel = torch.tensor([0.2, 0.6, 0.2])  # Simple smoothing kernel
    signal_conv = signal1[:50]  # Take first 50 samples
    
    # Direct convolution
    direct_conv = torch.nn.functional.conv1d(
        signal_conv.view(1, 1, -1),
        kernel.view(1, 1, -1),
        padding='same'
    ).squeeze()
    
    # FFT convolution
    n = len(signal_conv) + len(kernel) - 1
    X = torch.fft.fft(signal_conv, n=n)
    K = torch.fft.fft(kernel, n=n)
    fft_conv = torch.fft.ifft(X * K).real[:len(signal_conv)]
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Original signals
    axes[0, 0].plot(t[:50], signal1[:50], label='10 Hz Sine')
    axes[0, 0].plot(t[:50], signal2[:50], label='20 Hz Cosine', alpha=0.7)
    axes[0, 0].plot(t[:50], (signal1 + signal2)[:50], label='Sum', linewidth=2)
    axes[0, 0].set_title(f'Linearity Property: {linearity_check}')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # FFT of sum vs sum of FFTs
    axes[0, 1].plot(torch.abs(fft_sum)[:len(fft_sum)//2], label='FFT(s1+s2)', alpha=0.7)
    axes[0, 1].plot(torch.abs(fft_sum_indiv)[:len(fft_sum_indiv)//2], '--', label='FFT(s1)+FFT(s2)', alpha=0.7)
    axes[0, 1].set_title(f'FFT Linearity Verification (Difference: {linearity_check})')
    axes[0, 1].set_xlabel('Frequency Bin')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Convolution comparison
    axes[1, 0].plot(signal_conv.numpy(), label='Original Signal', alpha=0.7)
    axes[1, 0].plot(direct_conv.numpy(), label='Direct Convolution', alpha=0.7)
    axes[1, 0].plot(fft_conv.numpy(), '--', label='FFT Convolution', alpha=0.7)
    axes[1, 0].set_title(f'Convolution Theorem (Match: {torch.allclose(direct_conv, fft_conv, atol=1e-6)})')
    axes[1, 0].set_xlabel('Sample')
    axes[1, 0].set_ylabel('Amplitude')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error between direct and FFT convolution
    axes[1, 1].plot((direct_conv - fft_conv).abs().numpy())
    axes[1, 1].set_title('Absolute Error between Methods')
    axes[1, 1].set_xlabel('Sample')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Linearity test passed: {linearity_check}")
    print(f"Inverse FFT test passed: {inverse_check}")
    print(f"Convolution theorem test passed: {torch.allclose(direct_conv, fft_conv, atol=1e-6)}\n")

# ============ DEMO 3: Image Filtering with 2D FFT ============
def demo_image_filtering():
    print("=== Demo 3: Image Filtering with 2D FFT ===")
    
    # Create synthetic image
    size = 256
    x = torch.linspace(-1, 1, size)
    y = torch.linspace(-1, 1, size)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # Create image components
    circle = (X**2 + Y**2 < 0.3).float()  # Circle
    stripes = (torch.sin(2 * torch.pi * 10 * X) > 0).float()  # Vertical stripes
    gradient = (X + 1) / 2  # Horizontal gradient
    
    image = circle + 0.5 * stripes + 0.3 * gradient
    image = image / image.max()  # Normalize
    image += 0.05 * torch.randn_like(image)  # Add noise
    
    # Compute 2D FFT and shift
    fft_image = torch.fft.fft2(image)
    fft_shifted = torch.fft.fftshift(fft_image)
    
    # Create filters
    rows, cols = size, size
    crow, ccol = rows // 2, cols // 2
    
    # Low-pass filter (circular)
    mask_lp = torch.zeros((rows, cols))
    radius = 30
    y_grid, x_grid = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
    mask_lp[(y_grid - crow)**2 + (x_grid - ccol)**2 <= radius**2] = 1
    
    # High-pass filter (inverse of low-pass)
    mask_hp = 1 - mask_lp
    
    # Band-pass filter (ring)
    mask_bp = torch.zeros((rows, cols))
    r1, r2 = 20, 60
    dist_sq = (y_grid - crow)**2 + (x_grid - ccol)**2
    mask_bp[(dist_sq >= r1**2) & (dist_sq <= r2**2)] = 1
    
    # Apply filters
    fft_lp = fft_shifted * mask_lp
    fft_hp = fft_shifted * mask_hp
    fft_bp = fft_shifted * mask_bp
    
    # Inverse FFT
    image_lp = torch.fft.ifft2(torch.fft.ifftshift(fft_lp)).real
    image_hp = torch.fft.ifft2(torch.fft.ifftshift(fft_hp)).real
    image_bp = torch.fft.ifft2(torch.fft.ifftshift(fft_bp)).real
    
    # Plotting
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Original image
    axes[0, 0].imshow(image.numpy(), cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # FFT magnitude spectrum
    axes[0, 1].imshow(torch.log(torch.abs(fft_shifted) + 1).numpy(), cmap='gray')
    axes[0, 1].set_title('FFT Magnitude Spectrum (log scale)')
    axes[0, 1].axis('off')
    
    # FFT phase
    axes[0, 2].imshow(torch.angle(fft_shifted).numpy(), cmap='hsv')
    axes[0, 2].set_title('FFT Phase Spectrum')
    axes[0, 2].axis('off')
    
    # Low-pass filter
    axes[1, 0].imshow(mask_lp.numpy(), cmap='gray')
    axes[1, 0].set_title('Low-Pass Filter Mask')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(image_lp.numpy(), cmap='gray')
    axes[1, 1].set_title('Low-Pass Filtered (Smooth)')
    axes[1, 1].axis('off')
    
    # High-pass filter
    axes[1, 2].imshow(mask_hp.numpy(), cmap='gray')
    axes[1, 2].set_title('High-Pass Filter Mask')
    axes[1, 2].axis('off')
    
    axes[2, 0].imshow(image_hp.numpy(), cmap='gray')
    axes[2, 0].set_title('High-Pass Filtered (Edges)')
    axes[2, 0].axis('off')
    
    # Band-pass filter
    axes[2, 1].imshow(mask_bp.numpy(), cmap='gray')
    axes[2, 1].set_title('Band-Pass Filter Mask')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(image_bp.numpy(), cmap='gray')
    axes[2, 2].set_title('Band-Pass Filtered')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Filters applied: Low-pass (smoothing), High-pass (edge detection), Band-pass\n")

# ============ DEMO 4: Real FFT for Efficient Processing ============
def demo_real_fft():
    print("=== Demo 4: Real FFT (rfft) for Efficiency ===")
    
    # Create real signal
    fs = 500
    t = torch.arange(0, 1, 1/fs)
    signal = torch.sin(2 * torch.pi * 30 * t) + 0.3 * torch.sin(2 * torch.pi * 80 * t)
    
    # Compare standard FFT vs Real FFT
    fft_complex = torch.fft.fft(signal)
    rfft_result = torch.fft.rfft(signal)
    
    # Inverse transforms
    signal_from_fft = torch.fft.ifft(fft_complex).real
    signal_from_rfft = torch.fft.irfft(rfft_result, n=len(signal))
    
    # Frequency axes
    freqs_fft = torch.fft.fftfreq(len(signal), 1/fs)
    freqs_rfft = torch.fft.rfftfreq(len(signal), 1/fs)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    
    # Original signal
    axes[0, 0].plot(t[:200], signal[:200])
    axes[0, 0].set_title('Original Signal (Time Domain)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Compare magnitude spectra
    axes[0, 1].plot(freqs_fft[:len(freqs_fft)//2].numpy(), 
                    torch.abs(fft_complex[:len(fft_complex)//2]).numpy(), 
                    label='Standard FFT', alpha=0.7)
    axes[0, 1].plot(freqs_rfft.numpy(), torch.abs(rfft_result).numpy(), 
                    '--', label='Real FFT (rfft)', alpha=0.7)
    axes[0, 1].set_title('Magnitude Spectrum Comparison')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Magnitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Memory and speed comparison
    memory_fft = fft_complex.numel() * fft_complex.element_size()
    memory_rfft = rfft_result.numel() * rfft_result.element_size()
    
    axes[1, 0].bar(['Standard FFT', 'Real FFT (rfft)'], 
                   [memory_fft / 1024, memory_rfft / 1024])
    axes[1, 0].set_title('Memory Usage Comparison (KB)')
    axes[1, 0].set_ylabel('Memory (KB)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Reconstruction error
    axes[1, 1].plot(t[:200], (signal - signal_from_fft)[:200], label='FFT Error', alpha=0.7)
    axes[1, 1].plot(t[:200], (signal - signal_from_rfft)[:200], '--', label='rFFT Error', alpha=0.7)
    axes[1, 1].set_title('Reconstruction Error')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Error')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Standard FFT output size: {fft_complex.shape} ({fft_complex.dtype})")
    print(f"Real FFT output size: {rfft_result.shape} ({rfft_result.dtype})")
    print(f"Memory saved: {(1 - memory_rfft/memory_fft)*100:.1f}%")
    print(f"Reconstruction error (max): {torch.max(torch.abs(signal - signal_from_rfft)):.6f}\n")

# ============ DEMO 5: Spectrogram and Time-Frequency Analysis ============
def demo_spectrogram():
    print("=== Demo 5: Spectrogram (Time-Frequency Analysis) ===")
    
    # Create non-stationary signal (frequency changes over time)
    fs = 1000
    t = torch.arange(0, 2, 1/fs)
    
    # Frequency modulation: chirp signal
    f0, f1 = 10, 100  # Start and end frequencies
    chirp = torch.sin(2 * torch.pi * (f0 + (f1 - f0) * t / 2) * t)
    
    # Add burst signals
    burst1 = 0.5 * torch.sin(2 * torch.pi * 200 * t) * (t < 0.3) * (t > 0.2)
    burst2 = 0.5 * torch.sin(2 * torch.pi * 50 * t) * (t < 1.5) * (t > 1.4)
    
    signal = chirp + burst1.float() + burst2.float()
    
    # Compute spectrogram using STFT
    n_fft = 256
    hop_length = 64
    
    # Using torch.stft
    stft_result = torch.stft(signal, n_fft=n_fft, hop_length=hop_length, 
                             window=torch.hann_window(n_fft), 
                             return_complex=True)
    spectrogram = torch.abs(stft_result)
    
    # Time and frequency axes
    times = torch.arange(spectrogram.shape[1]) * hop_length / fs
    freqs = torch.fft.rfftfreq(n_fft, 1/fs)
    
    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Time domain signal
    axes[0, 0].plot(t.numpy(), signal.numpy())
    axes[0, 0].set_title('Time Domain Signal (Frequency-varying)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Spectrogram
    im = axes[0, 1].imshow(20 * torch.log10(spectrogram + 1e-10).numpy(), 
                            aspect='auto', origin='lower',
                            extent=[times[0], times[-1], freqs[0], freqs[-1]])
    axes[0, 1].set_title('Spectrogram (Time-Frequency Representation)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Frequency (Hz)')
    plt.colorbar(im, ax=axes[0, 1], label='Magnitude (dB)')
    
    # FFT of entire signal (average frequency content)
    full_fft = torch.fft.rfft(signal)
    full_freqs = torch.fft.rfftfreq(len(signal), 1/fs)
    
    axes[1, 0].plot(full_freqs.numpy(), torch.abs(full_fft).numpy())
    axes[1, 0].set_title('Global Frequency Spectrum (time-averaged)')
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Magnitude')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Power over time at specific frequencies
    freq_indices = [torch.argmin(torch.abs(freqs - f)).item() for f in [50, 100, 200]]
    
    for idx, freq in zip(freq_indices, [50, 100, 200]):
        power_over_time = spectrogram[idx, :]
        axes[1, 1].plot(times.numpy(), power_over_time.numpy(), label=f'{freq} Hz')
    
    axes[1, 1].set_title('Frequency Power over Time')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Magnitude')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Spectrogram shape: {spectrogram.shape} (frequencies × time windows)")
    print(f"Frequency resolution: {freqs[1] - freqs[0]:.1f} Hz")
    print(f"Time resolution: {times[1] - times[0]:.3f} seconds\n")

# ============ Run all demos ============
if __name__ == "__main__":
    demo_basic_fft()
    # demo_fft_properties()
    # demo_image_filtering()
    # demo_real_fft()
    # demo_spectrogram()
    
    print("All demonstrations completed!")