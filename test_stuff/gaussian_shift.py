import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import shift as scipy_shift
from scipy.fft import fft2, ifft2
import time

def generate_gaussian_2d(center, shape, sigma):
    """Generate 2D Gaussian using voxel centers (add 0.5 offset)."""
    y = np.arange(shape[0]) + 0.5
    x = np.arange(shape[1]) + 0.5
    yy, xx = np.meshgrid(y, x, indexing='ij')
    
    dy = yy - center[0]
    dx = xx - center[1]
    r_squared = dx**2 + dy**2
    gaussian = np.exp(-r_squared / (2 * sigma**2))
    return gaussian

def method1_scipy_shift(gaussian, shift):
    """Method 1: scipy.ndimage.shift with cubic interpolation."""
    return scipy_shift(gaussian, shift=shift, order=3, mode='constant', cval=0.0)

def method2_fourier_shift(array, shift):
    """Method 2: Fourier shift theorem."""
    f = fft2(array)
    ny, nx = array.shape
    freq_y = np.fft.fftfreq(ny)
    freq_x = np.fft.fftfreq(nx)
    fy, fx = np.meshgrid(freq_y, freq_x, indexing='ij')
    
    phase_shift = np.exp(-2j * np.pi * (fy * shift[0] + fx * shift[1]))
    shifted = np.real(ifft2(f * phase_shift))
    return shifted

def method3_analytical(original_center, sigma, shape, shift):
    """Method 3: Analytical recomputation."""
    new_center = original_center + np.array(shift)
    return generate_gaussian_2d(new_center, shape, sigma)

def benchmark_method(method_func, *args, n_runs=500):
    """Benchmark a method."""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        result = method_func(*args)
        end = time.perf_counter()
        times.append((end - start) * 1000)
    return np.mean(times), np.std(times), result

# Test parameters
shape = (32, 32)
sigma = 2.0
base_center = 16.0

# Different starting positions (offsets from base center)
center_offsets = [0.0, 0.1, 0.25, 0.5, 0.75]

# Shift amounts to test
shift_amounts = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

print("=" * 80)
print("Testing shift quality for different starting positions")
print(f"Base center: {base_center}")
print(f"Center offsets: {center_offsets}")
print(f"Shift amounts: {shift_amounts}")
print("=" * 80)

all_results = {}

for center_offset in center_offsets:
    center = np.array([base_center + center_offset, base_center + center_offset])
    
    print(f"\n{'='*80}")
    print(f"Starting position: ({center[0]:.2f}, {center[1]:.2f})")
    print(f"{'='*80}")
    
    # Generate original Gaussian at this center
    gaussian_original = generate_gaussian_2d(center, shape, sigma)
    
    results_for_center = []
    
    for shift_val in shift_amounts:
        shift = np.array([shift_val, shift_val])
        
        # Ground truth
        ground_truth = generate_gaussian_2d(center + shift, shape, sigma)
        
        # Benchmark methods
        t_scipy, _, res_scipy = benchmark_method(method1_scipy_shift, gaussian_original, shift)
        t_fourier, _, res_fourier = benchmark_method(method2_fourier_shift, gaussian_original, shift)
        t_analytical, _, res_analytical = benchmark_method(method3_analytical, center, sigma, shape, shift)
        
        # Compute errors
        err_scipy = np.abs(res_scipy - ground_truth).max()
        err_fourier = np.abs(res_fourier - ground_truth).max()
        err_analytical = np.abs(res_analytical - ground_truth).max()
        
        results_for_center.append({
            "shift": shift_val,
            "t_scipy": t_scipy,
            "t_fourier": t_fourier,
            "t_analytical": t_analytical,
            "err_scipy": err_scipy,
            "err_fourier": err_fourier,
            "err_analytical": err_analytical,
        })
        
        print(f"  Shift {shift_val:4.2f}: Scipy {t_scipy:6.4f}ms (err {err_scipy:.2e}) | "
              f"Fourier {t_fourier:6.4f}ms (err {err_fourier:.2e}) | "
              f"Analytical {t_analytical:6.4f}ms (err {err_analytical:.2e})")
    
    all_results[center_offset] = results_for_center

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

colors = ['blue', 'green', 'red', 'purple', 'orange']
linestyles = ['-', '--', '-.', ':', '-']

# Top left: Scipy error vs shift amount for different starting positions
ax = axes[0, 0]
for i, offset in enumerate(center_offsets):
    results = all_results[offset]
    shifts = [r["shift"] for r in results]
    errors = [r["err_scipy"] for r in results]
    ax.semilogy(shifts, errors, color=colors[i], linestyle=linestyles[i], 
                linewidth=2, markersize=6, marker='o', 
                label=f'Center offset: {offset:.2f}')
ax.axhline(y=1e-10, color='k', linestyle='--', alpha=0.3, label='~Machine precision')
ax.set_xlabel('Shift Amount (pixels)', fontsize=11)
ax.set_ylabel('Max Absolute Error (log scale)', fontsize=11)
ax.set_title('Scipy Shift - Accuracy vs Shift Amount', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Top right: Fourier error vs shift amount
ax = axes[0, 1]
for i, offset in enumerate(center_offsets):
    results = all_results[offset]
    shifts = [r["shift"] for r in results]
    errors = [r["err_fourier"] for r in results]
    ax.semilogy(shifts, errors, color=colors[i], linestyle=linestyles[i], 
                linewidth=2, markersize=6, marker='o', 
                label=f'Center offset: {offset:.2f}')
ax.axhline(y=1e-10, color='k', linestyle='--', alpha=0.3, label='~Machine precision')
ax.set_xlabel('Shift Amount (pixels)', fontsize=11)
ax.set_ylabel('Max Absolute Error (log scale)', fontsize=11)
ax.set_title('Fourier Shift - Accuracy vs Shift Amount', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom left: Scipy timing vs shift amount
ax = axes[1, 0]
for i, offset in enumerate(center_offsets):
    results = all_results[offset]
    shifts = [r["shift"] for r in results]
    times = [r["t_scipy"] for r in results]
    ax.plot(shifts, times, color=colors[i], linestyle=linestyles[i], 
            linewidth=2, markersize=6, marker='o', 
            label=f'Center offset: {offset:.2f}')
ax.set_xlabel('Shift Amount (pixels)', fontsize=11)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Scipy Shift - Computation Time', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Bottom right: Comparison of methods (averaged across all starting positions)
ax = axes[1, 1]
avg_scipy_time = []
avg_fourier_time = []
avg_analytical_time = []
avg_scipy_err = []
avg_fourier_err = []

for shift_val in shift_amounts:
    scipy_times = [all_results[offset][i]["t_scipy"] 
                   for offset in center_offsets 
                   for i, r in enumerate(all_results[offset]) if r["shift"] == shift_val]
    fourier_times = [all_results[offset][i]["t_fourier"] 
                     for offset in center_offsets 
                     for i, r in enumerate(all_results[offset]) if r["shift"] == shift_val]
    analytical_times = [all_results[offset][i]["t_analytical"] 
                        for offset in center_offsets 
                        for i, r in enumerate(all_results[offset]) if r["shift"] == shift_val]
    
    scipy_errs = [all_results[offset][i]["err_scipy"] 
                  for offset in center_offsets 
                  for i, r in enumerate(all_results[offset]) if r["shift"] == shift_val]
    fourier_errs = [all_results[offset][i]["err_fourier"] 
                    for offset in center_offsets 
                    for i, r in enumerate(all_results[offset]) if r["shift"] == shift_val]
    
    avg_scipy_time.append(np.mean(scipy_times))
    avg_fourier_time.append(np.mean(fourier_times))
    avg_analytical_time.append(np.mean(analytical_times))
    avg_scipy_err.append(np.max(scipy_errs))
    avg_fourier_err.append(np.max(fourier_errs))

ax.plot(shift_amounts, avg_scipy_time, 'ro-', linewidth=2, markersize=6, label='Scipy')
ax.plot(shift_amounts, avg_fourier_time, 'go-', linewidth=2, markersize=6, label='Fourier')
ax.plot(shift_amounts, avg_analytical_time, 'mo-', linewidth=2, markersize=6, label='Analytical')
ax.set_xlabel('Shift Amount (pixels)', fontsize=11)
ax.set_ylabel('Time (ms)', fontsize=11)
ax.set_title('Method Comparison - Average Time Across All Starting Positions', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('shift_analysis_starting_positions.png', dpi=150, bbox_inches='tight')
print("\n✓ Visualization saved as 'shift_analysis_starting_positions.png'")
plt.show()

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

# Check if starting position affects error
print("\nDoes starting position affect error?")
for shift_val in [0.5, 1.0, 2.0, 5.0]:
    scipy_errs = [all_results[offset][i]["err_scipy"] 
                  for offset in center_offsets 
                  for i, r in enumerate(all_results[offset]) if r["shift"] == shift_val]
    print(f"  Shift {shift_val:.1f}: Scipy error range [{min(scipy_errs):.2e}, {max(scipy_errs):.2e}]")

print(f"\nAverage timing (across all starting positions and shift amounts):")
all_scipy_times = [r["t_scipy"] for offset in center_offsets for r in all_results[offset]]
all_fourier_times = [r["t_fourier"] for offset in center_offsets for r in all_results[offset]]
all_analytical_times = [r["t_analytical"] for offset in center_offsets for r in all_results[offset]]

print(f"  Scipy:      {np.mean(all_scipy_times):.4f} ± {np.std(all_scipy_times):.4f} ms")
print(f"  Fourier:    {np.mean(all_fourier_times):.4f} ± {np.std(all_fourier_times):.4f} ms")
print(f"  Analytical: {np.mean(all_analytical_times):.4f} ± {np.std(all_analytical_times):.4f} ms")

print(f"\nMax error (scipy):")
all_scipy_errs = [r["err_scipy"] for offset in center_offsets for r in all_results[offset]]
print(f"  Across all tests: {max(all_scipy_errs):.2e}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("=" * 80)
print("✓ Use scipy.shift(heatmap, shift, order=3)")
print("\nReasons:")
print(f"  1. Consistent accuracy across all starting positions")
print(f"  2. Fast: {np.mean(all_scipy_times):.4f} ms average")
print(f"  3. Max error: {max(all_scipy_errs):.2e}")
print("  4. Single function call, no need to store centers/sigmas")
print("=" * 80)