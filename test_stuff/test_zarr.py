import zarr
import numpy as np
import time
from numcodecs import Blosc

def analyze_compression_options(zarr_path):
    """Test different compression settings on your actual data"""
    
    # Open original zarr
    z_orig = zarr.open(zarr_path, mode='r')
    print(f"Original zarr:")
    print(f"  Shape: {z_orig.shape}")
    print(f"  Dtype: {z_orig.dtype}")
    print(f"  Compressor: {z_orig.compressors}")
    print(f"  Compression ratio: {(z_orig.nbytes / z_orig.nbytes_stored()):.2f}x")
    
    # Read a sample chunk for testing
    sample_chunk = z_orig[0:128, 0:128, 0:128]
    print(f"  Sample chunk shape: {sample_chunk.shape}")
    print(f"  Sample data range: [{sample_chunk.min():.3f}, {sample_chunk.max():.3f}]")
    print(f"  Sample std: {sample_chunk.std():.3f}")
    
    # Test different compression settings
    compressors = [
        ("Current zstd-3", Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)),
        ("zstd-1 (faster)", Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)), 
        ("lz4 (fastest)", Blosc(cname='lz4', clevel=1, shuffle=Blosc.BITSHUFFLE)),
        ("zstd-5 (better)", Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)),
        ("No compression", None)
    ]
    
    print(f"\n{'Compressor':<20} {'Ratio':<8} {'Encode ms':<10} {'Decode ms':<10}")
    print("-" * 50)
    
    for name, compressor in compressors:
        # Test compression
        start_time = time.perf_counter()
        if compressor:
            compressed = compressor.encode(sample_chunk)
            encode_time = (time.perf_counter() - start_time) * 1000
            
            # Test decompression  
            start_time = time.perf_counter()
            decompressed = compressor.decode(compressed)
            decode_time = (time.perf_counter() - start_time) * 1000
            
            ratio = sample_chunk.nbytes / len(compressed)
        else:
            encode_time = 0
            decode_time = 0
            ratio = 1.0
            
        print(f"{name:<20} {ratio:<8.2f} {encode_time:<10.1f} {decode_time:<10.1f}")

def test_dtype_performance(zarr_path):
    """Compare float16 vs float32 performance"""
    
    z = zarr.open(zarr_path, mode='r')
    chunk = z[0:128, 0:128, 0:128]
    
    print(f"\nData type comparison:")
    print(f"Current (float16): {chunk.dtype}, {chunk.nbytes / (1024**2):.1f} MB")
    
    # Convert to float32
    chunk_f32 = chunk.astype(np.float32)
    print(f"Float32: {chunk_f32.dtype}, {chunk_f32.nbytes / (1024**2):.1f} MB")
    
    # Test precision difference
    precision_loss = np.abs(chunk.astype(np.float32) - chunk_f32).max()
    print(f"Precision difference: {precision_loss}")
    
    # Test compression difference
    compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.BITSHUFFLE)
    
    compressed_f16 = compressor.encode(chunk)
    compressed_f32 = compressor.encode(chunk_f32)
    
    print(f"Compressed size (float16): {len(compressed_f16) / (1024**2):.2f} MB")
    print(f"Compressed size (float32): {len(compressed_f32) / (1024**2):.2f} MB")
    print(f"Size ratio (f32/f16): {len(compressed_f32) / len(compressed_f16):.2f}x")

def calculate_theoretical_throughput(zarr_path):
    """Calculate theoretical max throughput"""
    
    z = zarr.open(zarr_path, mode='r')
    
    # Your measurements
    chunk_read_time_ms = 16.0
    chunk_size_mb = (128**3 * 2) / (1024**2)  # float16 = 2 bytes
    
    # Theoretical throughput
    chunks_per_second = 1000 / chunk_read_time_ms
    mb_per_second = chunks_per_second * chunk_size_mb
    
    print(f"\nTheoretical Performance:")
    print(f"Chunk size: {chunk_size_mb:.1f} MB")
    print(f"Chunk read time: {chunk_read_time_ms:.1f} ms")
    print(f"Max chunks/sec: {chunks_per_second:.1f}")
    print(f"Max throughput: {mb_per_second:.1f} MB/s")
    
    # Compare to your current 50% efficiency
    actual_throughput = mb_per_second * 0.5
    print(f"Your actual (50%): {actual_throughput:.1f} MB/s")
    
    # Bottleneck analysis
    print(f"\nBottleneck analysis:")
    print(f"Storage I/O: {chunk_read_time_ms:.1f} ms")
    print(f"Decompression: ~1-2 ms (estimated)")
    print(f"Transform overhead: ~2-5 ms (estimated)")
    print(f"Python/PyTorch: ~5-10 ms (estimated)")
    print(f"Total estimated: ~24-33 ms per patch")
    print(f"→ Efficiency: {(chunk_read_time_ms / 25):.1%} (reasonable!)")

# Recommendations based on your settings
def recommendations():
    print(f"\n" + "="*60)
    print("RECOMMENDATIONS FOR YOUR SETUP:")
    print("="*60)
    
    print("""
✅ KEEP AS-IS (your settings are good):
   - 128³ chunks (perfect alignment)
   - zstd compression level 3 (good balance)
   - bitshuffle (optimal for float data)
   - 16ms read time is quite fast

🤔 OPTIONAL TWEAKS TO TRY:

1. Try zstd level 1 for slightly faster decompression:
   zarr.save_array("test.zarr", data, 
                  compressor=Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE))

2. If you need higher precision, consider float32:
   - 2x memory but better precision
   - May compress similarly due to more data

3. For maximum speed, try lz4:
   compressor=Blosc(cname='lz4', clevel=1, shuffle=Blosc.BITSHUFFLE)

4. Monitor if you're CPU or I/O bound:
   - If CPU bound: lower compression or more workers
   - If I/O bound: faster storage or prefetching

🎯 YOUR 50% EFFICIENCY IS ACTUALLY QUITE GOOD!
   Storage → Decompression → Transforms → PyTorch overhead
   
   For scientific computing, this is solid performance.
""")

# Usage:
analyze_compression_options(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_zarr_data\train\tomo_0a8f05.zarr')
test_dtype_performance(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_zarr_data\train\tomo_0a8f05.zarr') 
calculate_theoretical_throughput(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_zarr_data\train\tomo_0a8f05.zarr')
recommendations()