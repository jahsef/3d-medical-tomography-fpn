import zarr
import time

def analyze_zarr_compression(zarr_path):
    """Check compression settings of your zarr files"""
    z = zarr.open(zarr_path, mode='r')
    print(f"Compression: {z.compressors}")
    print(f"Chunks: {z.chunks}")
    print(f"Dtype: {z.dtype}")
    print(f"Compression ratio: {z.nbytes / z.nbytes_stored()}x")
    
    # Test read speed
    import time
    start = time.time()
    chunk = z[0:128, 0:128, 0:128]  # Read one chunk
    end = time.time()
    print(f"Single chunk read time: {(end-start)*1000:.1f}ms")
    

analyze_zarr_compression(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_zarr_data\train\tomo_0a8f05.zarr')
