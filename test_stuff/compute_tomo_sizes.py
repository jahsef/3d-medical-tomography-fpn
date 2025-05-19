from pathlib import Path
import imageio as iio
import numpy as np

master_dir = Path(r'C:\Users\kevin\Documents\GitHub\kaggle BYU bacteria motor\original_data\train')
poop_list = []

for tomo_dir in master_dir.iterdir():
    if tomo_dir.is_dir():
        files = list(tomo_dir.rglob('*'))
        num_files = len(files)
        test_image = iio.imread_v2(files[0])
        shape = test_image.shape
        num_pixels = shape[0] * shape[1]
        poop_list.append((num_files, num_pixels))

array = np.asarray(poop_list, dtype = np.float32)
#n, 2

mean = array.mean(axis = 0)
print(mean)
#4.1542285e+02 9.1516481e+05
#avg tomo stats: 415 depth, 956 h/w