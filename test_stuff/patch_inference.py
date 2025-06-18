from pathlib import Path
import sys
import torch
current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs.motoridentifier import MotorIdentifier
import time
from natsort import natsorted
import imageio.v3 as iio
import numpy as np

import matplotlib.pyplot as plt

device = torch.device('cuda')
model = MotorIdentifier().to(device)
model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\heatmap\curriculum\run5\best.pt'))
model.eval()
tomo_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train\tomo_be4a3a')

def normalize_tomogram(tomo_array):
    """Normalize tomogram: convert to float16, scale to [0,1], then standardize."""
    # Convert to float16 and scale to [0,1] (assuming input is uint8 0-255)
    tomo_normalized = tomo_array.astype(np.float16) / 255.0
    
    # Apply z-score normalization
    tomo_normalized = (tomo_normalized - 0.479915) / 0.224932
    
    return tomo_normalized

def load_tomogram(src:Path ):
    IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    """Process a single tomogram with memory-efficient approach."""

    print(f'loading tomogram: {src.name}')
    
    # Load images
    files = [
        f for f in src.rglob('*')
        if f.is_file() and f.suffix.lower() in IMAGE_EXTS
    ]
    
    files = natsorted(files, key=lambda x: x.name)
    
    if not files:
        print(f"No image files found in {src}")
        return
    
    # Load all images into memory
    imgs = []
    for file in files:
        img = iio.imread(file, mode="L")
        imgs.append(img)
    
    # Stack into array and keep as numpy for now
    tomo_array = np.stack(imgs)
    
    # Apply normalization at tomogram level
    tomo_array = normalize_tomogram(tomo_array)
    return torch.as_tensor(tomo_array)

tomo = load_tomogram(tomo_path)
tomo:torch.Tensor
original_shape = tomo.shape
tomo = tomo.reshape(1,1, *original_shape).to(device)

results = model.inference(tomo, batch_size=4, patch_size= 128, overlap = 0, device = device, tqdm_progress = True)
print(results.shape)
fart = results.view(results.shape[2:]).cpu().numpy()
#1//60
#tomo_d7475d,143.0
#tomo_51a47f,112.0
#1//10
#tomo_101279  161.0         295.0         585.0
#tomo_be4a3a 111.0         602.0         896.0

plt.imshow(fart[111, ...], cmap='plasma')
plt.show()



#load patch to bcdhw
#1/10 SUBSET ['tomo_bdc097', 'tomo_d7475d', 'tomo_51a47f', 'tomo_2c607f', 'tomo_975287', 'tomo_51a77e', 'tomo_3e7407', 'tomo_412d88', 'tomo_91beab', 'tomo_cc65a9', 'tomo_1f0e78', 'tomo_e71210', 'tomo_00e463', 'tomo_f36495', 'tomo_6943e6', 'tomo_711fad', 'tomo_aff073', 'tomo_fe050c', 'tomo_24795a', 'tomo_c46d3c', 'tomo_be4a3a', 'tomo_0d4c9e', 'tomo_821255', 'tomo_47ac94', 'tomo_ac4f0d', 'tomo_12f896', 'tomo_675583', 'tomo_20a9ed', 'tomo_b2b342', 'tomo_28f9c1', 'tomo_94c173', 'tomo_935f8a', 'tomo_746d88', 'tomo_8e4919', 'tomo_da79d8', 'tomo_40b215', 'tomo_c36b4b', 'tomo_1af88d', 'tomo_a2a928', 'tomo_13973d', 'tomo_c4db00', 'tomo_568537', 'tomo_101279', 'tomo_512f98', 'tomo_7fbc49', 'tomo_0333fa', 'tomo_f2fa4a', 'tomo_a37a5c', 'tomo_ec607b', 'tomo_a8bf76', 'tomo_dfc627', 'tomo_7a9b64', 'tomo_8b6795', 'tomo_23a8e8', 'tomo_651ecd', 'tomo_67565e', 'tomo_e9fa5f', 'tomo_2bb588', 'tomo_3a0914', 'tomo_10c564', 'tomo_8e30f5']
patch_dict = torch.load(Path.cwd() / 'patch_pt_data' / 'tomo_d7475d' /'patch_112_160_384.pt' )

# tomo_d7475d,patch_112_160_384.pt,True
# tomo_d7475d,patch_112_160_400.pt,True
# tomo_d7475d,patch_112_160_416.pt,True
# tomo_d7475d,patch_112_160_432.pt,True
# tomo_d7475d,patch_112_176_384.pt,True
# tomo_d7475d,patch_112_176_400.pt,True
# tomo_d7475d,patch_112_176_416.pt,True
# tomo_d7475d,patch_112_176_432.pt,True

patch = patch_dict['patch'].to(device)
# print(patch.shape)
patch = patch.reshape(1, 1, 64, 64, 64)
# print(patch.shape)

with torch.amp.autocast(device_type = 'cuda'):
    with torch.no_grad():
        results = torch.sigmoid_(model.forward(patch))
        # results = model.forward(patch)
    
label = patch_dict['labels']
# print(f'sparse label: {label[..., :3]}')

d = label[0, 0]#(1,4)
print(f'd: {d}')
results = results.reshape(results.shape[2:]).cpu().numpy()
print(results.shape)



target_depth_slice = results[d, ...]

argmax = np.asarray(np.unravel_index(np.argmax(results), shape = results.shape))

print(f'global max indices: {argmax}')
print(f'value: {results[*argmax]}')


slice_max = np.unravel_index(np.argmax(target_depth_slice), shape = target_depth_slice.shape)
print(f'slice_max indices: {slice_max}')
print(f'value at slice_max: {target_depth_slice[slice_max]}')


gt_coords = label[0, :3].numpy()
print(f'gt_coords: {gt_coords}')
print(f'value at gt: {results[*gt_coords]}')

print(f'DISTANCE BETWEEN GLOBAL MAX AND GT: {np.sqrt(np.sum((gt_coords - argmax)**2))}')




plt.figure(0, figsize=(11,11))
plt.imshow(target_depth_slice, cmap='plasma')

# Add circle at ground truth coordinates
from matplotlib.patches import Circle

gt = Circle((gt_coords[2], gt_coords[1]), radius=3, fill=False, color='green', linewidth=2)

pred = Circle((slice_max[1], slice_max[0]), radius=3, fill=False, color='red', linewidth=2)

plt.gca().add_patch(gt)

plt.gca().add_patch(pred)

plt.show()


