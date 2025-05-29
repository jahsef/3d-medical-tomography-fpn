from pathlib import Path
import sys
import torch
current_dir = Path.cwd()
sys.path.append(str(Path.cwd()))
#added model_defs to path
from model_defs.motoridentifier import MotorIdentifier
import time

device = torch.device('cuda')
model = MotorIdentifier(max_motors=5).to(device)
model.load_state_dict(torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\small_custom_cnn\deeper_skinny\best.pt'))
tomo_path = r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_val_fulltomo\tomo_00e463.pt'

tomo = torch.load(tomo_path)
tomo:torch.Tensor
original_shape = tomo.shape
tomo = tomo.reshape(1,1, *original_shape).to(device)
print(tomo.dtype)

start = time.perf_counter()
results = model.inference(tomo, num_patches_per_batch= 69420, patch_size= 64, stride = int(64*7/8), conf_threshold= 0.9925)

end = time.perf_counter()


print(f'runtime: {end - start}')
if results is None:
    print('results none')
else:
    print(results.shape)

    print(results)
