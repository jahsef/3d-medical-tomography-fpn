import torch
import imageio as iio


tensor = torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train\tomo_0a8f05.pt')
print(tensor[0])

image = iio.v3.imread(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\original_data\train\tomo_0a8f05\slice_0000.jpg')
print(image)