import torch


num_patches = 5
c = 1
d,h,w = 64,64,64
patches = torch.empty(size = (num_patches,c,d,h,w ))
patch = torch.zeros(size = (c,d,h,w))

print('before')
print(patches.shape)
print(patch.shape)

for i in range(num_patches):
    patches[i] = patch.clone().contiguous()

print('after')
print(patches.shape)
