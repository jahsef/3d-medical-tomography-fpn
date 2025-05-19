import torch



tensor = torch.load(r'C:\Users\kevin\Documents\GitHub\kaggle BYU bacteria motor\normalized_pt_data\train\tomo_2acf68.pt')
print(tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(tensor[0].numpy(), cmap = 'Grays_r')
plt.show()
