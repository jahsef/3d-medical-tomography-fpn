import torch

fart = torch.load(r"C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\normalized_pt_data\train\tomo_0c2749\patch_0_0_60.pt")
print(fart)


# fart = torch.zeros(size = (512,512,512))

# poop = fart[:64, :64, :64].clone().contiguous()

# torch.save(poop, './poop.pt')