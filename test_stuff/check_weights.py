import torch


poo  = torch.load(r"C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\models\fpn_comparison\old_fpn2\weights\best.pt")

fart = [poo[key].max().item() for key in poo.keys()]


fart_arr = torch.tensor(fart)
print(f'all key max stats distribution')
print(f'mean: {fart_arr.mean()}')
print(f'median: {fart_arr.median()}')
print(f'std: {fart_arr.std()}')
print(f'min: {fart_arr.min()}')
print(f'max: {fart_arr.max()}')


def printshit(tensor):
    print(f'mean: {tensor.mean()}')
    print(f'median: {tensor.median()}')
    print(f'std: {tensor.std()}')
    print(f'min: {tensor.min()}')
    print(f'max: {tensor.max()}')

print("\n")
stem_keys = list(poo.keys())[:2]
for key in stem_keys:
    print(f'key: {key}')
    printshit(poo[key])