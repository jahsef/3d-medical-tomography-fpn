import torch
from pathlib import Path

class TomoDataset(torch.utils.data.Dataset):
    def __init__(self, list_tuple_path_labels: list[tuple], transform):
        """_summary_
        Args:
            list_tuple_path_labels (list[tuple]): list of tuple of path, labels
        """
        
        super().__init__()
        self.list_tuple_path_labels = list_tuple_path_labels
        self.transform = transform

    def __len__(self):
        return len(self.list_tuple_path_labels)

    def __getitem__(self, idx):
        path, labels = self.list_tuple_path_labels[idx]
        
        # Better error messages
        assert path.stem == labels['tomo_id'], \
            f"ID mismatch: tomo={path.stem}, label={labels['tomo_id']}"

        tensor = torch.load(path, mmap=True)
        
        if tensor.ndim == 3:
            # print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            
            tensor = tensor.unsqueeze(0)  # shape becomes [1, D, H, W]
            # print(tensor.shape)
            
            
        # print(tensor.shape)
        assert tensor.shape[1:] == labels['shape'], \
            f"Shape mismatch in {path.stem}: got {tensor.shape[1:]}, expected {labels['shape']}"
        
        return self.transform(tensor), labels['xyzconf']