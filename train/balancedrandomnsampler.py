import torch
import numpy as np


class RandomNSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, n):
        """
        Args:
            data_source: Your dataset
            n: Total number of samples to draw
        """
        self.data_source = data_source
        self.n = n
        self.num_samples = len(data_source)
        
        if self.n > self.num_samples:
            raise ValueError(f"n={n} is greater than dataset size={self.num_samples}")
        
    def __iter__(self):
        # Sample n indices randomly without replacement
        indices = np.random.choice(self.num_samples, size=self.n, replace=False)
        return iter(indices.tolist())
    
    def __len__(self):
        return self.n


# Usage example:
# sampler = RandomNSampler(
#     data_source=your_dataset, 
#     n=1000  # Sample 1000 total samples randomly
# )