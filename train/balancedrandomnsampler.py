import torch
import numpy as np


class BalancedRandomNSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, n, class_labels=None, balance_ratio=0.5):
        """
        Args:
            data_source: Your dataset
            n: Total number of samples to draw
            class_labels: List/array of class labels for each sample in dataset
                         If None, assumes data_source has a 'labels' or 'targets' attribute
            balance_ratio: Ratio of positive samples (0.5 = equal balance)
        """
        self.data_source = data_source
        self.n = n
        self.num_samples = len(data_source)
        self.balance_ratio = balance_ratio
        
        if self.n > self.num_samples:
            raise ValueError(f"n={n} is greater than dataset size={self.num_samples}")
        
        
        # Get indices for each class
        
        self.class_labels = np.array(class_labels)
        self.positive_indices = np.where(self.class_labels == 1)[0]  # Assuming 1 = positive class
        self.negative_indices = np.where(self.class_labels == 0)[0]  # Assuming 0 = negative class

        
    def __iter__(self):
        # Calculate how many of each class to sample
        n_positive = int(self.n * self.balance_ratio)
        n_negative = self.n - n_positive
        
        # Sample from each class
        if n_positive > len(self.positive_indices):
            # Not enough positive samples, take all and fill with negatives
            sampled_positive = self.positive_indices.copy()
            n_negative = self.n - len(sampled_positive)
        else:
            sampled_positive = np.random.choice(self.positive_indices, size=n_positive, replace=False)
            
        if n_negative > len(self.negative_indices):
            # Not enough negative samples, take all and fill with positives  
            sampled_negative = self.negative_indices.copy()
            remaining = self.n - len(sampled_negative) - len(sampled_positive)
            if remaining > 0:
                additional_positive = np.random.choice(
                    self.positive_indices, size=remaining, replace=True
                )
                sampled_positive = np.concatenate([sampled_positive, additional_positive])
        else:
            sampled_negative = np.random.choice(self.negative_indices, size=n_negative, replace=False)
        
        # Combine and shuffle
        all_indices = np.concatenate([sampled_positive, sampled_negative])
        np.random.shuffle(all_indices)
        
        return iter(all_indices.tolist())
    
    def __len__(self):
        return self.n


# Usage example:
# sampler = BalancedRandomNSampler(
#     data_source=your_dataset, 
#     n=1000,  # Sample 1000 total samples
#     class_labels=your_labels,  # Or let it auto-detect from dataset
#     balance_ratio=0.5  # 50/50 split
# )