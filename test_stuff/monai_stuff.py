#!/usr/bin/env python3

import torch
import sys
import os
from monai import transforms

def test_transforms_on_pt(filepath):
    """Load a .pt file and test transform shape changes"""
    
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} not found!")
        return
    
    try:
        # Load the .pt file
        data = torch.load(filepath, map_location='cpu')
        
        # Extract patch tensor (adjust key name as needed)
        if isinstance(data, dict):
            # Try common keys
            patch = None
            for key in ['patch', 'image', 'data', 'input']:
                if key in data:
                    patch = data[key]
                    print(f"Using key '{key}' from dict")
                    break
            
            if patch is None:
                print("Available keys:", list(data.keys()))
                key = input("Enter the key name for your patch data: ")
                patch = data[key]
        else:
            patch = data
        
        print(f"Original patch shape: {patch.shape}")
        print(f"Original patch dtype: {patch.dtype}")
        print("-" * 50)
        
        # Test different rotation axes
        test_data = {"patch": patch}
        
        axes_to_test = [[0,1], [0,2], [1,2]]
        
        for axes in axes_to_test:
            try:
                transform = transforms.RandRotate90d(
                    keys=["patch"], 
                    prob=1.0,  # Always apply
                    spatial_axes=axes
                )
                
                result = transform(test_data.copy())
                new_shape = result["patch"].shape
                
                print(f"Axes {axes}: {patch.shape} -> {new_shape}")
                
                if new_shape == patch.shape:
                    print(f"  ✓ SAFE - shape preserved")
                else:
                    print(f"  ✗ UNSAFE - shape changed!")
                    
            except Exception as e:
                print(f"Axes {axes}: ERROR - {e}")
        
        print("-" * 50)
        print("Use the 'SAFE' axes pair in your RandRotate90d transform")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":

    test_transforms_on_pt(r"C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\patch_pt_data\tomo_00e047\patch_0_576_504.pt")
