
from pathlib import Path
import torch
from collections import defaultdict


def analyze_dataset_balance(data_dir: Path):
    """Analyze the balance of positive/negative patches in the dataset."""
    stats = defaultdict(lambda: {'positive': 0, 'negative': 0})
    
    for tomo_dir in data_dir.iterdir():
        print(f'checking: {tomo_dir}')
        if not tomo_dir.is_dir():
            continue
            
        tomo_id = tomo_dir.name
        
        # Look for .pt files directly in the tomo directory
        for pt_file in tomo_dir.glob('*.pt'):
            try:
                data = torch.load(pt_file, map_location='cpu')
                label_data = data['labels']
                has_motor = False
                
                if label_data.numel() > 0:
                    # Check if any row has non-zero coordinates or confidence
                    if label_data.dim() == 2:
                        has_motor = torch.any(label_data != 0).item()
                    else:
                        has_motor = torch.any(label_data != 0).item()
                
                if has_motor:
                    stats[tomo_id]['positive'] += 1
                else:
                    stats[tomo_id]['negative'] += 1
            except Exception as e:
                print(f"Error loading {pt_file}: {e}")
    
    # Print statistics
    total_positive = sum(s['positive'] for s in stats.values())
    total_negative = sum(s['negative'] for s in stats.values())
    total_patches = total_positive + total_negative
    
    print(f"\nDataset Statistics:")
    print(f"Total patches: {total_patches}")
    if total_patches > 0:
        print(f"Positive patches: {total_positive} ({total_positive/total_patches:.1%})")
        print(f"Negative patches: {total_negative} ({total_negative/total_patches:.1%})")

def prune_empty_dirs(master_tomo_dir: Path):
    """Remove empty directories."""
    print('Removing empty directories...')
    removed_count = 0
    tomo_dirs = [x for x in master_tomo_dir.iterdir() if x.is_dir()]
    for dir in tomo_dirs:
        # Check if directory has any .pt files
        pt_files = list(dir.glob('*.pt'))
        
        if len(pt_files) == 0:
            dir.rmdir()
            removed_count += 1
    print(f'Removed {removed_count} empty directories')
    

if __name__ == '__main__':
    dir = Path('patch_pt_data')
    analyze_dataset_balance(dir)