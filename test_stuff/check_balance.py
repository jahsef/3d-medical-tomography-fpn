from pathlib import Path
from collections import Counter
import torch

patches_dir = Path('data/processed/relabeled_patches')

patch_types = Counter()
tomo_counts = {}

for pt_file in patches_dir.rglob('*.pt'):
    tomo_id = pt_file.parent.name
    data = torch.load(pt_file, weights_only=False)
    patch_type = data['patch_type']

    patch_types[patch_type] += 1

    if tomo_id not in tomo_counts:
        tomo_counts[tomo_id] = Counter()
    tomo_counts[tomo_id][patch_type] += 1

print("=== Overall Distribution ===")
total = sum(patch_types.values())
for ptype, count in sorted(patch_types.items()):
    print(f"{ptype}: {count} ({100*count/total:.1f}%)")
print(f"Total: {total}")

print(f"\n=== Per-Tomogram Stats ({len(tomo_counts)} tomos) ===")
for ptype in sorted(patch_types.keys()):
    counts = [tc[ptype] for tc in tomo_counts.values()]
    print(f"{ptype}: min={min(counts)}, max={max(counts)}, avg={sum(counts)/len(counts):.1f}")
