from pathlib import Path
import pandas as pd
import numpy as np
from collections import defaultdict

csv_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train_labels.csv')
patch_size = (160, 288, 288)

df = pd.read_csv(csv_path)

# Group motors by tomo_id
motors_by_tomo = defaultdict(list)
for _, row in df.iterrows():
    if row['Number of motors'] > 0:
        tomo_id = row['tomo_id']
        coords = (row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2'])
        motors_by_tomo[tomo_id].append(coords)

# Check for overlapping motors (within patch_size of each other)
overlap_counts = {}
for tomo_id, motors in motors_by_tomo.items():
    if len(motors) < 2:
        continue

    count = 0
    motors_arr = np.array(motors)
    for i in range(len(motors_arr)):
        for j in range(i + 1, len(motors_arr)):
            diff = np.abs(motors_arr[i] - motors_arr[j])
            if np.all(diff < patch_size):
                count += 1

    if count > 0:
        overlap_counts[tomo_id] = count

print(f"Tomograms with motors that could share a patch (within {patch_size}):\n")
for tomo_id, count in sorted(overlap_counts.items(), key=lambda x: -x[1]):
    print(f"  {tomo_id}: {count} overlapping pairs")

print(f"\nTotal: {len(overlap_counts)} tomograms with potential overlap")
print(f"Total overlapping pairs: {sum(overlap_counts.values())}")
