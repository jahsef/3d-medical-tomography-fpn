from pathlib import Path
import pandas as pd
import numpy as np
import ast

original_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\train_labels.csv')
relabeled_path = Path(r'C:\Users\kevin\Documents\GitHub\kaggle-byu-bacteria-motor-comp\data\original_data\relabeled_data.csv')

# Load original data - group by tomo_id to get motor counts and coords
orig_df = pd.read_csv(original_path)
orig_motors = {}
for tomo_id, group in orig_df.groupby('tomo_id'):
    motors = []
    for _, row in group.iterrows():
        if row['Number of motors'] > 0:
            motors.append((row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2']))
    orig_motors[tomo_id] = motors

# Load relabeled data - parse normalized coords and convert to absolute
relab_df = pd.read_csv(relabeled_path)
relab_motors = {}
for _, row in relab_df.iterrows():
    tomo_id = row['tomo_id']
    coords_str = row['coordinates']
    shape = (row['z_shape'], row['y_shape'], row['x_shape'])

    coords_list = ast.literal_eval(coords_str) if coords_str != '[]' else []
    motors = [(c[0] * shape[0], c[1] * shape[1], c[2] * shape[2]) for c in coords_list]
    relab_motors[tomo_id] = motors

# Motor count distributions
orig_counts = np.array([len(m) for m in orig_motors.values()])
relab_counts = np.array([len(m) for m in relab_motors.values()])

def print_stats(name, arr):
    print(f"{name}:")
    print(f"  mean={arr.mean():.2f}, median={np.median(arr):.2f}, std={arr.std():.2f}, min={arr.min()}, max={arr.max()}")

print("=== Motor Count Distributions ===")
print_stats("Original", orig_counts)
print_stats("Relabeled", relab_counts)

# Count disagreements - original as GT
DIST_THRESH = 50  # voxels - motors within this distance are considered same
disagreements = {}

for tomo_id in orig_motors:
    if tomo_id not in relab_motors:
        disagreements[tomo_id] = len(orig_motors[tomo_id])  # all missing
        continue

    orig = np.array(orig_motors[tomo_id]) if orig_motors[tomo_id] else np.empty((0, 3))
    relab = np.array(relab_motors[tomo_id]) if relab_motors[tomo_id] else np.empty((0, 3))

    # Count unmatched original motors
    unmatched = 0
    for o in orig:
        if len(relab) == 0:
            unmatched += 1
        else:
            dists = np.linalg.norm(relab - o, axis=1)
            if dists.min() > DIST_THRESH:
                unmatched += 1

    # Count extra relabeled motors
    extra = 0
    for r in relab:
        if len(orig) == 0:
            extra += 1
        else:
            dists = np.linalg.norm(orig - r, axis=1)
            if dists.min() > DIST_THRESH:
                extra += 1

    if unmatched + extra > 0:
        disagreements[tomo_id] = unmatched + extra

print(f"\n=== Disagreements (thresh={DIST_THRESH} voxels) ===")
print(f"Tomograms with disagreements: {len(disagreements)}/{len(orig_motors)}")
print(f"Total disagreement count: {sum(disagreements.values())}")

if disagreements:
    print("\nTop disagreements:")
    for tomo_id, count in sorted(disagreements.items(), key=lambda x: -x[1])[:10]:
        print(f"  {tomo_id}: {count}")
