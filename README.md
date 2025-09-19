# Bacterial Flagellar Motor Detection in Cryo-ET

Competition project for detecting bacterial flagellar motors in cryo-electron tomography data.
**Hardware**: RTX 3080 10GB, 32GB RAM, 2TB NVMe SSD

## Technical Implementation

### Architecture
- Feature Pyramid Network (FPN) with multi-scale feature fusion
- PreAct Resnet for simple testing
- Downsampled heatmap outputs (16x reduction) for compute efficiency

### Training Pipeline
- Patch-based training on 160x288x288 volumes
- GPU-accelerated Gaussian blob annotation (200 angstrom sigma)
- Gradient accumulation for large effective batch sizes
- Custom continuous weighted BCE loss for class imbalance

### Data Processing
- Medium data scale: 70GB JPG → 300GB raw tensors


### Results
Training logs: `models/resnet_10m/custom_weight10_bce/run2/logs/training_progress.png`

Example inference output: `_saved_images/example_output.png`
