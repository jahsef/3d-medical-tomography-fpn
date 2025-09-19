# Bacterial Flagellar Motor Detection in Cryo-ET

Competition project for detecting bacterial flagellar motors in cryo-electron tomography data.

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
- Efficient pipeline: 70GB JPG → 300GB preprocessed tensors
- Memory-optimized dataset class with lazy loading
- PyTorch tensor caching for fast iteration

### Results
Training logs: `models/resnet_10m/custom_weight10_bce/run2/logs/training_progress.png`

Example inference output: `_saved_images/example_output.png`