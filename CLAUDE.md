# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition project for bacterial flagellar motor detection in cryo-electron tomography (cryo-ET) data. The project implements a deep learning pipeline using PyTorch and MONAI for 3D computer vision on tomographic image volumes.

## Key Architecture Components

### Core Model Architecture
- **MotorIdentifier**: Main FPN neural network in `model_defs/motoridentifier.py`
- **nnblock.py**: Custom PyTorch building blocks including PreActResBlock3d and normalization layers

### Training Pipeline
- **Trainer**: Located in `train/trainer.py` - comprehensive training loop with metrics tracking
- **PatchTomoDataset**: Custom dataset class for loading 3D tomogram patches
- **Loss Functions**: Multiple loss variants including WeightedBCELoss, FocalLoss, BCETopKLoss

### Data Organization
```
data/
├── original_data/          # Raw Kaggle competition data
├── processed/patch_pt_data/ # Preprocessed patch data in PyTorch format
models/                     # Model experiments organized by architecture type
├── fpn/                   # Feature Pyramid Network experiments  
├── simple_resnet/         # ResNet-based experiments
├── resnet/                # Other ResNet variants
```

## Common Development Commands

### Training Models
```bash
python train/train.py
```

### Model Evaluation
Key metrics implemented in trainer.py:
- `soft_dice_score()`: Soft Dice coefficient for heatmap evaluation
- `comprehensive_heatmap_metric()`: Combined correlation, spatial, and confidence scoring
- `topk_accuracy()`: Top-K accuracy for peak detection

## Architecture Details

### Feature Pyramid Network Design
The MotorIdentifier uses an FPN with encoder-decoder structure and multi-scale feature fusion.

### Training Configuration
- **Patch Size**: Typically 160x288x288 voxels
- **Downsampling Factor**: 16x for heatmap targets  
- **Gaussian Blob Sigma**: 200 angstroms for motor annotation
- **Loss**: Weighted BCE with high positive weight (420x) due to extreme class imbalance

## File Structure Navigation

- `model_defs/`: Core model definitions and neural network blocks
- `train/`: Training scripts, dataset classes, and utilities  
- `test_stuff/`: Analysis utilities and visualization tools
- `inference_results/`: Output CSV files from model inference
- `models/`: Organized experiment directories with saved weights

## Development Notes

- Models are saved in experiment-specific subdirectories under `models/`
- Training uses gradient accumulation (`batches_per_step`) for large effective batch sizes
- Cosine annealing with warmup for learning rate scheduling
- Differential learning rates for backbone vs head components