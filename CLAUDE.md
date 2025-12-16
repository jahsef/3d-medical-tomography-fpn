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
├── original_data/
│   ├── train/                    # Raw tomogram images (tomo_N/0.jpg, 1.jpg, ...)
│   ├── train_labels.csv          # Original labels (deprecated)
│   └── RELABELED_DATA.csv        # New relabeled data with folds
├── processed/
│   ├── patch_pt_data/            # Old preprocessed patches
│   └── relabeled_patch_pt_data/  # New patches from relabeled data
models/                           # Model experiments organized by architecture type
├── fpn/                          # Feature Pyramid Network experiments
├── simple_resnet/                # ResNet-based experiments
├── resnet/                       # Other ResNet variants
```

### Data Conversion Pipeline
`test_stuff/convert_pt.py` - Converts raw tomograms to training patches

**Architecture**: 3 producer processes load tomograms in parallel, main process consumes and saves

**Label Format (RELABELED_DATA.csv)**:
- Columns: `tomo_id, z, y, x, z_shape, y_shape, x_shape, voxel_spacing, coordinates, fold`
- `fold` column provides pre-set cross-validation splits (use instead of random splitting)
- `fold=-1` indicates excluded samples

**Patch Sampling Strategy** (per tomogram):
- **Single motor**: N patches per motor, centered around motor location
- **Multi motor**: up to `total_motors * Y` patches containing 2+ motors
- **Hard negative**: M patches per motor, close but not containing motor
- **Random negative**: X patches per motor, anywhere not in other categories

Each patch dict includes `patch_type`: 'single', 'multi', 'hard_negative', 'negative'

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


Core Principles

FAIL FAST: Explicit errors are better than silent failures
Self-documenting code: Clear names and structure over excessive comments
Readability: Reduce nesting, keep functions focused

Function Design

Type hints required at minimum in function signatures
NO default arguments - force explicit parameter passing
Size limit: Functions >100 lines should be split into smaller pieces
Private helpers: Prefix with underscore _helper_function() if not meant for public API
At least one assertion in major functions to validate assumptions

Data Access

NEVER use dict.get() except for kwargs or special edge cases
Access dictionaries directly: config['learning_rate'] not config.get('learning_rate')
Document any exceptions to this rule with inline comments explaining why

Naming Conventions

Descriptive over abbreviated: batch_size not bs, num_epochs not ne
Exception: Standard ML conventions (x, y, loss, optimizer) are acceptable
Variables should clearly communicate intent

Comments

Minimize noise: Don't comment self-explanatory code
Comment complexity: Explain non-obvious logic or algorithms
Document unconventional choices: If code deviates from standard patterns, explain why
Let the code speak for itself when possible

Code Organization

No globals: All state passed explicitly as function parameters
Reduce nesting: Early returns, guard clauses, helper functions

Error Handling

Explicit failures: Let errors propagate rather than catch and suppress
Meaningful assertions: assert x.shape[0] == batch_size, f"Expected {batch_size}, got {x.shape[0]}"
Validate assumptions early and loudly