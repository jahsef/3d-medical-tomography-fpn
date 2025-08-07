# Gemini Project Context: Kaggle BYU Bacteria Motor Competition

## Project Overview

This project is for the Kaggle competition on identifying bacterial motors in tomogram data. It appears to be a machine learning project focused on image classification or segmentation.

**Key Technologies:**

*   **Language:** Python
*   **ML Framework:** PyTorch
*   **Specialized Libraries:** MONAI (inferred from `model_defs/test_monai.py` and `model_defs/motoridentifier.py`)

**Architecture:**

The project is structured to handle large image datasets (tomograms) and train deep learning models.

*   `original_data/`: Contains the raw competition data.
*   `patch_pt_data/`, `relabel_data/`: These directories likely store processed or augmented versions of the data for training.
*   `model_defs/`: Contains the Python source code for the model architectures (e.g., `motoridentifier.py`, `nnblock.py`). The models seem to be based on U-Net or FPN architectures, common for medical imaging tasks.
*   `models/`: Likely stores the saved model checkpoints.
*   `inference_results/`: Stores the output of model predictions on test data.

## Building and Running

*TODO: Add specific commands for training, inference, and data processing.*

**Example (placeholder):**

```bash
# Train the model
python train.py --model fpn_optimized --epochs 10 --data_path patch_pt_data/

# Run inference
python inference.py --model_path models/fpn_optimized/best_model.pth --test_data_path original_data/test/
```

## Development Conventions

*   **Data:** Raw data is kept separate from processed data.
*   **Models:** Model definitions are in `model_defs/`. Saved models are in `models/`.
*   **Configuration:** No explicit configuration files were found. Configuration may be handled through command-line arguments in the training/inference scripts.
*   **Dependencies:** No `requirements.txt` was found. It is recommended to create one to track project dependencies.
