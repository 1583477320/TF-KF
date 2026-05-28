# TF-KF: Transformer-based Kalman Filter for 3D Human Pose Estimation

This project implements a neural Kalman Filter approach for 3D human pose estimation from 2D CNN features. Instead of using fixed Kalman filter parameters, a Transformer or LSTM dynamically predicts the state transition matrix **F**, process noise covariance **Q**, measurement noise covariance **R**, and Kalman gain **K** at each time step, enabling the filter to adapt to complex human motion patterns.

## Model Variants

| Model | Description |
|-------|-------------|
| `kfl_QRFf_transformer` | Transformer encoder predicts Q, R, F matrices for Kalman filter |
| `kfl_QRFf` | LSTM predicts Q, R, F matrices for Kalman filter |
| `kfl_QRf` | LSTM predicts Q, R matrices (F fixed) |
| `kfl_K` | LSTM predicts Kalman gain K directly |
| `lstm` | Pure LSTM baseline (no Kalman filter) |
| `pure_kalman` | Classic Kalman filter with fixed F, Q, R matrices |
| `Inception` | End-to-end InceptionResNetV2 CNN from images |

All sequence models operate on 2D CNN features extracted by InceptionResNetV2 (2048-d or 51-d keypoints), and support two prediction modes:
- **Current frame**: input frame *t* → predict 3D pose of frame *t*
- **Next frame**: input frame *t* → predict 3D pose of frame *t+1*

## Project Structure

```
TF-KF/
├── helper/
│   ├── config.py                  # Global configuration parameters
│   ├── dt_utils.py                # Data loading, preprocessing, H36M dataset preparation
│   ├── train_helper.py            # Batch preparation and training utilities
│   ├── utils.py                   # Logging, state management, data shuffling
│   ├── checkpoint.py              # Checkpoint save/load
│   └── data_lodaer.py             # Image transforms for Inception model
├── model_runner/
│   ├── klstm/
│   │   ├── kfl_QRFf_transformer.py  # Transformer + Kalman Filter (Q, R, F)
│   │   ├── kfl_QRFf.py              # LSTM + Kalman Filter (Q, R, F)
│   │   ├── kfl_QRf.py               # LSTM + Kalman Filter (Q, R)
│   │   ├── kfl_K.py                 # LSTM + Kalman Filter (K only)
│   │   └── pure_kalman.py           # Pure Kalman Filter (fixed matrices)
│   ├── lstm/
│   │   └── pt_lstm.py               # Pure LSTM baseline
│   ├── cnn/
│   │   ├── inception_train.py       # InceptionResNetV2 training loop
│   │   └── inception_eval.py        # InceptionResNetV2 evaluation
│   └── model_provider.py            # Model factory
├── nets/
│   └── inception_resnet_v2.py       # InceptionResNetV2 architecture
├── train_h36m.py                    # Main training script (Transformer/LSTM KF models)
├── train_transformer.py             # Transformer-specific training script
├── train_pure_kalman.py             # Pure Kalman filter training & evaluation
├── train.py                         # InceptionResNetV2 CNN training
├── evaluate_mpjpe.py                # Per-action MPJPE evaluation with formatted tables
├── compare_models.py                # Multi-model prediction comparison & visualization
├── analyze_model_weights.py         # Model weight analysis tool
├── optimize_kalman_params.py        # Kalman filter hyperparameter grid search
├── use_best_params.py               # Apply best Kalman parameters from optimization
├── generate_latex_table.py          # Generate LaTeX results tables
├── h5.py                            # H36M dataset conversion utilities
├── mt.py                            # 3D pose visualization tool
├── data/h36m/                       # Human3.6M dataset (images, annotations, cache)
└── model/                           # Trained model checkpoints
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy, h5py, Matplotlib, Pillow, tqdm
- TensorBoard (for training visualization)

## Dataset

This project uses the [Human3.6M](http://vision.imar.ro/human3.6m/) dataset. Place it under `data/h36m/`:

```
data/h36m/
├── annot/
│   ├── train.h5
│   ├── valid.h5
│   └── ...
├── images/
│   ├── train/
│   └── test/
└── cache/
    ├── train.npz
    └── valid.npz
```

## Usage

### Training a Kalman Filter model

```bash
# Train Transformer + Kalman Filter (default)
python train_h36m.py

# Train LSTM + Kalman Filter
# Edit model_name in main() to "kfl_QRFf"
```

Key training features: AdamW optimizer, ReduceLROnPlateau scheduler, gradient clipping, early stopping, TensorBoard logging, and automatic checkpoint saving (`model_final.pth`).

### Training the Inception CNN

```bash
python train.py --mode 1
```

### Training pure Kalman filter

```bash
python train_pure_kalman.py
```

### Evaluation

```bash
# Per-action MPJPE evaluation with formatted result table
python evaluate_mpjpe.py

# Multi-model comparison with visualizations
python compare_models.py --sample_idx 100 --output_dir output/comparison
```

### Analysis tools

```bash
# Analyze model weight statistics
python analyze_model_weights.py

# Grid search for optimal Kalman filter parameters
python optimize_kalman_params.py

# Apply best found parameters
python use_best_params.py

# Generate LaTeX formatted result table
python generate_latex_table.py
```

## Metric

**MPJPE (Mean Per Joint Position Error)** in millimeters — the Euclidean distance between predicted and ground-truth 3D joint positions, averaged over all 17 joints. Evaluated under Human3.6M Protocol #1 (no rigid alignment).

## License

MIT License
