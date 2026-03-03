# TF-KF: Transformer-based Kalman Filter for 3D Human Pose Estimation

This project implements a Transformer-based Kalman Filter approach for 3D human pose estimation from 2D images.

## Features

- **Multiple Model Architectures**:
  - `kfl_QRFf_transformer`: Transformer + Kalman Filter (Q, R, F matrices)
  - `kfl_QRFf`: LSTM + Kalman Filter (Q, R, F matrices)
  - `kfl_QRf`: LSTM + Kalman Filter (Q, R matrices)
  - `kfl_K`: LSTM + Kalman Filter (K only)
  - `lstm`: Pure LSTM baseline

- **Two Prediction Modes**:
  - Current frame prediction: Input frame t → Predict 3D pose of frame t
  - Next frame prediction: Input frame t → Predict 3D pose of frame t+1

- **Mixed Precision Training**: Supports FP16 training for faster computation

## Project Structure

```
TF-KF/
├── helper/
│   ├── config.py           # Configuration parameters
│   ├── dt_utils.py         # Data loading and preprocessing
│   ├── train_helper.py     # Training utilities
│   ├── utils.py            # General utilities
│   ├── checkpoint.py       # Checkpoint management
│   └── data_lodaer.py      # Data loader for Inception
├── model_runner/
│   └── klstm/
│       ├── kfl_QRFf_transformer.py
│       ├── kfl_QRFf.py
│       ├── kfl_QRf.py
│       ├── kfl_K.py
│       └── lstm.py
├── nets/
│   └── inception_resnet_v2.py
├── train_h36m.py           # Training script
├── evaluate_mpjpe.py       # Evaluation script
├── compare_models.py       # Model comparison tool
└── model/                  # Model checkpoints (not included)
```

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- Pillow
- tqdm

## Dataset

This project uses the Human3.6M dataset. Please download and place it in the `data/h36m/` directory.

## Usage

### Training

```bash
python train_h36m.py
```

### Evaluation

```bash
python evaluate_mpjpe.py
```

### Model Comparison

```bash
# Compare all models for a specific sample
python compare_models.py --sample_idx 100

# Specify output directory
python compare_models.py --sample_idx 100 --output_dir my_output
```

## Model Performance

| Model | MPJPE (mm) | Description |
|-------|------------|-------------|
| Inception (baseline) | ~89 | End-to-end CNN |
| kfl_QRFf_transformer | ~113 | Transformer + Kalman Filter |
| kfl_QRFf | - | LSTM + Kalman Filter |

## Citation

If you find this project useful, please cite:

```bibtex
@misc{tf-kf,
  author = {Zhao},
  title = {TF-KF: Transformer-based Kalman Filter for 3D Human Pose Estimation},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/1583477320/TF-KF}
}
```

## License

MIT License
