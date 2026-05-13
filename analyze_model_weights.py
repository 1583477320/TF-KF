import torch
import numpy as np
import os
from collections import OrderedDict

def analyze_model_weights(model_path):
    """
    分析模型权重
    
    Args:
        model_path: 模型文件路径
    """
    print("=" * 80)
    print(f"Analyzing Model Weights: {model_path}")
    print("=" * 80)
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nCheckpoint type: New format (contains model_state_dict)")
        print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"Loss: {checkpoint.get('loss', 'N/A')}")
        print(f"Metrics: {checkpoint.get('metrics', {})}")
    else:
        state_dict = checkpoint
        print(f"\nCheckpoint type: Old format (direct state_dict)")
    
    print(f"\nTotal parameters: {len(state_dict)}")
    
    print("\n" + "=" * 80)
    print("Parameter Analysis")
    print("=" * 80)
    
    param_info = []
    total_params = 0
    total_nan = 0
    total_inf = 0
    
    for name, param in state_dict.items():
        param_shape = param.shape
        param_size = param.numel()
        total_params += param_size
        
        has_nan = torch.isnan(param).any().item()
        has_inf = torch.isinf(param).any().item()
        
        if has_nan:
            total_nan += param_size
        if has_inf:
            total_inf += param_size
        
        param_dtype = param.dtype
        param_mean = param.float().mean().item()
        param_std = param.float().std().item()
        param_min = param.float().min().item()
        param_max = param.float().max().item()
        
        info = {
            'name': name,
            'shape': tuple(param_shape),
            'size': param_size,
            'dtype': str(param_dtype),
            'mean': param_mean,
            'std': param_std,
            'min': param_min,
            'max': param_max,
            'has_nan': has_nan,
            'has_inf': has_inf
        }
        param_info.append(info)
        
        status = "OK"
        if has_nan:
            status = "❌ NaN"
        elif has_inf:
            status = "❌ Inf"
        
        print(f"\n{name:50s}")
        print(f"  Shape: {param_shape}")
        print(f"  Size: {param_size:,} ({param_size / 1e6:.2f}M)")
        print(f"  Dtype: {param_dtype}")
        print(f"  Mean: {param_mean:8.4f}")
        print(f"  Std:  {param_std:8.4f}")
        print(f"  Min:  {param_min:8.4f}")
        print(f"  Max:  {param_max:8.4f}")
        print(f"  Status: {status}")
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"Parameters with NaN: {total_nan:,} ({total_nan / total_params * 100:.2f}%)")
    print(f"Parameters with Inf: {total_inf:,} ({total_inf / total_params * 100:.2f}%)")
    print(f"Valid parameters: {total_params - total_nan - total_inf:,} ({(total_params - total_nan - total_inf) / total_params * 100:.2f}%)")
    
    if total_nan > 0 or total_inf > 0:
        print("\n" + "!" * 80)
        print("WARNING: Model weights contain NaN or Inf values!")
        print("!" * 80)
        print("\nThis indicates:")
        print("  1. Training instability (gradient explosion)")
        print("  2. Numerical issues in the model")
        print("  3. Learning rate too high")
        print("  4. Missing gradient clipping")
        print("\nRecommendations:")
        print("  1. Retrain the model with lower learning rate")
        print("  2. Add gradient clipping")
        print("  3. Use more conservative hyperparameters")
        print("  4. Check for NaN/Inf during training")
        print("\n" + "!" * 80)
    
    print("\n" + "=" * 80)
    
    return {
        'total_params': total_params,
        'total_nan': total_nan,
        'total_inf': total_inf,
        'param_info': param_info
    }


def compare_models(model_paths, model_names):
    """
    比较多个模型的权重
    
    Args:
        model_paths: 模型路径列表
        model_names: 模型名称列表
    """
    print("=" * 80)
    print("Comparing Multiple Models")
    print("=" * 80)
    
    all_results = []
    
    for model_path, model_name in zip(model_paths, model_names):
        print(f"\n\n{'=' * 80}")
        print(f"Model: {model_name}")
        print(f"{'=' * 80}")
        
        result = analyze_model_weights(model_path)
        result['model_name'] = model_name
        all_results.append(result)
    
    print("\n\n" + "=" * 80)
    print("Comparison Summary")
    print("=" * 80)
    
    print(f"\n{'Model':<30s} {'Total Params':<15s} {'NaN %':<10s} {'Inf %':<10s} {'Valid %':<10s}")
    print("-" * 80)
    
    for result in all_results:
        total_params = result['total_params']
        total_nan = result['total_nan']
        total_inf = result['total_inf']
        valid_params = total_params - total_nan - total_inf
        
        nan_pct = total_nan / total_params * 100 if total_params > 0 else 0
        inf_pct = total_inf / total_params * 100 if total_params > 0 else 0
        valid_pct = valid_params / total_params * 100 if total_params > 0 else 0
        
        print(f"{result['model_name']:<30s} {total_params:>15,} {nan_pct:>9.2f}% {inf_pct:>9.2f}% {valid_pct:>9.2f}%")
    
    print("=" * 80)


def main():
    """
    主函数
    """
    print("=" * 80)
    print("Model Weight Analyzer")
    print("=" * 80)
    
    model_dir = "/home/zhao/pyproject/TF-KF/model"
    
    print("\nAvailable models:")
    print("-" * 80)
    
    models_to_analyze = []
    
    kfl_QRFf_transformer_path = f"{model_dir}/kfl_QRFf_transformer/model_final.pth"
    if os.path.exists(kfl_QRFf_transformer_path):
        models_to_analyze.append((kfl_QRFf_transformer_path, "kfl_QRFf_transformer"))
        print(f"  ✓ kfl_QRFf_transformer")
    else:
        print(f"  ✗ kfl_QRFf_transformer (not found)")
    
    kfl_QRFf_path = f"{model_dir}/kfl_QRFf/model_final.pth"
    if os.path.exists(kfl_QRFf_path):
        models_to_analyze.append((kfl_QRFf_path, "kfl_QRFf"))
        print(f"  ✓ kfl_QRFf")
    else:
        print(f"  ✗ kfl_QRFf (not found)")
    
    print("-" * 80)
    
    if not models_to_analyze:
        print("\nNo models found to analyze!")
        return
    
    if len(models_to_analyze) == 1:
        analyze_model_weights(models_to_analyze[0][0])
    else:
        model_paths = [m[0] for m in models_to_analyze]
        model_names = [m[1] for m in models_to_analyze]
        compare_models(model_paths, model_names)
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
