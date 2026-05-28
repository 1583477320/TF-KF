#!/usr/bin/env python3
"""
生成 LaTeX 格式的模型对比表格并保存到 md 文件
"""

import sys
import os

sys.path.insert(0, '/home/zhao/pyproject/TF-KF')

from helper import config
from helper import dt_utils as dut
from evaluate_mpjpe import evaluate_model_by_action, load_model, find_model_files, H36M_ACTIONS
from compare_models import load_pure_kalman_model
from model_runner.klstm.pure_kalman import PureKalmanFilter

import torch
import numpy as np
from tqdm import tqdm
from helper import utils as ut
from helper import train_helper as th

def generate_latex_table(results_dict, output_file="results.md"):
    """
    生成 LaTeX 格式的结果表格并保存到 md 文件中
    
    Args:
        results_dict: 包含各模型结果的字典
        output_file: 输出文件路径（默认：results.md）
    """
    models = list(results_dict.keys())
    
    # 生成 LaTeX 表格内容
    latex_content = "\\begin{table*}[htbp]\n"
    latex_content += "  \\centering\n"
    latex_content += "  \\caption{Average 3D joint error on Human3.6M for test subjects. The error is given in [mm].}\n"
    latex_content += "  \\label{tab:h36m_results}\n"
    latex_content += "  \\resizebox{\\textwidth}{!}{\n"
    latex_content += "  \\begin{tabular}{l" + "c" * (len(H36M_ACTIONS) + 1) + "}\n"
    latex_content += "    \\toprule\n"
    
    # 表头
    header = "    Model "
    for action in H36M_ACTIONS:
        header += f"& {action} "
    header += "& Mean \\\\\n"
    latex_content += header
    
    latex_content += "    \\midrule\n"
    
    # 表格内容
    for model in models:
        row = f"    {model} "
        for action in H36M_ACTIONS:
            mpjpe = results_dict[model]["action_mpjpe"].get(action)
            if mpjpe is not None:
                row += f"& {mpjpe:.2f} "
            else:
                row += "& - "
        
        overall = results_dict[model]["overall_mpjpe"]
        if overall is not None:
            row += f"& {overall:.2f} \\\\\n"
        else:
            row += "& - \\\\\n"
        latex_content += row
    
    latex_content += "    \\bottomrule\n"
    latex_content += "  \\end{tabular}\n"
    latex_content += "  }\n"
    latex_content += "\\end{table*}\n"
    
    # 保存到 md 文件
    with open(output_file, 'w') as f:
        f.write("# Human3.6M Model Evaluation Results\n\n")
        f.write("## MPJPE Comparison (Mean Per Joint Position Error)\n\n")
        f.write("The following table shows the MPJPE results for different models on the Human3.6M dataset.\n\n")
        f.write("```latex\n")
        f.write(latex_content)
        f.write("```\n\n")
        f.write("### Key Notes:\n")
        f.write("- Error metric: MPJPE (Mean Per Joint Position Error)\n")
        f.write("- Unit: millimeters (mm)\n")
        f.write("- Lower values indicate better performance\n")
    
    print(f"\nLaTeX table saved to {output_file}")


def extract_action_from_path(frame_path):
    """从帧路径中提取动作名称"""
    parts = frame_path.replace('\\', '/').split('/')
    for part in parts:
        if 'Directions' in part:
            return 'Directions'
        elif 'Discussion' in part:
            return 'Discussion'
        elif 'Eating' in part:
            return 'Eating'
        elif 'Greeting' in part:
            return 'Greeting'
        elif 'Phoning' in part:
            return 'Phoning'
        elif 'Photo' in part:
            return 'Photo'
        elif 'Posing' in part:
            return 'Posing'
        elif 'Purchases' in part:
            return 'Purchases'
        elif 'Sitting' in part:
            return 'Sitting'
        elif 'SittingDown' in part:
            return 'SittingDown'
        elif 'Smoking' in part:
            return 'Smoking'
        elif 'Waiting' in part:
            return 'Waiting'
        elif 'WalkDog' in part:
            return 'WalkDog'
        elif 'Walking' in part:
            return 'Walking'
        elif 'WalkTogether' in part:
            return 'WalkTogether'
    return 'Unknown'


def compute_mpjpe(pred, gt):
    """计算 MPJPE"""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    
    if pred.ndim == 2:
        pred = pred.reshape(-1, 17, 3)
    if gt.ndim == 2:
        gt = gt.reshape(-1, 17, 3)
    
    joint_errors = np.sqrt(np.sum((pred - gt) ** 2, axis=2))
    return np.mean(joint_errors) * 1000


def evaluate_kalman_by_action(model, params, X, Y, index_list, S_list, R_L_list, F_list, batch_size, device):
    """
    评估纯卡尔曼滤波器模型（按动作类别）
    使用CNN特征作为输入
    """
    model.eval()
    
    NOUT = params["n_output"]
    
    params["reset_state"] = -1
    n_batches = len(index_list) // batch_size
    
    all_preds = []
    all_gts = []
    all_actions = []
    
    # 获取CNN特征用于输入
    X_input = X
    
    eval_pbar = tqdm(range(n_batches), desc="Evaluating Kalman", unit="batch", leave=False)
    for minibatch_index in eval_pbar:
        start_idx = minibatch_index * batch_size
        end_idx = start_idx + batch_size
        
        # 获取当前批次的数据
        batch_indices = index_list[start_idx:end_idx]
        
        # 获取姿态真值
        batch_y = []
        batch_actions = []
        for seq_idx in batch_indices:
            if seq_idx < len(Y):
                batch_y.append(Y[seq_idx])
                # 从路径提取动作类别
                if seq_idx < len(F_list):
                    frame_paths = F_list[seq_idx]
                    if len(frame_paths) > 0:
                        action = extract_action_from_path(frame_paths[0])
                        batch_actions.append(action)
                    else:
                        batch_actions.append("Unknown")
                else:
                    batch_actions.append("Unknown")
            else:
                batch_y.append(np.zeros((1, NOUT)))
                batch_actions.append("Unknown")
        
        if len(batch_y) == 0:
            continue
            
        # 将批次数据堆叠
        batch_y = np.stack(batch_y)
        batch_X = []
        for seq_idx in batch_indices:
            if seq_idx < len(X_input):
                batch_X.append(X_input[seq_idx])
            else:
                batch_X.append(np.zeros((1, NOUT)))
        batch_X = np.stack(batch_X)
        
        # 转换为张量
        x = torch.from_numpy(batch_X).float().to(device)
        y = torch.from_numpy(batch_y).float().to(device)
        
        # 创建掩码（假设所有数据都有效）
        repeat_data = torch.ones(x.shape[0], x.shape[1]).to(device)
        
        actual_bsz = x.shape[0]
        
        # 初始化卡尔曼滤波器状态
        x_init = torch.zeros(actual_bsz, NOUT).to(device)
        P_init = torch.eye(NOUT).unsqueeze(0).expand(actual_bsz, -1, -1).to(device)
        
        # 执行卡尔曼滤波
        with torch.no_grad():
            x_filtered, _ = model.forward(x, repeat_data, x_init, P_init)
        
        # 获取最后一个时间步的输出
        pred = x_filtered[:, -1, :]
        gt = y[:, -1, :]
        
        all_preds.append(pred.cpu().numpy())
        all_gts.append(gt.cpu().numpy())
        all_actions.extend(batch_actions)
    
    if len(all_preds) == 0:
        return {
            "action_mpjpe": {action: None for action in H36M_ACTIONS},
            "overall_mpjpe": None,
            "num_samples": 0
        }
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    action_results = {action: [] for action in H36M_ACTIONS}
    
    for i, action in enumerate(all_actions):
        if action in action_results:
            action_results[action].append(i)
    
    action_mpjpe = {}
    for action in H36M_ACTIONS:
        indices = action_results[action]
        if len(indices) > 0:
            action_preds = all_preds[indices]
            action_gts = all_gts[indices]
            action_mpjpe[action] = compute_mpjpe(action_preds, action_gts)
        else:
            action_mpjpe[action] = None
    
    overall_mpjpe = compute_mpjpe(all_preds, all_gts)
    
    return {
        "action_mpjpe": action_mpjpe,
        "overall_mpjpe": overall_mpjpe,
        "num_samples": len(all_preds)
    }


def main():
    print("=" * 60)
    print("Generating LaTeX Table for Human3.6M Results")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    params = config.get_params()
    params["device"] = device
    params["batch_size"] = 5
    params["seq_length"] = 50
    params["reset_state"] = 5
    params["normalise_data"] = 4
    params = config.update_params(params)
    
    checkpoint_dir = params.get("cp_file", "model")
    
    print("\nSearching for model files...")
    model_files = find_model_files(checkpoint_dir)
    
    if not model_files:
        print(f"No model files found in {checkpoint_dir}")
        return
    
    print(f"Found models: {list(model_files.keys())}")
    
    first_model_path = list(model_files.values())[0][0]
    first_checkpoint = torch.load(first_model_path, map_location=device)
    saved_params = first_checkpoint.get('params', {})
    predict_next_frame = saved_params.get('predict_next_frame', True)
    params['predict_next_frame'] = predict_next_frame
    print(f"Detected predict_next_frame mode: {predict_next_frame}")
    
    print("\nPreparing dataset...")
    (params, X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list,
            X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list) = \
            dut.prepare_training_set(params)
    
    (index_train_list, S_Train_list) = dut.get_seq_indexes(params, S_Train_list)
    (index_test_list, S_Test_list) = dut.get_seq_indexes(params, S_Test_list)
    
    batch_size = params['batch_size']
    
    results_dict = {}
    
    for model_type, model_paths in model_files.items():
        if not model_paths:
            continue
        
        model_path = model_paths[0]
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_type}")
        print(f"Model: {os.path.basename(model_path)}")
        print("=" * 60)
        
        params["model"] = model_type
        
        try:
            # Kalman 模型使用不同的评估方法
            if model_type == "Kalman":
                model = load_pure_kalman_model(model_path, device)
                results = evaluate_kalman_by_action(
                    model, params, X_test, Y_test, index_test_list,
                    S_Test_list, R_L_Test_list, F_list_test, batch_size, device
                )
            else:
                model = load_model(model_path, model_type, params, device)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Total parameters: {total_params:,}")
                results = evaluate_model_by_action(
                    model, params, X_test, Y_test, index_test_list,
                    S_Test_list, R_L_Test_list, F_list_test, batch_size, device
                )
            
            results_dict[model_type] = results
            
            # 检查 overall_mpjpe 是否为 None
            if results['overall_mpjpe'] is not None:
                print(f"\nOverall MPJPE: {results['overall_mpjpe']:.2f} mm")
            else:
                print("\nOverall MPJPE: N/A")
            
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error evaluating {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if results_dict:
        # 生成 LaTeX 表格
        generate_latex_table(results_dict)

if __name__ == "__main__":
    main()
