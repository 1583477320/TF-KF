import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import argparse
from tqdm import tqdm

from helper import config
from helper import dt_utils as dut
from helper import utils as ut
from helper import train_helper as th
from helper.data_lodaer import build_inception_test_transform
from model_runner.klstm.kfl_QRFf_transformer import Model as kfl_QRFf_transformer
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.lstm.pt_lstm import Model as lstm
from nets.inception_resnet_v2 import InceptionResNetV2

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10), (8, 11), (11, 12),
    (12, 13), (8, 14), (14, 15), (15, 16)
]

MODEL_COLORS = {
    "kfl_QRFf_transformer": "#FF6B6B",
    "kfl_QRFf": "#4ECDC4",
    "kfl_QRf": "#45B7D1",
    "kfl_K": "#96CEB4",
    "lstm": "#FFEAA7",
    "Inception": "#9B59B6",
    "GT": "#3498DB"
}


def find_model_files(base_checkpoint_dir):
    """
    查找所有模型文件并按类型分组
    """
    model_files = {}
    
    if not os.path.exists(base_checkpoint_dir):
        return model_files
    
    has_subdirs = False
    for model_type in ["kfl_QRFf_transformer", "kfl_QRFf", "kfl_QRf", "kfl_K", "lstm"]:
        model_dir = os.path.join(base_checkpoint_dir, model_type)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            has_subdirs = True
            break
    
    if has_subdirs:
        for model_type in ["kfl_QRFf_transformer", "kfl_QRFf", "kfl_QRf", "kfl_K", "lstm"]:
            model_dir = os.path.join(base_checkpoint_dir, model_type)
            
            if not os.path.exists(model_dir):
                continue
            
            best_files = []
            regular_files = []
            
            for f in os.listdir(model_dir):
                if f.endswith(".ckpt") or f.endswith(".pth"):
                    full_path = os.path.join(model_dir, f)
                    
                    if f.startswith("best_"):
                        best_files.append(full_path)
                    elif f not in ["resume_checkpoint.ckpt", "optimizer_final.pth"]:
                        regular_files.append(full_path)
            
            all_files = sorted(best_files, key=lambda x: os.path.getmtime(x), reverse=True) + \
                       sorted(regular_files, key=lambda x: os.path.getmtime(x), reverse=True)
            
            if all_files:
                model_files[model_type] = all_files
    else:
        exclude_files = ["optimizer_final.pth", "inceptionresnetv2-520b38e4.pth", "model_final.pth"]
        for f in os.listdir(base_checkpoint_dir):
            if f in exclude_files:
                continue
            if f.endswith(".ckpt") or f.endswith(".pth"):
                full_path = os.path.join(base_checkpoint_dir, f)
                model_type = detect_model_type(full_path)
                
                if model_type is None:
                    continue
                
                if model_type not in model_files:
                    model_files[model_type] = []
                
                if f.startswith("best_"):
                    model_files[model_type].insert(0, full_path)
                else:
                    model_files[model_type].append(full_path)
    
    inception_path = os.path.join(base_checkpoint_dir, "model_final.pth")
    if os.path.exists(inception_path):
        model_files["Inception"] = [inception_path]
    
    return model_files


def detect_model_type(model_path):
    """
    检测模型类型
    """
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    
    if isinstance(checkpoint, dict):
        keys = list(checkpoint.keys())
        
        if 'model_state_dict' in checkpoint:
            keys = list(checkpoint['model_state_dict'].keys())
        
        if any('transformer_F' in k for k in keys):
            return "kfl_QRFf_transformer"
        elif any('lstms_F' in k for k in keys):
            if any('lstms_K' in k for k in keys):
                return "kfl_K"
            elif any('lstms_Q' in k for k in keys) and any('lstms_R' in k for k in keys):
                return "kfl_QRFf"
            elif any('lstms_Q' in k for k in keys):
                return "kfl_QRf"
            else:
                return "lstm"
    
    if 'transformer' in model_path.lower():
        return "kfl_QRFf_transformer"
    elif 'kfl_k' in model_path.lower():
        return "kfl_K"
    elif 'kfl_qrff' in model_path.lower():
        return "kfl_QRFf"
    elif 'kfl_qrf' in model_path.lower():
        return "kfl_QRf"
    elif 'lstm' in model_path.lower():
        return "lstm"
    
    return "kfl_QRFf"


def load_model(model_path, model_type, params, device):
    """
    加载模型
    """
    if model_type == "kfl_QRFf_transformer":
        model = kfl_QRFf_transformer(params=params)
    elif model_type == "kfl_QRFf":
        model = kfl_QRFf(params=params)
    elif model_type == "kfl_QRf":
        model = kfl_QRf(params=params)
    elif model_type == "kfl_K":
        model = kfl_K(params=params)
    elif model_type == "lstm":
        model = lstm(params=params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        saved_params = checkpoint.get('params', {})
        if 'predict_next_frame' in saved_params:
            params['predict_next_frame'] = saved_params['predict_next_frame']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def compute_mpjpe(pred, gt):
    """
    计算 MPJPE (mm)
    """
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


def transform_coords(p):
    """坐标转换用于可视化"""
    return p[:, 0], p[:, 2], -p[:, 1]


def visualize_single_model_comparison(img_path, gt_3d, pred_3d, model_name, mpjpe, save_path=None, add_legend=True):
    """
    单个模型与真实值对比可视化 - 包含3D和2D对比
    
    Args:
        img_path: 原始图片路径
        gt_3d: 真实 3D 坐标 (17, 3)
        pred_3d: 预测 3D 坐标 (17, 3)
        model_name: 模型名称
        mpjpe: MPJPE 值
        save_path: 保存路径
        add_legend: 是否添加图例
    """
    fig = plt.figure(figsize=(12, 5))
    
    ax_img = fig.add_subplot(1, 3, 1)
    try:
        raw_img = Image.open(img_path)
        ax_img.imshow(raw_img)
        ax_img.set_title(f"Input Image\n{os.path.basename(img_path)}")
    except:
        ax_img.text(0.5, 0.5, "No Image", ha='center', va='center')
        ax_img.set_title("Input Image")
    ax_img.axis('off')
    
    ax_3d = fig.add_subplot(1, 3, 2, projection='3d')
    ax_3d.set_title(f"3D Pose Comparison\n{model_name} | MPJPE: {mpjpe:.2f} mm")
    
    gt_x, gt_y, gt_z = transform_coords(gt_3d)
    pred_x, pred_y, pred_z = transform_coords(pred_3d)
    color = MODEL_COLORS.get(model_name, "#888888")
    
    for s, e in SKELETON_EDGES:
        ax_3d.plot([gt_x[s], gt_x[e]], [gt_y[s], gt_y[e]], [gt_z[s], gt_z[e]], 
                   color=MODEL_COLORS["GT"], linestyle='-', linewidth=2.5)
        ax_3d.plot([pred_x[s], pred_x[e]], [pred_y[s], pred_y[e]], [pred_z[s], pred_z[e]], 
                   color=color, linestyle='--', linewidth=2)
    
    ax_3d.scatter(gt_x, gt_y, gt_z, color=MODEL_COLORS["GT"], s=50, marker='o', label='Ground Truth')
    ax_3d.scatter(pred_x, pred_y, pred_z, color=color, s=40, marker='^', label=model_name)
    
    if add_legend:
        ax_3d.legend(loc='upper left', fontsize=8)
    
    ax_3d.view_init(elev=15, azim=-70)
    
    ax_2d = fig.add_subplot(1, 3, 3)
    ax_2d.set_title("2D View (X-Y)")
    ax_2d.set_xlabel("X")
    ax_2d.set_ylabel("Y")
    
    for s, e in SKELETON_EDGES:
        ax_2d.plot([gt_3d[s, 0], gt_3d[e, 0]], [-gt_3d[s, 1], -gt_3d[e, 1]], 
                   color=MODEL_COLORS["GT"], linestyle='-', linewidth=2)
        ax_2d.plot([pred_3d[s, 0], pred_3d[e, 0]], [-pred_3d[s, 1], -pred_3d[e, 1]], 
                   color=color, linestyle='--', linewidth=1.5)
    
    ax_2d.scatter(gt_3d[:, 0], -gt_3d[:, 1], color=MODEL_COLORS["GT"], s=30, marker='o', label='Ground Truth')
    ax_2d.scatter(pred_3d[:, 0], -pred_3d[:, 1], color=color, s=25, marker='^', label=model_name)
    ax_2d.legend(fontsize=8)
    ax_2d.set_aspect('equal')
    ax_2d.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.close()


def visualize_multi_model_comparison(img_path, gt_3d, predictions, output_dir=None, add_legend=True):
    """
    多模型预测对比可视化 - 为每个模型生成单独的对比图
    
    Args:
        img_path: 原始图片路径
        gt_3d: 真实 3D 坐标 (17, 3)
        predictions: 字典 {model_name: pred_3d}
        output_dir: 输出目录
        add_legend: 是否添加图例
    """
    for model_name, pred_3d in predictions.items():
        mpjpe = compute_mpjpe(pred_3d.reshape(1, 17, 3), gt_3d.reshape(1, 17, 3))
        
        save_path = None
        if output_dir:
            safe_name = model_name.replace("/", "_")
            save_path = os.path.join(output_dir, f"pose_comparison_{safe_name}.png")
        
        visualize_single_model_comparison(
            img_path=img_path,
            gt_3d=gt_3d,
            pred_3d=pred_3d,
            model_name=model_name,
            mpjpe=mpjpe,
            save_path=save_path,
            add_legend=add_legend
        )


def visualize_error_comparison(gt_3d, predictions, save_path=None):
    """
    每个关节的误差对比图
    """
    joint_names = [
        "Hip", "RHip", "RKnee", "RFoot",
        "LHip", "LKnee", "LFoot",
        "Spine", "Thorax", "Neck", "Head",
        "LShoulder", "LElbow", "LWrist",
        "RShoulder", "RElbow", "RWrist"
    ]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    x = np.arange(len(joint_names))
    width = 0.8 / (len(predictions) + 1)
    
    gt_expanded = gt_3d.reshape(1, 17, 3)
    
    for i, (model_name, pred_3d) in enumerate(predictions.items()):
        pred_expanded = pred_3d.reshape(1, 17, 3)
        joint_errors = np.sqrt(np.sum((pred_expanded - gt_expanded) ** 2, axis=2)).flatten() * 1000
        
        color = MODEL_COLORS.get(model_name, "#888888")
        bars = ax.bar(x + i * width, joint_errors, width, label=model_name, color=color, alpha=0.8)
    
    ax.set_xlabel('Joint', fontsize=12)
    ax.set_ylabel('Error (mm)', fontsize=12)
    ax.set_title('Per-Joint Position Error Comparison', fontsize=14)
    ax.set_xticks(x + width * len(predictions) / 2)
    ax.set_xticklabels(joint_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to: {save_path}")
    
    plt.show()


@torch.no_grad()
def predict_with_model(model, params, X, Y, index_list, S_list, R_L_list, F_list, batch_size, device):
    """
    使用模型进行预测
    """
    model.eval()
    
    dic_state = ut.get_state_list(params)
    
    def to_tensor(v):
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float().to(device)
        elif isinstance(v, list):
            return [(to_tensor(t[0]), to_tensor(t[1])) for t in v]
        return v
    
    for k, v in dic_state.items():
        dic_state[k] = to_tensor(v)
    
    is_transformer = "transformer" in params.get("model", "").lower()
    if is_transformer:
        lstm_states = ["F_pre", "Q_pre", "R_pre", "K_pre"]
        for state_key in lstm_states:
            if state_key in dic_state:
                del dic_state[state_key]
    
    NOUT = params["n_output"]
    I = torch.eye(NOUT, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    params["reset_state"] = -1
    state_reset_counter_lst = [0 for _ in range(batch_size)]
    n_batches = len(index_list) // batch_size
    
    all_preds = []
    all_gts = []
    all_indices = []
    
    for minibatch_index in tqdm(range(n_batches), desc="Predicting", leave=False):
        state_reset_counter_lst = [s + 1 for s in state_reset_counter_lst]
        
        (
            dic_state, 
            x, y, r, f,
            curr_sid,
            state_reset_counter_lst,
            curr_id_lst
        ) = th.prepare_kfl_QRFf_batch(
            is_test=1,
            index_list=index_list,
            minibatch_index=minibatch_index,
            batch_size=batch_size,
            S_list=S_list,
            dic_state=dic_state,
            params=params,
            Y=Y,
            X=X,
            R_L_list=R_L_list,
            F_list=F_list,
            state_reset_counter_lst=state_reset_counter_lst,
            device=device
        )
        
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float().to(device)
        
        if r is None:
            repeat_data = torch.ones((x.shape[0], x.shape[1]), device=device)
        else:
            if not isinstance(r, torch.Tensor):
                repeat_data = torch.from_numpy(r).float().to(device)
            else:
                repeat_data = r.float()
        
        if repeat_data.dim() == 1:
            repeat_data = repeat_data.unsqueeze(1)
        
        actual_bsz = x.shape[0]
        
        if dic_state.get("_x_pre") is None:
            _x_inp = torch.zeros(actual_bsz, NOUT, device=device)
        else:
            _x_inp = dic_state["_x_pre"]
        
        if dic_state.get("PCov_pre") is None:
            _P_inp = I[:actual_bsz].clone()
        else:
            _P_inp = dic_state["PCov_pre"]
        
        result = model(
            _z=x,
            target_data=y,
            repeat_data=repeat_data,
            _x_inp=_x_inp,
            _P_inp=_P_inp,
            _I=I[:actual_bsz],
            state_dict=dic_state,
            is_training=False
        )
        
        if len(result) == 4:
            loss, new_states, final_output, y_target = result
        else:
            loss, new_states = result
            final_output, y_target = None, None
        
        if "F_t" in new_states:
            dic_state["F_pre"] = [(h.detach(), c.detach()) for h, c in new_states["F_t"]]
        if "Q_t" in new_states:
            dic_state["Q_pre"] = [(h.detach(), c.detach()) for h, c in new_states["Q_t"]]
        if "R_t" in new_states:
            dic_state["R_pre"] = [(h.detach(), c.detach()) for h, c in new_states["R_t"]]
        if "PCov_t" in new_states:
            dic_state["PCov_pre"] = new_states["PCov_t"].detach()
        if "_x_t" in new_states:
            dic_state["_x_pre"] = new_states["_x_t"].detach()
        
        if final_output is not None:
            pred = final_output
            gt = y_target
        else:
            pred = model.final_output
            gt = model.y
        
        all_preds.append(pred.cpu().numpy())
        all_gts.append(gt.cpu().numpy())
        all_indices.extend(curr_id_lst)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    return all_preds, all_gts, all_indices


@torch.no_grad()
def predict_with_inception(model, img_paths, Y, device, batch_size=128):
    """
    使用 Inception 模型直接从图像预测 3D 姿态
    
    Args:
        model: InceptionResNetV2 模型
        img_paths: 图像路径列表
        Y: 真实 3D 姿态 (N, 51)
        device: 设备
        batch_size: 批大小
    
    Returns:
        all_preds: 预测结果
        all_gts: 真实值
    """
    model.eval()
    if device.type == 'cuda':
        model = model.half()
    
    all_preds = []
    all_gts = []
    
    transform = build_inception_test_transform()
    
    n_samples = len(img_paths)
    
    for i in tqdm(range(0, n_samples, batch_size), desc="Inception predicting"):
        batch_paths = img_paths[i:i+batch_size]
        batch_gts = Y[i:i+batch_size]
        
        imgs = []
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                img = transform(img)
            except:
                img = torch.zeros(3, 299, 299)
            imgs.append(img)
        
        imgs = torch.stack(imgs).to(device)
        if device.type == 'cuda':
            imgs = imgs.half()
        
        preds, _ = model(imgs)
        preds = preds.float().cpu().numpy()
        
        all_preds.append(preds)
        all_gts.append(batch_gts)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    return all_preds, all_gts


def main():
    parser = argparse.ArgumentParser(description="多模型预测对比工具")
    parser.add_argument("--img", type=str, default=None, help="指定要对比的图片路径")
    parser.add_argument("--sample_idx", type=int, default=100, help="指定样本索引（默认：100）")
    parser.add_argument("--output_dir", type=str, default="output/comparison", help="输出目录（默认：output/comparison）")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Multi-Model Prediction Comparison")
    print("=" * 60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    params = config.get_params()
    params["device"] = device
    params["batch_size"] = 256
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
    first_checkpoint = torch.load(first_model_path, map_location='cpu')
    saved_params = first_checkpoint.get('params', {})
    predict_next_frame = saved_params.get('predict_next_frame', False)
    params['predict_next_frame'] = predict_next_frame
    print(f"Detected predict_next_frame mode: {predict_next_frame}")
    
    sample_idx = args.sample_idx
    
    print(f"\nPreparing dataset for sample {sample_idx}...")
    (params, X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list,
            X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list) = \
            dut.prepare_training_set(params)
    
    sample_idx_to_input_img = {}
    sample_idx_to_pred_img = {}
    sample_idx_to_gt = {}
    global_sample_idx = 0
    
    for seq_idx, frames in enumerate(F_list_test):
        seq_len = len(frames)
        Y_seq = Y_test[seq_idx] if seq_idx < len(Y_test) else None
        
        if predict_next_frame:
            for frame_idx in range(seq_len - 1):
                sample_idx_to_input_img[global_sample_idx] = frames[frame_idx]
                sample_idx_to_pred_img[global_sample_idx] = frames[frame_idx + 1]
                if Y_seq is not None and frame_idx + 1 < Y_seq.shape[0]:
                    sample_idx_to_gt[global_sample_idx] = Y_seq[frame_idx + 1]
                global_sample_idx += 1
        else:
            for frame_idx, frame_path in enumerate(frames):
                sample_idx_to_input_img[global_sample_idx] = frame_path
                sample_idx_to_pred_img[global_sample_idx] = frame_path
                if Y_seq is not None and frame_idx < Y_seq.shape[0]:
                    sample_idx_to_gt[global_sample_idx] = Y_seq[frame_idx]
                global_sample_idx += 1
    
    total_samples = len(sample_idx_to_input_img)
    print(f"Total samples: {total_samples}")
    
    if sample_idx >= total_samples:
        print(f"Error: sample_idx {sample_idx} is out of range (max: {total_samples - 1})")
        return
    
    input_img_path = sample_idx_to_input_img[sample_idx]
    pred_img_path = sample_idx_to_pred_img[sample_idx]
    gt_3d = sample_idx_to_gt.get(sample_idx)
    
    if gt_3d is None:
        print(f"Error: No ground truth found for sample {sample_idx}")
        return
    
    gt_3d = gt_3d.reshape(17, 3)
    
    print(f"\nSample {sample_idx}:")
    print(f"  Input frame: {os.path.basename(input_img_path)}")
    print(f"  Target frame: {os.path.basename(pred_img_path)}")
    
    predictions = {}
    
    for model_type, model_paths in model_files.items():
        if not model_paths:
            continue
        
        model_path = model_paths[0]
        print(f"\n{'='*60}")
        print(f"Predicting with: {model_type}")
        print("=" * 60)
        
        try:
            if model_type == "Inception":
                model = InceptionResNetV2(num_classes=51)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.to(device)
                if device.type == 'cuda':
                    model = model.half()
                model.eval()
                
                transform = build_inception_test_transform()
                
                # Inception 使用输入帧图片预测，但结果和下一帧 GT 对比
                try:
                    img = Image.open(input_img_path).convert("RGB")
                    img = transform(img).unsqueeze(0).to(device)
                    if device.type == 'cuda':
                        img = img.half()
                    
                    with torch.no_grad():
                        pred, _ = model(img)
                        pred = pred.float().cpu().numpy().reshape(17, 3)
                    
                    predictions[model_type] = pred
                    mpjpe = compute_mpjpe(pred.reshape(1, 17, 3), gt_3d.reshape(1, 17, 3))
                    print(f"MPJPE: {mpjpe:.2f} mm")
                except Exception as e:
                    print(f"Error processing image: {e}")
                
                del model
                torch.cuda.empty_cache()
            else:
                params["model"] = model_type
                model = load_model(model_path, model_type, params, device)
                
                cumsum = 0
                target_seq_idx = 0
                target_frame_idx = 0
                original_sample_idx = sample_idx
                
                for seq_idx, frames in enumerate(F_list_test):
                    seq_len = len(frames) - 1 if predict_next_frame else len(frames)
                    if original_sample_idx < cumsum + seq_len:
                        target_seq_idx = seq_idx
                        target_frame_idx = original_sample_idx - cumsum
                        break
                    cumsum += seq_len
                
                print(f"  Sequence: {target_seq_idx}, Frame: {target_frame_idx}")
                
                X_seq = X_test[target_seq_idx]
                Y_seq = Y_test[target_seq_idx]
                
                X_input = torch.from_numpy(X_seq).unsqueeze(0).float().to(device)
                
                with torch.no_grad():
                    I = torch.eye(params['n_output']).unsqueeze(0).to(device)
                    
                    dic_state = {}
                    dic_state["_x_pre"] = torch.zeros(1, params['n_output']).to(device)
                    dic_state["PCov_pre"] = I.clone()
                    
                    result = model(
                        _z=X_input,
                        target_data=torch.from_numpy(Y_seq).unsqueeze(0).float().to(device),
                        repeat_data=torch.ones(1, X_seq.shape[0]).to(device),
                        _x_inp=dic_state["_x_pre"],
                        _P_inp=dic_state["PCov_pre"],
                        _I=I.clone(),
                        state_dict=dic_state,
                        is_training=False
                    )
                    
                    if len(result) == 4:
                        loss, new_states, final_output, y_target = result
                        pred = final_output[target_frame_idx].cpu().numpy().reshape(17, 3)
                    else:
                        loss, new_states = result
                        pred = model.final_output[target_frame_idx].cpu().numpy().reshape(17, 3)
                    
                    predictions[model_type] = pred
                    mpjpe = compute_mpjpe(pred.reshape(1, 17, 3), gt_3d.reshape(1, 17, 3))
                    print(f"MPJPE: {mpjpe:.2f} mm")
                
                del model
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"Error with {model_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if predictions:
        print("\n" + "=" * 60)
        print("MPJPE Summary")
        print("=" * 60)
        
        print(f"\n{'Model':<25} {'MPJPE (mm)':<15}")
        print("-" * 40)
        
        for model_type, pred in predictions.items():
            mpjpe = compute_mpjpe(pred.reshape(1, 17, 3), gt_3d.reshape(1, 17, 3))
            print(f"{model_type:<25} {mpjpe:<15.2f}")
        
        visualize_multi_model_comparison(
            img_path=input_img_path,
            gt_3d=gt_3d,
            predictions=predictions,
            output_dir=args.output_dir
        )
        
        visualize_error_comparison(
            gt_3d=gt_3d,
            predictions=predictions,
            save_path=os.path.join(args.output_dir, "joint_error_comparison.png")
        )
    
    return predictions, gt_3d


if __name__ == "__main__":
    
    result = main()
    if result is not None:
        predictions, gts = result
    else:
        print("No models found for comparison.")
