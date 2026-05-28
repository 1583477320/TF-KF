import torch
import numpy as np
import os
from tqdm import tqdm
from helper import config
from helper import dt_utils as dut
from helper import utils as ut
from helper import train_helper as th
from model_runner.klstm.kfl_QRFf_transformer import Model as kfl_QRFf_transformer
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.klstm.pure_kalman import PureKalmanFilter
from model_runner.lstm.pt_lstm import Model as lstm
from nets.inception_resnet_v2 import InceptionResNetV2

H36M_ACTIONS = [
    "Directions", "Discussion", "Eating", "Greeting", 
    "Phoning", "Photo", "Posing", "Purchases",
    "Sitting", "SittingDown", "Smoking", "Waiting",
    "WalkDog", "Walking", "WalkTogether"
]


def extract_action_from_path(frame_path):
    """
    从帧路径中提取动作类别
    例如: data/images/train/S1_Directions_1.54138969_000076.jpg -> Directions
    """
    filename = os.path.basename(frame_path)
    parts = filename.split("_")
    if len(parts) >= 2:
        action = parts[1]
        action_map = {
            "SitDown": "SittingDown",
            "Walk": "Walking"
        }
        return action_map.get(action, action)
    return "Unknown"


def compute_mpjpe(pred, gt):
    """
    计算 MPJPE (Mean Per Joint Position Error)
    
    Args:
        pred: 预测值 (N, 51) 或 (N, 17, 3)
        gt: 真实值 (N, 51) 或 (N, 17, 3)
    
    Returns:
        mpjpe: 平均每个关节的位置误差 (mm)
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(gt, torch.Tensor):
        gt = gt.detach().cpu().numpy()
    
    # 检查是否为空数组
    if pred.shape[0] == 0 or gt.shape[0] == 0:
        return None
    
    if pred.ndim == 2:
        pred = pred.reshape(-1, 17, 3)
    if gt.ndim == 2:
        gt = gt.reshape(-1, 17, 3)
    
    joint_errors = np.sqrt(np.sum((pred - gt) ** 2, axis=2))
    
    # 检查是否为空
    if joint_errors.size == 0:
        return None
    
    mpjpe = np.mean(joint_errors)
    
    return mpjpe * 1000


def detect_model_type(model_path):
    """
    通过检查 state_dict 的键名来检测模型类型
    """
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
    except Exception:
        try:
            state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        except Exception as e2:
            print(f"Warning: Cannot load {model_path}: {e2}")
            return None
    
    keys = list(state_dict.keys())
    
    if any('transformer_F' in k for k in keys):
        return "kfl_QRFf_transformer"
    elif any('lstms_F' in k for k in keys):
        if any('lstms_K' in k for k in keys):
            return "kfl_K"
        elif any('lstms_Q' in k for k in keys) and any('lstms_R' in k for k in keys):
            return "kfl_QRFf"
        elif any('lstms_Q' in k for k in keys):
            return "kfl_QRf"
    elif 'inceptionresnetv2' in model_path.lower():
        return "inception"
    elif 'transformer' in model_path.lower():
        return "kfl_QRFf_transformer"
    elif 'pure_kalman_filter' in model_path.lower():
        return "Kalman"
    elif 'kfl_k' in model_path.lower():
        return "kfl_K"
    elif 'kfl_qrff' in model_path.lower():
        return "kfl_QRFf"
    elif 'kfl_qrf' in model_path.lower():
        return "kfl_QRf"
    elif 'lstm' in model_path.lower():
        return "lstm"
    else:
        return "kfl_QRFf"


def load_model(model_path, model_type, params, device):
    """
    加载训练好的模型
    支持新格式（包含 model_state_dict）和旧格式（直接的 state_dict）
    支持 Inception 模型和 Kalman
    """
    if model_type == "inception":
        model = InceptionResNetV2(num_classes=params['n_output'])
        model.to(device)
        model.eval()
        return model
    elif model_type == "Kalman":
        # 加载纯卡尔曼滤波器模型
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # 提取参数
        F = torch.from_numpy(checkpoint['F']).to(device)
        Q = torch.from_numpy(checkpoint['Q']).to(device)
        R = torch.from_numpy(checkpoint['R']).to(device)
        dim = checkpoint['dim']
        
        # 处理H矩阵
        H = None
        if checkpoint.get('H') is not None:
            H = torch.from_numpy(checkpoint['H']).to(device)
        
        # 创建模型
        model = PureKalmanFilter(F, Q, R, dim, H)
        return model
    elif model_type == "kfl_QRFf_transformer":
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
        epoch = checkpoint.get('epoch', 'N/A')
        loss = checkpoint.get('loss', 'N/A')
        saved_params = checkpoint.get('params', {})
        predict_next_frame = saved_params.get('predict_next_frame', True)
        print(f"  Checkpoint info: epoch={epoch}, loss={loss}, predict_next_frame={predict_next_frame}")
        params['predict_next_frame'] = predict_next_frame
    else:
        state_dict = checkpoint
    
    if model_type == "kfl_QRFf_transformer":
        if 'fc_F_mat.weight' in state_dict:
            saved_weight = state_dict['fc_F_mat.weight']
            saved_bias = state_dict['fc_F_mat.bias']
            model_weight = model.fc_F_mat.weight
            
            if saved_weight.shape[0] != model_weight.shape[0]:
                print(f"  Note: Adapting fc_F_mat from {saved_weight.shape} to {model_weight.shape}")
                nout = params['n_output']
                if saved_weight.shape[0] == nout * nout:
                    state_dict['fc_F_mat.weight'] = saved_weight[:nout, :]
                    state_dict['fc_F_mat.bias'] = saved_bias[:nout]
        
        for key in state_dict.keys():
            if isinstance(state_dict[key], torch.Tensor):
                if torch.isnan(state_dict[key]).any() or torch.isinf(state_dict[key]).any():
                    print(f"  Warning: {key} contains nan/inf values, reinitializing...")
                    if key.endswith('.weight') and state_dict[key].ndim >= 2:
                        torch.nn.init.xavier_normal_(state_dict[key])
                    elif key.endswith('.bias'):
                        torch.nn.init.zeros_(state_dict[key])
                    else:
                        state_dict[key] = torch.zeros_like(state_dict[key])
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model


def find_model_files(base_checkpoint_dir):
    """
    查找所有模型文件并按类型分组
    支持：
    1. 新版本：分模型保存的目录结构 (model/kfl_QRFf/..., model/lstm/...)
    2. 旧版本：单目录结构 (model/*.ckpt)
    """
    model_files = {}
    
    if not os.path.exists(base_checkpoint_dir):
        return model_files
    
    has_subdirs = False
    for model_type in ["kfl_QRFf_transformer", "pure_kalman", "kfl_QRFf", "kfl_QRf", "kfl_K", "lstm"]:
        model_dir = os.path.join(base_checkpoint_dir, model_type)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            has_subdirs = True
            break
    
    if has_subdirs:
        for model_type in ["kfl_QRFf_transformer", "pure_kalman", "kfl_QRFf", "kfl_QRf", "kfl_K", "lstm"]:
        # for model_type in ["kfl_QRFf"]:
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
                # 将目录名称映射到模型类型
                if model_type == "pure_kalman":
                    model_files["Kalman"] = all_files
                else:
                    model_files[model_type] = all_files
    else:
        exclude_files = ["optimizer_final.pth", "inceptionresnetv2-520b38e4.pth"]
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
    
    return model_files


@torch.no_grad()
def evaluate_inception_model(model, params, X, Y, index_list, S_list, R_L_list, F_list, batch_size, device):
    """
    评估 Inception 模型（直接从图像预测 3D 姿态）
    
    Args:
        model: Inception 模型
        params: 参数字典
        X: 输入特征
        Y: 目标姿态
        index_list: 索引列表
        S_list: 序列ID列表
        R_L_list: 重复掩码列表
        F_list: 帧路径列表
        batch_size: 批次大小
        device: 设备
    
    Returns:
        all_preds: 所有预测结果
        all_gts: 所有真实值
    """
    model.eval()
    
    all_preds = []
    all_gts = []
    
    n_batches = len(index_list) // batch_size
    
    from helper.data_lodaer import build_inception_test_transform
    
    for minibatch_index in tqdm(range(n_batches), desc="Evaluating Inception", leave=False):
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
            dic_state={},
            params=params,
            Y=Y,
            X=X,
            R_L_list=R_L_list,
            F_list=F_list,
            state_reset_counter_lst=[0 for _ in range(batch_size)],
            device=device
        )
        
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float().to(device)
        
        with torch.no_grad():
            pred, _ = model(x)
        
        pred_np = pred.detach().cpu().numpy()
        gt_np = y.detach().cpu().numpy()
        
        all_preds.append(pred_np)
        all_gts.append(gt_np)
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_gts = np.concatenate(all_gts, axis=0)
    
    return all_preds, all_gts


@torch.no_grad()
def evaluate_model_by_action(model, params, X, Y, index_list, S_list, R_L_list, F_list, batch_size, device):
    """
    按动作类别评估模型性能
    
    Args:
        model: 要评估的模型
        params: 参数字典
        X: 输入特征
        Y: 目标姿态
        index_list: 索引列表
        S_list: 序列ID列表
        R_L_list: 重复掩码列表
        F_list: 帧路径列表
        batch_size: 批次大小
        device: 设备
    
    Returns:
        results: 包含各动作 MPJPE 的字典
    """
    model.eval()
    
    # 检查是否是Kalman模型
    is_pure_kalman = isinstance(model, PureKalmanFilter)
    
    if is_pure_kalman:
        # 纯卡尔曼滤波器的评估逻辑
        NOUT = params["n_output"]
        n_batches = len(index_list) // batch_size
        
        all_preds = []
        all_gts = []
        all_actions = []
        
        eval_pbar = tqdm(range(n_batches), desc="Evaluating Pure Kalman", unit="batch", leave=False)
        for minibatch_index in eval_pbar:
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
                dic_state={},
                params=params,
                Y=Y,
                X=X,  # 使用CNN特征作为输入
                R_L_list=R_L_list,
                F_list=F_list,
                state_reset_counter_lst=[0 for _ in range(batch_size)],
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
            
            # 初始化状态
            _x_inp = torch.zeros(actual_bsz, NOUT, device=device)
            _P_inp = torch.eye(NOUT, device=device).unsqueeze(0).repeat(actual_bsz, 1, 1)
            
            # 执行纯卡尔曼滤波
            x_filtered, P_filtered = model.forward(x, repeat_data, _x_inp, _P_inp)
            
            # 应用掩码
            flt = torch.squeeze(torch.reshape(repeat_data, [-1, 1]), 1)
            indices = torch.where(torch.not_equal(flt, 0))[0]
            
            xres_flat = torch.reshape(x_filtered, [-1, NOUT])
            pred = torch.index_select(xres_flat, 0, indices)
            target_flat = torch.reshape(y, [-1, NOUT])
            gt = torch.index_select(target_flat, 0, indices)
            
            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            
            for seq_idx in curr_id_lst:
                if seq_idx < len(F_list):
                    frame_paths = F_list[seq_idx]
                    if len(frame_paths) > 0:
                        action = extract_action_from_path(frame_paths[0])
                        all_actions.append(action)
                    else:
                        all_actions.append("Unknown")
                else:
                    all_actions.append("Unknown")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
    else:
        # 其他模型的评估逻辑
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
        all_actions = []
        
        eval_pbar = tqdm(range(n_batches), desc="Evaluating", unit="batch", leave=False)
        for minibatch_index in eval_pbar:
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
            
            try:
                loss, new_states, final_output, y = model(
                    _z=x,
                    target_data=y,
                    repeat_data=repeat_data,
                    _x_inp=_x_inp,
                    _P_inp=_P_inp,
                    _I=I[:actual_bsz],
                    state_dict=dic_state,
                    is_training=False
                )
                pred = final_output
                gt = y
            except ValueError:
                loss, new_states = model(
                    _z=x,
                    target_data=y,
                    repeat_data=repeat_data,
                    _x_inp=_x_inp,
                    _P_inp=_P_inp,
                    _I=I[:actual_bsz],
                    state_dict=dic_state,
                    is_training=False
                )
                pred = model.final_output
                gt = y
            
            all_preds.append(pred.cpu().numpy())
            all_gts.append(gt.cpu().numpy())
            
            for seq_idx in curr_id_lst:
                if seq_idx < len(F_list):
                    frame_paths = F_list[seq_idx]
                    if len(frame_paths) > 0:
                        action = extract_action_from_path(frame_paths[0])
                        all_actions.append(action)
                    else:
                        all_actions.append("Unknown")
                else:
                    all_actions.append("Unknown")
            
            eval_pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        all_preds = np.concatenate(all_preds, axis=0)
        all_gts = np.concatenate(all_gts, axis=0)
    
    action_results = {action: [] for action in H36M_ACTIONS}
    for i, action in enumerate(all_actions):
        if action in action_results:
            action_results[action].append(i)
    
    action_mpjpe = {}
    for action in H36M_ACTIONS:
        indices = action_results[action]
        if len(indices) > 0 and len(all_preds) > 0:
            action_preds = all_preds[indices]
            action_gts = all_gts[indices]
            action_mpjpe[action] = compute_mpjpe(action_preds, action_gts)
        else:
            action_mpjpe[action] = None
    
    overall_mpjpe = compute_mpjpe(all_preds, all_gts) if len(all_preds) > 0 else None
    
    return {
        "action_mpjpe": action_mpjpe,
        "overall_mpjpe": overall_mpjpe,
        "num_samples": len(all_preds)
    }


@torch.no_grad()
def evaluate_model_by_action(model, params, X, Y, index_list, S_list, R_L_list, F_list, batch_size, device):
    """
    按动作类别评估模型性能
    
    Returns:
        results: 包含各动作 MPJPE 的字典
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
    all_actions = []
    
    eval_pbar = tqdm(range(n_batches), desc="Evaluating", unit="batch", leave=False)
    for minibatch_index in eval_pbar:
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
        
        try:
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
                pred = final_output
                gt = y_target
            else:
                loss, new_states = result
                pred = model.final_output
                gt = model.y
        except Exception as e:
            print(f"Model forward error: {e}")
            continue
        
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
        
        all_preds.append(pred.cpu().numpy())
        all_gts.append(gt.cpu().numpy())
        
        for seq_idx in curr_id_lst:
            if seq_idx < len(F_list):
                frame_paths = F_list[seq_idx]
                if len(frame_paths) > 0:
                    action = extract_action_from_path(frame_paths[0])
                    all_actions.append(action)
                else:
                    all_actions.append("Unknown")
            else:
                all_actions.append("Unknown")
        
        eval_pbar.set_postfix(loss=f"{loss.item():.4f}")
    
    # 检查是否有任何预测结果
    if len(all_preds) == 0 or len(all_gts) == 0:
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
        if len(indices) > 0 and len(all_preds) > 0:
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


def print_results_table(results_dict):
    """
    打印格式化的结果表格
    行：不同模型
    列：测试场景（动作）+ Mean
    """
    models = list(results_dict.keys())
    
    col_width = 12
    model_width = 22
    
    total_width = model_width + (len(H36M_ACTIONS) + 1) * col_width
    
    print("\n" + "=" * total_width)
    print("Human3.6M Protocol #1 - MPJPE (Mean Per Joint Position Error) in mm")
    print("=" * total_width)
    
    header = f"{'Model':<{model_width}}"
    for action in H36M_ACTIONS:
        header += f"{action:<{col_width}}"
    header += f"{'Mean':<{col_width}}"
    print(header)
    print("-" * total_width)
    
    for model in models:
        row = f"{model:<{model_width}}"
        values = []
        for action in H36M_ACTIONS:
            mpjpe = results_dict[model]["action_mpjpe"].get(action)
            if mpjpe is not None:
                row += f"{mpjpe:<{col_width}.2f}"
                values.append(mpjpe)
            else:
                row += f"{'N/A':<{col_width}}"
        
        overall = results_dict[model]["overall_mpjpe"]
        row += f"{overall:<{col_width}.2f}"
        print(row)
    
    print("=" * total_width)


def main():
    print("=" * 60)
    print("Human3.6M Model Evaluation")
    print("Metric: MPJPE (Mean Per Joint Position Error)")
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
            model = load_model(model_path, model_type, params, device)
            
            # 只对神经网络模型计算参数量
            if model_type != "Kalman":
                total_params = sum(p.numel() for p in model.parameters())
                print(f"Total parameters: {total_params:,}")
            else:
                print("Total parameters: N/A (Pure Kalman Filter)")
            
            if model_type == "inception":
                all_preds, all_gts = evaluate_inception_model(
                    model, params, X_test, Y_test, index_test_list,
                    S_Test_list, R_L_Test_list, F_list_test, batch_size, device
                )
            else:
                results = evaluate_model_by_action(
                    model, params, X_test, Y_test, index_test_list,
                    S_Test_list, R_L_Test_list, F_list_test, batch_size, device
                )
            
            results_dict[model_type] = results
            
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
        print_results_table(results_dict)
    
    return results_dict


if __name__ == "__main__":
    results = main()
