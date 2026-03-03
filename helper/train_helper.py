from helper import utils as ut
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper.data_lodaer import H36MDataset, build_inception_train_transform, build_inception_test_transform,H36MSequenceDataset
import os


def prepare_batch(is_test, index_list, minibatch_index, batch_size, S_list, dic_state,
                           params, Y, X, R_L_list, F_list, state_reset_counter_lst):
    if params["model"] == "lstm":
        return prepare_lstm_batch(is_test, index_list, minibatch_index, batch_size, S_list, dic_state, params, Y, X, R_L_list, F_list, state_reset_counter_lst)
    elif params["model"] == "kfl_QRFf_transformer":
        return prepare_kfl_QRFf_batch(is_test, index_list, minibatch_index, batch_size, S_list, dic_state,
                           params, Y, X, R_L_list, F_list, state_reset_counter_lst, device=params.get('device', 'cpu'))
    else:
        return prepare_kfl_QRFf_batch(is_test, index_list, minibatch_index, batch_size, S_list, dic_state,
                           params, Y, X, R_L_list, F_list, state_reset_counter_lst, device=params.get('device', 'cpu'))

def prepare_lstm_batch(is_test, index_list, minibatch_index, batch_size, S_list, dic_state, params, Y, X, R_L_list, F_list, state_reset_counter_lst):
    # Logic remains identical as it manipulates numpy arrays for state management
    if is_test == 1:
        reset_state = -1
    else:
        reset_state = params['reset_state']

    curr_id_lst = index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    
    # Handle first batch case where pre_id_lst would be out of bounds
    if minibatch_index > 0:
        pre_id_lst = index_list[(minibatch_index - 1) * batch_size: (minibatch_index) * batch_size]
    else:
        pre_id_lst = [-1] * batch_size # Dummy values for first batch

    curr_sid = S_list[curr_id_lst]
    pre_sid = S_list[pre_id_lst] if minibatch_index > 0 else np.array([-1]*batch_size)
    
    new_S = ut.get_zero_state(params)

    if minibatch_index > 0:
        for idx in range(batch_size):
            state_reset_counter = state_reset_counter_lst[idx]
            if state_reset_counter % reset_state == 0 and reset_state > 0:
                # Reset state
                for s in range(params['nlayer']):
                    new_S[s][0][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx] = 0
            elif pre_sid[idx] != curr_sid[idx]: # Sequence changed
                for s in range(params['nlayer']):
                    new_S[s][0][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
                    new_S[s][1][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
                state_reset_counter_lst[idx] = 0
            elif curr_id_lst[idx] == pre_id_lst[idx]: # Repeated value
                state_reset_counter_lst[idx] = state_reset_counter - 1
                for s in range(params['nlayer']):
                    new_S[s][0][idx] = dic_state["lstm_pre"][s][0][idx]
                    new_S[s][1][idx] = dic_state["lstm_pre"][s][1][idx]
            else: # Normal progression
                for s in range(params['nlayer']):
                    new_S[s][0][idx] = dic_state["lstm_t"][s][0][idx]
                    new_S[s][1][idx] = dic_state["lstm_t"][s][1][idx]
    
    x = X[curr_id_lst]
    y = Y[curr_id_lst]
    if R_L_list is not None:
        r = R_L_list[curr_id_lst]
    else:
        r = None
    f = F_list[curr_id_lst]
    dic_state["lstm_pre"] = new_S

    return (dic_state, x, y, r, f, curr_sid, state_reset_counter_lst, curr_id_lst)

# def prepare_kfl_QRFf_batch(is_test, index_list, minibatch_index, batch_size, S_list, dic_state,
#                            params, Y, X, R_L_list, F_list, state_reset_counter_lst):
#     # Logic remains identical
#     if is_test == 1:
#         reset_state = -1
#         P_mul = 0.01
#     else:
#         reset_state = params['reset_state']
#         P_mul = params['P_mul']

#     curr_id_lst = index_list[minibatch_index * batch_size: (minibatch_index + 1) * batch_size]
    
#     if minibatch_index > 0:
#         pre_id_lst = index_list[(minibatch_index - 1) * batch_size: (minibatch_index) * batch_size]
#     else:
#         pre_id_lst = [-1] * batch_size

#     curr_sid = S_list[curr_id_lst]
#     pre_sid = S_list[pre_id_lst] if minibatch_index > 0 else np.array([-1]*batch_size)
    
#     new_state_F = ut.get_zero_state(params)
#     x = X[curr_id_lst]
#     y = Y[curr_id_lst]
#     new_P = np.asarray([np.diag([1.0] * params['n_output']) for i in range(params["batch_size"])], dtype=np.float32) * P_mul
#     _x = np.copy(x[:, 0, :])
    
#     if "Q" in params["model"]:
#         new_state_Q = ut.get_zero_state(params, 'Q')
#     if "R" in params["model"]:
#         new_state_R = ut.get_zero_state(params, 'R')
#     if "K" in params["model"]:
#         new_state_K = ut.get_zero_state(params, 'K')

#     if minibatch_index > 0:
#         for idx in range(batch_size):
#             state_reset_counter = state_reset_counter_lst[idx]
#             if state_reset_counter % reset_state == 0 and reset_state > 0:
#                 # Reset states
#                 for s in range(params['nlayer']):
#                     new_state_F[s][0][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
#                     new_state_F[s][1][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
#                 if "Q" in params["model"]:
#                     for s in range(params['Qnlayer']):
#                         new_state_Q[s][0][idx] = np.zeros(shape=(1, params['Qn_hidden']), dtype=np.float32)
#                         new_state_Q[s][1][idx] = np.zeros(shape=(1, params['Qn_hidden']), dtype=np.float32)
#                 if "R" in params["model"]:
#                     for s in range(params['Rnlayer']):
#                         new_state_R[s][0][idx] = np.zeros(shape=(1, params['Rn_hidden']), dtype=np.float32)
#                         new_state_R[s][1][idx] = np.zeros(shape=(1, params['Rn_hidden']), dtype=np.float32)
#                 if "K" in params["model"]:
#                     for s in range(params['Knlayer']):
#                         new_state_K[s][0][idx] = np.zeros(shape=(1, params['Kn_hidden']), dtype=np.float32)
#                         new_state_K[s][1][idx] = np.zeros(shape=(1, params['Kn_hidden']), dtype=np.float32)
                
#                 new_P = np.asarray([np.diag([1.0] * params['n_output']) for i in range(params["batch_size"])]) * P_mul
#                 _x[idx] = dic_state["_x_pre"][idx]
#                 state_reset_counter_lst[idx] = 0
                
#             elif pre_sid[idx] != curr_sid[idx]: # Sequence change
#                 for s in range(params['nlayer']):
#                     new_state_F[s][0][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
#                     new_state_F[s][1][idx] = np.zeros(shape=(1, params['n_hidden']), dtype=np.float32)
#                 if "Q" in params["model"]:
#                     for s in range(params['Qnlayer']):
#                         new_state_Q[s][0][idx] = np.zeros(shape=(1, params['Qn_hidden']), dtype=np.float32)
#                         new_state_Q[s][1][idx] = np.zeros(shape=(1, params['Qn_hidden']), dtype=np.float32)
#                 if "R" in params["model"]:
#                     for s in range(params['Rnlayer']):
#                         new_state_R[s][0][idx] = np.zeros(shape=(1, params['Rn_hidden']), dtype=np.float32)
#                         new_state_R[s][1][idx] = np.zeros(shape=(1, params['Rn_hidden']), dtype=np.float32)
#                 if "K" in params["model"]:
#                     for s in range(params['Knlayer']):
#                         new_state_K[s][0][idx] = np.zeros(shape=(1, params['Kn_hidden']), dtype=np.float32)
#                         new_state_K[s][1][idx] = np.zeros(shape=(1, params['Kn_hidden']), dtype=np.float32)
                
#                 new_P[idx] = np.diag([1.0] * params['n_output']) * P_mul
#                 state_reset_counter_lst[idx] = 0
                
#             elif curr_id_lst[idx] == pre_id_lst[idx]: # Repeat
#                 state_reset_counter_lst[idx] = state_reset_counter - 1
#                 for s in range(params['nlayer']):
#                     new_state_F[s][0][idx] = dic_state["F_pre"][s][0][idx]
#                     new_state_F[s][1][idx] = dic_state["F_pre"][s][1][idx]
#                 if "Q" in params["model"]:
#                     for s in range(params['Qnlayer']):
#                         new_state_Q[s][0][idx] = dic_state["Q_pre"][s][0][idx]
#                         new_state_Q[s][1][idx] = dic_state["Q_pre"][s][1][idx]
#                 if "R" in params["model"]:
#                     for s in range(params['Rnlayer']):
#                         new_state_R[s][0][idx] = dic_state["R_pre"][s][0][idx]
#                         new_state_R[s][1][idx] = dic_state["R_pre"][s][1][idx]
#                 if "K" in params["model"]:
#                     for s in range(params['Knlayer']):
#                         new_state_K[s][0][idx] = dic_state["K_pre"][s][0][idx]
#                         new_state_K[s][1][idx] = dic_state["K_pre"][s][1][idx]
#                 new_P[idx] = dic_state["PCov_pre"][idx]
#                 _x[idx] = dic_state["_x_pre"][idx]
                
#             else: # Normal
#                 for s in range(params['nlayer']):
#                     new_state_F[s][0][idx] = dic_state["F_t"][s][0][idx]
#                     new_state_F[s][1][idx] = dic_state["F_t"][s][1][idx]
#                 if "Q" in params["model"]:
#                     for s in range(params['Qnlayer']):
#                         new_state_Q[s][0][idx] = dic_state["Q_t"][s][0][idx]
#                         new_state_Q[s][1][idx] = dic_state["Q_t"][s][1][idx]
#                 if "R" in params["model"]:
#                     for s in range(params['Rnlayer']):
#                         new_state_R[s][0][idx] = dic_state["R_t"][s][0][idx]
#                         new_state_R[s][1][idx] = dic_state["R_t"][s][1][idx]
#                 if "K" in params["model"]:
#                     for s in range(params['Knlayer']):
#                         new_state_K[s][0][idx] = dic_state["K_t"][s][0][idx]
#                         new_state_K[s][1][idx] = dic_state["K_t"][s][1][idx]
#                 new_P[idx] = dic_state["PCov_t"][idx]
#                 _x[idx] = dic_state["_x_t"][idx]

#     dic_state["F_pre"] = new_state_F
#     dic_state["PCov_pre"] = new_P
#     if "Q" in params["model"]:
#         dic_state["Q_pre"] = new_state_Q
#     if "R" in params["model"]:
#         dic_state["R_pre"] = new_state_R
#     if "K" in params["model"]:
#         dic_state["K_pre"] = new_state_K

#     dic_state["_x_pre"] = _x

#     if R_L_list is not None:
#         r = R_L_list[curr_id_lst]
#     else:
#         r = None
#     f = F_list[curr_id_lst]

#     return (dic_state, x, y, r, f, curr_sid, state_reset_counter_lst, curr_id_lst)


def get_zero_state_torch(nlayer, batch_size, hidden_size, device):
    """
    返回:
        state = [
            [h, c],   # layer 0
            [h, c],   # layer 1
            ...
        ]
    h, c shape: (batch_size, hidden_size)
    """
    state = []
    for _ in range(nlayer):
        h = torch.zeros(batch_size, hidden_size, device=device)
        c = torch.zeros(batch_size, hidden_size, device=device)
        state.append([h, c])
    return state


def prepare_kfl_QRFf_batch(
    is_test,
    index_list,
    minibatch_index,
    batch_size,
    S_list,
    dic_state,
    params,
    Y,
    X,
    R_L_list,
    F_list,
    state_reset_counter_lst,
    device
):
    # ---------------------------
    # Step 1: 参数获取
    # ---------------------------
    if is_test == 1:
        reset_state = -1
        P_mul = 0.01
    else:
        reset_state = params['reset_state']
        P_mul = params['P_mul']

    # ---------------------------
    # Step 2: 数据索引获取
    # ---------------------------
    start_idx = minibatch_index * batch_size
    end_idx = (minibatch_index + 1) * batch_size
    
    curr_id_lst = index_list[start_idx : end_idx]

    if minibatch_index > 0:
        pre_id_lst = index_list[start_idx - batch_size : start_idx]
    else:
        pre_id_lst = [-1] * batch_size

    # ---------------------------
    # Step 3: 序列 ID 转换
    # ---------------------------
    # 修复：确保 S_list 支持列表索引，并将结果转为 Tensor
    # 注意：S_list[curr_id_lst] 如果 curr_id_lst 包含 -1 可能会有问题，这里假设 S_list 是 list 或 np.array
    curr_sid = torch.as_tensor(S_list[curr_id_lst], device=device)
    
    if minibatch_index > 0:
        pre_sid = torch.as_tensor(S_list[pre_id_lst], device=device)
    else:
        # 修复：第一帧没有前序，sid 设为 -1
        pre_sid = torch.full((batch_size,), -1, device=device, dtype=torch.long)

    # ---------------------------
    # Step 4: 初始化新状态容器
    # ---------------------------
    # Check if model uses LSTM or Transformer
    is_transformer = "transformer" in params.get("model", "").lower()
    
    if not is_transformer:
        new_state_F = get_zero_state_torch(
            params['nlayer'], batch_size, params['n_hidden'], device
        )

        if "Q" in params["model"]:
            new_state_Q = get_zero_state_torch(
                params['Qnlayer'], batch_size, params['Qn_hidden'], device
            )
        if "R" in params["model"]:
            new_state_R = get_zero_state_torch(
                params['Rnlayer'], batch_size, params['Rn_hidden'], device
            )
        if "K" in params["model"]:
            new_state_K = get_zero_state_torch(
                params['Knlayer'], batch_size, params['Kn_hidden'], device
            )

    # Kalman P 初始化
    eye = torch.eye(params['n_output'], device=device)
    new_P = eye.unsqueeze(0).repeat(batch_size, 1, 1) * P_mul

    # 准备 Batch 数据
    x = torch.as_tensor(X[curr_id_lst], device=device)
    y = torch.as_tensor(Y[curr_id_lst], device=device)

    # _x 初始化 (默认为 0，后续逻辑会根据情况覆盖)
    _x = torch.zeros(batch_size, params['n_output'], device=device)

    # ---------------------------
    # Step 5: 状态继承 / Reset 逻辑
    # ---------------------------
    # Check if model uses LSTM or Transformer
    is_transformer = "transformer" in params.get("model", "").lower()
    
    if not is_transformer and minibatch_index > 0:
        
        # --- 辅助函数：安全获取上一时刻的状态 (_t) ---
        # 优先读取 dic_state["XXX_t"]，如果不存在（KeyError），则读取 dic_state["XXX_pre"] 作为后备
        # 如果都不存在，返回 None (触发默认重置逻辑)
        
        def get_prev_state(key_t, key_pre, default_val=None):
            val = dic_state.get(key_t)
            if val is None:
                val = dic_state.get(key_pre)
            if val is None:
                val = default_val
            return val

        # 获取 KF 状态
        prev_P_t = get_prev_state("PCov_t", "PCov_pre", eye.unsqueeze(0).repeat(batch_size, 1, 1) * P_mul)
        prev_x_t = get_prev_state("_x_t", "_x_pre", torch.zeros(batch_size, params['n_output'], device=device))

        # 获取 LSTM 状态
        prev_F_t = get_prev_state("F_t", "F_pre")
        prev_Q_t = get_prev_state("Q_t", "Q_pre") if "Q" in params["model"] else None
        prev_R_t = get_prev_state("R_t", "R_pre") if "R" in params["model"] else None
        prev_K_t = get_prev_state("K_t", "K_pre") if "K" in params["model"] else None

        for idx in range(batch_size):
            state_reset_counter = state_reset_counter_lst[idx]
            is_same_seq = (pre_sid[idx] == curr_sid[idx])

            # ---- Case 1: 序列切换 ----
            if not is_same_seq:
                # 新序列开始：重置所有状态
                # P 重置
                new_P[idx] = eye * P_mul
                # x 重置为 0
                _x[idx] = 0.0 
                # 计数器重置
                state_reset_counter_lst[idx] = 0
                # LSTM 状态保持 Zero (已在 Step 4 初始化，无需操作)

            # ---- Case 2: 强制周期 Reset (Counter) ----
            # 仅在同序列内生效
            elif reset_state > 0 and (state_reset_counter % reset_state == 0):
                # 物理继承：为了保持时间序列连续性，继承上一时刻的 _t 状态
                # 但重置 P 以打破长程误差积累
                new_P[idx] = eye * P_mul
                
                # 继承上一帧的输出
                if prev_x_t is not None:
                    _x[idx] = prev_x_t[idx]
                if prev_P_t is not None:
                    new_P[idx] = prev_P_t[idx]
                
                # 修正：根据通常逻辑，周期 reset 意味着重置协方差矩阵以防止发散
                # 这里选择重置 P
                new_P[idx] = eye * P_mul
                if prev_x_t is not None:
                    _x[idx] = prev_x_t[idx]

                # 继承 LSTM 状态
                if prev_F_t is not None:
                    for s in range(params['nlayer']):
                        new_state_F[s][0][idx] = prev_F_t[s][0][idx]
                        new_state_F[s][1][idx] = prev_F_t[s][1][idx]

                if prev_Q_t is not None and "Q" in params["model"]:
                    for s in range(params['Qnlayer']):
                        new_state_Q[s][0][idx] = prev_Q_t[s][0][idx]
                        new_state_Q[s][1][idx] = prev_Q_t[s][1][idx]

                if prev_R_t is not None and "R" in params["model"]:
                    for s in range(params['Rnlayer']):
                        new_state_R[s][0][idx] = prev_R_t[s][0][idx]
                        new_state_R[s][1][idx] = prev_R_t[s][1][idx]

                if prev_K_t is not None and "K" in params["model"]:
                    for s in range(params['Knlayer']):
                        new_state_K[s][0][idx] = prev_K_t[s][0][idx]
                        new_state_K[s][1][idx] = prev_K_t[s][1][idx]

                state_reset_counter_lst[idx] = 0

            # ---- Case 3: 正常继承 ----
            else:
                # 同序列内，正常继承上一帧的 _t 状态
                if prev_P_t is not None:
                    new_P[idx] = prev_P_t[idx]
                if prev_x_t is not None:
                    _x[idx] = prev_x_t[idx]
                
                if prev_F_t is not None:
                    for s in range(params['nlayer']):
                        new_state_F[s][0][idx] = prev_F_t[s][0][idx]
                        new_state_F[s][1][idx] = prev_F_t[s][1][idx]

                if prev_Q_t is not None and "Q" in params["model"]:
                    for s in range(params['Qnlayer']):
                        new_state_Q[s][0][idx] = prev_Q_t[s][0][idx]
                        new_state_Q[s][1][idx] = prev_Q_t[s][1][idx]

                if prev_R_t is not None and "R" in params["model"]:
                    for s in range(params['Rnlayer']):
                        new_state_R[s][0][idx] = prev_R_t[s][0][idx]
                        new_state_R[s][1][idx] = prev_R_t[s][1][idx]

                if prev_K_t is not None and "K" in params["model"]:
                    for s in range(params['Knlayer']):
                        new_state_K[s][0][idx] = prev_K_t[s][0][idx]
                        new_state_K[s][1][idx] = prev_K_t[s][1][idx]
    elif is_transformer and minibatch_index > 0:
        # Transformer model: only handle KF states
        def get_prev_state(key_t, key_pre, default_val=None):
            val = dic_state.get(key_t)
            if val is None:
                val = dic_state.get(key_pre)
            if val is None:
                val = default_val
            return val

        prev_P_t = get_prev_state("PCov_t", "PCov_pre", eye.unsqueeze(0).repeat(batch_size, 1, 1) * P_mul)
        prev_x_t = get_prev_state("_x_t", "_x_pre", torch.zeros(batch_size, params['n_output'], device=device))

        for idx in range(batch_size):
            state_reset_counter = state_reset_counter_lst[idx]
            is_same_seq = (pre_sid[idx] == curr_sid[idx])

            if not is_same_seq:
                new_P[idx] = eye * P_mul
                _x[idx] = 0.0
                state_reset_counter_lst[idx] = 0
            elif reset_state > 0 and (state_reset_counter % reset_state == 0):
                new_P[idx] = eye * P_mul
                if prev_x_t is not None:
                    _x[idx] = prev_x_t[idx]
                state_reset_counter_lst[idx] = 0
            else:
                if prev_P_t is not None:
                    new_P[idx] = prev_P_t[idx]
                if prev_x_t is not None:
                    _x[idx] = prev_x_t[idx]

    # ---------------------------
    # Step 6: 更新 dic_state (_pre)
    # ---------------------------
    # Check if model uses LSTM or Transformer
    is_transformer = "transformer" in params.get("model", "").lower()
    
    if not is_transformer:
        dic_state["F_pre"] = new_state_F
        if "Q" in params["model"]:
            dic_state["Q_pre"] = new_state_Q
        if "R" in params["model"]:
            dic_state["R_pre"] = new_state_R
        if "K" in params["model"]:
            dic_state["K_pre"] = new_state_K
    
    dic_state["PCov_pre"] = new_P
    dic_state["_x_pre"] = _x

    # ---------------------------
    # Step 7: 获取 r (mask) 和 f
    # ---------------------------
    r = (
        torch.as_tensor(R_L_list[curr_id_lst], device=device)
        if R_L_list is not None
        else None
    )
    f = F_list

    return (
        dic_state,
        x, y, r, f,
        curr_sid,
        state_reset_counter_lst,
        curr_id_lst
    )

def get_feed(params, r, x, y, I, dic_state, is_training=0, device='cpu'):
    """
    PyTorch 版本的 get_feed。
    不再返回 feed_dict，而是返回一个包含 PyTorch Tensors 的字典。
    调用者可以使用 model(**feed) 将这些数据传入模型。
    
    注意：这里假设 PyTorch 模型的 forward 方法接受与字典键相匹配的参数名。
    """
    
    # 辅助函数：将 state 字典转换为 Tensor 字典
    def convert_states(state_dict):
        return {k: torch.from_numpy(v).float().to(device) for k, v in state_dict.items()}

    # 将基本数据转换为 Tensor
    # 注意：r 可能为 None
    x_tensor = torch.from_numpy(x).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)
    I_tensor = torch.from_numpy(I).float().to(device)
    r_tensor = torch.from_numpy(r).float().to(device) if r is not None else None
    
    # 将状态字典转换为 Tensor
    state_tensor_dict = convert_states(dic_state)
    
    # Dropout 概率设置
    output_keep_prob = params['rnn_keep_prob'] if is_training == 1 else 1.0
    input_keep_prob = params['input_keep_prob'] if is_training == 1 else 1.0

    # 构建返回的输入字典
    # 为了兼容性，我们尽量保持与原 TF feed_dict 类似的键名，但通常 PyTorch 模型使用参数名
    # 这里构建一个通用字典，train.py 中可以直接解包传入 model
    
    inputs = {
        'x': x_tensor,
        'y': y_tensor,
        'r': r_tensor,
        'I': I_tensor,
        'state_dict': state_tensor_dict,
        'is_training': is_training,
        'keep_prob': output_keep_prob,
        'input_keep_prob': input_keep_prob
    }
    
    # 针对不同模型的特定参数处理
    # 如果你的 PyTorch 模型 forward 方法签名需要特定的参数名，可以在这里映射
    # 这里我们添加具体的键，以便在 train.py 中可以直接访问
    
    if params["model"] == "kfl_K":
        inputs['initial_state'] = state_tensor_dict.get("F_pre")
        inputs['initial_state_K'] = state_tensor_dict.get("K_pre")
        
    elif params["model"] == "kfl_QRf":
        inputs['initial_state'] = state_tensor_dict.get("F_pre")
        inputs['initial_state_Q'] = state_tensor_dict.get("Q_pre")
        inputs['initial_state_R'] = state_tensor_dict.get("R_pre")
        inputs['_P_inp'] = state_tensor_dict.get("PCov_pre")
        inputs['_x_inp'] = state_tensor_dict.get("_x_pre")
        
    elif params["model"] == "kfl_Rf":
        inputs['initial_state'] = state_tensor_dict.get("F_pre")
        inputs['initial_state_R'] = state_tensor_dict.get("R_pre")
        inputs['_P_inp'] = state_tensor_dict.get("PCov_pre")
        
    elif params["model"] == "kfl_f":
        inputs['initial_state'] = state_tensor_dict.get("F_pre")
        inputs['_P_inp'] = state_tensor_dict.get("PCov_pre")
        
    elif params["model"] == "kfl_QRFf":
        inputs['initial_state'] = state_tensor_dict.get("F_pre")
        inputs['initial_state_Q'] = state_tensor_dict.get("Q_pre")
        inputs['initial_state_R'] = state_tensor_dict.get("R_pre")
        inputs['_P_inp'] = state_tensor_dict.get("PCov_pre")
        
    elif params["model"] == "lstm":
        inputs['initial_state'] = state_tensor_dict.get("lstm_pre")
        # LSTM 模型通常不需要 I, _P_inp 等卡尔曼滤波相关变量
        
    elif params["model"] == "kf_QR":
        inputs['initial_state_Q'] = state_tensor_dict.get("Q_pre")
        inputs['initial_state_R'] = state_tensor_dict.get("R_pre")
        # H 和 F 矩阵通常直接从 params 获取，或者是模型的一部分，这里作为参数传入
        inputs['H'] = torch.from_numpy(params['H_mat']).float().to(device) if 'H_mat' in params else None
        inputs['F'] = torch.from_numpy(params['F_mat']).float().to(device) if 'F_mat' in params else None

    return inputs

# 假设 H36MSequenceDataset 类定义在其他地方（例如 dataset.py），
# 或者你可以直接将上面的类定义粘贴到这个文件中。
# from dataset import H36MSequenceDataset 

def get_dataloader(params, is_training=True):
    """
    根据参数配置生成 DataLoader。
    
    Args:
        params (dict): 包含配置信息的字典。
        is_training (bool): 是否为训练模式。
            - True: 加载训练集，开启 shuffle，计算并保存 normalization stats。
            - False: 加载测试/验证集，不 shuffle，加载已有的 stats。
            
    Returns:
        torch.utils.data.DataLoader
    """
    # 1. 确定模式
    # Dataset 类中通过 mode == "train" 判断是否计算并保存归一化参数
    mode = 'train' if is_training else 'test'
    
    data_path_key = 'train_data_path' if is_training else 'test_data_path'
    h5_path = params.get(data_path_key)
    
    if h5_path is None:
        raise ValueError(f"params 字典中缺少 '{data_path_key}'，请检查配置。")

    # 获取其他超参数，并设置默认值以防 params 中缺失
    seq_length = params.get('seq_length', 10)
    step = params.get('step', 5)
    batch_size = params.get('batch_size', 32)
    num_workers = params.get('num_workers', 4) # 建议设为 4 或 8，根据 CPU 核心数调整
    pin_memory = torch.cuda.is_available() # 如果有 GPU 则开启 pin_memory 以加速传输

    train_transform = build_inception_train_transform()
    DATA_ROOT = "data/h36m"

    H5_TRAIN  = os.path.join(DATA_ROOT, "annot", "train.h5")
    IMG_ROOT  = os.path.join(DATA_ROOT, "images", "train")
    IMG_TRAIN = os.path.join(DATA_ROOT, "annot", "train_images.txt")

    train_dataset = H36MDataset(
        img_txt=IMG_TRAIN,
        h5_path=H5_TRAIN,
        img_root=IMG_ROOT,
        transform=train_transform
    )

    # 4. 实例化 DataLoader
    loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=is_training,   # 训练时打乱数据，测试/验证时不需要
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=is_training  # 训练时通常丢弃最后一个不完整的 batch
    )
    
    return loader

import torch
import numpy as np

def initialize_model_states(params, device):
    """
    初始化 Kalman 状态和 LSTM 隐藏状态
    """
    batch_size = params['batch_size']
    nout = params['n_output']
    
    # 1. 初始化 Kalman 状态 x (Batch, NOUT) 和 协方差 P (Batch, NOUT, NOUT)
    # 通常 x 初始化为 0，P 初始化为单位矩阵
    x_inp = torch.zeros((batch_size, nout)).to(device)
    P_inp = torch.stack([torch.eye(nout) for _ in range(batch_size)]).to(device)
    
    # 2. 初始化单位矩阵 I (Batch, NOUT, NOUT)
    I_matrix = torch.stack([torch.eye(nout) for _ in range(batch_size)]).to(device)
    
    # 3. 初始化 LSTM 隐藏状态 (h, c)
    # 注意：模型内部使用 ModuleList，状态需要是 [(h, c), (h, c), ...] 格式
    def get_lstm_state(num_layers, hidden_size):
        states = []
        for _ in range(num_layers):
            h = torch.zeros(batch_size, hidden_size).to(device)
            c = torch.zeros(batch_size, hidden_size).to(device)
            states.append((h, c))
        return states

    state_dict = {
        "F_pre": get_lstm_state(params['nlayer'], params['n_hidden']),
        "Q_pre": get_lstm_state(params['Qnlayer'], params['Qn_hidden']),
        "R_pre": get_lstm_state(params['Rnlayer'], params['Rn_hidden'])
    }
    
    return x_inp, P_inp, I_matrix, state_dict


def detach_state(v):
    """
    支持:
      - Tensor
      - list[(Tensor, Tensor)]
    """
    if torch.is_tensor(v):
        return v.detach()

    elif isinstance(v, list):
        out = []
        for item in v:
            # LSTM state: (h, c)
            if isinstance(item, (tuple, list)):
                h, c = item
                out.append((
                    h.detach(),
                    c.detach()
                ))
            else:
                raise TypeError(f"Unsupported list item type: {type(item)}")
        return out

    else:
        raise TypeError(f"Unsupported state type: {type(v)}")