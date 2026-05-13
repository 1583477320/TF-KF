import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim

from helper import train_helper as th
from helper import config
from model_runner.klstm.pure_kalman import PureKalmanFilter
from helper import dt_utils as dut
from helper import utils as ut


@torch.no_grad()
def test_pure_kalman(
    kf,
    params,
    X, Y,
    index_list,
    S_list,
    R_L_list,
    batch_size,
    device
):
    """
    测试纯卡尔曼滤波器
    """
    NOUT = params["n_output"]
    I = torch.eye(NOUT, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    dic_state = ut.get_state_list(params)
    
    # 移除LSTM状态
    lstm_states = ["F_pre", "Q_pre", "R_pre", "K_pre", "F_t", "Q_t", "R_t", "K_t"]
    for state_key in lstm_states:
        if state_key in dic_state:
            del dic_state[state_key]
    
    # 确保params中不包含LSTM相关参数
    params['nlayer'] = 0
    params['Qnlayer'] = 0
    params['Rnlayer'] = 0
    params['Knlayer'] = 0
    
    n_batches = len(index_list) // batch_size
    
    total_loss = 0.0
    total_count = 0
    
    for minibatch_index in range(n_batches):
        (dic_state, x, y, r, f, curr_sid, state_reset_counter_lst, curr_id_lst) = \
            th.prepare_kfl_QRFf_batch(
                is_test=1,
                index_list=index_list,
                minibatch_index=minibatch_index,
                batch_size=batch_size,
                S_list=S_list,
                dic_state=dic_state,
                params=params,
                Y=Y,
                X=X,  # 使用CNN特征作为输入
                R_L_list=R_L_list,
                F_list=None,
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
        
        actual_bsz = x.shape[0]
        
        if dic_state.get("_x_pre") is None:
            _x_inp = torch.zeros(actual_bsz, NOUT, device=device)
        else:
            _x_inp = dic_state["_x_pre"]
        
        if dic_state.get("PCov_pre") is None:
            _P_inp = I[:actual_bsz].clone()
        else:
            _P_inp = dic_state["PCov_pre"]
        
        # 执行纯卡尔曼滤波
        x_filtered, P_filtered = kf.forward(x, repeat_data, _x_inp, _P_inp)
        
        # 计算损失
        flt = torch.squeeze(torch.reshape(repeat_data, [-1, 1]), 1)
        indices = torch.where(torch.not_equal(flt, 0))[0]
        
        xres_flat = torch.reshape(x_filtered, [-1, NOUT])
        final_output = torch.index_select(xres_flat, 0, indices)
        target_flat = torch.reshape(y, [-1, NOUT])
        y_target = torch.index_select(target_flat, 0, indices)
        
        mse_loss = torch.mean(torch.square(final_output - y_target))
        
        batch_count = final_output.shape[0]
        total_loss += mse_loss.item() * batch_count
        total_count += batch_count
        
        # 更新状态
        dic_state["_x_pre"] = x_filtered[:, -1, :]
        dic_state["PCov_pre"] = P_filtered[:, -1, :, :]
    
    if total_count > 0:
        avg_loss = total_loss / total_count
    else:
        avg_loss = 0.0
    
    print(f"[Test Pure Kalman] Avg MSE Loss = {avg_loss:.6f}")
    return avg_loss


def train_pure_kalman(kf, params, device, X_train, Y_train, index_train_list, S_Train_list, R_L_Train_list, X_test, Y_test, index_test_list, S_Test_list, R_L_Test_list, batch_size):
    """
    训练纯卡尔曼滤波器
    
    参数:
    kf: PureKalmanFilter 实例
    params: 参数字典
    device: 设备
    X_train, Y_train: 训练数据
    index_train_list: 训练数据索引
    S_Train_list: 训练序列列表
    R_L_Train_list: 训练数据的R_L列表
    X_test, Y_test: 测试数据
    index_test_list: 测试数据索引
    S_Test_list: 测试序列列表
    R_L_Test_list: 测试数据的R_L列表
    batch_size: 批次大小
    
    返回:
    best_test_loss: 最佳测试损失
    """
    NOUT = params["n_output"]
    I = torch.eye(NOUT, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    n_train_batches = len(index_train_list) // batch_size
    n_test_batches = len(index_test_list) // batch_size
    
    print('Training pure Kalman filter')
    
    best_test_loss = float('inf')
    
    for e in range(10):  # 运行10个epoch进行评估
        total_train_loss = 0
        state_reset_counter_lst = [0 for _ in range(batch_size)]
        dic_state = ut.get_state_list(params)
        
        # 移除LSTM状态
        lstm_states = ["F_pre", "Q_pre", "R_pre", "K_pre", "F_t", "Q_t", "R_t", "K_t"]
        for state_key in lstm_states:
            if state_key in dic_state:
                del dic_state[state_key]
        
        # 确保params中不包含LSTM相关参数
        params['nlayer'] = 0
        params['Qnlayer'] = 0
        params['Rnlayer'] = 0
        params['Knlayer'] = 0
        
        for minibatch_index in range(n_train_batches):
            (dic_state, x, y, r_batch, f_batch, _, state_reset_counter_lst, _) = \
                th.prepare_kfl_QRFf_batch(
                    is_test=0, 
                    index_list=index_train_list, 
                    minibatch_index=minibatch_index,
                    batch_size=params['batch_size'],
                    S_list=S_Train_list,
                    dic_state=dic_state,
                    params=params,
                    Y=Y_train,
                    X=X_train,  # 使用CNN特征作为输入
                    R_L_list=R_L_Train_list,
                    F_list=None,
                    state_reset_counter_lst=state_reset_counter_lst,
                    device=device
                )
            
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x).float().to(device)
            if not isinstance(y, torch.Tensor):
                y = torch.from_numpy(y).float().to(device)
            
            if r_batch is None:
                repeat_data = torch.ones((x.shape[0], x.shape[1]), device=device)
            else:
                if not isinstance(r_batch, torch.Tensor):
                    repeat_data = torch.from_numpy(r_batch).float().to(device)
                else:
                    repeat_data = r_batch.float()
            
            actual_bsz = x.shape[0]
            
            if dic_state.get("_x_pre") is None:
                _x_inp = torch.zeros(actual_bsz, NOUT, device=device)
            else:
                _x_inp = dic_state["_x_pre"]
            
            if dic_state.get("PCov_pre") is None:
                _P_inp = I[:actual_bsz].clone()
            else:
                _P_inp = dic_state["PCov_pre"]
            
            # 执行纯卡尔曼滤波
            x_filtered, P_filtered = kf.forward(x, repeat_data, _x_inp, _P_inp)
            
            # 计算损失
            # 应用掩码
            flt = torch.squeeze(torch.reshape(repeat_data, [-1, 1]), 1)
            indices = torch.where(torch.not_equal(flt, 0))[0]
            
            xres_flat = torch.reshape(x_filtered, [-1, NOUT])
            final_output = torch.index_select(xres_flat, 0, indices)
            target_flat = torch.reshape(y, [-1, NOUT])
            y_target = torch.index_select(target_flat, 0, indices)
            
            diff = final_output - y_target
            loss = 0.5 * torch.sum(diff ** 2)
            
            if diff.shape[0] > 0:
                loss = loss / diff.shape[0]
            
            total_train_loss += loss.item()
            
            # 更新状态
            dic_state["_x_pre"] = x_filtered[:, -1, :]
            dic_state["PCov_pre"] = P_filtered[:, -1, :, :]
            
            if (minibatch_index % 100 == 0):
                print("Training batch loss: (%i / %i / %i)  %f"%(e, minibatch_index, n_train_batches, loss.item()))
        
        total_train_loss = total_train_loss / n_train_batches
        s = 'TRAIN (Pure Kalman) --> epoch %i | error %f'%(e, total_train_loss)
        ut.log_write(s, params)
        
        # 测试
        pre_test = "TEST_Data (Pure Kalman)"
        test_eval_loss = test_pure_kalman(kf, params, X_test, Y_test, index_test_list, S_Test_list, R_L_Test_list, batch_size, device)
        
        s = 'TEST (Pure Kalman) --> epoch %i | error %f'%(e, test_eval_loss)
        ut.log_write(s, params)
        
        # 更新最佳测试损失
        if test_eval_loss < best_test_loss:
            best_test_loss = test_eval_loss
    
    return best_test_loss


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 配置参数
    params = config.get_params()
    params['normalise_data'] = 4
    params['reset_state'] = 5
    params['seq_length'] = 50
    params["reload_data"] = 0
    params['batch_size'] = 5
    params = config.update_params(params)
    params["device"] = device
    params["model"] = "pure_kalman"

    # 准备数据
    (params, X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list,
            X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list) = \
            dut.prepare_training_set(params)

    (index_train_list, S_Train_list) = dut.get_seq_indexes(params, S_Train_list)
    (index_test_list, S_Test_list) = dut.get_seq_indexes(params, S_Test_list)

    batch_size = params['batch_size']
    n_train_batches = len(index_train_list) // batch_size
    n_test_batches = len(index_test_list) // batch_size
    params['training_size'] = len(X_train) * params['seq_length']
    params['test_size'] = len(X_test) * params['seq_length']
    
    # 初始化纯卡尔曼滤波器
    NOUT = params["n_output"]
    # 初始化固定的 F, Q, R 矩阵
    F = torch.eye(NOUT, device=device)
    Q = torch.eye(NOUT, device=device) * 0.1  # 过程噪声协方差
    R = torch.eye(NOUT, device=device) * 1.0  # 测量噪声协方差
    
    pure_kf = PureKalmanFilter(F, Q, R, NOUT)
    
    # 训练纯卡尔曼滤波器
    print("=== Training Pure Kalman Filter ===")
    best_test_loss = train_pure_kalman(pure_kf, params, device, X_train, Y_train, index_train_list, S_Train_list, R_L_Train_list, 
                     X_test, Y_test, index_test_list, S_Test_list, R_L_Test_list, batch_size)
    
    # 保存模型到 model/pure_kalman 目录
    model_dir = "/home/zhao/pyproject/TF-KF/model/pure_kalman"
    os.makedirs(model_dir, exist_ok=True)
    
    # 保存模型参数
    model_state = {
        'F': pure_kf.F.cpu().numpy(),
        'Q': pure_kf.Q.cpu().numpy(),
        'R': pure_kf.R.cpu().numpy(),
        'H': pure_kf.H.cpu().numpy() if pure_kf.H is not None else None,
        'dim': pure_kf.dim,
        'best_test_loss': best_test_loss,
        'params': {
            'n_output': NOUT,
            'seq_length': params['seq_length'],
            'batch_size': params['batch_size'],
            'normalise_data': params['normalise_data'],
            'reset_state': params['reset_state']
        }
    }
    
    model_path = os.path.join(model_dir, 'pure_kalman_model.pth')
    torch.save(model_state, model_path)
    print(f"Model saved to {model_path}")
    
    # 保存模型配置信息
    config_path = os.path.join(model_dir, 'model_config.txt')
    with open(config_path, 'w') as f:
        f.write(f"Pure Kalman Filter Model Configuration\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"State Dimension: {pure_kf.dim}\n")
        f.write(f"Best Test Loss: {best_test_loss:.6f}\n")
        f.write(f"\nMatrix Shapes:\n")
        f.write(f"F shape: {pure_kf.F.shape}\n")
        f.write(f"Q shape: {pure_kf.Q.shape}\n")
        f.write(f"R shape: {pure_kf.R.shape}\n")
        if pure_kf.H is not None:
            f.write(f"H shape: {pure_kf.H.shape}\n")
        f.write(f"\nParameters:\n")
        for key, value in model_state['params'].items():
            f.write(f"{key}: {value}\n")
    print(f"Model config saved to {config_path}")
