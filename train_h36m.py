import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from helper import train_helper as th
from helper import config
from helper import checkpoint as cp
from model_runner.klstm.kfl_QRf import Model as kfl_QRf
from model_runner.klstm.kfl_QRFf import Model as kfl_QRFf
from model_runner.klstm.kfl_QRFf_transformer import Model as kfl_QRFf_transformer
from model_runner.klstm.kfl_K import Model as kfl_K
from model_runner.lstm.pt_lstm import Model as lstm 
from helper import dt_utils as dut
from helper import utils as ut


@torch.no_grad()
def test_data(
    model,
    params,
    X, Y,
    index_list,
    S_list,
    R_L_list,
    F_list,
    batch_size,
    device
):
    model.eval()

    # -------------------------
    # 初始化 Kalman 状态
    # -------------------------
    dic_state = ut.get_state_list(params)

    # --- 修复点1：将 dic_state 中的 numpy 数组初始化为 Tensor ---
    # 因为 helper 函数现在要求输入必须是 Tensor
    def to_tensor(v):
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float().to(device)
        elif isinstance(v, list):
             # 处理 LSTM state list [(h, c), ...]
            return [(to_tensor(t[0]), to_tensor(t[1])) for t in v]
        return v

    for k, v in dic_state.items():
        dic_state[k] = to_tensor(v)
    
    # Check if model uses LSTM or Transformer
    is_transformer = "transformer" in params.get("model", "").lower()
    
    # For transformer models, we don't need LSTM states
    if is_transformer:
        # Remove LSTM states if they exist
        lstm_states = ["F_pre", "Q_pre", "R_pre", "K_pre"]
        for state_key in lstm_states:
            if state_key in dic_state:
                del dic_state[state_key]

    NOUT = params["n_output"]
    I = torch.eye(NOUT, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

    params["reset_state"] = -1 

    state_reset_counter_lst = [0 for _ in range(batch_size)]
    n_batches = len(index_list) // batch_size

    total_loss = 0.0
    total_count = 0

    # -------------------------
    # batch loop with tqdm
    # -------------------------
    test_pbar = tqdm(range(n_batches), desc="Testing", unit="batch", leave=False)
    for minibatch_index in test_pbar:

        state_reset_counter_lst = [s + 1 for s in state_reset_counter_lst]

        # ------------------------------------------------
        # 1. 取 batch (helper 返回的应该是 Tensor)
        # ------------------------------------------------
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

        # --- 修复点2：显式确保输入数据是 Tensor ---
        # 虽然 helper 可能已经做了转换，但为了保险起见再次检查
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float().to(device)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float().to(device)

        # 处理 Mask (repeat_data)
        if r is None:
            repeat_data = torch.ones((x.shape[0], x.shape[1]), device=device)
        else:
            if not isinstance(r, torch.Tensor):
                repeat_data = torch.from_numpy(r).float().to(device)
            else:
                repeat_data = r.float()
        
        # 确保 repeat_data 形状正确 (B, T)
        if repeat_data.dim() == 1:
            repeat_data = repeat_data.unsqueeze(1)

        # 准备初始状态 _x_inp 和 _P_inp
        # 使用实际 batch size (x.shape[0]) 以防最后一个 batch 不满
        actual_bsz = x.shape[0]
        
        if dic_state.get("_x_pre") is None:
            _x_inp = torch.zeros(actual_bsz, NOUT, device=device)
        else:
            _x_inp = dic_state["_x_pre"]

        if dic_state.get("PCov_pre") is None:
            _P_inp = I[:actual_bsz].clone()
        else:
            _P_inp = dic_state["PCov_pre"]

        # ------------------------------------------------
        # 3. 模型推理
        # ------------------------------------------------
        # 传入 _I 时也要注意 batch size 切片
        cost, new_states = model(
            _z=x,
            target_data=y,
            repeat_data=repeat_data,
            _x_inp=_x_inp,
            _P_inp=_P_inp,
            _I=I[:actual_bsz],
            state_dict=dic_state,
            is_training=False
        )

        # ------------------------------------------------
        # 4. 更新状态 (保持 Tensor 格式)
        # ------------------------------------------------
        # --- 修复点3：不再转 numpy，而是 detach ---
        # LSTM 状态是列表
        if "F_t" in new_states:
            dic_state["F_pre"] = [(h.detach(), c.detach()) for h, c in new_states["F_t"]]
        if "Q_t" in new_states:
            dic_state["Q_pre"] = [(h.detach(), c.detach()) for h, c in new_states["Q_t"]]
        if "R_t" in new_states:
            dic_state["R_pre"] = [(h.detach(), c.detach()) for h, c in new_states["R_t"]]
        
        # 普通状态
        if "PCov_t" in new_states:
            dic_state["PCov_pre"] = new_states["PCov_t"].detach()
        if "_x_t" in new_states:
            dic_state["_x_pre"] = new_states["_x_t"].detach()

        # ------------------------------------------------
        # 5. 累积 Loss (计算 MSE)
        # ------------------------------------------------
        pred = model.final_output
        gt = model.y
        mse_loss = torch.mean(torch.square(pred - gt))
        
        batch_count = pred.shape[0]
        total_loss += mse_loss.item() * batch_count
        total_count += batch_count

    if total_count > 0:
        avg_loss = total_loss / total_count
    else:
        avg_loss = 0.0

    print(f"[Test] Avg MSE Loss = {avg_loss:.6f}")
    return avg_loss


def train(model, params, device, X_train, Y_train, X_test, Y_test, index_train_list, index_test_list, S_Train_list, S_Test_list, R_L_Train_list, R_L_Test_list, F_list_train, F_list_test):
    batch_size = params["batch_size"]
    num_epochs = 100
    decay_rate = 0.9
    show_every = 100
    deca_start = 3
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    scaler = GradScaler('cuda')
    
    actual_seq_length = X_train.shape[1]
    I_np = torch.eye(params['n_output']).repeat(params["batch_size"], 1, 1).to(device)   

    print('Training model: ' + params["model"])
    print(f'Batch size: {batch_size}')
    print(f'Sequence length: {actual_seq_length}')
    print(f'Mixed precision: Enabled')
    if params.get('predict_next_frame', False):
        print('Mode: Predict Next Frame')
    noise_std = params['noise_std']
    new_noise_std = 0.0

    n_train_batches = len(index_train_list) // batch_size
    
    base_checkpoint_dir = params.get("cp_file", "model")
    checkpoint_dir = os.path.join(base_checkpoint_dir, params["model"])
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_manager = cp.ModelCheckpoint(
        checkpoint_dir=checkpoint_dir,
        model_name=params["model"],
        max_checkpoints=5,
        save_best_only=False
    )
    
    start_epoch = 0
    resume_checkpoint = os.path.join(checkpoint_dir, "resume_checkpoint.ckpt")
    if os.path.exists(resume_checkpoint):
        try:
            ckpt = torch.load(resume_checkpoint, map_location=device)
            
            saved_predict_next_frame = ckpt.get('params', {}).get('predict_next_frame', False)
            current_predict_next_frame = params.get('predict_next_frame', False)
            
            if saved_predict_next_frame != current_predict_next_frame:
                print(f"[Checkpoint] Mode changed (predict_next_frame: {saved_predict_next_frame} -> {current_predict_next_frame}), starting fresh")
                start_epoch = 0
            else:
                model.load_state_dict(ckpt['model_state_dict'])
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                start_epoch = ckpt.get('epoch', 0) + 1
                print(f"[Checkpoint] Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"[Checkpoint] Failed to resume: {e}")
            start_epoch = 0

    epoch_pbar = tqdm(range(start_epoch, num_epochs), desc="Training", unit="epoch")
    for e in epoch_pbar:
        # 学习率衰减
        if e > (deca_start-1):
            new_lr = params['lr'] * (decay_rate ** (e))
        else:
            new_lr = params['lr']
        
        for g in optimizer.param_groups:
            g['lr'] = new_lr

        total_train_loss = 0

        state_reset_counter_lst = [0 for i in range(batch_size)]
        index_train_list_s = index_train_list
        dic_state = ut.get_state_list(params)
        
        # Check if model uses LSTM or Transformer
        is_transformer = "transformer" in params.get("model", "").lower()
        
        # For transformer models, we don't need LSTM states
        if is_transformer:
            # Remove LSTM states if they exist
            lstm_states = ["F_pre", "Q_pre", "R_pre", "K_pre"]
            for state_key in lstm_states:
                if state_key in dic_state:
                    del dic_state[state_key]
        
        if params["shufle_data"]==1 and params['reset_state']==1:
            index_train_list_s = ut.shufle_data(index_train_list)

        model.train() # 设置为训练模式
        batch_pbar = tqdm(range(n_train_batches), desc=f"Epoch {e}", unit="batch", leave=False)
        for minibatch_index in batch_pbar:
            # 1. 调用你写的准备函数 (包含了重置逻辑)
            # 注意：这里的 dic_state 会根据 case 1-4 自动决定是 reset 还是继承
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
                    X=X_train,
                    R_L_list=R_L_Train_list,
                    F_list=None,
                    state_reset_counter_lst=state_reset_counter_lst,
                    device=device
                )

            # 噪声注入
            if noise_std > 0.0:
               u_cnt = e*n_train_batches + minibatch_index
               if u_cnt in params['noise_schedule']:
                   if u_cnt==params['noise_schedule'][0]:
                     new_noise_std=noise_std
                   else:
                       new_noise_std = noise_std * (u_cnt / (params['noise_schedule'][1]))

                   s = 'NOISE --> u_cnt %i | error %f' % (u_cnt, new_noise_std)
                   ut.log_write(s, params)
               if new_noise_std>0.0:
                   noise = np.random.normal(0.0, new_noise_std, x.shape)
                   x = noise + x

            # 模型前向传播 (混合精度)
            with autocast('cuda'):
                loss, new_states_t = model(
                    _z=x,
                    target_data=y,
                    repeat_data=r_batch,
                    _x_inp=dic_state["_x_pre"],
                    _P_inp=dic_state["PCov_pre"],
                    _I=I_np,
                    state_dict=dic_state, 
                    is_training=True
                )

            # 3. 更新梯度 (混合精度)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # 4. 关键：将模型输出的 _t 状态写回 dic_state 的 _t 槽位
            for k, v in new_states_t.items():
                dic_state[k] = th.detach_state(v)

            total_train_loss += loss.item()
            batch_pbar.set_postfix(loss=f"{loss.item():.4f}")

        total_train_loss = total_train_loss / n_train_batches
        epoch_pbar.set_postfix(train_loss=f"{total_train_loss:.4f}")
        s = 'TRAIN --> epoch %i | error %f'%(e, total_train_loss)
        ut.log_write(s, params)

        pre_test = "TRAINING_Data"
        train_eval_loss = test_data(model, params, X_train, Y_train, index_train_list, S_Train_list, R_L_Train_list,
                                   F_list_train, batch_size, device)

        pre_test = "TEST_Data"
        test_eval_loss = test_data(model, params, X_test, Y_test, index_test_list, S_Test_list, R_L_Test_list, F_list_test, batch_size, device)
        
        epoch_pbar.set_postfix(train_loss=f"{total_train_loss:.4f}", test_loss=f"{test_eval_loss:.4f}")
        
        metrics = {
            'train_loss': total_train_loss,
            'train_eval_loss': train_eval_loss,
            'test_eval_loss': test_eval_loss,
            'learning_rate': new_lr
        }
        
        checkpoint_manager.save(
            model=model,
            optimizer=optimizer,
            epoch=e,
            loss=test_eval_loss,
            params=params,
            metrics=metrics
        )
        
        resume_ckpt_path = os.path.join(checkpoint_dir, "resume_checkpoint.ckpt")
        torch.save({
            'epoch': e,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': test_eval_loss
        }, resume_ckpt_path)

if __name__ == "__main__":
    # GPU Configuration
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    torch.autograd.set_detect_anomaly(True)
    print(f"Using device: {device}")
    # Main Execution Block
    rnn_keep_prob_lst = [0.8]
    rnn_input_prob_lst = [1.0]
    seq_lst = [50]
    reset_state = [5, 100, 20]
    normalise_data_lst = [3]
    params = config.get_params()
    # params["mfile"] = '/mnt/Data1/hc/tt/cp/lstm_nostate1/cp/' # 保持原样或注释掉

    rnn_keep_prob = 0.8
    input_keep_prob = 1.0
    params['rnn_keep_prob'] = rnn_keep_prob
    params['input_keep_prob'] = input_keep_prob
    seq = 50
    res = 5

    # PyTorch 不需要 tf.Graph().as_default()
    print("seq: ============== %s  ============" % seq)
    print("reset_state: ============== %s  ============" % res)
    print("rnn_keep_prob: ============== %s  ============" % rnn_keep_prob)

    params['normalise_data'] = 4
    params['reset_state'] = res
    params['seq_length'] = seq
    params["reload_data"] = 0
    params['batch_size'] = 256
    params['lr']=0.0001
    params['predict_next_frame'] = True
    params = config.update_params(params)
    params["model"] = "kfl_QRFf"
    params["device"] = device

    # 初始化 PyTorch 模型
    if params["model"] == "lstm":
        tracker = lstm(params=params)
    elif params["model"] == "kfl_QRf":
        tracker = kfl_QRf(params=params)
    # elif params["model"] == "kfl_Rf":
    #     tracker = kfl_Rf(params=params)
    elif params["model"] == "kfl_QRFf":
        tracker = kfl_QRFf(params=params)
    elif params["model"] == "kfl_QRFf_transformer":
        tracker = kfl_QRFf_transformer(params=params)
    elif params["model"] == "kfl_K":
        tracker = kfl_K(params=params)

    # 将模型移动到 GPU
    tracker.to(device)

    params["rn_id"] = "dobuleloss081500_nrm4_seq%i_res%i_keep%f_lr%f"%(seq, res, rnn_keep_prob, params["lr"])
    params = config.update_params(params)

    (params, X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list,
            X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list) = \
            dut.prepare_training_set(params)

    # show_every = 1
    (index_train_list, S_Train_list) = dut.get_seq_indexes(params, S_Train_list)
    (index_test_list, S_Test_list) = dut.get_seq_indexes(params, S_Test_list)

    batch_size = params['batch_size']
    n_train_batches = len(index_train_list)
    n_train_batches //= batch_size

    n_test_batches = len(index_test_list)
    n_test_batches //= batch_size
    params['training_size'] = len(X_train) * params['seq_length']
    params['test_size'] = len(X_test) * params['seq_length']

    # ut.start_log(params)
    # ut.log_write("Model training started", params)

    train(tracker, params, device, X_train, Y_train, X_test, Y_test, index_train_list, index_test_list, S_Train_list, S_Test_list, R_L_Train_list, R_L_Test_list, F_list_train, F_list_test)