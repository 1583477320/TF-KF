import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler

from helper import train_helper as th
from helper import config
from model_runner.klstm.kfl_QRFf_transformer import Model as kfl_QRFf_transformer
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

    dic_state = ut.get_state_list(params)

    def to_tensor(v):
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float().to(device)
        elif isinstance(v, list):
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

    for minibatch_index in range(n_batches):

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


def train(model, params, device):
    batch_size = params["batch_size"]
    num_epochs = 10  # Reduce for testing
    decay_rate = 0.9
    show_every = 100
    deca_start = 3
    pre_best_loss = 10000
    
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])
    
    I_np = torch.tensor([np.diag([1.0]*params['n_output']) for i in range(params["batch_size"])]).float().to(device)
    
    print('Training model:'+params["model"])
    noise_std = params['noise_std']
    new_noise_std = 0.0
    scaler = GradScaler()

    for e in range(num_epochs):
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

        model.train()
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
                    X=X_train,
                    R_L_list=R_L_Train_list,
                    F_list=None,
                    state_reset_counter_lst=state_reset_counter_lst,
                    device=device
                )

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

            # Disable mixed precision for transformer model
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

            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params.get("grad_clip", 1.0))
            
            optimizer.step()
            optimizer.zero_grad()

            for k, v in new_states_t.items():
                dic_state[k] = th.detach_state(v)

            total_train_loss += loss.item()
                
            if (minibatch_index % show_every == 0):
                print("Training batch loss: (%i / %i / %i)  %f"%(e, minibatch_index, n_train_batches, loss.item()))

        total_train_loss = total_train_loss / n_train_batches
        s = 'TRAIN --> epoch %i | error %f'%(e, total_train_loss)
        ut.log_write(s, params)

        pre_test = "TRAINING_Data"
        train_eval_loss = test_data(model, params, X_train, Y_train, index_train_list, S_Train_list, R_L_Train_list,
                                   F_list_train, batch_size, device)

        pre_test = "TEST_Data"
        test_eval_loss = test_data(model, params, X_test, Y_test, index_test_list, S_Test_list, R_L_Test_list, F_list_test, batch_size, device)
        
        base_cp_path = params["cp_file"] + "/"

        lss_str = '%.5f' % test_eval_loss
        model_name = lss_str + "_" + str(e) + "_" + str(params["rn_id"]) + params["model"] + "_model.ckpt"
        save_path = base_cp_path + model_name
        
        is_saved = False
        if pre_best_loss > test_eval_loss:
            pre_best_loss = test_eval_loss
            model_name = lss_str + "_" + str(e) + "_" + str(params["rn_id"]) + params["model"] + "_best_model.ckpt"
            save_path = base_cp_path + model_name
            torch.save(model.state_dict(), save_path)
            is_saved = True
        else:
            if e % 3.0 == 0:
                torch.save(model.state_dict(), save_path)
                is_saved = True
                
        if is_saved:
            s = 'MODEL_Saved --> epoch %i | error %f path %s' % (e, test_eval_loss, save_path)
            ut.log_write(s, params)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    rnn_keep_prob_lst = [0.8]
    rnn_input_prob_lst = [1.0]
    seq_lst = [50]
    reset_state = [5, 100, 20]
    normalise_data_lst = [3]
    params = config.get_params()

    rnn_keep_prob = 0.8
    input_keep_prob = 1.0
    params['rnn_keep_prob'] = rnn_keep_prob
    params['input_keep_prob'] = input_keep_prob
    seq = 50
    res = 5

    print("seq: ============== %s  ============" % seq)
    print("reset_state: ============== %s  ============" % res)
    print("rnn_keep_prob: ============== %s  ============" % rnn_keep_prob)

    params['normalise_data'] = 4
    params['reset_state'] = res
    params['seq_length'] = seq
    params["reload_data"] = 0
    params['batch_size'] = 5
    params = config.update_params(params)
    params["model"] = "kfl_QRFf_transformer"
    params["device"] = device

    tracker = kfl_QRFf_transformer(params=params)
    tracker.to(device)

    params["rn_id"] = "transformer_test_seq%i_res%i_keep%f_lr%f"%(seq, res, rnn_keep_prob, params["lr"])
    params = config.update_params(params)

    (params, X_train, Y_train, F_list_train, G_list_train, S_Train_list, R_L_Train_list,
            X_test, Y_test, F_list_test, G_list_test, S_Test_list, R_L_Test_list) = \
            dut.prepare_training_set(params)

    (index_train_list, S_Train_list) = dut.get_seq_indexes(params, S_Train_list)
    (index_test_list, S_Test_list) = dut.get_seq_indexes(params, S_Test_list)

    batch_size = params['batch_size']
    n_train_batches = len(index_train_list)
    n_train_batches //= batch_size

    n_test_batches = len(index_test_list)
    n_test_batches //= batch_size
    params['training_size'] = len(X_train) * params['seq_length']
    params['test_size'] = len(X_test) * params['seq_length']

    train(tracker, params, device)
