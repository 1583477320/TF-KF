import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, params, is_training=True):
        super(Model, self).__init__()
        self.is_training = is_training
        # 注意：batch_size 实际上应该根据输入动态获取，params中的主要用于初始化参考
        self.NOUT = params['n_output']
        
        # Transition LSTM parameters
        self.num_layers = params['nlayer']
        self.rnn_size = params['n_hidden']
        
        # Q Noise LSTM parameters
        self.Qn_hidden = params['Qn_hidden']
        self.Qnlayer = params['Qnlayer']
        
        # R Noise LSTM parameters
        self.Rn_hidden = params['Rn_hidden']
        self.Rnlayer = params['Rnlayer']

        # --- Layers ---
        self.lstms_F = nn.ModuleList()
        for i in range(self.num_layers):
            input_size = self.NOUT if i == 0 else self.rnn_size
            self.lstms_F.append(nn.LSTMCell(input_size, self.rnn_size))

        self.lstms_Q = nn.ModuleList()
        for i in range(self.Qnlayer):
            input_size_Q = self.NOUT if i == 0 else self.Qn_hidden
            self.lstms_Q.append(nn.LSTMCell(input_size_Q, self.Qn_hidden))

        self.lstms_R = nn.ModuleList()
        for i in range(self.Rnlayer):
            input_size_R = self.NOUT if i == 0 else self.Rn_hidden
            self.lstms_R.append(nn.LSTMCell(input_size_R, self.Rn_hidden))

        # Output Layers
        self.fc_x = nn.Linear(self.rnn_size, self.NOUT)
        self.fc_Q = nn.Linear(self.Qn_hidden, self.NOUT)
        self.fc_F = nn.Linear(self.Qn_hidden, self.NOUT)
        self.fc_R = nn.Linear(self.Rn_hidden, self.NOUT)

        self._init_weights()

    def _init_zero_state(self, batch_size, hidden_size, num_layers, device):
        return [
            (
                torch.zeros(batch_size, hidden_size, device=device),
                torch.zeros(batch_size, hidden_size, device=device)
            )
            for _ in range(num_layers)
        ]

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTMCell):
                nn.init.xavier_uniform_(module.weight_ih)
                nn.init.xavier_uniform_(module.weight_hh)
                if module.bias is not None:
                    nn.init.zeros_(module.bias_ih)
                    nn.init.zeros_(module.bias_hh)

    def forward(self, _z, target_data, repeat_data, _x_inp, _P_inp, _I, state_dict, is_training):
        device = _z.device
        
        # 修复1：提前获取 batch_size
        batch_size = _z.shape[0]
        seq_length = _z.shape[1]

        # 修复2：先从 state_dict 获取状态，再判断是否为 None
        state_F = state_dict.get("F_pre")
        state_Q = state_dict.get("Q_pre")
        state_R = state_dict.get("R_pre")

        # 如果状态为 None（第一次运行），则初始化
        if state_F is None:
            state_F = self._init_zero_state(batch_size, self.rnn_size, self.num_layers, device)
        if state_Q is None:
            state_Q = self._init_zero_state(batch_size, self.Qn_hidden, self.Qnlayer, device)
        if state_R is None:
            state_R = self._init_zero_state(batch_size, self.Rn_hidden, self.Rnlayer, device)

        x = _x_inp
        P = _P_inp

        xres_lst, pres_lst, tres_lst, kres_lst = [], [], [], []

        for time_step in range(seq_length):
            z_t = _z[:, time_step, :]
            
            # --- 1. 状态转移 Cell (预测 x) ---
            new_state_F = []
            out_F = x
            for i, lstm in enumerate(self.lstms_F):
                h, c = state_F[i]
                h, c = lstm(out_F, (h, c))
                out_F = h
                new_state_F.append((h, c))
            
            if is_training:
                out_F = F.dropout(out_F, p=0.2, training=is_training)
            
            state_F = new_state_F
            pred_x = self.fc_x(out_F) 
            
            # --- 2. Q 噪声 Cell (预测 Q 和 F 矩阵) ---
            new_state_Q = []
            out_Q = pred_x 
            for i, lstm in enumerate(self.lstms_Q):
                h, c = state_Q[i]
                h, c = lstm(out_Q, (h, c))
                out_Q = h
                new_state_Q.append((h, c))
            
            if is_training:
                out_Q = F.dropout(out_Q, p=0.2, training=is_training)
            
            state_Q = new_state_Q
            pred_Q = self.fc_Q(out_Q)
            pred_F_diag = self.fc_F(out_Q)

            # --- 4. 卡尔曼滤波数学计算 (预测步) ---
            # 注意：F_mat 用于协方差传递，这里假设为对角矩阵
            F_mat = torch.diag_embed(F.softplus(pred_F_diag) + 0.01)
            Q_mat = torch.diag_embed(F.softplus(pred_Q) + 0.01)

            # 确保 P 是正定的
            P = (P + torch.transpose(P, 1, 2)) / 2
            P_diag = torch.diagonal(P, dim1=1, dim2=2)
            P_diag = torch.clamp(P_diag, min=1e-4, max=100.0)
            P = torch.diag_embed(P_diag)

            # 预测协方差
            P_pred = torch.matmul(F_mat, torch.matmul(P, torch.transpose(F_mat, 1, 2))) + Q_mat
            
            # 确保 P_pred 是正定的
            P_pred_diag = torch.diagonal(P_pred, dim1=1, dim2=2)
            P_pred_diag = torch.clamp(P_pred_diag, min=1e-4, max=100.0)
            P_pred = torch.diag_embed(P_pred_diag)

            # 修复3：检查 repeat_data (Mask)
            # repeat_data[:, time_step] 形状
            # 如果 repeat_data 为 1，说明是有效数据，执行更新步；否则跳过（保持预测值）
            mask_t = repeat_data[:, time_step] # (Batch,)
            
            # --- 3. R Noise Cell (预测 R) ---
            # 只有在有效观测时，R-LSTM 才有物理意义，但在工程上为了 batch 维度一致，通常一起计算
            new_state_R = []
            out_R = z_t
            for i, lstm in enumerate(self.lstms_R):
                h, c = state_R[i]
                h, c = lstm(out_R, (h, c))
                out_R = h
                new_state_R.append((h, c))
            
            if is_training:
                out_R = F.dropout(out_R, p=0.2, training=is_training)
            
            state_R = new_state_R
            pred_R = self.fc_R(out_R)
            R_mat = torch.diag_embed(F.softplus(pred_R) + 0.01)

            # --- 更新步 ---
            y = z_t - pred_x
            S = P_pred + R_mat
            
            # 确保 S 是正定的
            S_diag = torch.diagonal(S, dim1=1, dim2=2)
            S_diag = torch.clamp(S_diag, min=1e-4, max=100.0)
            S_reg = torch.diag_embed(S_diag)
            
            if torch.isnan(S_reg).any() or torch.isinf(S_reg).any():
                print("警告: S_reg 发生数值爆炸，包含 NaN 或 Inf！")
                S_reg = torch.eye(S.shape[-1], device=device, dtype=S.dtype).unsqueeze(0).expand_as(S)
            
            try:
                K = torch.linalg.solve(S_reg, P_pred.transpose(-2, -1)).transpose(-2, -1)
            except RuntimeError:
                K = torch.matmul(P_pred, torch.linalg.inv(S_reg))
            
            x_k_update = pred_x + torch.squeeze(torch.matmul(K, y.unsqueeze(-1)), -1)
            
            # 检查 x_k_update 是否有 NaN
            if torch.isnan(x_k_update).any():
                x_k_update = pred_x
            
            I_K = _I.to(K.dtype) - K
            P_k_update = torch.matmul(I_K, torch.matmul(P_pred, torch.transpose(I_K, 1, 2))) + \
                        torch.matmul(K, torch.matmul(R_mat, torch.transpose(K, 1, 2)))
            
            # 确保 P_k_update 是正定的
            P_k_update_diag = torch.diagonal(P_k_update, dim1=1, dim2=2)
            P_k_update_diag = torch.clamp(P_k_update_diag, min=1e-4, max=100.0)
            P_k_update = torch.diag_embed(P_k_update_diag)

            mask_b1 = mask_t.unsqueeze(1).to(x_k_update.dtype)
            mask_b11 = mask_t.unsqueeze(1).unsqueeze(1).to(P_k_update.dtype)

            # 如果 mask_t == 1, 使用 x_k_update; 否则使用 pred_x
            x = x_k_update * mask_b1 + pred_x * (1.0 - mask_b1)
            P = P_k_update * mask_b11 + P_pred * (1.0 - mask_b11)
            
            # 注意：对于 LSTM State (h, c)，如果我们要极度严谨，也应该在 mask=0 时不更新 state_R。
            # 但为了保持 RNN 的连续性和简化代码，这里通常允许 RNN 继续跑，只要输出 x 和 P 是正确的即可。

            # 存储结果
            xres_lst.append(x)
            pres_lst.append(P)
            tres_lst.append(pred_x)
            kres_lst.append(K)

        # 结果堆叠与转置 (Batch, Seq, Dim)
        xres = torch.transpose(torch.stack(xres_lst), 0, 1)
        target_data_flat = torch.reshape(target_data, [-1, self.NOUT])
        
        # 掩码过滤 (用于 Loss 计算)
        flt = torch.squeeze(torch.reshape(repeat_data, [-1, 1]), 1)
        indices = torch.where(torch.not_equal(flt, 0))[0]

        xres_flat = torch.reshape(xres, [-1, self.NOUT])
        self.final_output = torch.index_select(xres_flat, 0, indices)
        self.y = torch.index_select(target_data_flat, 0, indices)

        # 损失计算
        diff = self.final_output - self.y
        loss = 0.5 * torch.sum(diff ** 2)
        
        # 防止除以0（例如所有数据都被mask了，虽然不太可能）
        if diff.shape[0] > 0:
            loss = loss / diff.shape[0]
        
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters()) * 1e-4
        self.cost = loss + l2_reg

        # 封装状态
        new_states = {
            "F_t": state_F,
            "Q_t": state_Q,
            "R_t": state_R,
            "PCov_t": P,
            "_x_t": x
        }

        return self.cost, new_states