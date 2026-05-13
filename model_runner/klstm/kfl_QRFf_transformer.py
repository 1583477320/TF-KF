import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].to(dtype=x.dtype)


def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask


class SharedTransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(SharedTransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, causal_mask=None, src_key_padding_mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=src_key_padding_mask)
        return x


class Model(nn.Module):
    """
    修正版模型：基于动力学的卡尔曼滤波
    
    改进点：
    1. 状态预测遵循 x_pred = F * x_prev + b，保证了状态与协方差预测的一致性。
    2. Transformer 用于预测动力学参数 (F, b, Q, R)，而不是直接预测状态。
    3. 形成了闭环：观测 -> Transformer -> 参数 -> KF预测 -> 更新 -> 下一时刻。
    """
    def __init__(self, params):
        super(Model, self).__init__()
        self.NOUT = params['n_output']
        self.n_joints = 17
        self.joint_dim = 3

        self.d_model = params.get('d_model', 512)
        self.nhead = params.get('nhead', 8)
        self.num_layers = params.get('num_layers', 3)
        self.dim_feedforward = params.get('dim_feedforward', 2048)
        self.dropout = params.get('dropout', 0.1)
        
        self.max_seq_len = params.get('max_seq_len', 100)
        self.register_buffer('cached_causal_mask', generate_causal_mask(self.max_seq_len, 'cpu'))

        self.shared_transformer = SharedTransformerEncoder(
            input_size=self.NOUT,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )

        # 输出头
        # 预测偏置/增量 b (NOUT)
        self.fc_b = nn.Linear(self.d_model, self.NOUT)
        # 预测 Q, F, R 的 Cholesky 参数 (n_joints * 6)
        self.fc_Q_diag = nn.Linear(self.d_model, self.n_joints * 6)
        self.fc_F_diag = nn.Linear(self.d_model, self.n_joints * 6)
        self.fc_R_diag = nn.Linear(self.d_model, self.n_joints * 6)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def build_block_diag_from_cholesky(self, L_params, batch_size,matrix_type='covariance'):
        """
        Args:
            matrix_type: 'covariance' 用于 Q, R (需正定);
                         'transition' 用于 F (需稳定，特征值接近1)
        """
        
        seq_len = L_params.shape[1]
        input_dtype = L_params.dtype
        device = L_params.device
        L_params = L_params.view(batch_size, seq_len, self.n_joints, 6)
        
        eps = torch.tensor(0.001, device=device, dtype=input_dtype)
        # 针对 F 矩阵的特殊处理：防止协方差爆炸
        if matrix_type == 'transition':
            # 核心修改：使用 Sigmoid 或 Tanh 将对角线限制在 1 附近
            # 方案 A (推荐): Sigmoid，范围 (0, 1)，代表有阻尼的平滑运动
            F_diag = torch.sigmoid(L_params[..., :3]) 
            
            # 方案 B (更强约束): 强制接近 1，允许微小震荡
            # F_diag = 1.0 + 0.1 * torch.tanh(L_params[..., :3]) # 范围 (0.9, 1.1)
            
            # 方案 C (兼容原代码逻辑但加 Cap): 使用 softplus 但截断上限
            # F_diag = F.softplus(L_params[..., :3]) + eps
            # F_diag = torch.clamp(F_diag, min=eps.item(), max=2.0) # 强制限制最大值为 2.0，防止爆炸
            
            L_off = L_params[..., 3:6] * torch.tensor(0.1, device=device, dtype=input_dtype)
            L_diag = F_diag
        else:
            # Q, R 矩阵保持原逻辑，必须正定
            L_diag = F.softplus(L_params[..., :3]) + eps
            L_off = L_params[..., 3:6] * torch.tensor(0.1, device=device, dtype=input_dtype)

        L_diag = torch.clamp(L_diag, min=eps.item(), max=100.0)
        L_off = torch.clamp(L_off, min=-10.0, max=10.0)
        
        L_blocks = torch.zeros(batch_size, seq_len, self.n_joints, 3, 3, device=device, dtype=input_dtype)
        L_blocks[..., 0, 0] = L_diag[..., 0]
        L_blocks[..., 1, 0] = L_off[..., 0]
        L_blocks[..., 1, 1] = L_diag[..., 1]
        L_blocks[..., 2, 0] = L_off[..., 1]
        L_blocks[..., 2, 1] = L_off[..., 2]
        L_blocks[..., 2, 2] = L_diag[..., 2]
        
        block_diag = torch.zeros(batch_size, seq_len, self.NOUT, self.NOUT, device=device, dtype=input_dtype)
        for j in range(self.n_joints):
            start_idx = j * 3
            end_idx = start_idx + 3
            block_diag[:, :, start_idx:end_idx, start_idx:end_idx] = L_blocks[:, :, j]
        
        return block_diag

    def forward(self, _z, target_data, repeat_data, _x_inp, _P_inp, _I, state_dict=None, is_training=False):
        device = _z.device
        batch_size = _z.shape[0]
        seq_length = _z.shape[1]

        # 初始状态
        x = _x_inp
        P = _P_inp
        
        input_dtype = _z.dtype

        # 处理 NaN (保持原逻辑的安全检查)
        if torch.isnan(P).any() or torch.isinf(P).any():
            P = torch.eye(self.NOUT, device=device, dtype=P.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        # 1. Transformer 编码观测历史，预测所有时间步的动力学参数
        causal_mask = self.cached_causal_mask[:seq_length, :seq_length].to(device)
        shared_features = self.shared_transformer(_z, causal_mask=causal_mask)
        
        # 预测参数
        b_all = self.fc_b(shared_features)                 # 偏置/运动增量
        Q_L_params = self.fc_Q_diag(shared_features)
        F_L_params = self.fc_F_diag(shared_features)
        R_L_params = self.fc_R_diag(shared_features)
        
        # 构建块对角矩阵
        Q_all = self.build_block_diag_from_cholesky(Q_L_params.to(input_dtype), batch_size,matrix_type='covariance')
        F_all = self.build_block_diag_from_cholesky(F_L_params.to(input_dtype), batch_size,matrix_type='transition')
        R_all = self.build_block_diag_from_cholesky(R_L_params.to(input_dtype), batch_size,matrix_type='covariance')

        xres_tensor = torch.zeros(batch_size, seq_length, self.NOUT, device=device, dtype=input_dtype)

        eye_NOUT = torch.eye(self.NOUT, device=device, dtype=input_dtype)
        eye_batch = eye_NOUT.unsqueeze(0).expand(batch_size, -1, -1)
        eps = torch.tensor(1e-2, device=device, dtype=input_dtype)
        half = torch.tensor(0.5, device=device, dtype=input_dtype)
        
        # 用于存储监控数据的列表
        p_norms = []
        # 2. 串行 KF 循环 (基于动力学模型的递归更新)
        for time_step in range(seq_length):
            # 获取当前时刻的参数
            b_t = b_all[:, time_step]
            Q_t = Q_all[:, time_step]
            F_t = F_all[:, time_step]
            R_t = R_all[:, time_step]
            
            # 当前观测
            z_t = _z[:, time_step, :]
            mask_t = repeat_data[:, time_step]

            # --- 预测步 ---
            # 核心修正：使用 F_t 和 b_t 计算状态预测
            # x_pred = F_t * x + b_t
            x_pred = torch.baddbmm(b_t.unsqueeze(-1), F_t, x.unsqueeze(-1)).squeeze(-1)
            
            # 协方差预测: P_pred = F_t * P * F_t^T + Q_t
            P_pred = torch.baddbmm(Q_t, F_t, torch.bmm(P, F_t.transpose(-1, -2)))
            P_pred = half * (P_pred + P_pred.transpose(-1, -2))
            P_pred = P_pred + eps * eye_batch
            
            # 计算当前批次 P_pred 的 Frobenius 范数均值
            current_p_norm = torch.norm(P_pred, p='fro', dim=(-2, -1)).mean().item()
            p_norms.append(current_p_norm)
            
            # 如果发现异常，立即打印警告（调试阶段用）
            if current_p_norm > 1e6 or np.isnan(current_p_norm):
                print(f"Warning: P explosion/NaN at step {time_step}. Norm: {current_p_norm:.2e}")
            # --- 更新步 ---
            innovation = z_t - x_pred
            
            # S = P_pred + R_t
            S = P_pred + R_t
            S = half * (S + S.transpose(-1, -2))
            S = S + eps * eye_batch
            
            # 计算 Kalman Gain
            S_chol = torch.linalg.cholesky_ex(S, upper=False).L
            S_inv = torch.cholesky_inverse(S_chol)
            K = torch.bmm(P_pred, S_inv)
            
            # 更新状态
            x_k_update = x_pred + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
            
            # 更新协方差
            P_k_update = torch.bmm(eye_batch - K, P_pred)
            P_k_update = half * (P_k_update + P_k_update.transpose(-1, -2))
            P_k_update = P_k_update + eps * eye_batch

            # --- 掩码处理 ---
            # 如果是无效帧，保持预测值；如果是有效帧，使用更新值
            mask_b1 = mask_t.unsqueeze(1).to(input_dtype)
            mask_b11 = mask_t.unsqueeze(1).unsqueeze(1).to(input_dtype)

            x = x_k_update * mask_b1 + x_pred * (1 - mask_b1)
            P = (P_k_update * mask_b11 + P_pred * (1 - mask_b11))
            
            xres_tensor[:, time_step, :] = x
        
        # 循环结束后，计算平均范数用于日志记录
        avg_p_norm = np.mean(p_norms) if p_norms else 0.
        # 3. 计算 Loss
        xres_flat = torch.reshape(xres_tensor, [-1, self.NOUT])
        target_flat = torch.reshape(target_data, [-1, self.NOUT])
        mask_flat = torch.reshape(repeat_data, [-1])
        
        valid_indices = torch.where(mask_flat != 0)[0]
        
        if valid_indices.shape[0] > 0:
            final_output = torch.index_select(xres_flat, 0, valid_indices)
            y = torch.index_select(target_flat, 0, valid_indices)
            
            diff = final_output - y
            diff_reshaped = diff.view(-1, self.n_joints, 3)
            
            mpjpe = torch.mean(torch.sqrt(torch.sum(diff_reshaped ** 2, dim=-1) + 1e-6))            
            loss = mpjpe 
        else:
            loss = xres_tensor.sum() * 0.0
            final_output = xres_tensor
            y = target_data

        new_states = {
            "PCov_t": P,
            "_x_t": x,
            "P_pred_norm": avg_p_norm
        }
        
        return loss, new_states, final_output, y