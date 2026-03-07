import torch
import torch.nn as nn
import math
import torch.nn.functional as F


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
    """
    共享的 Transformer Encoder
    
    优化：使用一个共享的 Transformer 处理输入，然后用不同的线性层输出不同特征
    """
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
    优化版本：共享 Transformer Encoder
    
    改进：
    1. 使用一个共享的 Transformer 处理输入
    2. 不同的线性层输出 F, Q, R 特征
    3. 计算量和显存占用减少 3 倍
    4. 缓存 causal mask 避免重复生成
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

        self.bptt_truncate = params.get('bptt_truncate', 10)
        
        self.max_seq_len = params.get('max_seq_len', 100)

        # Cache causal mask for max sequence length
        self.register_buffer('cached_causal_mask', generate_causal_mask(self.max_seq_len, 'cpu'))

        self.shared_transformer = SharedTransformerEncoder(
            input_size=self.NOUT,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )

        self.fc_x = nn.Linear(self.d_model, self.NOUT)
        self.fc_Q_diag = nn.Linear(self.d_model, self.n_joints * 6)
        self.fc_F_diag = nn.Linear(self.d_model, self.n_joints * 6)
        self.fc_R_diag = nn.Linear(self.d_model, self.n_joints * 6)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def build_block_diag_from_cholesky(self, L_params, batch_size):
        seq_len = L_params.shape[1]
        input_dtype = L_params.dtype
        device = L_params.device
        L_params = L_params.view(batch_size, seq_len, self.n_joints, 6)
        
        eps = torch.tensor(0.001, device=device, dtype=input_dtype)
        L_diag = F.softplus(L_params[..., :3]) + eps
        L_off = L_params[..., 3:6] * torch.tensor(0.1, device=device, dtype=input_dtype)
        
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

        x = _x_inp
        P = _P_inp

        if torch.isnan(P).any() or torch.isinf(P).any():
            P = torch.eye(self.NOUT, device=device, dtype=P.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        input_dtype = _z.dtype
        # Use cached causal mask, sliced to current sequence length
        causal_mask = self.cached_causal_mask[:seq_length, :seq_length].to(device)

        shared_features = self.shared_transformer(_z, causal_mask=causal_mask)
        pred_x_all = self.fc_x(shared_features)
        Q_L_params = self.fc_Q_diag(shared_features)
        F_L_params = self.fc_F_diag(shared_features)
        R_L_params = self.fc_R_diag(shared_features)
        
        input_dtype = _z.dtype
        Q_all = self.build_block_diag_from_cholesky(Q_L_params.to(input_dtype), batch_size)
        F_all = self.build_block_diag_from_cholesky(F_L_params.to(input_dtype), batch_size)
        R_all = self.build_block_diag_from_cholesky(R_L_params.to(input_dtype), batch_size)

        xres_lst = []

        eye_NOUT = torch.eye(self.NOUT, device=device, dtype=input_dtype)
        eye_batch = eye_NOUT.unsqueeze(0).expand(batch_size, -1, -1)
        eps = torch.tensor(1e-1, device=device, dtype=input_dtype)
        half = torch.tensor(0.5, device=device, dtype=input_dtype)
        clamp_min = torch.tensor(-10.0, device=device, dtype=input_dtype)
        clamp_max = torch.tensor(10.0, device=device, dtype=input_dtype)
        one = torch.tensor(1.0, device=device, dtype=input_dtype)
        
        P = P.to(input_dtype)
        
        xres_tensor = torch.zeros(batch_size, seq_length, self.NOUT, device=device, dtype=input_dtype)

        for time_step in range(seq_length):
            pred_x = pred_x_all[:, time_step, :]
            Q_t = Q_all[:, time_step]
            F_t = F_all[:, time_step]
            R_t = R_all[:, time_step]

            P_pred = torch.baddbmm(Q_t, F_t, torch.bmm(P, F_t.transpose(-1, -2)))
            P_pred = half * (P_pred + P_pred.transpose(-1, -2))
            P_pred = P_pred + eps * eye_batch

            mask_t = repeat_data[:, time_step]
            z_t = _z[:, time_step, :]
            innovation = z_t - pred_x
            
            S = P_pred + R_t
            S = half * (S + S.transpose(-1, -2))
            S = S + eps * eye_batch
            
            S_chol = torch.linalg.cholesky_ex(S, upper=False).L
            S_inv = torch.cholesky_inverse(S_chol)
            
            K = torch.bmm(P_pred, S_inv)
            K = torch.clamp(K, min=clamp_min.item(), max=clamp_max.item())

            x_k_update = pred_x + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(-1)
            
            x_k_update = torch.where(
                torch.isnan(x_k_update) | torch.isinf(x_k_update),
                pred_x,
                x_k_update
            )

            P_k_update = torch.bmm(eye_batch - K, P_pred)
            P_k_update = half * (P_k_update + P_k_update.transpose(-1, -2))
            P_k_update = P_k_update + eps * eye_batch

            mask_b1 = mask_t.unsqueeze(1).to(input_dtype)
            mask_b11 = mask_t.unsqueeze(1).unsqueeze(1).to(input_dtype)

            x = x_k_update * mask_b1 + pred_x * (one - mask_b1)
            P = (P_k_update * mask_b11 + P_pred * (one - mask_b11))

            xres_tensor[:, time_step, :] = x

        xres_flat = torch.reshape(xres_tensor, [-1, self.NOUT])
        target_flat = torch.reshape(target_data, [-1, self.NOUT])
        mask_flat = torch.reshape(repeat_data, [-1])

        valid_indices = torch.where(mask_flat != 0)[0]

        if valid_indices.shape[0] > 0:
            final_output = torch.index_select(xres_flat, 0, valid_indices)
            y = torch.index_select(target_flat, 0, valid_indices)
            
            diff = final_output - y
            diff_reshaped = diff.view(-1, self.n_joints, 3)
            
            eps = torch.tensor(1e-6, device=device, dtype=input_dtype)
            joint_errors = torch.sqrt(torch.sum(diff_reshaped ** 2, dim=-1) + eps)
            mpjpe = joint_errors.mean().to(input_dtype)
            
            smooth_loss = torch.mean(torch.sqrt(diff ** 2 + eps)).to(input_dtype)
            
            loss = mpjpe + torch.tensor(0.1, device=device, dtype=input_dtype) * smooth_loss
        else:
            final_output = torch.empty(0, self.NOUT, device=device, dtype=input_dtype)
            y = torch.empty(0, self.NOUT, device=device, dtype=input_dtype)
            loss = xres_tensor.sum() * torch.tensor(0.0, device=device, dtype=input_dtype)

        new_states = {
            "PCov_t": P,
            "_x_t": x
        }

        return loss, new_states, final_output, y
