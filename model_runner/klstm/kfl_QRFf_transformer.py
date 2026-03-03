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
        # x shape: (Batch, Seq_len, Dim)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
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
        # 保持输出维度与 d_model 一致
        self.output_projection = nn.Linear(d_model, d_model)
        
    def forward(self, x, mask=None):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x = self.output_projection(x)
        return x

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.NOUT = params['n_output']
        
        # --- Transformer Parameters ---
        self.d_model = params.get('d_model', 512)
        self.nhead = params.get('nhead', 8)
        self.num_layers = params.get('num_layers', 3)
        self.dim_feedforward = params.get('dim_feedforward', 2048)
        self.dropout = params.get('dropout', 0.1)
        
        # Q Transformer Parameters
        self.Q_d_model = params.get('Q_d_model', 256)
        self.Q_nhead = params.get('Q_nhead', 4)
        self.Q_num_layers = params.get('Q_num_layers', 1)
        
        # R Transformer Parameters
        self.R_d_model = params.get('R_d_model', 256)
        self.R_nhead = params.get('R_nhead', 4)
        self.R_num_layers = params.get('R_num_layers', 1)
        
        # --- Modules ---
        # 1. State Transition Transformer (F): Predicts next state x_pred
        self.transformer_F = TransformerEncoder(
            input_size=self.NOUT,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.fc_x = nn.Linear(self.d_model, self.NOUT)
        
        # 2. Q Noise Transformer: Predicts Q (process noise) and F (transition matrix for covariance)
        self.transformer_Q = TransformerEncoder(
            input_size=self.NOUT,
            d_model=self.Q_d_model,
            nhead=self.Q_nhead,
            num_layers=self.Q_num_layers,
            dim_feedforward=self.Q_d_model * 4,
            dropout=self.dropout
        )
        # Output Q (diagonal elements for stability)
        self.fc_Q = nn.Linear(self.Q_d_model, self.NOUT)
        # Output F (Diagonal elements for stability)
        self.fc_F_mat = nn.Linear(self.Q_d_model, self.NOUT)
        
        # 3. R Noise Transformer: Predicts R (measurement noise)
        self.transformer_R = TransformerEncoder(
            input_size=self.NOUT,
            d_model=self.R_d_model,
            nhead=self.R_nhead,
            num_layers=self.R_num_layers,
            dim_feedforward=self.R_d_model * 4,
            dropout=self.dropout
        )
        self.fc_R = nn.Linear(self.R_d_model, self.NOUT)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, _z, target_data, repeat_data, _x_inp, _P_inp, _I, state_dict=None, is_training=False):
        """
        Args:
            _z: Observations (Batch, Seq_len, NOUT)
            target_data: Ground truth states (Batch, Seq_len, NOUT)
            repeat_data: Mask (Batch, Seq_len), 1 for valid observation, 0 for missing
            _x_inp: Initial state (Batch, NOUT)
            _P_inp: Initial covariance (Batch, NOUT, NOUT)
            _I: Identity matrix (NOUT, NOUT)
            state_dict: Dictionary of states (not used in transformer model)
            is_training: Boolean indicating if model is in training mode
        """
        device = _z.device
        batch_size = _z.shape[0]
        seq_length = _z.shape[1]
        
        # Initialize states
        x = _x_inp
        P = _P_inp
        
        # Ensure P is initialized as identity if it contains NaN
        if torch.isnan(P).any() or torch.isinf(P).any():
            P = torch.eye(self.NOUT, device=device, dtype=P.dtype).unsqueeze(0).expand(batch_size, -1, -1)
        
        xres_lst = []
        
        # --- Kalman Filtering Loop ---
        for time_step in range(seq_length):
            z_t = _z[:, time_step, :]
            
            # Check for NaN in input
            if torch.isnan(z_t).any():
                z_t = torch.zeros_like(z_t)
            
            # 1. Predict State (x_pred) using Previous State (x)
            # Input x is (Batch, NOUT) -> unsqueeze to (Batch, 1, NOUT)
            F_output = self.transformer_F(x.unsqueeze(1))
            pred_x = self.fc_x(F_output.squeeze(1))
            
            # 2. Predict Parameters (F_mat and Q) using Previous State (x)
            Q_output = self.transformer_Q(x.unsqueeze(1))
            Q_output_vec = Q_output.squeeze(1)
            
            # Process Noise Q (Diagonal elements only)
            pred_Q = self.fc_Q(Q_output_vec)
            q_diag = F.softplus(pred_Q) + 0.01
            
            # Transition Matrix F (Diagonal elements only)
            pred_F_diag = self.fc_F_mat(Q_output_vec)
            f_diag = F.softplus(pred_F_diag) + 0.01
            
            # 3. Predict Measurement Noise (R) using Observation (z_t)
            R_output = self.transformer_R(z_t.unsqueeze(1))
            pred_R = self.fc_R(R_output.squeeze(1))
            r_diag = F.softplus(pred_R) + 0.01
            
            # --- Covariance Prediction (Diagonal only) ---
            p_diag = torch.diagonal(P, dim1=1, dim2=2)
            p_diag = torch.clamp(p_diag, min=1e-6, max=1e6)
            p_pred_diag = f_diag * p_diag * f_diag + q_diag
            p_pred_diag = torch.clamp(p_pred_diag, min=1e-6, max=1e6)
            
            # --- Measurement Update ---
            mask_t = repeat_data[:, time_step]
            
            # Innovation y = z - Hx (H=I here)
            y = z_t - pred_x
            
            # Innovation Covariance S = P_pred + R (diagonal)
            s_diag = p_pred_diag + r_diag
            
            # Kalman Gain K = P_pred / S (element-wise for diagonal)
            k_diag = p_pred_diag / (s_diag + 1e-6)
            k_diag = torch.clamp(k_diag, min=0.0, max=1.0)
            
            # State Update (element-wise)
            x_k_update = pred_x + k_diag * y
            
            # Check for NaN in update
            if torch.isnan(x_k_update).any():
                x_k_update = pred_x
            
            # Covariance Update (diagonal): P_new = (1 - K) * P_pred
            p_k_update_diag = (1.0 - k_diag) * p_pred_diag
            p_k_update_diag = torch.clamp(p_k_update_diag, min=1e-6, max=1e6)
            
            # Reconstruct full matrices for output
            K = torch.diag_embed(k_diag)
            P_pred = torch.diag_embed(p_pred_diag)
            P_k_update = torch.diag_embed(p_k_update_diag)
            
            # Apply Mask
            mask_b1 = mask_t.unsqueeze(1).to(x_k_update.dtype)
            mask_b11 = mask_t.unsqueeze(1).unsqueeze(1).to(P_k_update.dtype)
            
            x = x_k_update * mask_b1 + pred_x * (1.0 - mask_b1)
            P = P_k_update * mask_b11 + P_pred * (1.0 - mask_b11)
            
            # Store results
            xres_lst.append(x)
        
        # Stack results
        xres = torch.stack(xres_lst, dim=1) # (Batch, Seq, NOUT)
        
        # --- Loss Calculation ---
        # Flatten
        xres_flat = torch.reshape(xres, [-1, self.NOUT])
        target_flat = torch.reshape(target_data, [-1, self.NOUT])
        mask_flat = torch.reshape(repeat_data, [-1])
        
        # Filter valid observations
        valid_indices = torch.where(mask_flat != 0)[0]
        
        if valid_indices.shape[0] > 0:
            final_output = torch.index_select(xres_flat, 0, valid_indices)
            y = torch.index_select(target_flat, 0, valid_indices)
            
            diff = final_output - y
            loss = 0.5 * torch.mean(diff ** 2)
        else:
            final_output = torch.empty(0, self.NOUT, device=device)
            y = torch.empty(0, self.NOUT, device=device)
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Store outputs for easy access
        self.final_output = final_output
        self.y = y
        
        # L2 Regularization
        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters()) * 1e-4
        loss = loss + l2_reg
        
        # Package states
        new_states = {
            "PCov_t": P,
            "_x_t": x
        }
        
        return loss, new_states