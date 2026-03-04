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

        # Transformer parameters
        self.d_model = params.get('d_model', 512)
        self.nhead = params.get('nhead', 8)
        self.num_layers = params.get('num_layers', 3)
        self.dim_feedforward = params.get('dim_feedforward', 2048)
        self.dropout = params.get('dropout', 0.1)

        # Q branch parameters
        self.Q_d_model = params.get('Q_d_model', 256)
        self.Q_nhead = params.get('Q_nhead', 4)
        self.Q_num_layers = params.get('Q_num_layers', 1)

        # R branch parameters
        self.R_d_model = params.get('R_d_model', 256)
        self.R_nhead = params.get('R_nhead', 4)
        self.R_num_layers = params.get('R_num_layers', 1)

        # Temporal context size for attention (key improvement)
        self.context_window = params.get('context_window', 16)

        # 1) Transition branch: x_{t-1..} -> pred_x_t
        self.transformer_F = TransformerEncoder(
            input_size=self.NOUT,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout
        )
        self.fc_x = nn.Linear(self.d_model, self.NOUT)

        # 2) Process branch: x_{t-1..} -> q_diag / f_diag
        self.transformer_Q = TransformerEncoder(
            input_size=self.NOUT,
            d_model=self.Q_d_model,
            nhead=self.Q_nhead,
            num_layers=self.Q_num_layers,
            dim_feedforward=self.Q_d_model * 4,
            dropout=self.dropout
        )
        self.fc_Q = nn.Linear(self.Q_d_model, self.NOUT)
        self.fc_F_mat = nn.Linear(self.Q_d_model, self.NOUT)

        # 3) Measurement branch: z_{..t} -> r_diag
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

    def _get_window(self, history_list, max_len):
        if len(history_list) <= max_len:
            return torch.stack(history_list, dim=1)
        return torch.stack(history_list[-max_len:], dim=1)

    def forward(self, _z, target_data, repeat_data, _x_inp, _P_inp, _I, state_dict=None, is_training=False):
        device = _z.device
        batch_size = _z.shape[0]
        seq_length = _z.shape[1]

        x = _x_inp
        P = _P_inp

        if torch.isnan(P).any() or torch.isinf(P).any():
            P = torch.eye(self.NOUT, device=device, dtype=P.dtype).unsqueeze(0).expand(batch_size, -1, -1)

        xres_lst = []
        state_history = [x]
        obs_history = []

        for time_step in range(seq_length):
            z_t = _z[:, time_step, :]
            if torch.isnan(z_t).any() or torch.isinf(z_t).any():
                z_t = torch.zeros_like(z_t)

            # Build temporal windows (real attention over recent context)
            x_win = self._get_window(state_history, self.context_window)
            obs_history.append(z_t)
            z_win = self._get_window(obs_history, self.context_window)

            # F branch
            F_output = self.transformer_F(x_win)
            pred_x = self.fc_x(F_output[:, -1, :])

            # Q/F covariance branch
            Q_output = self.transformer_Q(x_win)
            Q_output_vec = Q_output[:, -1, :]
            pred_Q = self.fc_Q(Q_output_vec)
            q_diag = F.softplus(pred_Q) + 0.01

            pred_F_diag = self.fc_F_mat(Q_output_vec)
            f_diag = F.softplus(pred_F_diag) + 0.01

            # R branch
            R_output = self.transformer_R(z_win)
            pred_R = self.fc_R(R_output[:, -1, :])
            r_diag = F.softplus(pred_R) + 0.01

            # Covariance predict (diagonal approximation)
            p_diag = torch.diagonal(P, dim1=1, dim2=2)
            p_diag = torch.clamp(p_diag, min=1e-6, max=1e6)
            p_pred_diag = f_diag * p_diag * f_diag + q_diag
            p_pred_diag = torch.clamp(p_pred_diag, min=1e-6, max=1e6)

            # Measurement update
            mask_t = repeat_data[:, time_step]
            innovation = z_t - pred_x
            s_diag = p_pred_diag + r_diag
            k_diag = p_pred_diag / (s_diag + 1e-6)
            k_diag = torch.clamp(k_diag, min=0.0, max=1.0)

            x_k_update = pred_x + k_diag * innovation
            if torch.isnan(x_k_update).any() or torch.isinf(x_k_update).any():
                x_k_update = pred_x

            p_k_update_diag = (1.0 - k_diag) * p_pred_diag
            p_k_update_diag = torch.clamp(p_k_update_diag, min=1e-6, max=1e6)

            P_pred = torch.diag_embed(p_pred_diag)
            P_k_update = torch.diag_embed(p_k_update_diag)

            mask_b1 = mask_t.unsqueeze(1).to(x_k_update.dtype)
            mask_b11 = mask_t.unsqueeze(1).unsqueeze(1).to(P_k_update.dtype)

            x = x_k_update * mask_b1 + pred_x * (1.0 - mask_b1)
            P = P_k_update * mask_b11 + P_pred * (1.0 - mask_b11)

            xres_lst.append(x)
            state_history.append(x)

        xres = torch.stack(xres_lst, dim=1)

        xres_flat = torch.reshape(xres, [-1, self.NOUT])
        target_flat = torch.reshape(target_data, [-1, self.NOUT])
        mask_flat = torch.reshape(repeat_data, [-1])

        valid_indices = torch.where(mask_flat != 0)[0]

        if valid_indices.shape[0] > 0:
            final_output = torch.index_select(xres_flat, 0, valid_indices)
            y = torch.index_select(target_flat, 0, valid_indices)
            diff = final_output - y
            loss = 0.5 * torch.mean(diff ** 2)
        else:
            final_output = torch.empty(0, self.NOUT, device=device)
            y = torch.empty(0, self.NOUT, device=device)
            # Keep differentiability while representing no supervised element
            loss = (xres.sum() * 0.0)

        self.final_output = final_output
        self.y = y

        l2_reg = sum(torch.sum(param ** 2) for param in self.parameters()) * 1e-4
        loss = loss + l2_reg

        new_states = {
            "PCov_t": P,
            "_x_t": x
        }

        return loss, new_states
