import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.num_layers = params['nlayer']
        self.rnn_size = params['n_hidden']
        self.NOUT = params['n_output']
        
        # Dropout probabilities
        self.output_keep_prob = params.get('rnn_keep_prob', 1.0)
        self.input_keep_prob = params.get('input_keep_prob', 1.0)

        # --- Transition LSTM (F) ---
        self.lstms_F = nn.ModuleList()
        for i in range(self.num_layers):
            in_dim = self.NOUT if i == 0 else self.rnn_size
            self.lstms_F.append(nn.LSTMCell(in_dim, self.rnn_size))

        # --- Q Noise LSTM ---
        self.Qn_hidden = params['Qn_hidden']
        self.Qnlayer = params['Qnlayer']
        self.lstms_Q = nn.ModuleList()
        for i in range(self.Qnlayer):
            in_dim = self.NOUT if i == 0 else self.rnn_size
            self.lstms_Q.append(nn.LSTMCell(in_dim, self.Qn_hidden))

        # --- R Noise LSTM ---
        self.Rn_hidden = params['Rn_hidden']
        self.Rnlayer = params['Rnlayer']
        self.lstms_R = nn.ModuleList()
        for i in range(self.Rnlayer):
            in_dim = self.NOUT if i == 0 else self.rnn_size
            self.lstms_R.append(nn.LSTMCell(in_dim, self.Rn_hidden))

        # --- Transition Output Layers (MLP) ---
        self.mlp_F = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.NOUT)
        )

        # --- Noise Output Layers ---
        self.fc_Q = nn.Linear(self.Qn_hidden, self.NOUT)
        self.fc_R = nn.Linear(self.Rn_hidden, self.NOUT)

        # Weights initialization (Xavier)
        self._init_weights(self.mlp_F)
        self._init_weights(self.fc_Q)
        self._init_weights(self.fc_R)
        for lstm in self.lstms_F:
            self._init_weights(lstm)
        for lstm in self.lstms_Q:
            self._init_weights(lstm)
        for lstm in self.lstms_R:
            self._init_weights(lstm)

    def _init_weights(self, module):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, _z, target_data, repeat_data, _x_inp, _P_inp, state_dict):
        """
        Args:
            _z: Measurement input (Batch, Seq, Features)
            target_data: Ground truth (Batch, Seq, Features)
            repeat_data: Mask for valid timesteps (Batch, Seq)
            _x_inp: Initial state estimate (Batch, Features)
            _P_inp: Initial covariance diagonal (Batch, Features) - Changed to vector for efficiency
            state_dict: Dictionary containing 'F_pre', 'Q_pre', 'R_pre' (Lists of (h, c) tuples)
        """
        
        batch_size = _z.shape[0]
        seq_length = _z.shape[1]

        # Unpack states
        state_F = state_dict.get("F_pre") 
        state_Q = state_dict.get("Q_pre")
        state_R = state_dict.get("R_pre")

        # Initialize containers
        x = _x_inp 
        P = _P_inp # Shape (Batch, NOUT) - Diagonal elements only

        xres_lst = []
        xpred_lst = []
        qres_lst = []
        rres_lst = []
        kres_lst = []
        tres_lst = []

        # Loop over time steps
        for time_step in range(seq_length):
            z_t = _z[:, time_step, :]
            
            # --- 1. Transition Model (F) ---
            inp_F = x
            
            if self.input_keep_prob < 1.0 and self.training:
                inp_F = F.dropout(inp_F, p=(1.0-self.input_keep_prob), training=self.training)

            new_state_F = []
            out = inp_F
            for i, lstm_cell in enumerate(self.lstms_F):
                h, c = state_F[i]
                h, c = lstm_cell(out, (h, c))
                out = h
                new_state_F.append((h, c))
                
                if self.training and self.output_keep_prob < 1.0:
                    out = F.dropout(out, p=(1.0-self.output_keep_prob), training=self.training)
            
            state_F = new_state_F
            
            # MLP Projection
            pred = self.mlp_F(out) # Predicted next state

            # --- 2. Process Noise Model (Q) ---
            new_state_Q = []
            out_Q = x
            for i, lstm_cell in enumerate(self.lstms_Q):
                h, c = state_Q[i]
                h, c = lstm_cell(out_Q, (h, c))
                out_Q = h
                new_state_Q.append((h, c))
                if self.training and self.output_keep_prob < 1.0:
                    out_Q = F.dropout(out_Q, p=(1.0-self.output_keep_prob), training=self.training)
            state_Q = new_state_Q
            pred_Q = self.fc_Q(out_Q)

            # --- 3. Measurement ---
            meas_z = z_t
            tres_lst.append(meas_z)

            # --- 4. Measurement Noise Model (R) ---
            new_state_R = []
            out_R = meas_z
            for i, lstm_cell in enumerate(self.lstms_R):
                h, c = state_R[i]
                h, c = lstm_cell(out_R, (h, c))
                out_R = h
                new_state_R.append((h, c))
                if self.training and self.output_keep_prob < 1.0:
                    out_R = F.dropout(out_R, p=(1.0-self.output_keep_prob), training=self.training)
            state_R = new_state_R
            pred_R = self.fc_R(out_R)

            # --- 5. Kalman Filter Update (Optimized for Diagonal Covariance) ---
            
            # Construct Q and R diagonals (ensure positivity)
            # Q: (B, NOUT), R: (B, NOUT)
            Q_diag = torch.exp(pred_Q) 
            R_diag = torch.exp(pred_R) 

            # Predict
            # x = pred
            x = pred
            
            # P = P + Q
            P = P + Q_diag

            # Update
            # y = meas_z - x
            y = meas_z - x
            # S = P + R (Scalar diagonal)
            S_diag = P + R_diag
            # K = P / S (Element-wise division for diagonal matrices)
            K_diag = P / S_diag
            
            # x = x + K * y (Element-wise multiplication since K is diagonal vector)
            x = x + K_diag * y

            xpred_lst.append(pred) 
            xres_lst.append(x)
            
            # Store diagonals for visualization/loss if needed
            kres_lst.append(K_diag)
            qres_lst.append(Q_diag)
            rres_lst.append(R_diag)
            
            # Update P (Joseph form for stability, simplified for diagonal)
            # P = (I - K) * P + K * R * K  -> Element wise for diagonal
            # P = (1 - K) * P + K * R * K
            # This is the simplified Joseph form equivalent for diagonal matrices
            P = (1.0 - K_diag) * P + K_diag * R_diag * K_diag

        # --- Post Processing ---
        xres = torch.stack(xres_lst) # (Seq, Batch, NOUT)
        xpred = torch.stack(xpred_lst)
        qres = torch.stack(qres_lst)
        rres = torch.stack(rres_lst)
        kres = torch.stack(kres_lst)
        tres = torch.stack(tres_lst)

        # Transpose to (Batch, Seq, NOUT)
        xres = torch.transpose(xres, 0, 1)
        xpred = torch.transpose(xpred, 0, 1)
        qres = torch.transpose(qres, 0, 1)
        rres = torch.transpose(rres, 0, 1)
        kres = torch.transpose(kres, 0, 1)
        tres = torch.transpose(tres, 0, 1)

        # Masking
        # repeat_data shape: (Batch, Seq)
        mask = repeat_data > 0
        # Expand mask to features: (Batch, Seq, 1)
        mask_expanded = mask.unsqueeze(-1)
        
        # Apply mask manually (or use index_select like original)
        # Using masked_select for efficiency
        self.final_output = torch.masked_select(xres, mask_expanded)
        self.final_pred_output = torch.masked_select(xpred, mask_expanded)
        self.y = torch.masked_select(target_data, mask_expanded)

        # --- Loss Calculation ---
        # Note: masked_select returns 1D tensor, so reshape might be needed if we want mean per batch
        # but standard MSE over all valid steps is fine.
        
        loss = torch.mean((self.final_output - self.y) ** 2)
        loss_pred = torch.mean((self.final_pred_output - self.y) ** 2)

        # L2 Regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        l2_reg = l2_reg * 1e-4

        self.cost = loss + l2_reg + 0.8 * loss_pred

        # Pack states
        new_states = {
            "F_t": state_F,
            "Q_t": state_Q,
            "R_t": state_R,
            "_x_t": x, 
            "PCov_t": P
        }

        return self.cost, new_states
    

if __name__ == '__main__':
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = {
        "batch_size": 16,
        "nlayer": 2,
        "n_hidden": 32,
        "n_output": 2,

        "Qn_hidden": 16,
        "Qnlayer": 1,
        "Rn_hidden": 16,
        "Rnlayer": 1,

        "rnn_keep_prob": 0.9,
        "input_keep_prob": 0.9
    }

    SEQ_LEN = 20
    EPOCHS = 200
    LR = 1e-5

    def generate_fake_data(batch, seq_len, state_dim):
        # 初始真实状态
        x = torch.randn(batch, state_dim)

        velocity = torch.tensor([0.3, -0.1]).to(x)

        target = []
        for _ in range(seq_len):
            process_noise = 0.05 * torch.randn(batch, state_dim)
            x = x + velocity + process_noise
            target.append(x)

        target = torch.stack(target, dim=1)  # (B,T,N)

        measurement_noise = 0.2 * torch.randn(batch, seq_len, state_dim)
        z = target + measurement_noise

        repeat = torch.ones(batch, seq_len)

        return z, target, repeat

    def init_lstm_state(num_layers, batch, hidden, device):
        return [
            (torch.zeros(batch, hidden, device=device),
             torch.zeros(batch, hidden, device=device))
            for _ in range(num_layers)
        ]

    model = Model(params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3) # Slightly higher LR usually fine for Adam

    loss_history = []

    for epoch in range(EPOCHS):
        model.train() # Important: Sets self.training = True
        optimizer.zero_grad()

        z, target, repeat = generate_fake_data(
            params["batch_size"], SEQ_LEN, params["n_output"]
        )
        z = z.to(device)
        target = target.to(device)
        repeat = repeat.to(device)

        # --- Adjusted Initialization ---
        x0 = torch.zeros(params["batch_size"], params["n_output"], device=device)
        
        # P0 is now a vector (diagonal elements)
        P0 = torch.ones(params["batch_size"], params["n_output"], device=device) 

        # _I is no longer needed in the optimized forward pass
        # I = torch.eye(...).to(device) 

        state_dict = {
            "F_pre": init_lstm_state(params["nlayer"], params["batch_size"], params["n_hidden"], device),
            "Q_pre": init_lstm_state(params["Qnlayer"], params["batch_size"], params["Qn_hidden"], device),
            "R_pre": init_lstm_state(params["Rnlayer"], params["batch_size"], params["Rn_hidden"], device)
        }

        # --- forward ---
        # Removed is_training argument, relying on model.train()
        # Removed _I argument
        loss, _ = model(
            _z=z,
            target_data=target,
            repeat_data=repeat,
            _x_inp=x0,
            _P_inp=P0,
            state_dict=state_dict
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        loss_history.append(loss.item())

        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss = {loss.item():.6f}")