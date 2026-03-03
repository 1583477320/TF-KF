import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, params, is_training=True):
        super(Model, self).__init__()
        self.is_training = is_training
        self.batch_size = params["batch_size"]
        self.NOUT = params['n_output']
        
        # Transition LSTM parameters
        self.num_layers = params['nlayer']
        self.rnn_size = params['n_hidden']
        
        # K Gain LSTM parameters
        self.Kn_hidden = params['Kn_hidden']
        self.Knlayer = params['Knlayer']
        # Dimension for the embedding layer before K-LSTM
        self.K_inp = params.get('K_inp', self.rnn_size) 

        # --- Layers ---

        # 1. Transition LSTM (F_cell)
        # Predicts the state x
        self.lstms_F = nn.ModuleList()
        for i in range(self.num_layers):
            self.lstms_F.append(nn.LSTMCell(self.NOUT, self.rnn_size))

        # 2. K Gain LSTM (K_cell)
        # Predicts the Kalman Gain K
        self.lstms_K = nn.ModuleList()
        for i in range(self.Knlayer):
            self.lstms_K.append(nn.LSTMCell(self.K_inp, self.Kn_hidden))

        # Output Layers
        # F-cell output -> State prediction x
        self.fc_x = nn.Linear(self.rnn_size, self.NOUT)
        
        # Input Embedding for K LSTM
        # Input is concat(x, z) -> dim 2 * NOUT
        self.fc_K_emb = nn.Linear(2 * self.NOUT, self.K_inp)
        
        # K-cell output -> Gain K
        self.fc_K = nn.Linear(self.Kn_hidden, self.NOUT)

        self._init_weights()

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
                    nn.init.zeros_(module.bias)

    def forward(self, _z, target_data, repeat_data, _x_inp, _P_inp, _I, state_dict, is_training):
        """
        Args:
            _z: Measurement input (Batch, Seq, NOUT)
            target_data: Ground truth (Batch, Seq, NOUT)
            repeat_data: Mask (Batch, Seq)
            _x_inp: Initial state (Batch, NOUT)
            _P_inp: (Ignored in this model, kept for interface consistency)
            _I: (Ignored in this model, kept for interface consistency)
            state_dict: Dictionary with 'F_pre', 'K_pre'
            is_training: Boolean
        """
        
        batch_size = _z.shape[0]
        seq_length = _z.shape[1]

        # Unpack states
        state_F = state_dict.get("F_pre")
        state_K = state_dict.get("K_pre")

        xres_lst = []
        # pres_lst = [] # kfl_K does not track P
        # tres_lst = [] # kfl_K does not explicitly track measurements in output list other than input
        kres_lst = []

        # Initial x
        x = _x_inp

        for time_step in range(seq_length):
            z_t = _z[:, time_step, :]
            
            # --- 1. Transition Cell (Predict State x) ---
            # Input: x
            new_state_F = []
            out_F = x
            for i, lstm in enumerate(self.lstms_F):
                h, c = state_F[i]
                out_F, (h, c) = lstm(out_F, (h, c))
                new_state_F.append((h, c))
                # Dropout
                if self.is_training:
                    out_F = F.dropout(out_F, p=0.2, training=self.is_training)
            
            state_F = new_state_F
            # Predict State x
            pred_x = self.fc_x(out_F) 

            # --- 2. Gain Cell (Predict K) ---
            # Input: Concat(pred_x, z_t)
            inp_K = torch.cat([pred_x, z_t], dim=-1) # (Batch, 2 * NOUT)
            
            # Embedding
            emb = F.relu(self.fc_K_emb(inp_K))
            
            new_state_K = []
            out_K = emb
            for i, lstm in enumerate(self.lstms_K):
                h, c = state_K[i]
                out_K, (h, c) = lstm(out_K, (h, c))
                new_state_K.append((h, c))
                if self.is_training:
                    out_K = F.dropout(out_K, p=0.2, training=self.is_training)
            
            state_K = new_state_K
            
            # Predict K (Gain)
            # Use Tanh activation as in TF code
            K = torch.tanh(self.fc_K(out_K))

            # --- 3. Update Step (Simplified KF) ---
            # y = z - x
            y = z_t - pred_x
            
            # x = x + K * y
            x = pred_x + K * y

            # Store results
            xres_lst.append(x)
            kres_lst.append(K)

        # Stack and Transpose results
        # Stacking returns (Seq, Batch, Dim), transpose to (Batch, Seq, Dim)
        xres = torch.transpose(torch.stack(xres_lst), 0, 1)
        kres = torch.transpose(torch.stack(kres_lst), 0, 1)

        # Masking
        flt = torch.squeeze(torch.reshape(repeat_data, [-1, 1]), 1)
        where_flt = torch.not_equal(flt, 0)
        indices = torch.where(where_flt)[0]

        # Flatten
        xres_flat = torch.reshape(xres, [-1, self.NOUT])
        y_flat = torch.reshape(target_data, [-1, self.NOUT])
        kres_flat = torch.reshape(kres, [-1, self.NOUT])

        # Gather valid
        self.final_output = torch.index_select(xres_flat, 0, indices)
        self.y = torch.index_select(y_flat, 0, indices)
        # self.final_K = torch.index_select(kres_flat, 0, indices) # Optional

        # Loss Calculation
        # L2 Loss
        tmp = self.final_output - self.y
        loss = torch.mean(tmp ** 2)
        
        # L2 Regularization
        l2_reg = 0.0
        for param in self.parameters():
            l2_reg += torch.sum(param ** 2)
        l2_reg = l2_reg * 1e-4

        self.cost = loss + l2_reg

        # Pack states
        new_states = {
            "F_t": state_F,
            "K_t": state_K,
            "_x_t": x # Last state
        }

        return self.cost, new_states