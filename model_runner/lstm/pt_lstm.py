import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, params, is_training=True):
        super(Model, self).__init__()
        self.is_training = is_training
        self.batch_size = params["batch_size"]
        self.num_layers = params['nlayer']
        self.rnn_size = params['n_hidden']
        self.n_input = params['n_input']
        self.NOUT = params['n_output']
        
        self.output_keep_prob = params.get('rnn_keep_prob', 1.0)
        self.input_keep_prob = params.get('input_keep_prob', 1.0)

        # --- LSTM Cells ---
        self.lstms = nn.ModuleList()
        for i in range(self.num_layers):
            # First layer takes n_input, subsequent layers take rnn_size
            input_dim = self.n_input if i == 0 else self.rnn_size
            self.lstms.append(nn.LSTMCell(input_dim, self.rnn_size))

        # --- Output Layers (MLP) ---
        # Corresponds to output_w1/b1, output_w2/b2, output_w3/b3
        self.mlp = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(),
            nn.Linear(self.rnn_size, self.NOUT)
        )

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

    def forward(self, input_data, target_data, repeat_data, state_dict, is_training):
        """
        Args:
            input_data: (Batch, Seq, N_input)
            target_data: (Batch, Seq, N_output)
            repeat_data: (Batch, Seq) mask
            state_dict: Dictionary containing 'lstm_pre' (List of (h, c) tuples)
            is_training: Boolean
        """
        
        batch_size = input_data.shape[0]
        seq_length = input_data.shape[1]

        # Unpack states
        # state_dict['lstm_pre'] is a list of (h, c) tuples
        state = state_dict.get("lstm_pre")
        
        outputs = []

        # Loop over time steps
        for t in range(seq_length):
            x_t = input_data[:, t, :]
            
            # Input Dropout (if specified)
            if self.is_training and self.input_keep_prob < 1.0:
                x_t = F.dropout(x_t, p=(1.0 - self.input_keep_prob), training=self.is_training)

            new_state = []
            out = x_t
            
            # Loop over layers
            for i, cell in enumerate(self.lstms):
                h, c = state[i]
                out, (h, c) = cell(out, (h, c))
                
                new_state.append((h, c))
                
                # Output Dropout
                # In TF code: if i > -1 and is_training ... apply dropout
                if self.is_training and self.output_keep_prob < 1.0:
                    out = F.dropout(out, p=(1.0 - self.output_keep_prob), training=self.is_training)
            
            state = new_state
            outputs.append(out)

        # Stack outputs: [Seq, Batch, Hidden]
        outputs = torch.stack(outputs)
        # Transpose to [Batch, Seq, Hidden]
        outputs = torch.transpose(outputs, 0, 1)
        # Reshape to [Batch * Seq, Hidden]
        outputs = torch.reshape(outputs, [-1, self.rnn_size])

        # Pass through MLP
        final_output = self.mlp(outputs)

        # --- Masking ---
        # repeat_data shape: (Batch, Seq)
        flt = torch.squeeze(torch.reshape(repeat_data, [-1, 1]), 1)
        where_flt = torch.not_equal(flt, 0)
        indices = torch.where(where_flt)[0]

        # Gather valid items
        # Flatten target to match
        y_flat = torch.reshape(target_data, [-1, self.NOUT])
        
        self.final_output = torch.index_select(final_output, 0, indices)
        self.y = torch.index_select(y_flat, 0, indices)

        # --- Loss Calculation ---
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
            "lstm_t": state
        }

        return self.cost, new_states