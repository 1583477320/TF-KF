import torch
import numpy as np
from helper import config
from model_runner.klstm.kfl_QRFf_transformer import Model as kfl_QRFf_transformer
from helper import utils as ut

def test_transformer_model():
    print("Testing kfl_QRFf_transformer model...")
    
    # Get parameters
    params = config.get_params()
    params["model"] = "kfl_QRFf_transformer"
    params["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {params['device']}")
    
    # Initialize model
    model = kfl_QRFf_transformer(params=params)
    model.to(params["device"])
    
    # Create dummy data
    batch_size = 2
    seq_length = 10
    n_output = params['n_output']
    
    _z = torch.randn(batch_size, seq_length, n_output).to(params["device"])
    target_data = torch.randn(batch_size, seq_length, n_output).to(params["device"])
    repeat_data = torch.ones(batch_size, seq_length).to(params["device"])
    
    # Initialize states
    _x_inp = torch.zeros(batch_size, n_output).to(params["device"])
    _P_inp = torch.stack([torch.eye(n_output) for _ in range(batch_size)]).to(params["device"])
    _I = torch.stack([torch.eye(n_output) for _ in range(batch_size)]).to(params["device"])
    
    # Initialize state dict
    dic_state = ut.get_state_list(params)
    
    # Convert to tensors
    def to_tensor(v):
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).float().to(params["device"])
        elif isinstance(v, list):
            return [(to_tensor(t[0]), to_tensor(t[1])) for t in v]
        return v
    
    for k, v in dic_state.items():
        dic_state[k] = to_tensor(v)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        loss, new_states = model(
            _z=_z,
            target_data=target_data,
            repeat_data=repeat_data,
            _x_inp=_x_inp,
            _P_inp=_P_inp,
            _I=_I,
            state_dict=dic_state,
            is_training=False
        )
    
    print(f"Forward pass successful! Loss: {loss.item():.4f}")
    print(f"Output shape: {model.final_output.shape}")
    print(f"Target shape: {model.y.shape}")
    
    # Test training mode
    model.train()
    loss_train, new_states_train = model(
        _z=_z,
        target_data=target_data,
        repeat_data=repeat_data,
        _x_inp=_x_inp,
        _P_inp=_P_inp,
        _I=_I,
        state_dict=dic_state,
        is_training=True
    )
    
    print(f"Training mode forward pass successful! Loss: {loss_train.item():.4f}")
    
    # Test backward pass
    loss_train.backward()
    print("Backward pass successful!")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nAll tests passed! The kfl_QRFf_transformer model is working correctly.")

if __name__ == "__main__":
    test_transformer_model()