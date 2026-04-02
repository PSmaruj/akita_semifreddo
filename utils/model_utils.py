"""
utils/model_utils.py

Akita model loading and inference utilities for the AkitaSF pipeline.

Functions
---------
load_model         : instantiate SeqNN and load weights from a state dict
run_model          : run a single forward pass and return output on CPU
store_tower_output : run the convolutional tower on a sequence and save activations
make_target        : generate an optimization target by overwriting feature positions
"""

import torch
import torch.nn as nn


DEFAULT_MODEL_SRC = "/home1/smaruj/akita_pytorch/"

def load_model(model_weights_path: str, device: torch.device,
               model_src: str = DEFAULT_MODEL_SRC) -> torch.nn.Module:
    """Instantiate SeqNN and load weights from a state dict.

    Parameters
    ----------
    model_weights_path : str
        Path to the saved state dict (.pt file).
    device : torch.device
        Torch device to load the model onto.

    Returns
    -------
    torch.nn.Module
        Model in eval mode with all parameters frozen.
    """
    import sys
    sys.path.insert(0, model_src)
    from akita.model import SeqNN

    model = SeqNN()
    model.load_state_dict(
        torch.load(model_weights_path, map_location=device, weights_only=True)
    )
    model.eval()
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    return model


@torch.no_grad()
def run_model(model: torch.nn.Module, X_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run a single forward pass and return the output on CPU.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen model in eval mode.
    X_tensor : torch.Tensor
        Input tensor of shape (1, 4, seq_len).
    device : torch.device
        Device the model lives on.

    Returns
    -------
    torch.Tensor
        Model output of shape (1, 1, num_contacts) on CPU.
    """
    y = model(X_tensor.to(device))
    return y.cpu()


def store_tower_output(X_tensor: torch.Tensor, model, path: str) -> None:
    """Run the convolutional tower on a sequence and save activations to disk.

    Passes X_tensor through conv_block_1 and conv_tower only (no head),
    saves the resulting activation tensor to path, and frees GPU cache.

    Parameters
    ----------
    X_tensor : torch.Tensor
        Input sequence tensor of shape (1, 4, seq_len), on the model's device.
    model : torch.nn.Module
        Akita SeqNN instance in eval mode.
    path : str
        Output path for the saved activation tensor (.pt file).
    """
    with torch.no_grad():
        x = model.conv_block_1(X_tensor)
        x = model.conv_tower(x)
    torch.save(x.cpu(), path)
    torch.cuda.empty_cache()
    
    
def make_target(
    model,
    X_tensor: torch.Tensor,
    feature_mask: torch.Tensor,
    feature_vector: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Generate an optimization target by overwriting feature positions in the model output.

    Runs a full forward pass on X_tensor, then replaces the positions
    indicated by feature_mask with values from feature_vector. Used to
    construct the target contact map for boundary, dot, and flame optimization.

    Parameters
    ----------
    model : torch.nn.Module
        Frozen Akita SeqNN in eval mode.
    X_tensor : torch.Tensor
        Input sequence tensor of shape (1, 4, seq_len).
    feature_mask : torch.Tensor
        1-D boolean or integer index tensor marking positions to overwrite
        in the upper-tri output vector.
    feature_vector : torch.Tensor
        Target values of shape (N_triu,); only positions in feature_mask
        are used.
    device : torch.device
        Device the model lives on.

    Returns
    -------
    torch.Tensor
        Modified model output of shape (1, 1, N_triu) on CPU.
    """
    with torch.no_grad():
        y = model(X_tensor.to(device))

    y_bar = y.clone()
    y_bar[0, 0, feature_mask] = feature_vector[feature_mask].to(device)
    return y_bar.cpu()