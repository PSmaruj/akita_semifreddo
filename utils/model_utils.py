import torch

DEFAULT_MODEL_SRC = "/home1/smaruj/pytorch_akita/"

def load_model(model_weights_path: str, device: torch.device) -> torch.nn.Module:
    """Instantiate SeqNN and load weights.

    Parameters
    ----------
    model_weights_path:
        Path to the saved state dict (.pt file).
    device:
        Torch device to load the model onto.

    Returns
    -------
    torch.nn.Module
        Model in eval mode with all parameters frozen.
    """
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
    """Run a forward pass and return the output on CPU.

    Parameters
    ----------
    model:
        Frozen model in eval mode.
    X_tensor:
        Input tensor of shape (1, 4, seq_len).
    device:
        Device the model lives on.

    Returns
    -------
    torch.Tensor
        Model output of shape (1, 1, num_contacts) on CPU.
    """
    y = model(X_tensor.to(device))
    return y.cpu()