import torch
import torch.nn as nn

class LocalL1Loss(nn.Module):
    """L1 loss computed only over the indices specified by a boolean/index mask."""

    def __init__(self, mask: torch.Tensor, reduction: str = 'sum'):
        """
        Args:
            mask: 1-D boolean tensor of shape (N_triu,), or integer index tensor.
                  Loaded from boundary_mask.pt.
            reduction: 'sum' or 'mean', passed to the underlying L1Loss.
        """
        super().__init__()
        self.register_buffer("mask", mask)
        self.loss_fn = nn.L1Loss(reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target shape: (1, 1, N_triu)
        return self.loss_fn(pred[..., self.mask], target[..., self.mask])