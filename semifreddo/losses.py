"""
semifreddo/losses.py

Loss functions for Ledidi-based sequence optimization in the AkitaSF pipeline.

Classes
-------
LocalL1Loss                  : masked L1 loss scaled by inverse mask coverage
LocalL1LossWithCTCFPenalty   : LocalL1Loss with an additional FIMO-based CTCF penalty
MultiHeadLocalL1Loss         : LocalL1Loss summed across multiple model heads
"""

import torch
import torch.nn as nn
import sys
import os

sys.path.insert(0, os.path.abspath("/home1/smaruj/akita_semifreddo/"))
from utils.fimo_utils import run_fimo
from .semifreddo import CTCFAwareSemifreddoWrapper


class LocalL1Loss(nn.Module):
    """L1 loss computed over a masked subset of the upper-triangular contact vector.

    The loss is scaled by the inverse of the mask's fractional coverage
    (scale = N_triu / K, where K is the number of masked positions), so that
    a mask covering X% of the full upper-tri vector contributes the same
    magnitude as a full-map loss regardless of mask size.

    In the simplest use case the mask corresponds exactly to the positions
    where the desired feature is expected to appear on the contact map
    (e.g. the boundary stripe or dot position). However, the mask can also
    be larger than the feature itself to implement a *semi-local* loss: the
    additional positions outside the feature region are included in the loss
    to encourage the optimizer to preserve their contact values while still
    designing the target feature. For example, to strengthen a dot while
    keeping a nearby boundary intact, the mask can combine the dot positions
    with the boundary positions, effectively constraining both simultaneously.

    Parameters
    ----------
    mask : torch.Tensor
        1-D integer index tensor of shape (K,), indexing positions in the
        upper-tri vector. Typically loaded from boundary_mask.pt or
        flame_mask.pt.
    n_triu : int
        Total length of the upper-tri vector (N_triu); used to compute
        the coverage scale factor.
    reduction : str
        Reduction mode passed to the underlying nn.L1Loss: 'sum' or 'mean'
        (default 'sum').
    """

    def __init__(self, mask: torch.Tensor, n_triu: int, reduction: str = 'sum'):
        super().__init__()
        self.register_buffer("mask", mask)
        self.loss_fn = nn.L1Loss(reduction=reduction)

        k = mask.shape[0]
        self.scale = n_triu / k

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred / target shape: (1, 1, N_triu)
        return self.scale * self.loss_fn(pred[..., self.mask], target[..., self.mask])


class LocalL1LossWithCTCFPenalty(nn.Module):
    """
    Drop-in replacement for LocalL1Loss as Ledidi's output_loss, adding a
    CTCF penalty term:

        output_loss = LocalL1Loss(pred, target) + γ * Σ(FIMO scores)

    The current sequence is read automatically from seq_wrapper.last_x, which
    is populated by CTCFAwareSemifreddoWrapper on every forward pass — no
    external set_current_sequence() call required.

    Full loss as defined in the methods section:
        Loss = λ * input_loss + output_loss + γ * Σ(FIMO scores)
    where λ and input_loss are handled by Ledidi (L1Loss, λ=0.1 default).

    Parameters
    ----------
    local_loss_fn  : LocalL1Loss
        Pre-built masked L1 loss (carries mask and coverage scale).
    motifs_dict    : dict
        MEME motif dict passed to run_fimo(), e.g. {"CTCF": pwm_tensor}.
        PWM tensors must be shape (4, motif_len) as required by tangermeme.fimo.
    edited_slice   : slice
        Nucleotide positions in the sequence corresponding to the edited bin,
        used to restrict FIMO scanning to the edited region only.
    gamma          : float
        Weight on the FIMO score sum (default: 300).
    fimo_threshold : float
        p-value threshold passed to FIMO (default: 1e-4).
    seq_wrapper    : CTCFAwareSemifreddoWrapper
        Wrapper whose .last_x is read on every forward call. Must be the same
        instance passed to Ledidi as the model.
    """

    def __init__(
        self,
        local_loss_fn: nn.Module,
        motifs_dict: dict,
        edited_slice: slice,
        gamma: float = 300.0,
        fimo_threshold: float = 1e-4,
        seq_wrapper: CTCFAwareSemifreddoWrapper | None = None,
    ):
        super().__init__()
        self.local_loss_fn  = local_loss_fn
        self.motifs_dict    = motifs_dict
        self.edited_slice   = edited_slice
        self.gamma          = gamma
        self.fimo_threshold = fimo_threshold
        self.seq_wrapper    = seq_wrapper

    def _ctcf_penalty(self) -> torch.Tensor:        
        """Run FIMO on the edited region and return γ * Σ(scores) as a scalar."""
        if self.seq_wrapper is None or self.seq_wrapper.last_x is None:
            return torch.tensor(0.0)

        x_edited = self.seq_wrapper.last_x[:, :, self.edited_slice]   # (1, 4, edited_len)
        hits_df  = run_fimo(x_edited, self.motifs_dict, threshold=self.fimo_threshold)

        if hits_df is None or hits_df.empty:
            return torch.tensor(0.0, device=self.seq_wrapper.last_x.device)

        score_sum = float(hits_df["score"].sum())
        return torch.tensor(
            self.gamma * score_sum,
            dtype=torch.float32,
            device=self.seq_wrapper.last_x.device,
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        map_loss     = self.local_loss_fn(pred, target)
        ctcf_penalty = self._ctcf_penalty()
        return map_loss + ctcf_penalty


class MultiHeadLocalL1Loss(nn.Module):
    """Apply LocalL1Loss independently to each model head and sum the results.

    Used for ensemble optimization where predictions have shape
    (batch, n_models, N_triu) and each model head is penalized separately
    using the same mask.

    Parameters
    ----------
    mask : torch.Tensor
        1-D integer index tensor of shape (K,), passed to LocalL1Loss.
    n_triu : int
        Total length of the upper-tri vector (N_triu).
    n_models : int
        Number of model heads (size of dim=1 in predictions).
    reduction : str
        Reduction mode passed to LocalL1Loss (default 'sum').
    """
    def __init__(self, mask: torch.Tensor, n_triu: int, n_models: int,
                 reduction: str = 'sum'):
        super().__init__()
        self.n_models   = n_models
        self.single_loss = LocalL1Loss(mask, n_triu=n_triu, reduction=reduction)

    def forward(self, y_hat: torch.Tensor, y_bar: torch.Tensor) -> torch.Tensor:
        # y_hat, y_bar: (batch, n_models, N_triu)
        total = sum(
            self.single_loss(y_hat[:, i, :].unsqueeze(1),
                             y_bar[:, i, :].unsqueeze(1))
            for i in range(self.n_models)
        )
        return total