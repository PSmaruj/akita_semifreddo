import torch
import torch.nn as nn

from .fimo_utils import run_fimo
from semifreddo.semifreddo import CTCFAwareSemifreddoWrapper

class LocalL1Loss(nn.Module):
    """
    L1 loss computed only over the indices specified by a mask, scaled by the
    inverse of the mask's coverage so that a mask covering X% of the full
    upper-tri vector contributes the same magnitude as a full-map loss.

    Scale factor = N_triu / K, where K is the number of masked positions.
    E.g. a mask covering 10% of positions is multiplied by 10.
    """

    def __init__(self, mask: torch.Tensor, n_triu: int, reduction: str = 'sum'):
        """
        Args:
            mask      : 1-D integer index tensor of shape (K,).
                        Loaded from boundary_mask.pt / flame_mask.pt.
            n_triu    : Total length of the upper-tri vector (N_triu).
                        Used to compute the coverage scale factor.
            reduction : 'sum' or 'mean', passed to the underlying L1Loss.
        """
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