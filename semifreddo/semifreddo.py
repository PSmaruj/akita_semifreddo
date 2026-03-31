"""
semifreddo.py

Semifreddo-AkitaPT: a "half-frozen" Akita wrapper that caches convolutional
trunk activations and recomputes only the bins affected by sequence edits.

Instead of passing the full 1.3 Mb sequence through the convolutional tower on
every optimisation step, only a small padded window around the edited bin(s) is
recomputed. The receptive field of the convolutional tower is ~10 kb (~5 bins at
2048 bp/bin), so ±5 bins of context around the edit are sufficient to avoid
zero-padding contamination. Only the central 5 bins of that window are spliced
back into the cached activations.

This reduces convolutional tower computation from 640 bins to 11 bins per step,
giving >3× lower peak memory and ~25% faster runtime vs. the full model while
producing identical predictions (Pearson R = 1.0).

Classes
-------
Semifreddo
    Core half-frozen forward pass. Accepts pre-cached trunk activations and
    a small padded sequence slice; returns the full model output.

SemifreddoLedidiWrapper
    Thin nn.Module wrapper around Semifreddo that exposes a standard
    model(X) → y_hat interface compatible with Ledidi.
"""

import torch
import torch.nn as nn


# Number of bins of sequence context required on each side of the edited bin.
# Derived from the convolutional tower receptive field (~5 bins at 2048 bp/bin).
_CONTEXT_BINS = 5

# Slice of sub_x that contains the uncontaminated central 5 bins.
# For an 11-bin window (context_bins=5): indices 5-2=3 to 5+3=8 → [3:8].
_CENTER_SLICE = slice(
    _CONTEXT_BINS - 2,
    _CONTEXT_BINS + 3,
)

# Sequence length of one bin after the convolutional tower (bp).
_BIN_SIZE = 2048


# =============================================================================
# Helpers
# =============================================================================

def _splice_activations(
    x: torch.Tensor,
    sub_x: torch.Tensor,
    edited_bin_start: int,
    edited_bin_end: int,
) -> torch.Tensor:
    """Replace 5 bins around [edited_bin_start, edited_bin_end] in the cached
    activation tensor `x` with the recomputed central 5 bins from `sub_x`.

    Parameters
    ----------
    x               : (B, C, 640) cached full trunk activations
    sub_x           : (B, C, 11)  freshly computed window activations
    edited_bin_start: first edited bin index in the 640-bin space
    edited_bin_end  : last  edited bin index in the 640-bin space (inclusive)

    Returns
    -------
    x with the 5 central bins replaced in-place on a clone.
    """
    x = x.clone()
    # Replace ±2 bins around the edit (5 bins total) with uncontaminated values.
    x[:, :, edited_bin_start - 2 : edited_bin_end + 3] = sub_x[:, :, _CENTER_SLICE]

    return x


# =============================================================================
# Semifreddo
# =============================================================================

class Semifreddo:
    """Half-frozen Akita forward pass.

    Parameters
    ----------
    model                   : Akita SeqNN in eval mode
    slice_0_padded_seq      : (1, 4, 11*2048) sequence window around edit 0
    edited_indices_slice_0  : iterable of bin indices (in 640-bin space) for edit 0
    precomputed_full_output : (1, C, 640) cached trunk activations
    slice_1_padded_seq      : optional second edit window (same format as slice_0)
    edited_indices_slice_1  : optional bin indices for edit 1
    batch_size              : batch size (default 1)
    cropping_applied        : bins cropped from each side by Akita (default 64)
    """

    def __init__(
        self,
        model,
        slice_0_padded_seq: torch.Tensor,
        edited_indices_slice_0,
        precomputed_full_output: torch.Tensor,
        slice_1_padded_seq: torch.Tensor = None,
        edited_indices_slice_1=None,
        batch_size: int = 1,
        cropping_applied: int = 64,
    ):
        self.model                   = model.eval()
        self.slice_0_padded_seq      = slice_0_padded_seq
        self.slice_1_padded_seq      = slice_1_padded_seq
        self.edited_indices_slice_0  = edited_indices_slice_0
        self.edited_indices_slice_1  = edited_indices_slice_1
        self.precomputed_full_output = precomputed_full_output
        self.batch_size              = batch_size
        self.cropping_applied        = cropping_applied

    def forward(self) -> torch.Tensor:
        device = next(self.model.parameters()).device

        # --- Recompute convolutional tower for the edited window(s) only -----

        sub_x_0 = self.model.conv_block_1(self.slice_0_padded_seq.to(device))
        sub_x_0 = self.model.conv_tower(sub_x_0)           # (1, C, 11)

        if self.slice_1_padded_seq is not None:
            sub_x_1 = self.model.conv_block_1(self.slice_1_padded_seq.to(device))
            sub_x_1 = self.model.conv_tower(sub_x_1)       # (1, C, 11)

        # --- Splice recomputed activations into the cached full trunk ---------

        x = self.precomputed_full_output.clone()
        if x.shape[0] != self.batch_size:
            x = x.repeat(self.batch_size, 1, 1)

        bin_0_start = min(self.edited_indices_slice_0) + self.cropping_applied
        bin_0_end   = max(self.edited_indices_slice_0) + self.cropping_applied
        x = _splice_activations(x, sub_x_0, bin_0_start, bin_0_end)

        if self.slice_1_padded_seq is not None:
            bin_1_start = min(self.edited_indices_slice_1) + self.cropping_applied
            bin_1_end   = max(self.edited_indices_slice_1) + self.cropping_applied
            x = _splice_activations(x, sub_x_1, bin_1_start, bin_1_end)

        # --- Run the trunk of the model on the spliced activations -----------

        # stochastic_reverse_complement is called here only to obtain
        # reverse_bool, which upper_tri needs for correct output orientation.
        # In eval mode it is a no-op on x itself.
        x, reverse_bool = self.model.stochastic_reverse_complement(x)

        x = self.model.residual1d_block1(x)
        x = self.model.residual1d_block2(x)
        x = self.model.residual1d_block3(x)
        x = self.model.residual1d_block4(x)
        x = self.model.residual1d_block5(x)
        x = self.model.residual1d_block6(x)
        x = self.model.residual1d_block7(x)
        x = self.model.residual1d_block8(x)
        x = self.model.residual1d_block9(x)
        x = self.model.residual1d_block10(x)
        x = self.model.residual1d_block11(x)

        x = self.model.conv_reduce(x)
        x = self.model.one_to_two(x)
        x = self.model.conv2d_block(x)
        x = self.model.symmetrize_2d(x)

        x = self.model.residual2d_block1(x)
        x = self.model.residual2d_block2(x)
        x = self.model.residual2d_block3(x)
        x = self.model.residual2d_block4(x)
        x = self.model.residual2d_block5(x)
        x = self.model.residual2d_block6(x)

        x = self.model.squeeze_excite(x)
        x = self.model.cropping_2d(x)
        x = self.model.upper_tri(x, reverse_complement_flags=reverse_bool)
        x = self.model.final(x)

        return x


# =============================================================================
# Ledidi wrapper
# =============================================================================

class SemifreddoLedidiWrapper(nn.Module):
    """Wraps Semifreddo to expose a standard model(X) → y_hat interface for Ledidi.

    On each call, extracts the 11-bin padded sequence slice around the edited
    bin from the full input X, and delegates to Semifreddo.

    Parameters
    ----------
    model                   : Akita SeqNN
    precomputed_full_output : (1, C, 640) trunk activations cached from initial X
    edited_bin              : 0-indexed bin being optimised (in 640-bin space)
    context_bins            : bins of context on each side (default 5 → 11-bin window)
    cropping_applied        : bins cropped by Akita (default 64)
    """

    def __init__(self, model, precomputed_full_output, full_X,
                 edited_bin, context_bins=5, cropping_applied=64):
        super().__init__()
        self.model                   = model
        self.precomputed_full_output = precomputed_full_output
        self.edited_bin              = edited_bin
        self.context_bins            = context_bins
        self.cropping_applied        = cropping_applied
        self.seq_slice_start = (edited_bin - context_bins + self.cropping_applied) * _BIN_SIZE
        self.seq_slice_end   = (edited_bin + context_bins + self.cropping_applied + 1) * _BIN_SIZE
        # Central bin bp coordinates within the full sequence
        self.center_bp_start = (edited_bin + self.cropping_applied) * _BIN_SIZE
        self.center_bp_end   = (edited_bin + self.cropping_applied + 1) * _BIN_SIZE
        self.edited_indices  = [edited_bin]
        self.register_buffer('full_X', full_X.clone())

    def forward(self, X_center: torch.Tensor) -> torch.Tensor:
        # X_center is (1, 4, 2048) — only the central editable bin
        # Build the full 11-bin window from frozen X, splice in the edited center
        X_window = self.full_X.detach()[:, :, self.seq_slice_start:self.seq_slice_end].clone()
        center_offset = self.context_bins * _BIN_SIZE  # = 5 * 2048 = 10240
        X_window[:, :, center_offset : center_offset + _BIN_SIZE] = X_center

        sf = Semifreddo(
            model                   = self.model,
            slice_0_padded_seq      = X_window,
            edited_indices_slice_0  = self.edited_indices,
            precomputed_full_output = self.precomputed_full_output,
            batch_size              = 1,
            cropping_applied        = self.cropping_applied,
        )
        return sf.forward()


class TwoAnchorSemifreddoLedidiWrapper(nn.Module):
    """Semifreddo wrapper for optimising two sequence bins simultaneously.

    Expects X_anchors of shape (1, 4, 2*BIN_SIZE) — the two anchor bins
    concatenated — and internally routes each half to Semifreddo's
    slice_0 / slice_1 mechanism.
    """
    
    def __init__(self, model, precomputed_full_output, full_X,
                 bin_lo, bin_hi, context_bins=5, cropping_applied=64):
        super().__init__()
        self.model                   = model
        self.precomputed_full_output = precomputed_full_output
        self.context_bins            = context_bins
        self.cropping_applied        = cropping_applied
        self.bin_lo                  = bin_lo
        self.bin_hi                  = bin_hi

        # bp coordinates of the 11-bin padded windows in the full sequence
        self.seq_lo_start = (bin_lo - context_bins + cropping_applied) * _BIN_SIZE
        self.seq_lo_end   = (bin_lo + context_bins + cropping_applied + 1) * _BIN_SIZE
        self.seq_hi_start = (bin_hi - context_bins + cropping_applied) * _BIN_SIZE
        self.seq_hi_end   = (bin_hi + context_bins + cropping_applied + 1) * _BIN_SIZE

        # bp coordinates of just the central (editable) bin for each anchor
        self.bp_lo_start  = (bin_lo + cropping_applied) * _BIN_SIZE
        self.bp_lo_end    = (bin_lo + cropping_applied + 1) * _BIN_SIZE
        self.bp_hi_start  = (bin_hi + cropping_applied) * _BIN_SIZE
        self.bp_hi_end    = (bin_hi + cropping_applied + 1) * _BIN_SIZE

        self.register_buffer('full_X', full_X.clone())
    
    def forward(self, X_anchors: torch.Tensor) -> torch.Tensor:
        # Split concatenated input back into per-anchor bins
        X_lo_edit = X_anchors[:, :, :_BIN_SIZE]          # (1, 4, BIN_SIZE)
        X_hi_edit = X_anchors[:, :, _BIN_SIZE:]           # (1, 4, BIN_SIZE)

        full_X = self.full_X.detach()
        ctx    = self.context_bins * _BIN_SIZE             # = 10240 bp

        # Build 11-bin padded windows, splicing in the current edited bins
        win_lo = full_X[:, :, self.seq_lo_start:self.seq_lo_end].clone()
        win_lo[:, :, ctx : ctx + _BIN_SIZE] = X_lo_edit

        win_hi = full_X[:, :, self.seq_hi_start:self.seq_hi_end].clone()
        win_hi[:, :, ctx : ctx + _BIN_SIZE] = X_hi_edit

        sf = Semifreddo(
            model                   = self.model,
            slice_0_padded_seq      = win_lo,
            edited_indices_slice_0  = [self.bin_lo],
            precomputed_full_output = self.precomputed_full_output,
            slice_1_padded_seq      = win_hi,
            edited_indices_slice_1  = [self.bin_hi],
            batch_size              = 1,
            cropping_applied        = self.cropping_applied,
        )
        return sf.forward()


class CTCFAwareSemifreddoWrapper(nn.Module):
    def __init__(self, sf_wrapper: nn.Module):
        super().__init__()
        self.sf_wrapper = sf_wrapper
        self.last_x: torch.Tensor | None = None

    @property
    def center_bp_start(self):
        return self.sf_wrapper.center_bp_start

    @property
    def center_bp_end(self):
        return self.sf_wrapper.center_bp_end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.last_x = x
        return self.sf_wrapper(x)


class MultiBinSemifreddoLedidiWrapper(nn.Module):
    """Semifreddo wrapper for optimising a contiguous block of bins simultaneously.

    Designed for large features (e.g. fountains) where edits span many bins.
    The editable region is a contiguous block of `n_edit_bins` bins centered on
    `center_bin` in the 512-bin contact map space.

    The conv tower is run once over a padded window of
    (n_edit_bins + 2 * context_bins) bins. The recomputed activations are
    spliced back into the cached full-sequence tower output, replacing
    (n_edit_bins + 2 * splice_buffer) bins around the edited block to avoid
    boundary contamination.

    Parameters
    ----------
    model                   : Akita SeqNN in eval mode
    precomputed_full_output : (1, C, 640) cached trunk activations from initial X
    full_X                  : (1, 4, L) full one-hot encoded sequence
    center_bin              : centre of the editable block in 512-bin map space
                              (default 256, the middle of the map)
    n_edit_bins             : number of contiguous bins to optimise (default 50)
    context_bins            : conv tower context bins on each side (default 5)
    splice_buffer           : extra bins spliced back on each side of the edited
                              block to avoid boundary artefacts (default 2,
                              matching the ±2 convention in _splice_activations)
    cropping_applied        : bins cropped from each side by Akita (default 64)
    """

    def __init__(
        self,
        model,
        precomputed_full_output: torch.Tensor,
        full_X: torch.Tensor,
        center_bin: int      = 256,
        n_edit_bins: int     = 50,
        context_bins: int    = 5,
        splice_buffer: int   = 2,
        cropping_applied: int = 64,
    ):
        super().__init__()
        self.model                   = model
        self.precomputed_full_output = precomputed_full_output
        self.n_edit_bins             = n_edit_bins
        self.context_bins            = context_bins
        self.splice_buffer           = splice_buffer
        self.cropping_applied        = cropping_applied

        # ── Bin indices in 512-bin map space ──────────────────────────────────
        half          = n_edit_bins // 2
        self.bin_start = center_bin - half              # first edited bin (map space)
        self.bin_end   = self.bin_start + n_edit_bins   # one-past-last (exclusive)

        # ── Bin indices in 640-bin tower space (add cropping offset) ──────────
        self.tower_bin_start = self.bin_start + cropping_applied
        self.tower_bin_end   = self.bin_end   + cropping_applied  # exclusive

        # ── bp coordinates of the padded conv-tower input window ──────────────
        # window = context_bins left + n_edit_bins + context_bins right
        self.seq_window_start = (self.tower_bin_start - context_bins) * _BIN_SIZE
        self.seq_window_end   = (self.tower_bin_end   + context_bins) * _BIN_SIZE

        # ── bp coordinates of just the editable block within full_X ──────────
        self.edit_bp_start = self.tower_bin_start * _BIN_SIZE
        self.edit_bp_end   = self.tower_bin_end   * _BIN_SIZE

        # ── Splice range in 640-bin tower space ───────────────────────────────
        # Replace (n_edit_bins + 2 * splice_buffer) bins in the cached tensor
        self.splice_start = self.tower_bin_start - splice_buffer
        self.splice_end   = self.tower_bin_end   + splice_buffer  # exclusive

        # ── Slice of the recomputed sub_x to read splice values from ─────────
        # sub_x has shape (1, C, n_edit_bins + 2*context_bins)
        # The uncontaminated central region starts at context_bins - splice_buffer
        sub_x_splice_start = context_bins - splice_buffer
        sub_x_splice_end   = context_bins + n_edit_bins + splice_buffer
        self.sub_x_splice  = slice(sub_x_splice_start, sub_x_splice_end)

        self.register_buffer('full_X', full_X.clone())

        # Expose for notebooks / sanity checks
        print(
            f"MultiBinSemifreddoLedidiWrapper:\n"
            f"  editable bins (map space) : {self.bin_start}–{self.bin_end - 1}\n"
            f"  editable bp               : {self.edit_bp_start:,}–{self.edit_bp_end:,}\n"
            f"  conv-tower window bp      : {self.seq_window_start:,}–{self.seq_window_end:,}\n"
            f"  splice range (tower space): {self.splice_start}–{self.splice_end - 1}"
        )

    def forward(self, X_edit: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        X_edit : (1, 4, n_edit_bins * BIN_SIZE)
            One-hot encoded sequence for the editable block only.

        Returns
        -------
        y_hat : (1, 1, N_triu)  model prediction
        """
        full_X = self.full_X.detach()

        # Build padded window: frozen context left + edited block + frozen context right
        X_window = full_X[:, :, self.seq_window_start:self.seq_window_end].clone()

        # Splice the current edited block into the window
        context_bp = self.context_bins * _BIN_SIZE
        X_window[:, :, context_bp : context_bp + self.n_edit_bins * _BIN_SIZE] = X_edit

        device = next(self.model.parameters()).device
        X_window = X_window.to(device)

        # Run conv tower on the padded window only
        with torch.no_grad():
            # conv_block_1 and conv_tower must not accumulate gradients for
            # the frozen cached activations path; only X_edit needs grad flow
            pass

        sub_x = self.model.conv_block_1(X_window)
        sub_x = self.model.conv_tower(sub_x)   # (1, C, n_edit_bins + 2*context_bins)

        # Splice recomputed activations into cached full-sequence tensor
        x = self.precomputed_full_output.clone().to(device)
        x[:, :, self.splice_start:self.splice_end] = sub_x[:, :, self.sub_x_splice]

        # Run the rest of the model (identical to Semifreddo.forward)
        x, reverse_bool = self.model.stochastic_reverse_complement(x)

        x = self.model.residual1d_block1(x)
        x = self.model.residual1d_block2(x)
        x = self.model.residual1d_block3(x)
        x = self.model.residual1d_block4(x)
        x = self.model.residual1d_block5(x)
        x = self.model.residual1d_block6(x)
        x = self.model.residual1d_block7(x)
        x = self.model.residual1d_block8(x)
        x = self.model.residual1d_block9(x)
        x = self.model.residual1d_block10(x)
        x = self.model.residual1d_block11(x)

        x = self.model.conv_reduce(x)
        x = self.model.one_to_two(x)
        x = self.model.conv2d_block(x)
        x = self.model.symmetrize_2d(x)

        x = self.model.residual2d_block1(x)
        x = self.model.residual2d_block2(x)
        x = self.model.residual2d_block3(x)
        x = self.model.residual2d_block4(x)
        x = self.model.residual2d_block5(x)
        x = self.model.residual2d_block6(x)

        x = self.model.squeeze_excite(x)
        x = self.model.cropping_2d(x)
        x = self.model.upper_tri(x, reverse_complement_flags=reverse_bool)
        x = self.model.final(x)

        return x
    

class StackingDesignWrapper(nn.Module):
    """Combines multiple models by stacking outputs along dim=1.
    Each model must return (batch, 1, N_triu); result is (batch, n_models, N_triu).
    Use instead of tangermeme DesignWrapper which concatenates on dim=-1.
    """
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, X):
        return torch.cat([m(X) for m in self.models], dim=1)