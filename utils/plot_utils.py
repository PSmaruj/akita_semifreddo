"""
utils/plot_utils.py

Visualization utilities for contact maps and CTCF tracks in the AkitaSF pipeline.

Functions
---------
plot_matrix       : plot a single contact map and save as SVG
plot_ctcf_track   : bar plot of binned CTCF motif hits on both strands
save_movie_frame  : render a contact map + CTCF track as a PNG frame for animations
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def plot_matrix(matrix, save_path, vmin=-0.6, vmax=0.6, title=None):
    """Plot a contact map and save it as an SVG.

    Accepts either a NumPy array or a PyTorch tensor; tensors are
    converted to NumPy automatically.

    Parameters
    ----------
    matrix : np.ndarray or torch.Tensor
        2-D contact map to plot.
    save_path : str
        Output path for the SVG file.
    vmin : float
        Minimum value for the colormap (default -0.6).
    vmax : float
        Maximum value for the colormap (default 0.6).
    title : str or None
        Optional plot title.
    """  
    if hasattr(matrix, "detach"):
        matrix = matrix.detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.matshow(matrix.astype(np.float16), cmap='RdBu_r', vmin=vmin, vmax=vmax)
    
    if title:
        ax.set_title(title)
        
    plt.colorbar(im, fraction=0.046, pad=0.04)
    
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.show()
    plt.close(fig)


def plot_ctcf_track(ctcf_plus, ctcf_minus, title="CTCF", save_path=None):
    """Plot binned CTCF motif hit counts as a two-strand bar chart.

    Plus-strand hits are plotted upward (black) and minus-strand hits
    downward (red), with a shared zero baseline.

    Parameters
    ----------
    ctcf_plus : np.ndarray
        Shape (n_bins,); hit counts on the + strand per bin.
    ctcf_minus : np.ndarray
        Shape (n_bins,); hit counts on the − strand per bin.
    title : str
        Plot title (default 'CTCF').
    save_path : str or None
        If provided, save the figure as an SVG to this path.
    """
    n_bins = len(ctcf_plus)
    x = np.arange(n_bins)

    fig, ax = plt.subplots(figsize=(15, 3))
    ax.bar(x,  ctcf_plus,  width=1.0, color="black", alpha=0.7, label="+ strand")
    ax.bar(x, -ctcf_minus, width=1.0, color="red",   alpha=0.7, label="− strand")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_xlim(0, n_bins)
    ax.set_ylim(-4, 4)
    ax.set_xlabel("Input sequence (bin)")
    ax.set_ylabel("CTCF motif count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, format="svg", bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()
    plt.close(fig)
    

def save_movie_frame(
    contact_map, 
    ctcf_plus, 
    ctcf_minus, 
    frame_idx, 
    frames_dir="frames", 
    vmin=-0.6, 
    vmax=0.6, 
    max_ctcf=5
):
    """Render a contact map and CTCF track as a PNG frame for animations.

    Produces a two-panel figure: the contact map on top and the CTCF
    strand bar chart below, sharing the x-axis. Frames are named
    frame_NNN.png and saved to frames_dir.

    Accepts NumPy arrays or PyTorch tensors for all array inputs;
    tensors are converted to NumPy automatically.

    Parameters
    ----------
    contact_map : np.ndarray or torch.Tensor
        2-D contact map of shape (map_size, map_size).
    ctcf_plus : np.ndarray or torch.Tensor
        Shape (map_size,); + strand CTCF hit counts per bin.
    ctcf_minus : np.ndarray or torch.Tensor
        Shape (map_size,); − strand CTCF hit counts per bin.
    frame_idx : int
        Frame index used for zero-padded filename (e.g. 007 → frame_007.png).
    frames_dir : str
        Directory where PNG frames are saved (default 'frames').
    vmin : float
        Minimum value for the contact map colormap (default -0.6).
    vmax : float
        Maximum value for the contact map colormap (default 0.6).
    max_ctcf : int
        Y-axis limit for the CTCF track (default 5).

    Returns
    -------
    str
        Full path to the saved PNG frame.
    """
    # Ensure the directory exists
    os.makedirs(frames_dir, exist_ok=True)

    # Use a style-agnostic setup
    fig = plt.figure(figsize=(10, 14), dpi=100)
    gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.05)
    
    map_size = contact_map.shape[0]

    # Upper Plot: Contact Map
    ax0 = fig.add_subplot(gs[0])
    # Ensure it's on CPU/Numpy before plotting
    if hasattr(contact_map, "detach"):
        contact_map = contact_map.detach().cpu().numpy()
        
    ax0.matshow(contact_map.astype(np.float16), cmap="RdBu_r", vmin=vmin, vmax=vmax)
    ax0.set_xticks([]); ax0.set_yticks([])
    for spine in ax0.spines.values():
        spine.set_visible(False)

    # Lower Plot: CTCF Tracks
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    x = np.arange(map_size)
    
    # Handle tensor conversion for tracks too
    cp = ctcf_plus.detach().cpu().numpy() if hasattr(ctcf_plus, "detach") else ctcf_plus
    cm = ctcf_minus.detach().cpu().numpy() if hasattr(ctcf_minus, "detach") else ctcf_minus

    ax1.bar(x,  cp,  width=1.0, color="black", alpha=0.7, label="+ strand")
    ax1.bar(x, -cm, width=1.0, color="red",   alpha=0.7, label="− strand")
    ax1.axhline(0, color="gray", linewidth=0.5)
    
    ax1.set_xlim(0, map_size)
    ax1.set_ylim(-max_ctcf, max_ctcf)
    ax1.set_ylabel("CTCF motif count")
    ax1.set_xlabel(f"Sequence bin ({map_size}-bin window)")
    ax1.legend(loc="upper right", framealpha=0.5)

    # Save logic
    path = os.path.join(frames_dir, f"frame_{frame_idx:03d}.png")
    fig.savefig(path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    
    return path