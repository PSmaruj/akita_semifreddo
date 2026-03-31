"""
utils/data_utils.py

Low-level data utilities shared across the AkitaSF pipeline.

Functions
---------
one_hot_encode_sequence : encode a DNA string as a (1, 4, L) float32 array
get_sequence            : extract an uppercase DNA string from a pyfaidx Fasta
gc_content              : compute mean GC fraction over a batch of OHE sequences

Matrix utilities (Akita upper-triangular format)
------------------------------------------------
set_diag                      : set a matrix diagonal to a given value in-place
from_upper_triu               : reconstruct a symmetric contact matrix from a flat vector
from_upper_triu_batch         : reconstruct a batch of contact matrices
upper_triangular_to_vector    : extract the upper-tri portion of a matrix into a vector
fragment_indices_in_upper_triangular : map a 2-D fragment mask to upper-tri indices
"""

import numpy as np
import random
import torch


def one_hot_encode_sequence(sequence_obj: object) -> np.ndarray:
    """One-hot encode a pyfaidx Sequence object.

    Unknown bases (N, etc.) are replaced by a randomly chosen valid base.

    Parameters
    ----------
    sequence_obj:
        A pyfaidx.Sequence (or any object whose str() is a DNA string).

    Returns
    -------
    np.ndarray
        Shape (1, 4, seq_len), dtype float32.
        Channel order: A=0, C=1, G=2, T=3.
    """
    sequence = str(sequence_obj).upper()
    base_to_int = {"A": 0, "C": 1, "G": 2, "T": 3}
    valid_bases = list(base_to_int.keys())

    encoded_indices = []
    for base in sequence:
        if base in base_to_int:
            encoded_indices.append(base_to_int[base])
        else:
            encoded_indices.append(base_to_int[random.choice(valid_bases)])

    encoded = np.array(encoded_indices, dtype=np.int64)
    ohe = np.zeros((4, len(encoded)), dtype=np.float32)
    ohe[encoded, np.arange(len(encoded))] = 1.0

    return ohe[np.newaxis, ...]  # (1, 4, seq_len)


def get_sequence(genome, chrom, start, end):
    return str(genome[chrom][start:end].seq.upper())


def upper_triangular_to_vector(matrix: np.ndarray, diagonal_offset: int = 2) -> np.ndarray:
    """Extract the upper triangular portion of a square matrix into a 1-D vector.

    Parameters
    ----------
    matrix:
        2-D numpy array of shape (N, N).
    diagonal_offset:
        Number of diagonals to skip from the main diagonal (default 2,
        matching Akita v2's UpperTri module).

    Returns
    -------
    np.ndarray
        1-D array of length N*(N-1)/2 - sum(skipped diagonals).
    """
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError(f"Expected square matrix, got {matrix.shape}.")
    return matrix[np.triu_indices(n, k=diagonal_offset)]


def fragment_indices_in_upper_triangular(
    matrix_size: int = 512,
    fragment_mask: np.ndarray = None,
    diagonal_offset: int = 2,
) -> np.ndarray:
    """Map a 2-D binary fragment mask to indices in the upper-tri vector.

    Parameters
    ----------
    matrix_size:
        Side length of the square contact matrix (default 512).
    fragment_mask:
        Boolean array of shape (matrix_size, matrix_size) marking the
        contacts of interest.
    diagonal_offset:
        Diagonal offset used when extracting the upper triangle (default 2).

    Returns
    -------
    np.ndarray
        Integer indices into the upper-triangular vector corresponding to
        the True positions of fragment_mask.
    """
    if fragment_mask is None:
        raise ValueError("fragment_mask must be provided.")
    if fragment_mask.shape != (matrix_size, matrix_size):
        raise ValueError(
            f"fragment_mask must have shape ({matrix_size}, {matrix_size}), "
            f"got {fragment_mask.shape}."
        )

    row_indices, col_indices = np.triu_indices(matrix_size, k=diagonal_offset)
    selected_indices = np.where(fragment_mask[row_indices, col_indices])[0]
    return selected_indices


def set_diag(matrix, value, k):
    """Set the k-th diagonal of a 2-D matrix to a given value, in-place.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array to modify in-place.
    value : float
        Value to assign to the diagonal (e.g. np.nan).
    k : int
        Diagonal offset: 0 = main diagonal, positive = above, negative = below.
    """
    # Explicitly set the diagonal to 'value' (in this case, np.nan) for each k
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value


def from_upper_triu(vector_repr, matrix_len, num_diags):
    """Reconstruct a symmetric contact matrix from an upper-triangular vector.

    Accepts either a NumPy array or a PyTorch tensor. The near-diagonal entries
    (within num_diags of the main diagonal) are set to np.nan to match Akita's
    output convention.

    Parameters
    ----------
    vector_repr : np.ndarray or torch.Tensor
        Flat upper-triangular vector of length N_triu.
    matrix_len : int
        Side length of the output square matrix.
    num_diags : int
        Number of diagonals to blank (set to np.nan), matching the diagonal
        offset used when extracting the upper triangle (typically 2 for Akita).

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape (matrix_len, matrix_len) with np.nan
        on the first num_diags diagonals.
    """
    # Ensure vector_repr is a NumPy array (if it's a PyTorch tensor, convert it)
    if isinstance(vector_repr, torch.Tensor):
        vector_repr = vector_repr.detach().flatten().cpu().numpy()  # Flatten and convert to NumPy array

    # Initialize a zero matrix of shape (matrix_len, matrix_len)
    z = np.zeros((matrix_len, matrix_len))

    # Get the indices for the upper triangular matrix
    triu_tup = np.triu_indices(matrix_len, num_diags)

    # Assign the values from the vector_repr to the upper triangular part of the matrix
    z[triu_tup] = vector_repr

    # Set the diagonals specified by num_diags to np.nan
    for i in range(-num_diags + 1, num_diags):
        set_diag(z, np.nan, i)

    # Ensure the matrix is symmetric
    return z + z.T


def from_upper_triu_batch(batch_vectors, matrix_len=512, num_diags=2):
    """Reconstruct a batch of symmetric contact matrices from upper-triangular vectors.

    Parameters
    ----------
    batch_vectors : np.ndarray or torch.Tensor
        Shape (B, N_triu). If a tensor, converted to NumPy automatically.
    matrix_len : int
        Side length of each output square matrix (default 512).
    num_diags : int
        Number of near-diagonal entries to set to np.nan (default 2).

    Returns
    -------
    np.ndarray
        Shape (B, matrix_len, matrix_len), dtype float32, with np.nan
        on the first num_diags diagonals of each matrix.
    """
    if isinstance(batch_vectors, torch.Tensor):
        batch_vectors = batch_vectors.detach().cpu().numpy()

    batch_size = batch_vectors.shape[0]
    matrices = np.zeros((batch_size, matrix_len, matrix_len), dtype=np.float32)

    triu_indices = np.triu_indices(matrix_len, num_diags)

    for i in range(batch_size):
        matrices[i][triu_indices] = batch_vectors[i]
        # Mirror to lower triangle
        matrices[i] = matrices[i] + matrices[i].T

        # Set diagonals to np.nan
        for k in range(-num_diags + 1, num_diags):
            set_diag(matrices[i], np.nan, k)

    return matrices  # shape: [B, 512, 512]


def gc_content(X: torch.Tensor, bp_start: int = 0, bp_end: int = -1) -> np.ndarray:
    """Mean GC fraction per sequence over the specified bp range."""
    region = X[:, :, bp_start:bp_end] if bp_end != -1 else X
    return (region[:, 1:3, :].sum(dim=(1, 2)) / region.sum(dim=(1, 2))).cpu().numpy()