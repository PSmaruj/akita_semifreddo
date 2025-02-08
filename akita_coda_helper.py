import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_map(matrix, vmin=-0.6, vmax=0.6, palette="RdBu_r", width=5, height=5, highlight_region=None):
    fig, axes = plt.subplots(1, 1, figsize=(width, height))

    sns.heatmap(
        matrix,
        vmin=vmin,
        vmax=vmax,
        cbar=False,
        cmap=palette,
        square=True,
        xticklabels=False,
        yticklabels=False,
        ax=axes
    )

    # Add a blue rectangle if highlight_region is provided
    if highlight_region:
        start_row, end_row, start_col, end_col = highlight_region
        rect = patches.Rectangle(
            (start_col, start_row),  # (x, y) -> starts at (col, row)
            end_col - start_col,  # width
            end_row - start_row,  # height
            linewidth=2,
            edgecolor='blue',
            facecolor='none'
        )
        axes.add_patch(rect)

    plt.tight_layout()
    plt.show()
    

# Helper function to set diagonal elements to a specific value
def set_diag(matrix, value, k):
    # Explicitly set the diagonal to 'value' (in this case, np.nan) for each k
    rows, cols = matrix.shape
    for i in range(rows):
        if 0 <= i + k < cols:
            matrix[i, i + k] = value


def from_upper_triu(vector_repr, matrix_len, num_diags):
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


def upper_triangular_to_vector_skip_diagonals(matrix, dim=64, diag=2):
    
    # Extract the upper triangular part excluding the first two diagonals
    upper_triangular_vector = matrix[np.triu_indices(dim, k=diag)]
    
    return upper_triangular_vector