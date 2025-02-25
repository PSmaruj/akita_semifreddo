import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler,ConcatDataset
import torch.nn.functional as F

import sys
import os

from helper import plot_map, from_upper_triu, upper_triangular_to_vector_skip_diagonals

sys.path.append(os.path.abspath("/home1/smaruj/pytorch_akita/"))
from model import SeqNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the entire model (architecture + weights)
model = torch.load("/home1/smaruj/pytorch_akita/model.pth")
model = model.to(device)
# Set the model to evaluation mode (important for inference)
model.eval()

X = torch.load("/scratch1/smaruj/ledidi_targets/full_X.pt", weights_only=True)
y_bar = torch.load("/scratch1/smaruj/ledidi_targets/full_modified_vector.pt", weights_only=True)

# to ensure the local, forked ledidi is used
# not the one installed using pip
import sys
sys.path.insert(0, "/home1/smaruj/ledidi")
from ledidi import Ledidi

start_index = 523264
end_index = 525312

seq_length = end_index - start_index

wrapper = Ledidi(model, verbose=True, batch_size=10,
                 input_loss=torch.nn.L1Loss(reduction='sum'), 
                 output_loss=torch.nn.L1Loss(reduction='sum'),
                 max_iter=4000,
                 early_stopping_iter=4000,
                 slice_length=seq_length, 
                 slice_index=224,
                 use_semifreddo=True,
                 saved_tmp_out="/scratch1/smaruj/ledidi_targets/full_tower_out.pt",
                 return_history=True
                 ).cuda()

# slice to be edited
slice_torch = X[:,:,start_index : end_index]
# flanking sequences
X_l_flank = X[:,:,start_index - 2048*2 : start_index]
X_r_flank = X[:,:,end_index : end_index + 2048*2]

x_bar, history = wrapper.fit_transform(slice_torch, y_bar, X_l_flank=X_l_flank, X_r_flank=X_r_flank)

