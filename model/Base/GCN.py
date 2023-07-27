import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv, JumpingKnowledge
from torch_sparse.tensor import SparseTensor
from typing import Tuple
from torch_scatter import scatter_add

class GCN(torch.nn.Module):
    def __init__(self,config, hidden_channels, num_features, num_classes):
        super().__init__()
        self.num_layers = config['num_layers']
        self.net_layers = nn.ModuleList([
            GCNConv(num_features, hidden_channels)]
        )
        self.net_layers.append(GCNConv(hidden_channels, num_classes))
        self.dropout = config['dropout']
        self.activation = getattr(F, config['activation'])

        # self.conv1 = ChebConv(data.num_features, 16, K=2)
        # self.conv2 = ChebConv(16, data.num_features, K=2)

    def forward(self, x, edge_index):
        for i in range(self.num_layers-1):
            x = self.net_layers[i](x, edge_index)
            x = self.activation(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.net_layers[-1](x, edge_index)
        return x