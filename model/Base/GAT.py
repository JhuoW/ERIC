import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATConv

class GAT(nn.Module):
    def __init__(self,config, hid_channels,  in_channels, out_channels):
        super(GAT, self).__init__()
        self.dropout = config['dropout']
        self.conv1 = GATConv(in_channels, hid_channels, heads=config['heads'], dropout=self.dropout)
        self.conv2 = GATConv(config['heads'] * hid_channels, out_channels, heads=1, concat=False,
                             dropout=self.dropout)
    
    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x