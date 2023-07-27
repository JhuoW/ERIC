from turtle import forward
import torch.nn.functional as F
import torch
import torch.nn as nn 


class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, num_layers = 2 ,use_bn=True):
        super(MLP, self).__init__()

        modules = []
        modules.append(nn.Linear(nfeat, nhid, bias=True))
        if use_bn:
            modules.append(nn.BatchNorm1d(nhid))
        modules.append(nn.ReLU())
        for i in range(num_layers-2):
            modules.append(nn.Linear(nhid, nhid, bias=True))
            if use_bn:
               modules.append(nn.BatchNorm1d(nhid)) 
            modules.append(nn.ReLU())


        modules.append(nn.Linear(nhid, nclass, bias=True))
        self.mlp_list = nn.Sequential(*modules)

    def forward(self, x):
        x = self.mlp_list(x)
        return x

class MLPLayers(nn.Module):
    def __init__(self, n_in, n_hid, n_out, num_layers = 2 ,use_bn=True):
        super(MLPLayers, self).__init__()
        modules = []
        modules.append(nn.Linear(n_in, n_hid))
        out = n_hid
        use_act = True
        for i in range(num_layers-1):  # num_layers = 3  i=0,1
            if i == num_layers-2:
                use_bn = False
                use_act = False
                out = n_out
            modules.append(nn.Linear(n_hid, out))
            if use_bn:
                modules.append(nn.BatchNorm1d(out)) 
            if use_act:
                modules.append(nn.ReLU())
        self.mlp_list = nn.Sequential(*modules)
    def forward(self,x):
        x = self.mlp_list(x)
        return x
