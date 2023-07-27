import imp
import torch
import torch.nn as nn
from torch_geometric.nn.glob import global_mean_pool, global_add_pool
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

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
    def __init__(self, n_in, n_hid, n_out, num_layers = 2 ,use_bn=True, act = 'relu'):
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

class AttentionModule2(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, config):
        """
        :param args: Arguments object.
        """
        super(AttentionModule2, self).__init__()
        self.config = config
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.config['filters_3'], self.config['filters_3'])
        )

    def init_parameters(self):
        """
        Initializing weights.
        """
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: Result of the GNN.
        :param size: Dimension size for scatter_
        :param batch: Batch vector, which assigns each node to a specific example
        :return representation: A graph level representation matrix.
        """
        size = batch[-1].item() + 1 if size is None else size
        mean = scatter_mean(x, batch, dim=0, dim_size=size)
        transformed_global = torch.tanh(torch.mm(mean, self.weight_matrix))

        coefs = torch.sigmoid((x * transformed_global[batch]).sum(dim=1))
        weighted = coefs.unsqueeze(-1) * x

        return scatter_add(weighted, batch, dim=0, dim_size=size)

    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))

class AttentionModule(torch.nn.Module):
    def __init__(self, config, dim_size):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.config = config
        self.dim_size = dim_size
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.dim_size, self.dim_size)
        )
        self.weight_matrix1 = torch.nn.Parameter(
            torch.Tensor(self.dim_size, self.dim_size)
        )

        channel = self.dim_size * 1
        reduction = 4
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Tanh(),
        )

        self.fc1 = nn.Linear(channel, channel)

    def init_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x, batch, size=None):
        attention = self.fc(x) # 节点的GIN feature 带入eq(4)
        x = attention * x + x  # eq(4)+ skip conn

        size = (
            batch[-1].item() + 1 if size is None else size
        )  # size is the quantity of batches: 128 eg
        mean = global_mean_pool(x, batch, size=size)  # dim of mean: 128 * 16  # 每个图做mean pool

        transformed_global = torch.tanh(
            torch.mm(mean, self.weight_matrix)
        )  # * global context c

        coefs = torch.sigmoid(
            (x * transformed_global[batch]).sum(dim=1)  # * graph-level embedding coef
        )  # transformed_global[batch]: 1128 * 16; coefs: 1128 * 0
        weighted = coefs.unsqueeze(-1) * x  # * graph level embedding

        return global_add_pool(weighted, batch, size = size)  # 128 * 16

    def get_coefs(self, x):
        mean = x.mean(dim=0)
        transformed_global = torch.tanh(torch.matmul(mean, self.weight_matrix))

        return torch.sigmoid(torch.matmul(x, transformed_global))

class TensorNetworkModule(torch.nn.Module):

    def __init__(self, config, filters):

        super(TensorNetworkModule, self).__init__()
        self.config = config
        self.filters = filters
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):

        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(
                self.filters, self.filters, self.config['tensor_neurons']
            )
        )
        self.weight_matrix_block = torch.nn.Parameter(
            torch.Tensor(self.config['tensor_neurons'], 2 * self.filters)
        )
        self.bias = torch.nn.Parameter(torch.Tensor(self.config['tensor_neurons'], 1))

    def init_parameters(self):

        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):

        batch_size = len(embedding_1)
        scoring = torch.matmul(
            embedding_1, self.weight_matrix.view(self.filters, -1)
        )
        scoring = scoring.view(batch_size, self.filters, -1).permute([0, 2, 1])
        scoring = torch.matmul(
            scoring, embedding_2.view(batch_size, self.filters, 1)
        ).view(batch_size, -1)
        combined_representation = torch.cat((embedding_1, embedding_2), 1)
        block_scoring = torch.t(
            torch.mm(self.weight_matrix_block, torch.t(combined_representation))
        )
        scores = F.relu(scoring + block_scoring + self.bias.view(-1))
        return scores



class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)
