from turtle import forward
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool, global_mean_pool
from model.layers import AttentionModule, MLPLayers, TensorNetworkModule


class GSC(nn.Module):
    def __init__(self, config, n_feat):
        super(GSC, self).__init__()
        self.config                     = config
        self.n_feat                     = n_feat
        self.setup_layers()
        self.setup_score_layer()

    def setup_layers(self):
        gnn_enc                         = self.config['gnn_encoder']
        self.filters                    = self.config['gnn_filters']
        self.n_layers_deepsets          = self.config['n_layers_deepsets']
        self.num_filter                 = len(self.filters)
        if self.config['fuse_type']     == 'stack':
            filters                     = []
            for i in range(self.num_filter):
                filters.append(self.filters[0])
            self.filters                = filters
        self.gnn_list                   = nn.ModuleList()
        self.mlp_list_inner             = nn.ModuleList()  # 内层GNN
        self.mlp_list_outer             = nn.ModuleList()  # 外层GNN
        self.NTN_list                   = nn.ModuleList()

        if gnn_enc                      == 'GCN':  # append
            self.gnn_list.append(GCNConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GCNConv(self.filters[i],self.filters[i+1]))
        elif gnn_enc                    == 'GAT':
            self.gnn_list.append(GATConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GATConv(self.filters[i],self.filters[i+1]))  
        elif gnn_enc                    == 'GIN':
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.n_feat, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ),eps=True))

            for i in range(self.num_filter-1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.filters[i],self.filters[i+1]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[i+1], self.filters[i+1]),
                torch.nn.BatchNorm1d(self.filters[i+1]),
            ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")
        # if not self.config['multi_deepsets']:
        if self.config['deepsets']:
            for i in range(self.num_filter):
                self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                self.act_inner                 = getattr(F, self.config.get('deepsets_inner_act', 'relu'))
                self.act_outer                 = getattr(F, self.config.get('deepsets_outer_act', 'relu'))
                if self.config['use_sim'] and self.config['NTN_layers'] != 1:
                    self.NTN_list.append(TensorNetworkModule(self.config, self.filters[i]))
            if self.config['use_sim'] and self.config['NTN_layers'] == 1:
                self.NTN = TensorNetworkModule(self.config, self.filters[self.num_filter-1])
            # if self.config['use_mlp_score']:
            #     self.mlp_out_score
            #     for i in range(self.num_filter):
            #         # 第i层GNN的MLP输出层
            if self.config['fuse_type']        == 'cat':
                self.channel_dim               = sum(self.filters)
                self.reduction                 = self.config['reduction']
                self.conv_stack                = nn.Sequential(
                                                                nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                                                                nn.ReLU(),
                                                                nn.Dropout(p = self.config['dropout']),
                                                                nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction) ),
                                                                nn.Dropout(p = self.config['dropout']),
                                                                nn.Tanh(),
                                                            )

            elif self.config['fuse_type']      == 'stack': 
                self.conv_stack                = nn.Sequential(
                    nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
                if self.config['use_sim']:
                    self.NTN                   = TensorNetworkModule(self.config, self.filters[0])
            elif self.config['fuse_type']      == 'add':
                pass
            else:
                raise RuntimeError(
                    'unsupported fuse type') 
        
            

    def setup_score_layer(self):
        if self.config['deepsets']:
            if self.config['fuse_type']             == 'cat':
                self.score_layer                    = nn.Sequential(nn.Linear((self.channel_dim // self.reduction) , 16),
                                                                        nn.ReLU(),
                                                                        nn.Linear(16 , 1))
            elif self.config['fuse_type']           == 'stack': 
                self.score_layer                    = nn.Linear(self.filters[0], 1)
            if self.config['use_sim']:
                if self.config['NTN_layers']!=1:
                    self.score_sim_layer                = nn.Sequential(nn.Linear(self.config['tensor_neurons'] * self.num_filter, self.config['tensor_neurons']),
                                                                        nn.ReLU(),
                                                                        nn.Linear(self.config['tensor_neurons'], 1))
                else:
                    self.score_sim_layer = nn.Sequential(nn.Linear(self.config['tensor_neurons'], self.config['tensor_neurons']),
                                                                        nn.ReLU(),
                                                                        nn.Linear(self.config['tensor_neurons'], 1))

    def convolutional_pass_level(self, enc, edge_index, x):
        feat = enc(x, edge_index)
        feat = F.relu(feat)
        feat = F.dropout(feat, p = self.config['dropout'], training=self.training)
        return feat

    def deepsets_outer(self, batch, feat, filter_idx, size = None):
        size = (
            batch[-1].item() + 1 if size is None else size   # 一个batch中的图数
        )
        sum_ = global_add_pool(feat, batch, size=size)
        return self.act_outer(self.mlp_list_outer[filter_idx](sum_))

    def forward(self,data):
        # DataBatch(edge_index=[2, 2304], i=[128], x=[1158, 29], num_nodes=1158, batch=[1158], ptr=[129])   batch中存放每个节点在那个batch
        edge_index_1 = data['g1'].edge_index.cuda()
        edge_index_2 = data['g2'].edge_index.cuda()
        features_1 = data["g1"].x.cuda()
        features_2 = data["g2"].x.cuda()
        batch_1 = (
            data["g1"].batch.cuda()
            if hasattr(data["g1"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).cuda()
        )
        batch_2 = (
            data["g2"].batch.cuda()
            if hasattr(data["g2"], "batch")
            else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).cuda()
        )
        
        conv_source_1 = torch.clone(features_1)
        conv_source_2 = torch.clone(features_2)
        for i in range(self.num_filter):  # 分层contrast
            conv_source_1 = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)  # 一个
            
            conv_source_2 = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)
            ## channel level contrast 聚合过一次后完全一样的节点对GED的贡献小一些， 聚合过两次后完全一样的节点对GED的贡献更小， 聚合过三次后完全一样的节点对GED几乎没有贡献
            # 特征分布上的差异来反映GED
            # Gaussian mixture model /置换不变的函数
            # Deep Set / Global
            if self.config['deepsets']: 
                # 不相似部分和nged做匹配
                deepsets_inner_1 = self.act_inner(self.mlp_list_inner[i](conv_source_1)) # [1147, 64]
                deepsets_inner_2 = self.act_inner(self.mlp_list_inner[i](conv_source_2))
                deepsets_outer_1 = self.deepsets_outer(batch_1, deepsets_inner_1,i)
                deepsets_outer_2 = self.deepsets_outer(batch_2, deepsets_inner_2,i)

                if self.config['fuse_type']=='cat':
                    diff_rep = torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat((diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2,2))), dim = 1)  
                    # score_rep = self.conv_stack(diff_rep)  #  [128, 64+32+16]
                elif self.config['fuse_type']=='stack':  # (128, 3, 1, 64)  batch_size = 128  channel  = num_filters, size= 1*64
                    # (128,1,64)
                    diff_rep = torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1) if i == 0 else torch.cat((diff_rep, torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1)), dim=1)   # (128,3,64)
            
                if self.config['use_sim'] and self.config['NTN_layers']!=1:
                    # 相似部分和 nged做匹配  NTN
                    sim_rep = self.NTN_list[i](deepsets_outer_1, deepsets_outer_2) if i == 0 else torch.cat((sim_rep, self.NTN_list[i](deepsets_outer_1, deepsets_outer_2)), dim = 1)  # (128, 16+16+16)
        
        score_rep = self.conv_stack(diff_rep).squeeze()  # (128,64)

        if self.config['use_sim'] and self.config['NTN_layers']==1:
            sim_rep = self.NTN(deepsets_outer_1, deepsets_outer_2)

        if self.config['use_sim']:
            sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())

        score = torch.sigmoid(self.score_layer(score_rep)).view(-1)
            
        # print(diff_rep.shape) [128, 64+32+16]
        if self.config['use_sim']:
            return (score + sim_score)/2
        else:
            return score
                


class Classifier(nn.Module):
    def __init__(self, config, in_dim):
        super(Classifier, self).__init__()
        self.score_layer = nn.Linear(in_dim, 1)

    def forward(self, score_vec):
        score = torch.sigmoid(self.score_layer(score_vec)).view(-1)
        return score





if __name__ == "__main__":
    pass