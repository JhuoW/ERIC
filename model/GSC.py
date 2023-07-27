from distutils.command.config import config
from platform import node
from turtle import forward
from gpustat import print_gpustat
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool, global_mean_pool
from model.layers import AttentionModule, MLPLayers, TensorNetworkModule, FF
from utils.gan_losses import get_negative_expectation, get_positive_expectation
from collections import OrderedDict, defaultdict
import numpy as np


class GSC(nn.Module):
    def __init__(self, config, n_feat):
        super(GSC, self).__init__()
        self.config                     = config
        self.n_feat                     = n_feat
        self.setup_layers()
        self.setup_score_layer()
        if config['dataset_name']== 'IMDBMulti':
            self.scale_init()

    def setup_layers(self):
        gnn_enc                         = self.config['gnn_encoder']
        self.filters                    = self.config['gnn_filters']
        self.num_filter                 = len(self.filters)
        self.use_ssl                    = self.config.get('use_ssl', False)


        if self.config['fuse_type']     == 'stack':
            filters                     = []
            for i in range(self.num_filter):
                filters.append(self.filters[0])
            self.filters                = filters
        self.gnn_list                   = nn.ModuleList()
        self.mlp_list_inner             = nn.ModuleList()  
        self.mlp_list_outer             = nn.ModuleList()  
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
                if self.config['inner_mlp']:
                    if self.config.get('inner_mlp_layers', 1) == 1:
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                    else:
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i], num_layers=self.config['inner_mlp_layers'], use_bn=False))
                if self.config.get('outer_mlp_layers', 1)     == 1:
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                else:
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i], num_layers=self.config['outer_mlp_layers'], use_bn=False))
                self.act_inner                 = getattr(F, self.config.get('deepsets_inner_act', 'relu'))
                self.act_outer                 = getattr(F, self.config.get('deepsets_outer_act', 'relu'))
                if self.config['use_sim'] and self.config['NTN_layers'] != 1:
                    self.NTN_list.append(TensorNetworkModule(self.config, self.filters[i]))
            if self.config['use_sim'] and self.config['NTN_layers'] == 1:
                self.NTN                       = TensorNetworkModule(self.config, self.filters[self.num_filter-1])

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
        if self.use_ssl:
            self.GCL_model                      = GCL(self.config, sum(self.filters))
            self.gamma                          = nn.Parameter(torch.Tensor(1)) 
            
            

    def setup_score_layer(self):
        if self.config['deepsets']:
            if self.config['fuse_type']                  == 'cat':
                self.score_layer                         = nn.Sequential(nn.Linear((self.channel_dim // self.reduction) , 16),
                                                                        nn.ReLU(),
                                                                        nn.Linear(16 , 1))
            elif self.config['fuse_type']                == 'stack': 
                self.score_layer                         = nn.Linear(self.filters[0], 1)
            if self.config['use_sim']:
                if self.config['NTN_layers']!=1:
                    self.score_sim_layer                 = nn.Sequential(nn.Linear(self.config['tensor_neurons'] * self.num_filter, self.config['tensor_neurons']),
                                                                        nn.ReLU(),
                                                                        nn.Linear(self.config['tensor_neurons'], 1))
                else:
                    self.score_sim_layer                 = nn.Sequential(nn.Linear(self.config['tensor_neurons'], self.config['tensor_neurons']),
                                                                        nn.ReLU(),
                                                                        nn.Linear(self.config['tensor_neurons'], 1))

        if self.config.get('output_comb', False):
            self.alpha                                   = nn.Parameter(torch.Tensor(1))
            self.beta                                    = nn.Parameter(torch.Tensor(1))

    def scale_init(self):
        nn.init.zeros_(self.gamma)
        nn.init.zeros_(self.alpha)
        nn.init.zeros_(self.beta)

    def convolutional_pass_level(self, enc, edge_index, x):
        feat                                             = enc(x, edge_index)
        feat                                             = F.relu(feat)
        feat                                             = F.dropout(feat, p = self.config['dropout'], training=self.training)
        return feat

    def deepsets_outer(self, batch, feat, filter_idx, size = None):
        size                                             = (batch[-1].item() + 1 if size is None else size)   # 一个batch中的图数

        pool                                             = global_add_pool(feat, batch, size=size) if self.config['pooling']=='add' else global_mean_pool(feat, batch, size=size)
        return self.act_outer(self.mlp_list_outer[filter_idx](pool))

    def collect_embeddings(self, all_graphs):
        node_embs_dict                                   = dict()  
        for g in all_graphs:
            feat = g.x.cuda()
            edge_index = g.edge_index.cuda()
            for i, gnn in enumerate(self.gnn_list):
                if i not in node_embs_dict.keys():
                    node_embs_dict[i] = dict()
                feat                                     = gnn(feat, edge_index)  
                feat                                     = F.relu(feat)        
                node_embs_dict[i][int(g['i'])] = feat
        return node_embs_dict

    def collect_graph_embeddings(self, all_graphs):
        node_embs_dict = self.collect_embeddings(all_graphs)
        graph_embs_dicts = dict()
        for i in node_embs_dict.keys():
            if i not in graph_embs_dicts.keys():
                graph_embs_dicts[i]          = dict()  
            for k, v in node_embs_dict[i].items():   
                deepsets_inner = self.act_inner(self.mlp_list_inner[i](v))
                g_emb            = torch.sum(deepsets_inner, dim=0)
                graph_embs_dicts[i][k] = g_emb   

        return graph_embs_dicts

    def forward(self,data):
        edge_index_1            = data['g1'].edge_index.cuda()
        edge_index_2            = data['g2'].edge_index.cuda()
        features_1              = data["g1"].x.cuda()
        features_2              = data["g2"].x.cuda()
        batch_1                 = (
                                    data["g1"].batch.cuda()
                                    if hasattr(data["g1"], "batch")
                                    else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).cuda()
                                    )
        batch_2                 = (
                                    data["g2"].batch.cuda()
                                    if hasattr(data["g2"], "batch")
                                    else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).cuda()
                                    )
        
        conv_source_1           = torch.clone(features_1)
        conv_source_2           = torch.clone(features_2)
        for i in range(self.num_filter):  # 分层contrast
            conv_source_1       = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)  # 一个
            
            conv_source_2       = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)
            if self.config['deepsets']: 
                if self.config.get('inner_mlp', True):
                    deepsets_inner_1 = self.act_inner(self.mlp_list_inner[i](conv_source_1)) # [1147, 64]
                    deepsets_inner_2 = self.act_inner(self.mlp_list_inner[i](conv_source_2))

                else:
                    deepsets_inner_1 = self.act_inner(conv_source_1)
                    deepsets_inner_2 = self.act_inner(conv_source_2)
                deepsets_outer_1     = self.deepsets_outer(batch_1, deepsets_inner_1,i)
                deepsets_outer_2     = self.deepsets_outer(batch_2, deepsets_inner_2,i)

                if self.config['fuse_type']=='cat':
                    diff_rep         = torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat((diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2,2))), dim = 1)  
                elif self.config['fuse_type']=='stack':  # (128, 3, 1, 64)  batch_size = 128  channel  = num_filters, size= 1*64
                    diff_rep         = torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1) if i == 0 else torch.cat((diff_rep, torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1)), dim=1)   # (128,3,64)
            
                if self.config['use_sim'] and self.config['NTN_layers']!=1:
                    sim_rep          = self.NTN_list[i](deepsets_outer_1, deepsets_outer_2) if i == 0 else torch.cat((sim_rep, self.NTN_list[i](deepsets_outer_1, deepsets_outer_2)), dim = 1)  # (128, 16+16+16)
                
                if self.use_ssl:
                    cat_node_embeddings_1   = conv_source_1 if i == 0 else torch.cat((cat_node_embeddings_1, conv_source_1), dim = 1)
                    cat_node_embeddings_2   = conv_source_2 if i == 0 else torch.cat((cat_node_embeddings_2, conv_source_2), dim = 1)
                    cat_global_embedding_1  = deepsets_outer_1 if i == 0 else torch.cat((cat_global_embedding_1, deepsets_outer_1), dim = 1)
                    cat_global_embedding_2  = deepsets_outer_2 if i == 0 else torch.cat((cat_global_embedding_2, deepsets_outer_2), dim = 1)

        L_cl = 0
        if not self.training:
            self.use_ssl = False
        if self.use_ssl:
            if self.config['use_deepsets']:
                L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2, g1 = cat_global_embedding_1, g2 = cat_global_embedding_2) * self.gamma
            else:
                L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2) * self.gamma

            if self.config.get('cl_loss_norm', False):
                if self.config.get('norm_type', 'sigmoid') == 'sigmoid':
                    L_cl = torch.sigmoid(L_cl)
                elif self.config.get('norm_type', 'sigmoid') == 'sum':
                    L_cl = torch.pow(L_cl, 2).sqrt()
                elif self.config.get('norm_type', 'sigmoid') == 'tanh':
                    L_cl = torch.tanh(L_cl)
                else:
                    raise "Norm Error"
        
        score_rep = self.conv_stack(diff_rep).squeeze()  # (128,64)

        if self.config['use_sim'] and self.config['NTN_layers']==1:
            sim_rep = self.NTN(deepsets_outer_1, deepsets_outer_2)

        if self.config['use_sim']:
            sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())

        score = torch.sigmoid(self.score_layer(score_rep)).view(-1)
            
        if self.config.get('use_sim', False):
            if self.config.get('output_comb', False):
                comb_score = self.alpha * score + self.beta * sim_score
                return comb_score, L_cl
            else:
                return (score + sim_score)/2 , L_cl
        else:
            return score , L_cl
                
class GCL(nn.Module):

    def __init__(self, config, embedding_dim):
        super(GCL, self).__init__()
        self.config        = config
        self.use_deepsets  = config['use_deepsets']
        self.use_ff        = config['use_ff']
        self.embedding_dim = embedding_dim
        self.measure       = config['measure']
        if self.use_ff:
            self.local_d   = FF(self.embedding_dim)   
            self.global_d  = FF(self.embedding_dim)
        self.init_emb()


    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch_1, batch_2, z1, z2, g1 = None, g2 = None):

        if not self.use_deepsets:
            g1       = global_add_pool(z1, batch_1)
            g2       = global_add_pool(z2, batch_2)

        num_graphs_1 = g1.shape[0]
        num_nodes_1  = z1.shape[0]
        pos_mask_1   = torch.zeros((num_nodes_1, num_graphs_1)).cuda()
        num_graphs_2 = g2.shape[0]
        num_nodes_2  = z2.shape[0]
        pos_mask_2   = torch.zeros((num_nodes_2, num_graphs_2)).cuda()
        for node_idx, graph_idices in enumerate(zip(batch_1, batch_2)): 
            g_idx_1, g_idx_2                 = graph_idices
            pos_mask_1[node_idx][g_idx_1]    = 1.
            pos_mask_2[node_idx][g_idx_2]    = 1.
        
        if self.config.get('norm', False):
            z1 = F.normalize(z1, dim=1)
            g1 = F.normalize(g1, dim=1)
            z2 = F.normalize(z2, dim=1)
            g2 = F.normalize(g2, dim=1)
        self_sim_1   = torch.mm(z1,g1.t())   * pos_mask_1  
        self_sim_2   = torch.mm(z2,g2.t())   * pos_mask_2
        cross_sim_12 = torch.mm(z1,g2.t())   * pos_mask_1   
        cross_sim_21 = torch.mm(z2,g1.t())   * pos_mask_2
        # get_positive_expectation(self_sim_1,  self.measure, average=False)

        if self.config['sep']:
            self_js_sim_11   = get_positive_expectation(self_sim_1,  self.measure, average=False).sum(1) 
            cross_js_sim_12  = get_positive_expectation(cross_sim_12,self.measure, average=False).sum(1)  

            self_js_sim_22   = get_positive_expectation(self_sim_2,  self.measure, average=False).sum(1)
            cross_js_sim_21  = get_positive_expectation(cross_sim_21, self.measure, average=False).sum(1)
            L_1              = (self_js_sim_11-cross_js_sim_12).pow(2).sum().sqrt()   
            L_2              = (self_js_sim_22-cross_js_sim_21).pow(2).sum().sqrt()
            return           L_1 + L_2
        else:
            L_1              = get_positive_expectation(self_sim_1,  self.measure, average=False).sum()- get_positive_expectation(cross_sim_12,self.measure, average=False).sum()
            L_2              = get_positive_expectation(self_sim_2,  self.measure, average=False).sum()- get_positive_expectation(cross_sim_21, self.measure, average=False).sum()
            return           L_1 - L_2



if __name__ == "__main__":
    pass