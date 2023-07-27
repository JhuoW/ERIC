import torch
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph, to_undirected, to_networkx
import random

def to_directed(edge_index):
    row, col = edge_index
    mask = row < col
    row, col = row[mask], col[mask]
    return torch.stack([row, col], dim=0)

def gen_pair(g, kl=None, ku=2):
    if kl is None:
        kl = ku

    directed_edge_index = to_directed(g.edge_index)

    n = g.num_nodes
    num_edges = directed_edge_index.size()[1]
    to_remove = random.randint(kl, ku) 

    edge_index_n = directed_edge_index[:, torch.randperm(num_edges)[to_remove:]]   
    if edge_index_n.size(1) != 0:  
        edge_index_n = to_undirected(edge_index_n)

    row, col = g.edge_index
    adj = torch.ones((n, n), dtype=torch.uint8)  
    adj[row, col] = 0  
    non_edge_index = adj.nonzero().t()  # 

    directed_non_edge_index = to_directed(non_edge_index)
    num_edges = directed_non_edge_index.size()[1]

    to_add = random.randint(kl, ku)

    edge_index_p = directed_non_edge_index[:, torch.randperm(num_edges)[:to_add]]  
    if edge_index_p.size(1):
        edge_index_p = to_undirected(edge_index_p)
    edge_index_p = torch.cat((edge_index_n, edge_index_p), 1)

    if hasattr(g, "i"):
        g2 = Data(x=g.x, edge_index=edge_index_p, i=g.i)
    else:
        g2 = Data(x=g.x, edge_index=edge_index_p)

    g2.num_nodes = g.num_nodes
    return g2, to_remove + to_add


def gen_pairs(graphs, kl=None, ku=2):

    gen_graphs_1 = []  
    gen_graphs_2 = []  

    count = len(graphs)
    mat = torch.full((count, count), float("inf")) 
    norm_mat = torch.full((count, count), float("inf"))

    for i, g in enumerate(graphs):
        g = g.clone()
        g.i = torch.tensor([i])  # 图的id
        g2, ged = gen_pair(g, kl, ku)
        gen_graphs_1.append(g)
        gen_graphs_2.append(g2)
        mat[i, i] = ged
        norm_mat[i, i] = ged / (0.5 * (g.num_nodes + g2.num_nodes))

    return gen_graphs_1, gen_graphs_2, mat, norm_mat


    