import imp
from .dataset_helper import dataset
from torch_geometric.datasets import Planetoid,GEDDataset
import torch_geometric.transforms as T
import torch
from .data_utils import *
from torch_geometric.transforms import OneHotDegree
from torch_geometric.utils import degree
from torch_geometric.loader import DataLoader
import numpy as np
from .CustomDataset import GEDDataset_Custom

class DatasetLocal(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    mask = None
    feat_transform = None
    recache = False

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)


    def get_data_mask(self, idx = None):
        if idx is None:
            return self.data
        else:
            mask = idx
            data = self.data.clone()
            data.train_mask = self.data.train_mask[:,mask]
            data.val_mask = self.data.val_mask[:,mask]
            data.test_mask = self.data.test_mask[:, mask]
            return data


    def load(self,config):
        if config['feat_norm']:
            self.feat_transform = T.NormalizeFeatures()

        if self.dataset_name in ['Cora']:
            dataset = Planetoid(root=self.dataset_source_folder_path, name = self.dataset_name, transform=self.feat_transform)
        
        if self.dataset_name in ['AIDS700nef', 'LINUX', 'IMDBMulti','ALKANE']:
            self.trainval_graphs = GEDDataset(   # 560
                self.dataset_source_folder_path + "/{}".format(self.dataset_name),
                self.dataset_name,
                train=True)
            if config['use_val']:
                val_ratio = config['val_ratio']
                num_trainval_gs = len(self.trainval_graphs)
                self.val_graphs = self.trainval_graphs[-int(num_trainval_gs*val_ratio):]   # 140
                self.training_graphs = self.trainval_graphs[0: -int(num_trainval_gs*val_ratio)]   # 420
        self.testing_graphs = GEDDataset(
            self.dataset_source_folder_path + "/{}".format(self.dataset_name),
            self.dataset_name,
            train=False)
        
        if self.dataset_name == 'ALKANE':
            self.testing_graphs = GEDDataset(
                self.dataset_source_folder_path + "/{}".format(self.dataset_name),
                self.dataset_name,
                train=True) 
        self.trainval_nged_matrix    = self.trainval_graphs.norm_ged
        self.trainval_ged_matrix     = self.trainval_graphs.ged
        self.real_trainval_data_size = self.trainval_nged_matrix.size(0)   # 700
        self.num_graphs              = len(self.trainval_graphs) + len(self.testing_graphs)
        self.num_train_graphs        = len(self.training_graphs)
        self.num_val_graphs          = len(self.val_graphs)
        self.num_test_graphs         = len(self.testing_graphs)


        # if config['use_val']:
        #     self.validation_triples = self.load_val_train_pairs()

        if config['synth']:
            self.synth_data_1, self.synth_data_2, _, synth_nged_matrix = gen_pairs(
                self.trainval_graphs.shuffle()[:500], 0, 3
            )
            real_data_size = self.nged_matrix.size(0) 
            synth_data_size = synth_nged_matrix.size(0)  
            self.nged_matrix = torch.cat(   
                (
                    self.nged_matrix,  
                    torch.full((real_data_size, synth_data_size), float("inf")),  
                ),
                dim=1,
            )
            synth_nged_matrix = torch.cat(
                (
                    torch.full((synth_data_size, real_data_size), float("inf")),  
                    synth_nged_matrix,
                ),
                dim=1,
            )

            """
            560*560 train  | 560*500 inf
           ----------------|-------------
            500*560 inf    | 500*500 diag     
            """
            self.nged_matrix = torch.cat((self.nged_matrix, synth_nged_matrix))  

        if self.trainval_graphs[0].x is None:  
            max_degree = 0
            for g in (
                self.trainval_graphs
                + self.testing_graphs
                + (self.synth_data_1 + self.synth_data_2 if config['synth'] else [])
            ):
                if g.edge_index.size(1) > 0:
                    max_degree = max(
                        max_degree, int(degree(g.edge_index[0]).max().item())
                    )
            one_hot_degree = OneHotDegree(max_degree, cat=False)
            self.trainval_graphs.transform = one_hot_degree
            self.val_graphs.transform = one_hot_degree
            self.training_graphs.transform = one_hot_degree
            self.testing_graphs.transform = one_hot_degree

            if config['synth']:
                for g in self.synth_data_1 + self.synth_data_2:
                    g = one_hot_degree(g)
                    g.i = g.i + real_data_size
        elif config['synth']:
            for g in self.synth_data_1 + self.synth_data_2:
                g.i = g.i + real_data_size

        self.number_of_labels = self.trainval_graphs.num_features
        self.input_dim = self.number_of_labels

    def load_custom_data(self, config, args):
        self.custom_dataset = GEDDataset_Custom(ged_main_dir=self.dataset_source_folder_path, config = config)


    def create_batches(self, config):

        if config['synth']:
            synth_data_ind = random.sample(range(len(self.synth_data_1)), 100)

        source_loader = DataLoader(
            self.training_graphs.shuffle()
            + (
                [self.synth_data_1[i] for i in synth_data_ind]
                if config['synth']
                else []
            ),
            batch_size=config['batch_size'], num_workers = config.get('num_works', 0)

        )

        target_loader = DataLoader(
            self.training_graphs.shuffle()
            + (
                [self.synth_data_2[i] for i in synth_data_ind]
                if config['synth']
                else []
            ),
            batch_size=config['batch_size'],num_workers = config.get('num_works', 0)
        )
        
        return list(zip(source_loader, target_loader))

    def transform_batch(self, batch, config):
        """
        Getting ged for graph pair and grouping with data into dictionary.
        :param data: Graph pair.
        :return new_data: Dictionary with data.
        """
        new_data = dict()

        new_data["g1"] = batch[0]  # DataBatch(edge_index=[2, 2254], i=[128], x=[1146, 29], num_nodes=1146, batch=[1146], ptr=[129])
        new_data["g2"] = batch[1] 

        normalized_ged = self.trainval_nged_matrix[
            batch[0]["i"].reshape(-1).tolist(), batch[1]["i"].reshape(-1).tolist()
        ].tolist()
        new_data["target"] = (
            torch.from_numpy(np.exp([(-el * config.get('scale', 1)) for el in normalized_ged])).view(-1).float()
        )
        new_data['norm_ged'] = (
            torch.from_numpy(np.exp([(el) for el in normalized_ged])).view(-1).float()
        )
        ged = self.trainval_ged_matrix[
            batch[0]["i"].reshape(-1).tolist(), batch[1]["i"].reshape(-1).tolist()
        ].tolist()

        new_data["target_ged"] = (
            torch.from_numpy(np.array([(el) for el in ged])).view(-1).float()   # nged
        )
        return new_data



    def load_val_train_pairs(self):
        val_len = len(self.val_graphs)
        train_len = len(self.training_graphs)

        val_pairs_triples = []
        for m in range(val_len):
            g1 = self.val_graphs[m]
            for n in range(train_len):
                g2 = self.training_graphs[n]
                nged = self.trainval_nged_matrix[g1["i"], g2["i"]]
                ged = self.trainval_ged_matrix[g1["i"], g2["i"]]
                val_pairs_triples.append([g1, g2, nged, ged])
        return val_pairs_triples

    def generate_all_val_gs(self, config):

        #print(self.val_graphs[0])Data(edge_index=[2, 20], i=[1], x=[10, 29], num_nodes=10)
        source_gs = []
        target_gs = []

        for i in range(len(self.validation_triples)):
            g1, g2, nged, ged = self.validation_triples[i]
            source_gs.append(g1)
            target_gs.append(g2)
        
        source_val_loader = DataLoader(source_gs, batch_size=config['val_batch_size'])
        
        target_val_loader = DataLoader(target_gs, batch_size=config['val_batch_size'])

        return list(zip(source_val_loader, target_val_loader))