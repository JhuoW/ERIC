import os
import DataHelper.simgnn_utils as simgnn_utils
import numpy as np
import networkx as nx

node_encoder_name_dict = {
    'aids700nef': 'OneHot',
    'linux': 'constant_1_2'
}
node_feat_name_dict = {
    'aids700nef': 'type',
    'linux': None
}
class GEDDataset_Custom(object):
    def __init__(self, ged_main_dir, config):
        self.config = config
        self.ged_main_dir = ged_main_dir
        data = simgnn_utils.SiameseDataSet(data_save_folder=os.path.join(self.ged_main_dir, 'data'), data_set_name=config['dataset_name'], validation_ratio=0.25, node_feat_name=node_feat_name_dict[config['dataset_name']],
                                           node_encoder_name=node_encoder_name_dict[config['dataset_name']])
        distance = simgnn_utils.DistanceCalculator(root_folder=os.path.join(self.ged_main_dir, 'save'), data_name=data.data_set_name)
        
        self.data = data
        self.input_dim = data.input_dim()
        self.dist_calculator = distance
        
        self.train_val_max_number_of_nodes = data.train_val_max_number_of_nodes
        self.testing_max_number_of_nodes = data.test_max_number_of_nodes
        max_nr_nodes = max(self.train_val_max_number_of_nodes, self.testing_max_number_of_nodes)
        self.train_val_graphs = self.preprocess(data.train_val_gs, max_nr_nodes)
        self.training_graphs = self.preprocess(data.train_gs, self.data.train_val_max_number_of_nodes)
        self.val_graphs = self.preprocess(data.val_gs, self.data.train_val_max_number_of_nodes)
        self.testing_graphs = self.preprocess(data.test_gs, max_nr_nodes)
        
        true_result_test_train = simgnn_utils.load_result(dataset=data.data_set_name.lower(), model='astar', result_folder=os.path.join(self.ged_main_dir, 'result'), sim=None, sim_mat=None, dist_mat=None, row_graphs=self.testing_graphs,
                                                          col_graphs=self.train_val_graphs, scale=1.0)
        self.ground_truth = true_result_test_train.dist_norm_mat_
        self.training_triples = self._load_train_triples(dist_calculator=self.dist_calculator)
        self.validation_triples = self._load_val_train_triples(dist_calculator=self.dist_calculator)
        self.test_triples = self._load_test_triples()
    
    def extract_test_matrices(self, row):
        row_idx, col_idx = self.test_triples[row]
        batch_rows_feature = []
        batch_rows_adjacent = []
        batch_rows_mask = []
        batch_cols_feature = []
        batch_cols_adjacent = []
        batch_cols_mask = []
        for row, col in zip(row_idx, col_idx):
            row_feature, row_adj, row_mask = self.testing_graphs[row].matrices
            col_feature, col_adj, col_mask = self.train_val_graphs[col].matrices
            batch_rows_feature.append(row_feature)
            batch_rows_adjacent.append(row_adj)
            batch_rows_mask.append(row_mask)
            batch_cols_feature.append(col_feature)
            batch_cols_adjacent.append(col_adj)
            batch_cols_mask.append(col_mask)
        return np.array(batch_rows_feature), np.array(batch_rows_adjacent), np.array(batch_rows_mask), np.array(batch_cols_feature), np.array(batch_cols_adjacent), np.array(
            batch_cols_mask)
    
    def get_training_batch(self):
        
        all_feat_1 = []
        all_adj_1 = []
        all_feat_2 = []
        all_adj_2 = []
        
        all_masks_1 = []
        all_masks_2 = []
        
        ged_pairs = []
        
        for i in range(self.config['batch_size']):
            m, n, norm_dist = self.training_triples.get_next_one()
            
            feature_1, adj_1, mask_1 = self.training_graphs[m].matrices
            feature_2, adj_2, mask_2 = self.training_graphs[n].matrices
            
            all_feat_1.append(feature_1)
            all_adj_1.append(adj_1)
            all_masks_1.append(mask_1)
            
            all_feat_2.append(feature_2)
            all_adj_2.append(adj_2)
            all_masks_2.append(mask_2)
            
            ged_pairs.append(norm_dist)
        
        return np.array(all_feat_1), np.array(all_adj_1), np.array(all_masks_1), np.array(all_feat_2), np.array(all_adj_2), np.array(all_masks_2), np.array(ged_pairs)
    
    def get_all_validation(self):
        
        all_feat_1 = []
        all_adj_1 = []
        all_feat_2 = []
        all_adj_2 = []
        
        all_mask_1 = []
        all_mask_2 = []
        
        ged_pair = []
        
        for i in range(len(self.validation_triples)):
            m, n, norm_dist = self.validation_triples[i]
            
            feature_1, adj_1, mask_1 = self.val_graphs[m].matrices
            feature_2, adj_2, mask_2 = self.training_graphs[n].matrices
            
            all_feat_1.append(feature_1)
            all_adj_1.append(adj_1)
            all_mask_1.append(mask_1)
            
            all_feat_2.append(feature_2)
            all_adj_2.append(adj_2)
            all_mask_2.append(mask_2)
            
            ged_pair.append(norm_dist)
        
        return np.array(all_feat_1), np.array(all_adj_1), np.array(all_mask_1), np.array(all_feat_2), np.array(all_adj_2), np.array(all_mask_2), np.array(ged_pair)
    
    def preprocess(self, gs, max_num_nodes):
        rtn = []
        for i in range(len(gs)):
            g1 = gs[i]
            feature_1, adj_1, mask_1 = self._graph_feature_adj(g1, max_num_nodes)
            g1.matrices = (feature_1, adj_1, mask_1)
            rtn.append(g1)
        return rtn
    
    def _graph_feature_adj(self, gs, max_nodes):
        
        feature = self.data.node_feat_encoder.encode(gs)
        adj = np.array(nx.to_numpy_matrix(gs))
        adj = adj + np.where(adj.transpose() > adj, 1, 0)
        adj = adj + np.eye(adj.shape[0])
        
        adj_normalized = self._dense_adj_normalization(adj)
        
        feature_padded = np.zeros((max_nodes, feature.shape[-1]))
        adj_padded = np.zeros((max_nodes, max_nodes))
        
        feature_padded[:feature.shape[0], :feature.shape[1]] = feature
        adj_padded[:adj_normalized.shape[0], :adj_normalized.shape[1]] = adj_normalized
        assert feature.shape[0] == adj_normalized.shape[0]
        masked = np.zeros(max_nodes)
        masked[:feature.shape[0]] = 1
        
        return feature_padded, adj_padded, masked
    
    @staticmethod
    def _dense_adj_normalization(adj):
        row_sum = np.array(adj.sum(1)).astype(np.float32)
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        
        d_mat_inv_sqrt = np.diag(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    
    def _load_train_triples(self, dist_calculator):
        train_length = len(self.training_graphs)
        train_pairs_triples = []
        for m in range(train_length):
            g1 = self.training_graphs[m]
            for n in range(train_length):
                g2 = self.training_graphs[n]
                _, norm_dist = dist_calculator.calculate_distance_btw_pairs(g1, g2, 1.0)
                train_pairs_triples.append((m, n, norm_dist))
        ret = simgnn_utils.SelfShuffleList(train_pairs_triples)
        return ret
    
    def _load_test_triples(self):
        row_len = len(self.testing_graphs)
        column_len = len(self.train_val_graphs)
        
        list_batches = []
        for row in range(row_len):
            batch_rows = []
            batch_cols = []
            
            for column in range(column_len):
                batch_rows.append(row)
            
            for column in range(column_len):
                batch_cols.append(column)
            
            list_batches.append((batch_rows, batch_cols))
        
        return list_batches
    
    def _load_val_train_triples(self, dist_calculator):
        val_length = len(self.val_graphs)
        train_length = len(self.training_graphs)
        val_pairs_triples = []
        for m in range(val_length):
            g1 = self.val_graphs[m]
            for n in range(train_length):
                g2 = self.training_graphs[n]
                dist, norm_dist = dist_calculator.calculate_distance_btw_pairs(g1, g2, 1.0)
                val_pairs_triples.append((m, n, norm_dist))
        
        return val_pairs_triples
