# Utilities in this file are all adopted from the implementation of SimGNN: https://github.com/yunshengb/SimGNN

import os
import pickle
import random
import re
from glob import glob
from os.path import basename
from os.path import isfile

import networkx as nx
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class SelfShuffleList(object):
    def __init__(self, li):
        assert (type(li) is list and list)
        self.li = li  # be careful! not a deep copy!
        self.idx = 0
        self._shuffle()  # shuffle at the beginning
    
    def get_next_one(self):
        if self.idx < len(self.li):
            rtn = self.li[self.idx]
            self.idx += 1
            return rtn
        else:
            self.idx = 0
            self._shuffle()
            return self.li[self.idx]
    
    def _shuffle(self):
        random.Random(123).shuffle(self.li)


class Data(object):
    def __init__(self, save_folder):
        print(save_folder)
        assert os.path.exists(save_folder)
        self.save_folder = save_folder
    
    def get_save_folder(self):
        return self.save_folder
    
    def get_dataset_folder_name(self):
        raise NotImplementedError
    
    def get_max_number_of_nodes(self):
        raise NotImplementedError


class AIDSData(Data):
    def __init__(self, save_folder, train_or_test):
        
        super(AIDSData, self).__init__(save_folder=save_folder)
        
        data_dir = '{}/{}/{}'.format(self.get_save_folder(), self.get_dataset_folder_name(), train_or_test)
        self.graphs = []
        self.graphs = iterate_get_graphs(data_dir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), data_dir))
        
        number_of_nodes = []
        for g in self.graphs:
            number_of_nodes.append(g.number_of_nodes())
        self.max_number_of_nodes = max(number_of_nodes)
        print('The max number of nodes among all graphs of {} is # {}'.format(self.get_dataset_folder_name(),
                                                                              self.get_max_number_of_nodes()))
        
        if 'nef' in self.get_dataset_folder_name():
            print('Removing edge features -> valence feature')
            for g in self.graphs:
                for n1, n2, d in g.edges(data=True):
                    d.pop('valence', None)
    
    def get_dataset_folder_name(self):
        return 'AIDS700nef'
    
    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]
    
    def get_max_number_of_nodes(self):
        return self.max_number_of_nodes


class LinuxData(Data):
    def __init__(self, save_folder, train_or_test):
        super(LinuxData, self).__init__(save_folder=save_folder)
        # Load all graphs
        data_dir = '{}/{}/{}'.format(self.get_save_folder(), self.get_dataset_folder_name(), train_or_test)
        self.graphs = []
        self.graphs = iterate_get_graphs(data_dir)
        print('Loaded {} graphs from {}'.format(len(self.graphs), data_dir))
        
        number_of_nodes = []
        for g in self.graphs:
            number_of_nodes.append(g.number_of_nodes())
        self.max_number_of_nodes = max(number_of_nodes)
        print('The max number of nodes among all graphs of {} is # {}'.format(self.get_dataset_folder_name(),
                                                                              self.get_max_number_of_nodes()))
    
    def get_dataset_folder_name(self):
        return 'LINUX'
    
    def get_gids(self):
        return [g.graph['gid'] for g in self.graphs]
    
    def get_max_number_of_nodes(self):
        return self.max_number_of_nodes



class SiameseDataSet(object):
    def __init__(self, data_save_folder, data_set_name, validation_ratio, node_feat_name, node_encoder_name):
        
        self.data_save_folder = data_save_folder
        self.data_set_name = data_set_name
        self.validation_ratio = validation_ratio
        self.node_feat_name = node_feat_name
        self.node_encoder_name = node_encoder_name
        
        # Load train and validation graphs
        print('\nprepare the train/validation data set and split')
        orig_train_data = self.load_data_set(data=self.data_set_name, train_or_test='train')
        self.train_val_gs = orig_train_data.graphs
        
        self.train_gs, self.val_gs = self._train_val_split(orig_train_data)
        self.train_val_max_number_of_nodes = orig_train_data.get_max_number_of_nodes()
        print('\t\ttrain_gs.len={} and val_gs.len={}, and their max number of nodes = {}'.format(len(self.train_gs),
                                                                                                 len(self.val_gs),
                                                                                                 self.train_val_max_number_of_nodes))
        print('\nprepare the testing data set')
        orig_test_data = self.load_data_set(data=self.data_set_name, train_or_test='test')
        self.test_gs = orig_test_data.graphs
        self.test_max_number_of_nodes = orig_test_data.get_max_number_of_nodes()
        
        print('\t\ttest_gs.len={}, and their max number of nodes = {}'.format(len(self.test_gs),
                                                                              self.test_max_number_of_nodes))
        
        self.node_feat_encoder = self._get_node_feature_encoder(orig_train_data.graphs + self.test_gs)
    
    def load_data_set(self, data, train_or_test):
        if data.lower() == 'aids700nef':
            return AIDSData(save_folder=self.data_save_folder, train_or_test=train_or_test)
        elif data.lower() == 'linux':
            return LinuxData(save_folder=self.data_save_folder, train_or_test=train_or_test)
        else:
            raise RuntimeError('!!! Not recognized data set of {}'.format(data))
    
    def input_dim(self):
        return self.node_feat_encoder.input_dim()
    
    def _train_val_split(self, orig_train_data):
        if self.validation_ratio < 0 or self.validation_ratio > 1:
            raise RuntimeError('ratio of validation {} must be in [0, 1]'.format(self.validation_ratio))
        gs = orig_train_data.graphs
        sp = int(len(gs) * (1 - self.validation_ratio))
        train_graphs = gs[0:sp]
        valid_graphs = gs[sp:]
        self._check_graphs_num(train_graphs, 'train')
        self._check_graphs_num(valid_graphs, 'validation')
        return train_graphs, valid_graphs
    
    @staticmethod
    def _check_graphs_num(graphs, label):
        if len(graphs) <= 2:
            raise RuntimeError('Insufficient {} graphs {}'.format(label, len(graphs)))
    
    def _get_node_feature_encoder(self, graphs):
        if self.node_encoder_name == 'OneHot':
            return NodeFeatureOneHotEncoder(graphs, self.node_feat_name)
        elif 'constant' in self.node_encoder_name:
            return NodeFeatureConstantEncoder(graphs, self.node_feat_name, self.node_encoder_name)
        else:
            raise RuntimeError('Unknown node_feat_encoder {}'.format(self.node_feat_encoder))


class NodeFeatureEncoder(object):
    def encode(self, g):
        raise NotImplementedError()
    
    def input_dim(self):
        raise NotImplementedError()


class NodeFeatureOneHotEncoder(NodeFeatureEncoder):
    def __init__(self, gs, node_feat_name):
        self.node_feat_name = node_feat_name
        inputs_set = set()
        for g in gs:
            inputs_set = inputs_set | set(self._node_feat_dic(g).values())
        self.feat_idx_dic = {feat: idx for idx, feat in enumerate(inputs_set)}
        
        self.oe = OneHotEncoder().fit(np.array(list(self.feat_idx_dic.values())).reshape(-1, 1))
    
    def encode(self, g):
        node_feat_dic = self._node_feat_dic(g)
        temp = [self.feat_idx_dic[node_feat_dic[n]] for n in g.nodes()]
        return self.oe.transform(np.array(temp).reshape(-1, 1)).toarray()
    
    def input_dim(self):
        return self.oe.transform([[0]]).shape[1]
    
    def _node_feat_dic(self, g):
        return nx.get_node_attributes(g, self.node_feat_name)


class NodeFeatureConstantEncoder(NodeFeatureEncoder):
    
    def __init__(self, _, node_feat_name=None, node_encoder_name='constant_1_2'):
        assert (node_feat_name is None)
        self.input_dim_ = int(node_encoder_name.split('_')[1])
        self.const = float(node_encoder_name.split('_')[2])
        print('A constant feature encoder where the input_dim is {} and input feature is {}'.format(self.input_dim_,
                                                                                                    self.const))
    
    def encode(self, g):
        rnt = np.full((g.number_of_nodes(), self.input_dim_), self.const)
        return rnt
    
    def input_dim(self):
        return self.input_dim_


def sorted_nicely(l):
    def try_int(s):
        try:
            return int(s)
        except:
            return s
    
    def alphanum_key(s):
        return [try_int(c) for c in re.split('([0-9]+)', s)]
    
    return sorted(l, key=alphanum_key)


def iterate_get_graphs(directory):
    graphs = []
    for file in sorted_nicely(glob(directory + '/*.gexf')):
        
        gid = int(basename(file).split('.')[0])
        
        g = nx.read_gexf(file)
        g.graph['gid'] = gid
        graphs.append(g)
        if not nx.is_connected(g):
            raise RuntimeError('{} not connected'.format(gid))
    return graphs


def normalized_exp_kernel_ged(d, g1, g2, scale):
    normalized = 2 * d / (g1.number_of_nodes() + g2.number_of_nodes())
    return np.exp(-scale * normalized)


def proc_filepath(filepath):
    if type(filepath) is not str:
        raise RuntimeError('Did you pass a file path to this function?')
    ext = '.pickle'
    if ext not in filepath:
        filepath += ext
    return filepath


def load(filepath):
    filepath = proc_filepath(filepath)
    if isfile(filepath):
        with open(filepath, 'rb') as handle:
            return pickle.load(handle)
    else:
        return None


def top_k_ids(query, qid, k, inclusive, rm=0):
    sort_id_mat = np.argsort(query, kind='mergesort')[:, ::-1]
    _, n = sort_id_mat.shape
    if k < 0 or k >= n:
        raise RuntimeError('Invalid k {}'.format(k))
    if not inclusive:
        return sort_id_mat[qid][:k]
    
    while k < n:
        cid = sort_id_mat[qid][k - 1]
        nid = sort_id_mat[qid][k]
        if abs(query[qid][cid] - query[qid][nid]) <= rm:
            k += 1
        else:
            break
    return sort_id_mat[qid][:k], k


def computing_precision_ks(trues, predictions, ks, inclusive=True, rm=0):
    assert trues.shape == predictions.shape
    m, n = trues.shape
    
    precision_ks = np.zeros((m, len(ks)))
    inclusive_final_true_ks = np.zeros((m, len(ks)))
    inclusive_final_pred_ks = np.zeros((m, len(ks)))
    
    for i in range(m):
        
        for k_idx, k in enumerate(ks):
            assert (type(k) is int and 0 < k < n)
            true_ids, true_k = top_k_ids(trues, i, k, inclusive, rm)
            pred_ids, pred_k = top_k_ids(predictions, i, k, inclusive, rm)
            precision_ks[i][k_idx] = min(len(set(true_ids).intersection(set(pred_ids))), k) / k
            inclusive_final_true_ks[i][k_idx] = true_k
            inclusive_final_pred_ks[i][k_idx] = pred_k
    return np.mean(precision_ks, axis=0), np.mean(inclusive_final_true_ks, axis=0), np.mean(inclusive_final_pred_ks,
                                                                                            axis=0)


def get_root_path():
    from os.path import dirname, abspath
    return dirname(dirname(abspath(__file__)))


def get_save_path():
    return get_root_path() + '/save'


def save_pkl(obj, handle):
    import pickle
    pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save(filepath, obj):
    with open(proc_filepath(filepath), 'wb') as handle:
        save_pkl(obj, handle)


class DistanceCalculator(object):
    
    def __init__(self, root_folder, data_name):
        
        self.sfn = '{}/{}_{}_{}_gidpair_dist_map'.format(root_folder, data_name.lower(), 'ged', 'astar')
        self.gid_pair_dist_map = load(self.sfn)
        print(type(self.gid_pair_dist_map))
        if not self.gid_pair_dist_map:
            raise RuntimeError('{} -> distance map does not exists!'.format(self.sfn))
        else:
            print('Loaded dist map from {} with {} entries'.format(self.sfn, len(self.gid_pair_dist_map)))
    
    def calculate_distance_btw_pairs(self, g1, g2, scale):
        gid1 = g1.graph['gid']
        gid2 = g2.graph['gid']
        pair = (gid1, gid2)
        d = self.gid_pair_dist_map.get(pair)
        if d is None:
            raise RuntimeWarning('{} distance is None'.format(d))
        return d, normalized_exp_kernel_ged(d, g1, g2, scale)


class Result(object):
    """
    The result object loads and stores the ranking result of a model
        for evaluation.
        Terminology:
            rtn: return value of a function.
            m: # of queries.
            n: # of database graphs.
    """
    
    def model(self):
        """
        :return: The model name.
        """
        return self.model_
    
    def m_n(self):
        return self.dist_sim_mat(norm=False).shape
    
    def dist_or_sim(self):
        raise NotImplementedError()
    
    def dist_sim_mat(self, norm):
        """
        Each result object stores either a distance matrix
            or a similarity matrix. It cannot store both.
        :param norm:
        :return: either the distance matrix or the similarity matrix.
        """
        raise NotImplementedError()
    
    def dist_sim(self, qid, gid, norm):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param norm:
        :return: (metric, dist or sim between qid and gid)
        """
        raise NotImplementedError()
    
    def sim_mat(self, sim_kernel, yeta, scale, norm):
        raise NotImplementedError()
    
    def top_k_ids(self, qid, k, norm, inclusive, rm):
        """
        :param qid: query id (0-indexed).
        :param k:
        :param norm:
        :param inclusive: whether to be tie inclusive or not.
            For example, the ranking may look like this:
            7 (sim_score=0.99), 5 (sim_score=0.99), 10 (sim_score=0.98), ...
            If tie inclusive, the top 1 results are [7, 9].
            Therefore, the number of returned results may be larger than k.
            In summary,
                len(rtn) == k if not tie inclusive;
                len(rtn) >= k if tie inclusive.
        :param rm:
        :return: for a query, the ids of the top k database graph
        ranked by this model.
        """
        sort_id_mat = self.sort_id_mat(norm)
        _, n = sort_id_mat.shape
        if k < 0 or k >= n:
            raise RuntimeError('Invalid k {}'.format(k))
        if not inclusive:
            return sort_id_mat[qid][:k]
        # Tie inclusive.
        dist_sim_mat = self.dist_sim_mat(norm)
        while k < n:
            cid = sort_id_mat[qid][k - 1]
            nid = sort_id_mat[qid][k]
            if abs(dist_sim_mat[qid][cid] - dist_sim_mat[qid][nid]) <= rm:
                k += 1
            else:
                break
        return sort_id_mat[qid][:k]
    
    def ranking(self, qid, gid, norm, one_based=True):
        """
        :param qid: query id (0-indexed).
        :param gid: database graph id (0-indexed) (NOT g.graph['gid']).
        :param norm:
        :param one_based: whether to return the 1-based or 0-based rank.
            True by default.
        :return: for a query, the rank of a database graph by this model.
        """
        # Assume self is ground truth.
        sort_id_mat = self.sort_id_mat(norm)
        finds = np.where(sort_id_mat[qid] == gid)
        assert (len(finds) == 1 and len(finds[0]) == 1)
        fid = finds[0][0]
        # Tie inclusive (always when find ranking).
        dist_sim_mat = self.dist_sim_mat(norm)
        while fid > 0:
            cid = sort_id_mat[qid][fid]
            pid = sort_id_mat[qid][fid - 1]
            if dist_sim_mat[qid][pid] == dist_sim_mat[qid][cid]:
                fid -= 1
            else:
                break
        if one_based:
            fid += 1
        return fid
    
    def classification_mat(self, thresh_pos, thresh_neg,
                           thresh_pos_sim, thresh_neg_sim, norm):
        raise NotImplementedError()
    
    def time(self, qid, gid):
        raise NotImplementedError()
    
    def time_mat(self):
        raise NotImplementedError()
    
    def mat(self, metric, norm):
        raise NotImplementedError()
    
    def sort_id_mat(self, norm):
        """
        :param norm:
        :return: a m by n matrix representing the ranking result.
            rtn[i][j]: For query i, the id of the j-th most similar
                       graph ranked by this model.
        """
        raise NotImplementedError()
    
    def ranking_mat(self, norm, one_based=True):
        """
        :param norm:
        :param one_based:
        :return: a m by n matrix representing the ranking result.
                 Note it is different from sort_id_mat.
            rtn[i][j]: For query i, the ranking of the graph j.
        """
        m, n = self.m_n()
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                rtn[i][j] = self.ranking(i, j, norm, one_based=one_based)
        return rtn


class DistanceModelResult(Result):
    def __init__(self, dataset, model, result_folder, dist_mat, row_graphs, col_graphs, scale):
        self.dataset = dataset
        self.model_ = model
        self.result_folder = result_folder
        if dist_mat is not None:
            self.dist_mat_ = dist_mat
            self.time_mat_ = None
        else:
            self.dist_mat_ = self._load_result_mat(self.dist_metric(), self.model_, len(row_graphs), len(col_graphs))
            print('self.dist_mat_.shape = ', self.dist_mat_.shape)
        self.dist_norm_mat_ = np.copy(self.dist_mat_)
        
        m, n = self.dist_mat_.shape
        assert (m == len(row_graphs))
        assert (n == len(col_graphs))
        for i in range(m):
            for j in range(n):
                self.dist_norm_mat_[i][j] = normalized_exp_kernel_ged(self.dist_mat_[i][j], row_graphs[i],
                                                                      col_graphs[j], scale)
        self.sort_id_mat_ = np.argsort(self.dist_mat_, kind='mergesort')
        self.dist_norm_sort_id_mat_ = np.argsort(self.dist_norm_mat_, kind='mergesort')
    
    def dist_metric(self):
        raise NotImplementedError()
    
    def dist_mat(self, norm):
        return self._select_dist_mat(norm)
    
    def dist_sim_mat(self, norm):
        return self.dist_mat(norm)
    
    def dist_or_sim(self):
        return 'dist'
    
    def dist_sim(self, qid, gid, norm):
        return self.dist_metric(), self._select_dist_mat(norm)[qid][gid]
    
    def time(self, qid, gid):
        return self.time_mat_[qid][gid]
    
    def time_mat(self):
        return self.time_mat_
    
    def mat(self, metric, norm):
        if metric == self.dist_metric():
            return self._select_dist_mat(norm)
        elif metric == 'time':
            return self.time_mat_
        else:
            raise RuntimeError('Unknown metric {} for model {}'.format(metric, self.model_))
    
    def sort_id_mat(self, norm):
        return self._select_sort_id_mat(norm)
    
    def _load_result_mat(self, metric, model, m, n):
        file_p = self.result_folder + '/{}/{}/{}_{}_mat_{}_{}_*.npy'.format(self.dataset, metric, self.dist_metric(),
                                                                            metric, self.dataset, model)
        print('load_result_mat\nfile_p: ', file_p)
        li = glob(file_p)
        if not li:
            if 'astar' in model:
                if self.dataset != 'imdbmulti':
                    raise RuntimeError('Not imdbmulti and no astar results!')
                return self._load_merged_astar_from_other_three(metric, m, n)
            else:
                raise RuntimeError('No results found {}'.format(file_p))
        file = self._choose_result_file(li, m, n)
        # file = '../results' + '/{}/{}/{}'.format(self.dataset, metric, 'ged_ged_mat_aids700nef_astar_2018-06-27T19_38_42_qilin_10cpus.npy')
        return np.load(file)
    
    def _load_merged_astar_from_other_three(self, metric, m, n):
        print("_load_merged_astar_from_other_three: beam80, hungarian, vj")
        self.model_ = 'merged_astar'
        merge_models = ['beam80', 'hungarian', 'vj']
        dms = [self._load_result_mat(self.dist_metric(), model, m, n)
               for model in merge_models]
        tms = [self._load_result_mat('time', model, m, n)
               for model in merge_models]
        for i in range(len(merge_models)):
            assert (dms[i].shape == (m, n))
            assert (tms[i].shape == (m, n))
        rtn = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                d_li = [dm[i][j] for dm in dms]
                d_min = np.min(d_li)
                if metric == self.dist_metric():
                    rtn[i][j] = d_min
                else:
                    assert (metric == 'time')
                    d_argmin = np.argmin(d_li)
                    rtn[i][j] = tms[d_argmin][i][j]
        return rtn
    
    def _choose_result_file(self, files, m, n):
        cands = []
        for file in files:
            temp = np.load(file)
            if temp.shape == (m, n):
                cands.append(file)
                if 'qilin' in file:
                    print(file)
                    return file
        if cands:
            return cands[0]
        raise RuntimeError('No result files in {}'.format(files))  # TODO: smart choice and cross-checking
    
    def _select_dist_mat(self, norm):
        return self.dist_norm_mat_ if norm else self.dist_mat_
    
    def _select_sort_id_mat(self, norm):
        return self.dist_norm_sort_id_mat_ if norm else self.sort_id_mat_


class PairwiseGEDModelResult(DistanceModelResult):
    def dist_metric(self):
        return 'ged'


def load_result(dataset, model, result_folder="../GEDResults", sim=None, sim_mat=None, dist_mat=None, row_graphs=None,
                col_graphs=None, time_mat=None, model_info=None, scale=None):
    if 'beam' in model or model in ['astar', 'hungarian', 'vj']:
        return PairwiseGEDModelResult(dataset, model, result_folder, dist_mat, row_graphs, col_graphs, scale)
    elif 'siamese' in model:
        raise NotImplementedError
    else:
        raise RuntimeError('Unknown model {}'.format(model))
