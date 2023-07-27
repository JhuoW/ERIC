import yaml
import numpy as np
from sklearn import metrics
from torch_geometric.data import Data
import torch
from torch_sparse import SparseTensor
from torch_scatter import scatter_mean
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal
import time, datetime
import os
import errno
from os.path import dirname
from warnings import warn

def get_config(config_path="config.yml"):
    with open(config_path, "r") as setting:
        config                             = yaml.load(setting, Loader=yaml.FullLoader)
    return config


def load_data(args, custom = False):
    if not custom:
        path                               = osp.join(args.data_dir)
        dataset                            = DatasetLocal(args.dataset, "")
        dataset.dataset_source_folder_path = path
        dataset.recache                    = args.recache
    else:
        path                               = osp.join(args.custom_data_dir)
        dataset                            = DatasetLocal(args.dataset, "")
        dataset.dataset_source_folder_path = path
        dataset.recache                    = args.recache
    return dataset
    

def create_dir_if_not_exists(dir):
    import os
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_fig(plt, dir, fn, print_path=True):
    plt_cnt = 0
    if dir is None or fn is None:
        return plt_cnt
    final_path_without_ext = '{}/{}'.format(dir, fn)
    for ext in ['pdf']:
        full_path = final_path_without_ext + '.' + ext
        create_dir_if_not_exists(dirname(full_path))
        try:
            plt.savefig(full_path, bbox_inches='tight')
        except:
            warn('savefig')
        if print_path:
            print('Saved to {}'.format(full_path))
        plt_cnt += 1
    return plt_cnt

def homophily(edge_index, y, method: str = 'edge'):
    assert method in ['edge', 'node']
    y                                  = y.squeeze(-1) if y.dim() > 1 else y

    if isinstance(edge_index, SparseTensor):
        col, row, _                    = edge_index.coo()
    else:
        row, col                       = edge_index

    if method                          == 'edge':
        return int((y[row]             == y[col]).sum()) / row.size(0)  # out neigh 的同质率
    else:
        out                            = torch.zeros_like(row, dtype=float)
        out[y[row]                     == y[col]] = 1.
        out                            = scatter_mean(out, col, 0, dim_size=y.size(0))
        return float(out.mean())



def accuracy(output, labels):
    preds                              = output.max(1)[1].type_as(labels)
    correct                            = preds.eq(labels).double()
    correct                            = correct.sum()
    return                     correct / len(labels)


def getneighborslst(data: Data):
    ei                                 = data.edge_index.numpy()
    lst                                = {i: set(ei[1][ei[0] == i]) for i in range(data.num_nodes)} 
    return lst

def get_device(cuda_id: int):
    device                             = torch.device('cuda' if cuda_id < 0 else 'cuda:%d' % cuda_id)
    return device


def Evaluation(output, labels):
    preds                              = output.cpu().detach().numpy()
    labels                             = labels.cpu().detach().numpy()
    '''
    binary_pred                        = preds
    binary_pred[binary_pred > 0.0]     = 1
    binary_pred[binary_pred <= 0.0]    = 0
    '''
    num_correct                        = 0
    binary_pred                        = np.zeros(preds.shape).astype('int')
    for i in range(preds.shape[0]):
        k                              = labels[i].sum().astype('int')
        topk_idx                       = preds[i].argsort()[-k:]
        binary_pred[i][topk_idx]       = 1
        for pos in list(labels[i].nonzero()[0]):
            if labels[i][pos] and labels[i][pos] == binary_pred[i][pos]:
                num_correct += 1

    # print('total number of correct is: {}'.format(num_correct))
    # print('preds max is: {0} and min is: {1}'.format(preds.max(),preds.min()))
    # '''
    return metrics.f1_score(labels, binary_pred, average="micro"), metrics.f1_score(labels, binary_pred, average="macro")

def print_config(config):
    print("**************** MODEL CONFIGURATION ****************")
    for key in sorted(config.keys()):
        val                           = config[key]
        keystr                        = "{}".format(key) + (" " * (24 - len(key)))
        print("{} -->   {}".format(keystr, val))
    print("**************** MODEL CONFIGURATION ****************")


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx                         = sparse_mx.tocoo().astype(np.float32)
    indices                           = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values                            = torch.from_numpy(sparse_mx.data)
    shape                             = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def edge_index_to_torch_sparse_tensor(edge_index, edge_weight = None):
    if edge_weight is None:
        edge_weight                   = torch.ones((edge_index.size(1),)).cuda()
    
    n_node                            = edge_index.max().item() + 1

    return torch.cuda.sparse.FloatTensor(edge_index, edge_weight, torch.Size((n_node, n_node)))


def get_classification_labels_from_dist_mat(dist_mat, thresh_pos, thresh_neg):
    m, n = dist_mat.shape
    label_mat                   = np.zeros((m, n))
    num_poses                   = 0
    num_negs                    = 0
    pos_pairs                   = []
    neg_pairs                   = []
    for i in range(m):
        num_pos                 = 0
        num_neg                 = 0
        for j in range(n):
            d                   = dist_mat[i][j]
            c                   = classify(d, thresh_pos, thresh_neg)
            if c                == 1:
                label_mat[i][j] = 1
                num_pos         += 1
                pos_pairs.append((i, j))
            elif c              == -1:
                label_mat[i][j] = -1
                num_neg         += 1
                neg_pairs.append((i, j))
        num_poses               += num_pos
        num_negs                += num_neg
    return label_mat, num_poses, num_negs, pos_pairs, neg_pairs

def classify(dist, thresh_pos, thresh_neg):
    if dist <= thresh_pos:
        return 1
    elif dist > thresh_neg:
        return -1
    else:
        return 0


###########################################################################################################################


def calculate_ranking_correlation(rank_corr_function, prediction, target):
    """
    Calculating specific ranking correlation for predicted values.
    :param rank_corr_function: Ranking correlation function.
    :param prediction: Vector of predicted values.
    :param target: Vector of ground-truth values.
    :return ranking: Ranking correlation value.
    """

    temp                              = prediction.argsort()
    r_prediction                      = np.empty_like(temp)
    r_prediction[temp]                = np.arange(len(prediction))

    temp                              = target.argsort()
    r_target                          = np.empty_like(temp)
    r_target[temp]                    = np.arange(len(target))
    
    return rank_corr_function(r_prediction, r_target).correlation


def _calculate_prec_at_k(k, target):
    target_increase                   = np.sort(target)
    target_value_sel                  = (target_increase <= target_increase[k-1]).sum()
    if target_value_sel               > k:
        best_k_target                 = target.argsort()[:target_value_sel]
    else:
        best_k_target                 = target.argsort()[:k]
    return best_k_target

def calculate_prec_at_k(k, prediction, target, target_ged):
    """
    Calculating precision at k.
    """
    best_k_pred                       = prediction.argsort()[::-1][:k]
    best_k_target                     = _calculate_prec_at_k(k, -target)
    best_k_target_ged                 = _calculate_prec_at_k(k, target_ged)

    
    return len(set(best_k_pred).intersection(set(best_k_target_ged))) / k



def save_model(config, dataset_name, model):
    PATH_MODEL                        = os.path.join(os.path.join(os.getcwd(),'model_saved'), dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(PATH_MODEL)
    except OSError as e:
        if e.errno                   != errno.EEXIST:
            raise "This was not a directory exist error"
    with open(os.path.join(PATH_MODEL, "config.yml"), 'w') as f:
        f.write('%s:\n'  % (config['model_name']))
        for key, value in config.items():
            f.write('%s: %s\n'         % (key, value))
    
    PATH_GSC                          = os.path.join(PATH_MODEL, config['model_name'] + "_" + dataset_name+'_checkpoint.pth' )
    torch.save(model.state_dict()     , PATH_GSC)
    print("Model Saved")


def save_best_val_model(config, dataset_name, model, PATH_MODEL):
    try:
        if not os.path.exists(PATH_MODEL):
            os.makedirs(PATH_MODEL)
    except OSError as e:
        if e.errno                    != errno.EEXIST:
            raise "This was not a directory exist error"    
    with open(os.path.join(PATH_MODEL, "config.yml"), 'w') as f:
        for key, value in config.items():
            f.write('%s: %s\n'         % (key, value))
    PATH_GSC                          = os.path.join(PATH_MODEL, config['model_name'] + "_" + dataset_name+'_checkpoint.pth' )
    torch.save(model.state_dict(),    PATH_GSC)
    return PATH_GSC


def save_best_val_model_metric(config, dataset_name, model, PATH_MODEL, metric):
    try:
        if not os.path.exists(PATH_MODEL):
            os.makedirs(PATH_MODEL)
    except OSError as e:
        if e.errno                    != errno.EEXIST:
            raise "This was not a directory exist error"    
    with open(os.path.join(PATH_MODEL, "config.yml"), 'w') as f:
        for key, value in config.items():
            f.write('%s: %s\n'         % (key, value))
    PATH_GSC                           = os.path.join(PATH_MODEL, config['model_name'] + "_" + dataset_name+'_checkpoint_{}.pth'.format(metric) )
    torch.save(model.state_dict(),     PATH_GSC)
    return PATH_GSC

def save_best_val_model_all(config, dataset_name, model, PATH_MODEL, current_metric, best_metric, best_val_paths, b_epoch):
    """
    current_metric: [rho, tau, p10, p20]
    """
    met           = ['mse', 'rho', 'tau', 'p10', 'p20']
    if current_metric[0]  <= best_metric[0]:
        best_val_mse      = current_metric[0]
        best_mse_path     = save_best_val_model_metric(config, dataset_name, model, PATH_MODEL, 'mse')
        best_metric[0]    = best_val_mse
        best_val_paths[0] = best_mse_path
        b_epoch           = current_metric[5]
    if current_metric[1]  >= best_metric[1]:
        best_rho          = current_metric[1]
        best_rho_path     = save_best_val_model_metric(config, dataset_name, model, PATH_MODEL, 'rho')
        best_metric[1]    = best_rho
        best_val_paths[1] = best_rho_path
    if current_metric[2]  >= best_metric[2]:
        best_tau          = current_metric[2]
        best_tau_path     = save_best_val_model_metric(config, dataset_name, model, PATH_MODEL, 'tau')
        best_metric[2]    = best_tau
        best_val_paths[2] = best_tau_path
    if current_metric[3]  >= best_metric[3]:
        best_p10          = current_metric[3]
        best_p10_path     = save_best_val_model_metric(config, dataset_name, model, PATH_MODEL, 'p10')
        best_metric[3]    = best_p10
        best_val_paths[3] = best_p10_path
    if current_metric[4]  >= best_metric[4]:
        best_p20          = current_metric[4]
        best_p20_path     = save_best_val_model_metric(config, dataset_name, model, PATH_MODEL, 'p20')
        best_metric[4]    = best_p20
        best_val_paths[4] = best_p20_path
    return best_metric, best_val_paths, b_epoch

def load_model(config, dataset_name, model):
    PATH_MODEL                        = os.path.join(os.path.join(os.getcwd(),'model_saved'), dataset_name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    PATH_GSC                          = os.path.join(PATH_MODEL, config['model_name'] + "_" + dataset_name+'_checkpoint.pth' )
    model.load_state_dict(torch.load(PATH_GSC))
    print("Model Loaded")

def load_model_all(dataset, model, loss_func, model_paths, T):
    test_mse                          = None
    test_rho                          = None
    test_tau                          = None
    test_prec_at_10                   = None
    test_prec_at_20                   = None
    met_test                          = [test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20]
    for i, path in enumerate(model_paths):
        model.load_state_dict(torch.load(path))
        res                           = T.evaluation(dataset.testing_graphs, dataset.trainval_graphs, model, loss_func, dataset)
        met_test[i]                   = res[i]
    return met_test

if __name__ == "__main__":
    config                           = get_config('config/AIDS700nef.yml')
    config['model_name']             = 'GSC_GNN'
    config                           = config['GSC_GNN'] 
    mydir                            = os.path.join(os.path.join(os.getcwd(),'model_saved'), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno                  != errno.EEXIST:
            raise  # This was not a "directory exist" error.
    with open(os.path.join(mydir, "config.txt"), 'w') as f:
        for key, value in config.items():
            f.write('%s:%s\n' % (key, value))

        