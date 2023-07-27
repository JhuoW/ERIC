import torch
from sklearn.metrics import f1_score
import torch.nn as nn
from utils.utils import Evaluation
import numpy as np
from torch_geometric.data import Batch
from DataHelper.DatasetLocal import DatasetLocal
from utils.utils import *
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm, trange

@torch.no_grad()
def get_eval_result(self, labels, pred_l, loss):

    if self.config['multilabel']:
        micro , macro              = Evaluation(pred_l , labels)
    else:
        micro                      = f1_score(labels.cpu(), pred_l.cpu(), average = "micro")
        macro                      = 0

    return {
        "micro": round(micro * 100 , 2) , # to percentage
        "macro": round(macro * 100 , 2)
    }


@torch.no_grad()
def evaluate(self, testing_graphs, training_graphs, model, loss_func, dataset: DatasetLocal, validation = False):
    model.eval()
    num_test_pairs                 = len(testing_graphs) * len(training_graphs)
    num_pair_per_node              = len(training_graphs)

    if validation:
        if not self.config.get('use_all_val', True):  
            num_test_pairs         = self.config['val_size']   
            num_pair_per_node      = num_test_pairs // len(testing_graphs)  
            
    scores                         = np.empty((len(testing_graphs), num_pair_per_node))  # (140, 420)
    ground_truth                   = np.empty((len(testing_graphs), num_pair_per_node))
    ground_truth_ged               = np.empty((len(testing_graphs), num_pair_per_node))
    prediction_mat                 = np.empty((len(testing_graphs), num_pair_per_node))

    rho_list                       = []
    tau_list                       = []
    prec_at_10_list                = []
    prec_at_20_list                = []




    t                              = tqdm(total=num_test_pairs)

    for i,g in enumerate(testing_graphs): 
        source_batch               = Batch.from_data_list([g] * num_pair_per_node)  
        if validation:
            if not                   self.config.get('use_all_val', True):   
                training_graphs    = training_graphs.shuffle()[: num_pair_per_node]   

        target_batch               = Batch.from_data_list(training_graphs)            
        data                       = dataset.transform_batch((source_batch, target_batch), self.config)
        target                     = data["target"]
        # target = data["norm_ged"]

        ground_truth[i]            = target
        target_ged                 = data["target_ged"]
        ground_truth_ged[i]        = target_ged
        prediction,_               = model(data)
        prediction_mat[i]          = prediction.cpu().detach().numpy()
        scores[i]                  = ( F.mse_loss(prediction.cpu().detach(), target, reduction="none").numpy())

        rho_list.append(
            calculate_ranking_correlation(
                spearmanr, prediction_mat[i], ground_truth[i]
            )
        )
        tau_list.append(
            calculate_ranking_correlation(
                kendalltau, prediction_mat[i], ground_truth[i]
            )
        )
        prec_at_10_list.append(
            calculate_prec_at_k(
                10, prediction_mat[i], ground_truth[i], ground_truth_ged[i]
            )
        )
        prec_at_20_list.append(
            calculate_prec_at_k(
                20, prediction_mat[i], ground_truth[i], ground_truth_ged[i]
            )
        )
        t.update(num_pair_per_node)

    rho                            = np.mean(rho_list).item()
    tau                            = np.mean(tau_list).item()
    prec_at_10                     = np.mean(prec_at_10_list).item()
    prec_at_20                     = np.mean(prec_at_20_list).item()
    model_mse_error                = np.mean(scores).item()


    return model_mse_error, rho, tau, prec_at_10, prec_at_20