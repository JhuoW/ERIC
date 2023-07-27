from utils.utils import *
from argparse import ArgumentParser
from model.GSC import GSC
import torch.nn as nn
import torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from torch_geometric.data import Batch

@torch.no_grad()
def evaluate(testing_graphs, training_graphs, model, loss_func, dataset: DatasetLocal, config):
    model.eval()

    scores                         = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth                   = np.empty((len(testing_graphs), len(training_graphs)))
    ground_truth_ged               = np.empty((len(testing_graphs), len(training_graphs)))
    prediction_mat                 = np.empty((len(testing_graphs), len(training_graphs)))

    rho_list                       = []
    tau_list                       = []
    prec_at_10_list                = []
    prec_at_20_list                = []

    num_test_pairs                 = len(testing_graphs) * len(training_graphs)
    t                              = tqdm(total=num_test_pairs)

    for i,g in enumerate(testing_graphs):
        source_batch               = Batch.from_data_list([g] * len(training_graphs))
        target_batch               = Batch.from_data_list(training_graphs)
        data                       = dataset.transform_batch((source_batch, target_batch), config)
        target                     = data["target"]
        # target = data["norm_ged"]

        ground_truth[i]            = target
        target_ged                 = data["target_ged"]
        ground_truth_ged[i]        = target_ged
        prediction, loss_cl        = model(data)
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
        t.update(len(training_graphs))

    rho                            = np.mean(rho_list).item()
    tau                            = np.mean(tau_list).item()
    prec_at_10                     = np.mean(prec_at_10_list).item()
    prec_at_20                     = np.mean(prec_at_20_list).item()
    model_mse_error                = np.mean(scores).item()


    return model_mse_error, rho, tau, prec_at_10, prec_at_20

if __name__ == '__main__':
    model_name = 'GSC_GNN'
    # dataset    = 'AIDS700nef'
    parser = ArgumentParser()
    parser.add_argument('--dataset',            type=str,             default= 'IMDBMulti') 
    parser.add_argument('--num_workers',        type=int,             default=8,                  choices=[0,8])
    parser.add_argument('--seed',               type=int,             default=1234,               choices=[0, 1, 1234])
    parser.add_argument('--data_dir',           type=str,             default="/data1/zhuowei/datasets/GED_Datasets/") 
    parser.add_argument('--custom_data_dir',    type=str,             default='datasets/GED/')
    parser.add_argument('--hyper_file',         type=str,             default= 'config/')
    parser.add_argument('--recache',          action="store_true",      help ="clean up the old adj data", default=True)   
    parser.add_argument('--no_dev',           action="store_true" ,  default = False)
    parser.add_argument('--patience',          type = int  ,         default = -1)
    parser.add_argument('--gpu_id',            type = int  ,         default = 3)
    parser.add_argument('--model',             type = str,           default ='GSC_GNN')  # GCN, GAT or other
    parser.add_argument('--train_first',       type = bool,          default = True)
    parser.add_argument('--save_model',        type = bool,          default = False)
    args = parser.parse_args()

    # CONFIG_PATH = "model_saved/LINUX/2022-03-20_09-55-10"
    # CONFIG_PATH = "model_saved/LINUX/2022-03-28_19-49-36"
    # CONFIG_PATH = "model_saved/AIDS700nef/2022-03-19_10-00-20"
    CONFIG_PATH = "model_saved/IMDBMulti/2022-03-31_11-21-55"
    config_path                  = osp.join(CONFIG_PATH, 'config' + '.yml')
    # config_path                  = osp.join('config/',args.dataset +'.yml')
    config                       = get_config(config_path)
    print(config)
    config                       = config[args.model] 
    config['dataset_name']       = args.dataset
    print(config)

    dataset                      = load_data(args, False)
    dataset.load(config)
    model                        = GSC(config, dataset.input_dim).cuda()


    best_val_model_path = osp.join(CONFIG_PATH, 'GSC_GNN_{}_checkpoint.pth'.format(args.dataset))
    model.load_state_dict(torch.load(best_val_model_path))
    model_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = evaluate(dataset.testing_graphs, dataset.trainval_graphs, model, nn.MSELoss(), dataset, config)

    def print_evaluation(model_mse,test_rho,test_tau,test_prec_at_10,test_prec_at_20):
        """
        Printing the error rates.
        """
        print("\nmse(10^-3): "   + str(round(model_mse * 1000, 5)) + ".")
        print("Spearman's rho: " + str(round(test_rho, 5)) + ".")
        print("Kendall's tau: "  + str(round(test_tau, 5)) + ".")
        print("p@10: "           + str(round(test_prec_at_10, 5)) + ".")
        print("p@20: "           + str(round(test_prec_at_20, 5)) + ".")
    print_evaluation(model_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20)