import imp
import os.path as osp
from argparse import ArgumentParser
import sys
from tkinter import Pack
from tqdm import tqdm
import torch
import random
from utils.logger import Logger
from utils.utils import *
from utils.random_seeder import set_random_seed
from training_procedure import Trainer
from DataHelper.DatasetLocal import DatasetLocal
from model.GSC import GSC

def main(args, config, logger: Logger, run_id: int, dataset: DatasetLocal):
    T                                = Trainer(config=config, args= args, logger= logger)

    model, optimizer, loss_func      = T.init(dataset)   # model of current split
    custom                           = config.get('custom', False)
    pbar                             = tqdm(range(config['epochs']), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    validation_data                  = None
    patience_cnt 		             = 0
    maj_metric 			             = "micro"   # or macro
    best_mse_metric                  = 100000.0
    best_metric 	  	             = 0
    best_metric_epoch 	             = -1 # best number on dev set
    report_mse_test 	             = 0
    report_rho_test                  = 0
    report_prec_at_10_test           = 0
    best_val_mse                     =  100000.
    best_val_tau                     = -100000.
    best_val_rho                     = -100000.
    best_val_p10                     = -100000.
    best_val_p20                     = -100000.
    best_val_epoch                   = -1
    loss_list                        = []
    monitor                          = config['monitor']
    best_val_paths                   = [None        , None        , None        , None        , None        ]
    best_val_metric                  = [best_val_mse, best_val_rho, best_val_tau, best_val_p10, best_val_p20]
    b_epoch                          = 0
    if config['save_best']:
        PATH_MODEL                   = os.path.join(os.path.join(os.getcwd(),'model_saved'), args.dataset, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))    
    for epoch in pbar:
        if not custom:
            batches                  = dataset.create_batches(config)   # 128对 graph-pair

        else:
            batch_feature_1, batch_adj_1, batch_mask_1, batch_feature_2, batch_adj_2, batch_mask_2, batch_ged = dataset.custom_dataset.get_training_batch()
        main_index                   = 0
        loss_sum                     = 0
        total_loss_sum               = 0 
        for batch_pair in batches:
            data                     = dataset.transform_batch(batch_pair, config)
            target                   = data["target"].cuda()
            model, loss              = T.train(data, model, loss_func, optimizer, target)   
            main_index               = main_index + batch_pair[0].num_graphs               
            loss_sum                 = loss_sum + loss                                    
        loss                         = loss_sum / main_index                              
        loss_list.append(loss)

        if config['use_val']:
            if epoch >= config['iter_val_start'] and epoch % config['iter_val_every'] ==0:
                model.eval()

                val_mse, val_rho, val_tau, val_prec_at_10, val_prec_at_20 = T.evaluation(dataset.val_graphs, dataset.training_graphs, model, loss_func, dataset, validation=True)
                logger.log("Validation Epoch = {}, MSE = {}(e-3), rho = {}, tau={}, prec_10 = {}, prec_20 = {}".format(epoch, val_mse*1000, val_rho, val_tau, val_prec_at_10, val_prec_at_20))
                if not config.get('save_best_all', False):  # run this
                    if best_mse_metric                >= val_mse:
                        best_mse_metric               = val_mse
                        best_val_epoch                = epoch
                        best_val_mse                  = val_mse
                        best_val_tau                  = val_tau
                        best_val_rho                  = val_rho
                        best_val_p10                  = val_prec_at_10
                        best_val_p20                  = val_prec_at_20
                        if config['save_best']:
                            best_val_model_path       = save_best_val_model(config, args.dataset, model, PATH_MODEL)

                else:
                    current_metric                    = [val_mse        , val_rho     , val_tau     , val_prec_at_10,  val_prec_at_20, epoch]
                    best_val_metric, best_val_paths, b_epoch = save_best_val_model_all(config, args.dataset, model, PATH_MODEL, current_metric, best_val_metric, best_val_paths, b_epoch, validation=True)
                    best_mse_metric                   = best_val_metric[0]
                    best_val_mse                      = best_val_metric[0]
                    best_val_rho                      = best_val_metric[1]
                    best_val_tau                      = best_val_metric[2]
                    best_val_p10                      = best_val_metric[3]
                    best_val_p20                      = best_val_metric[4]
                    best_val_epoch                    = b_epoch


        if epoch != config['epochs']-1:
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% ( 
                            epoch ,      loss)
            # pbar.set_postfix_str(postfix_str)
        elif epoch == config['epochs'] and config.get('show_last', False): 
            mse, rho, tau, prec_at_10, prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.training_graphs, model, loss_func, dataset)
            best_mse_metric                       = mse
            best_metric_epoch                     = epoch
            report_mse_test                       = mse
            report_rho_test                       = rho
            report_tau_test                       = tau
            report_prec_at_10_test                = prec_at_10
            report_prec_at_20_test                = prec_at_20
            
            postfix_str = "<Epoch %d> [Train Loss] %.4f [Cur Tes %s] %.4f <Best Epoch %d> [Best Tes mse] %.4f [rho] %.4f [tau] %.4f [prec_at_10] %.4f [prec_at_20] %.4f " % ( 
                            epoch ,      loss,         monitor,      eval(monitor),  
                            best_metric_epoch ,report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)
        else:
            postfix_str = "<Epoch %d> [Train Loss] %.5f"% ( 
                epoch ,      loss)
        if not args.train_first:
            mse, rho, tau, prec_at_10, prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.training_graphs, model, loss_func, dataset)  # return 2 list, 

    

            if monitor                         == 'mse':   # *↓
                if mse                         <= best_mse_metric:
                    best_mse_metric             = mse
                    best_metric_epoch           = epoch
                    report_mse_test             = mse
                    report_rho_test             = rho
                    report_tau_test             = tau
                    report_prec_at_10_test      = prec_at_10
                    report_prec_at_20_test      = prec_at_20
                    patience_cnt                = 0  
                else:
                    patience_cnt               += 1
            elif monitor in ['rho', 'tau', 'prec_at_10', 'prec_at_20']:   # *↑
                current_metric                  = eval(monitor)
                if best_metric                 <= current_metric: 
                    best_metric                 = current_metric
                    best_metric_epoch           = epoch
                    report_mse_test             = mse
                    report_rho_test             = rho
                    report_tau_test             = tau
                    report_prec_at_10_test      = prec_at_10
                    report_prec_at_20_test      = prec_at_20
                    patience_cnt                = 0  
                else:
                    patience_cnt               += 1               

            if config['patience'] > 0 and patience_cnt >= config['patience']:
                break

            postfix_str = "<Epoch %d> [Train Loss] %.4f [Cur Tes %s] %.4f <Last Epoch %d> [Last Tes mse] %.4f [rho] %.4f [tau] %.4f [prec_at_10] %.4f [prec_at_20] %.4f " % ( 
                            epoch ,      loss,         monitor, eval(monitor),  
                            best_metric_epoch ,report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)

        pbar.set_postfix_str(postfix_str)
    logger.add_line()
    logger.log("start testing using best val model")
    if not config.get('save_best_all', False):
        model.load_state_dict(torch.load(best_val_model_path))
        test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.trainval_graphs, model, loss_func, dataset)
    else:
        met_test                                                       = load_model_all(dataset, model, loss_func, best_val_paths, T)
        test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20 = met_test
    best_val_result = {
        'best_val_epoch': best_val_epoch,
        'best_val_mse'  : best_val_mse,
        'best_val_tau'  : best_val_tau,
        'best_val_rho'  : best_val_rho,
        'best_val_p10'  : best_val_p10,
        'best_val_p20'  : best_val_p20
    }
    return model, best_val_epoch , test_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20, loss, PATH_MODEL, best_val_result


def print_evaluation(model_error,rho,tau,prec_at_10,prec_at_20):
    """
    Printing the error rates.
    """
    print("\nmse(10^-3): "   + str(round(model_error * 1000, 5))         + ".")
    print("Spearman's rho: " + str(round(rho, 5))                        + ".")
    print("Kendall's tau: "  + str(round(tau, 5))                        + ".")
    print("p@10: "           + str(round(prec_at_10, 5))                 + ".")
    print("p@20: "           + str(round(prec_at_20, 5))                 + ".")





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset',           type = str,            default = 'LINUX') 
    parser.add_argument('--num_workers',       type = int,            default = 8,                  choices=[0,8])
    parser.add_argument('--seed',              type = int,            default = 1234,               choices=[0, 1, 1234])
    parser.add_argument('--data_dir',          type = str,            default = 'datasets/GED/') 
    parser.add_argument('--custom_data_dir',   type = str,            default = 'datasets/GED/')
    parser.add_argument('--hyper_file',        type = str,            default = 'config/')
    parser.add_argument('--recache',         action = "store_true",      help = "clean up the old adj data", default=True)   
    parser.add_argument('--no_dev',          action = "store_true" ,  default = False)
    parser.add_argument('--patience',          type = int  ,          default = -1)
    parser.add_argument('--gpu_id',            type = int  ,          default = 2)
    parser.add_argument('--model',             type = str,            default ='GSC_GNN')  # GCN, GAT or other
    parser.add_argument('--train_first',       type = bool,           default = True)
    parser.add_argument('--save_model',        type = bool,           default = False)
    parser.add_argument('--run_pretrain',    action ='store_true',    default = False)
    parser.add_argument('--pretrain_path',     type = str,            default = 'model_saved/LINUX/2022-03-20_03-01-57')
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu_id)

    logger                       = Logger(mode = [print])  
    logger.add_line              = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()
    config_path                  = osp.join(args.hyper_file, args.dataset + '.yml') if not args.run_pretrain else osp.join(args.pretrain_path, 'config' + '.yml')
    config                       = get_config(config_path)
    model_name                   = args.model
    config                       = config[model_name] 
    config['model_name']         = model_name
    config['dataset_name']       = args.dataset
    custom                       = config.get('custom', False)
    dev_ress                     = []
    tes_ress                     = []
    tra_ress                     = []
    if config.get('seed',-1)     > 0:
        set_random_seed(config['seed'])
        logger.log ("Seed set. %d" % (config['seed']))
    seeds                        = [random.randint(0,233333333) for _ in range(config['multirun'])]
    dataset                      = load_data(args, custom)
    if not custom:
        dataset.load(config)  # config dataset
    else:
        dataset.load_custom_data(config, args)
    print_config(config)
    all_org_wei                  = []
    all_gen_wei                  = []
    if args.run_pretrain:
        pretrain_model           = GSC(config, dataset.input_dim).cuda()
        pretrain_model_para      = osp.join(args.pretrain_path, 'GSC_GNN_{}_checkpoint.pth'.format(args.dataset))
        pretrain_model.            load_state_dict(torch.load(pretrain_model_para))
        T                        = Trainer(config=config, args= args, logger= logger)

        model_mse, test_rho, test_tau, \
            test_prec_at_10, test_prec_at_20 = T.evaluation(dataset.testing_graphs, dataset.trainval_graphs, pretrain_model, torch.nn.MSELoss(), dataset, config)
        print_evaluation(model_mse, test_rho, test_tau, test_prec_at_10, test_prec_at_20)
    else:
        print("total graphs = {}"                                   .format(dataset.num_graphs))
        print("train_gs.len={} and val_gs.len={} and test_gs.len={}".format(dataset.num_train_graphs, dataset.num_val_graphs, dataset.num_test_graphs))
        for run_id in range(config['multirun']):   # one mask
            logger.add_line()
            logger.log ("\t\t%d th Run" % run_id)
            logger.add_line()
            # set_random_seed(seeds[run_id])
            # logger.log ("Seed set to %d." % seeds[run_id])

            model, best_metric_epoch ,report_mse_test, report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test, loss,PATH_MODEL, best_val_results = main(args, config, logger, run_id, dataset)

            logger.add_line()
            print_evaluation(report_mse_test,report_rho_test,report_tau_test,report_prec_at_10_test,report_prec_at_20_test)

            test_results = {
                'mse'       : report_mse_test,
                'rho'       : report_rho_test,
                'tau'       : report_tau_test,
                'prec_at_10': report_prec_at_10_test,
                'prec_at_20': report_prec_at_20_test
            }
            with open(osp.join(PATH_MODEL, 'result.txt'), 'w') as f:
                f.write('\n')
                for k, v       in   best_val_results.items():
                    f.write('%s: %s\n'         % (k, v))
                f.write('\n')
                for key, value in   test_results.items():
                    f.write('%s: %s\n'         % (key, value))

            if args.save_model:
                save_model(config, args.dataset, model)

        logger.add_line()