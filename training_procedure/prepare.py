import torch
import torch.nn as nn
from importlib import import_module
import os.path as osp
from DataHelper.DatasetLocal import DatasetLocal
from model.GSC import GSC
from model.SImGNN import SimGNN
def prepare_train(self, model):
    config               = self.config
    optimizer            = getattr(torch.optim, config['optimizer'])(  params          = model.parameters(),
                                                                       lr              = config['lr'] ,
                                                                       weight_decay    = config.get('weight_decay', 0) )
    if config.get('lr_scheduler', False):
        scheduler        = torch.optim.lr_scheduler.StepLR(optimizer,  config['step_size'],gamma=config['gamma'])
    loss_func            = nn   .MSELoss(reduction='sum')
    return optimizer, loss_func

def prepare_model(self, dataset: DatasetLocal):
    config               = self.config
    model_name           = config['model_name']
    if model_name in ["GCN"]:
        Model_Class      = getattr    (import_module("model.GCN"), model_name)
        model            = Model_Class(config, config['hidden_dim'], dataset.nfeat, dataset.num_classes).cuda()
    if model_name in ["GAT"]:
        Model_Class      = getattr    (import_module("model.GAT"), model_name)
        model            = Model_Class(config, config['hidden_dim'], dataset.nfeat, dataset.num_classes).cuda()
    if model_name in ['GSC_GNN']:
        model            = GSC        (config, dataset.input_dim).cuda()
        if config['fuse_type']   == 'cat':
            in_dim       = sum        (config['gnn_filters'])
        elif config['fuse_type'] == 'stack':
            in_dim       = config['gnn_filters'][0]
        
    if model_name in ['SimGNN']:
        model            = SimGNN(config, )
    return model

def init(self, dataset):
    config               = self.config
    model                = self.prepare_model(dataset)
    optimizer, loss_func = self.prepare_train(model)
    
    return model, optimizer, loss_func