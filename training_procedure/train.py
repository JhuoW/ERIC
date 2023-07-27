
import torch.nn as nn
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm

def train(self, graph_batch, model, loss_func, optimizer, target, dataset = None):
    model.train(True)

    config                        = self.config   
    use_ssl                       = config.get('use_ssl', False)  
    optimizer.zero_grad()

    if config['model_name'] in ['GSC_GNN']: 
        # if not config['use_sim']:
        prediction, loss_cl       = model(graph_batch)
        loss                      = loss_func(prediction, target) if not use_ssl else loss_func(prediction, target)+loss_cl
        loss.backward()
        if self.config.get('clip_grad', False):
            nn.utils.clip_grad_norm_(model.parameters(), 1)
        # else:
        #     prediction_diff, prediction_sim = model(graph_batch)
        #     loss = (loss_func(prediction_diff, target) + loss_func(prediction_sim, target))/2
        # prediction = classifier(score_vec)
    elif config['model_name'] in ['SimGNN']:
        prediction                = model(graph_batch)
        loss                      = loss_func(prediction, target)
        loss.backward()

    optimizer.step()
    
    return model, float(loss)
