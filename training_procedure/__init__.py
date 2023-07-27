import pdb
from .prepare import prepare_model, prepare_train, init
from .train import train
from .evaluate import evaluate
import torch
import torch.nn as nn
from functools import partial


class Trainer:
    def __init__(self, args, config, logger):
        self.config     = config
        self.logger     = logger
        self.args       = args
        self.flags      = {}
        self.split_info = None

        
        self.prepare_train = partial(prepare_train, self)
        self.prepare_model = partial(prepare_model, self)
        # self.prepare_sage = partial(prepare_sage, self)
        # self.load_data     = partial(load_data, self)
        self.init          = partial(init, self)

        self.train 		   = partial(train, self)
        self.evaluation    = partial(evaluate, self)		