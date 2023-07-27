import abc
import torch
from torch_geometric.data import InMemoryDataset
import torch
import os
import os.path as osp
import numpy as np
from torch_geometric.utils import to_undirected
import shutil
import sys


class dataset:
    dataset_name = None
    dataset_descrition = None
    
    dataset_source_folder_path = None
    dataset_source_file_name = None
    
    data = None
    
    
    def __init__(self, dName=None, dDescription=None):
        self.dataset_name = dName
        self.dataset_descrition = dDescription
    
    def print_dataset_information(self):
        print('Dataset Name: ' + self.dataset_name)
        print('Dataset Description: ' + self.dataset_descrition)

    @abc.abstractmethod
    def load(self):
        return
    