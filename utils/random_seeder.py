import numpy as np
import torch as tc
import random

def set_random_seed(seed):		
	random .seed(seed)
	np     .random.seed(seed)
	tc     .manual_seed(seed)
	tc     .cuda.manual_seed_all(seed)
	tc     .backends.cudnn.deterministic = True
	tc     .backends.cudnn.benchmark     = False
