import torch
from types import SimpleNamespace

class Model: 
    def __init__(self): pass   

class Params:
    def __init__(self, config):
        
        #============================ project (data, models) ===============================
        self.dataset_name = config['dataset_name']
        
        self.torch_dtype = torch.bfloat16 if config['dtype'] == 16 else torch.float32 if config['dtype'] == 32 else None
