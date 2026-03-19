import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time
import gc

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from modules.uNet import uNet

class Trade26(nn.Module):
    def __init__(self, pm, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.pm = pm
        
        self.layer = nn.Linear(3, 3)
 
    def forward_backward_no_optimizer_step(self, batch, train_dict, accum_substeps):
        pass       
        
        # substep_loss, substep_metrics
        