import torch
import torch.nn as nn
from datetime import datetime, timedelta, timezone
import time
import gc

from pathlib import Path
import os, sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from autoformer.models import Autoformer

class Trade26(nn.Module):
    def __init__(self, pm, device='cpu', dtype=torch.bfloat16):
        super().__init__()
        self.pm = pm
        
        self.model = Autoformer.Model(pm)
        
        self.loss = nn.MSELoss()
        
        
    def predict(self, batch):
        values, marks = batch
        
        if len(values.shape) == 2: values = values.unsqueeze(-1)
        
        values = values.float().to(device='cuda')
        marks = marks.float().to(device='cuda')
        
        x_enc = values[:, :self.pm.seq_len, :]
        x_mark_enc = marks[:, :self.pm.seq_len, :]

        x_dec = torch.concat([x_enc[:, -self.pm.label_len:, :], torch.zeros_like(values[:, self.pm.seq_len : self.pm.seq_len + self.pm.pred_len, :])], dim=1)
        x_mark_dec = torch.concat([x_mark_enc[:, -self.pm.label_len:, :], marks[:, self.pm.seq_len : self.pm.seq_len + self.pm.pred_len, :]], dim=1)
        
        outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        pred = outputs[0] * 1 / 6       #------------------------------------------ 1 / 6
        true = values[:, self.pm.seq_len : self.pm.seq_len + self.pm.pred_len, :]
        
        return pred, true
        
 
    def forward_backward_no_optimizer_step(self, batch, train_dict, trainable, accum_substeps):
        pred, true = self.predict(batch)
        
        # values, marks = batch
        
        # if len(values.shape) == 2: values = values.unsqueeze(-1)
        
        # values = values.float().to(device='cuda')
        # marks = marks.float().to(device='cuda')
        
        # x_enc = values[:, :self.pm.seq_len, :]
        # x_mark_enc = marks[:, :self.pm.seq_len, :]
        # x_dec = torch.concat([x_enc[:, -self.pm.label_len:, :], torch.zeros_like(values[:, -self.pm.pred_len:, :])], dim=1)
        # x_mark_dec = torch.concat([x_mark_enc[:, -self.pm.label_len:, :], marks[:, -self.pm.pred_len:, :]], dim=1)
        
        # outputs = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # pred = outputs[0][:, -self.pm.pred_len:, :]
        # true = values[:, -self.pm.pred_len:, :]
        
        loss = self.loss(pred, true)
        
        if trainable:
            loss.backward()
        
        loss = loss.detach()
        metrics = loss.unsqueeze(0)
    
        return loss, metrics    # (), (C,)