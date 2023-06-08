import os, yaml
import numpy as np
from easydict import EasyDict

import torch
import torch.nn as nn

opts = EasyDict({
  'adaTrain': 1,
  'attention_size': 10,
  'batch_size': 8,
  'dropout': 0.5,
  'hidden_size': 128,
  'input_size': 3,
  'isBidirectional': True,
  'num_layers': 2,
  'nums_ada_epoch': 20,
  'nums_epoch': 200,
  'output_size': 1,
  'sequence_length': 300,
  'window_size': 30,
  'start_second': 30})

# LSTM model
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = torch.nn.LSTM(input_size = opts.input_size,
                                  hidden_size = opts.hidden_size, 
                                  num_layers = opts.num_layers,
                                  batch_first = True,
                                  bidirectional = opts.isBidirectional,
                                  dropout = opts.dropout)
        
        for name, param in self.lstm.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, 0.0)
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        self.out = torch.nn.Linear(opts.hidden_size*2, opts.output_size)
        
    def forward(self, sequence): # multi-to-one
        r_out,_ = self.lstm(sequence)
        out = self.out(r_out)

        return out