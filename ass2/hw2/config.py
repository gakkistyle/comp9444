#!/usr/bin/env python3
"""
config.py

UNSW COMP9444 Neural Networks and Deep Learning

This file deteremines which device PyTorch will utilise.
You may change the variable device if you wish.
"""

import torch

# Use a GPU if available, as it should be faster.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
