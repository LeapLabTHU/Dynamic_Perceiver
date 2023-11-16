import sys
import time
import torch
import math
import torch.nn as nn
import torch.nn.init as init
import logging
import os
from collections import OrderedDict
import torch.nn.functional as F
import numpy as np
import random


def free(torch_tensor):
    return torch_tensor.cpu().detach().numpy()

def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("Dynamic_Perceiver")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def aggregate_dicts(dict, key, val):
    if  key not in dict:
        dict[key] = [val]
    else:
        dict[key].append(val)