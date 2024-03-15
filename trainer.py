import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import matplotlib.pyplot as plt

import shutil
import argparse

from models.mlp_direct import *
from utils.pde_loss_functions import *
from utils.cavity_dataset import *

if __name__ == '__main__':
    
    # Dataset:
    dataset_args = dict()
    dataset_args['x_min'] = 0.0
    dataset_args['x_max'] = 1.0
    dataset_args['y_min'] = 0.0
    dataset_args['y_max'] = 1.0

    ub = np.array([dataset_args['x_max'], dataset_args['y_max']])
    lb = np.array([dataset_args['x_min'], dataset_args['y_min']])   
    
    dataset_args['ub'] = ub
    dataset_args['lb'] = lb

    dataset_args['N_b'] = 200       # inlet & outlet
    dataset_args['N_w'] = 400       # wall
    dataset_args['N_s'] = 200       # surface
    dataset_args['N_c'] = 40000     # collocation
    dataset_args['N_r'] = 10000

    # Dataset
    xy_col, xy_bnd, uv_bnd = getData_cavity(dataset_args)

    # Model
    pinn_model = MLP_cavity(ub=dataset_args['ub'], lb=dataset_args['lb'])
    
    # Phase 1 ADAM training:
    for i in range(1):
        pinn_model.closure_dyn(xy_bnd=xy_bnd, xy_col=xy_col, uv_bnd=uv_bnd)
        pinn_model.adam.step()
    
    # Phase 2 LBFGS training:
    pinn_model.lbfgs.step(pinn_model.closure_dyn(xy_bnd=xy_bnd, xy_col=xy_col, uv_bnd=uv_bnd))