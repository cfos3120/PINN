import numpy as np
from pyDOE import lhs
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def getData_cavity(dataset_args):

    x_min = dataset_args['x_min']
    x_max = dataset_args['x_max']
    y_min = dataset_args['y_min']
    y_max = dataset_args['y_max']
    # r = 0.05
    # xc = 0.5
    # yc = 0.5

    ub = dataset_args['ub']
    lb = dataset_args['lb']

    ### Data Prepareation ###
    N_b = dataset_args['N_b']  # inlet & outlet
    N_w = dataset_args['N_w']  # wall
    N_s = dataset_args['N_s']  # surface
    N_c = dataset_args['N_c']  # collocation
    N_r = dataset_args['N_r']


    inlet_x = np.random.uniform(x_min, x_max, (N_b, 1))
    inlet_y = np.ones((N_b, 1))*y_max
    inlet_u = np.ones((N_b, 1))
    inlet_v = np.zeros((N_b, 1))
    inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
    inlet_uv = np.concatenate([inlet_u, inlet_v], axis=1)

    lwall_xy = np.random.uniform([x_min, y_min], [x_min, y_max], (N_w, 2))
    rwall_xy = np.random.uniform([x_max, y_min], [x_max, y_max], (N_w, 2))
    bwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
    lwall_uv = np.zeros((N_w, 2))
    rwall_uv = np.zeros((N_w, 2))
    bwall_uv = np.zeros((N_w, 2))

    # cylinder surface, u=v=0
    # theta = np.linspace(0.0, 2 * np.pi, N_s)
    # cyl_x = (r * np.cos(theta) + xc).reshape(-1, 1)
    # cyl_y = (r * np.sin(theta) + yc).reshape(-1, 1)
    # cyl_xy = np.concatenate([cyl_x, cyl_y], axis=1)
    # cyl_uv = np.zeros((N_s, 2))

    # all boundary except outlet
    xy_bnd = np.concatenate([inlet_xy, lwall_xy, rwall_xy, bwall_xy], axis=0) #, cyl_xy], axis=0)
    uv_bnd = np.concatenate([inlet_uv, lwall_uv, rwall_uv, bwall_uv], axis=0) #, cyl_uv], axis=0)

    # Collocation
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # refine points around cylider
    # refine_ub = np.array([xc + 2 * r, yc + 2 * r])
    # refine_lb = np.array([xc - 2 * r, yc - 2 * r])

    # xy_col_refine = refine_lb + (refine_ub - refine_lb) * lhs(2, N_r)
    # xy_col = np.concatenate([xy_col, xy_col_refine], axis=0)

    # remove collocation points inside the cylinder
    # dst_from_cyl = np.sqrt((xy_col[:, 0] - xc) ** 2 + (xy_col[:, 1] - yc) ** 2)
    # xy_col = xy_col[dst_from_cyl > r].reshape(-1, 2)

    # concatenate all xy for collocation
    xy_col = np.concatenate((xy_col, xy_bnd), axis=0)

    # convert to tensor
    xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)
    return xy_col, xy_bnd, uv_bnd