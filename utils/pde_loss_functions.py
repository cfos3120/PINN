import numpy as np
import torch
from torch.autograd import grad

# this is not referenced by any models yet.
def pde_loss(xy, model):
    xy = xy.clone()
    xy.requires_grad = True

    model_output = model(xy)
    u = model_output[...,0:1]
    v = model_output[...,1:2]
    p = model_output[...,2:3]

    # calculate derivatives with respect to input coordinates
    u_out = grad(u.sum(), xy, create_graph=True)[0]
    v_out = grad(v.sum(), xy, create_graph=True)[0]
    p_out = grad(p.sum(), xy, create_graph=True)[0]

    # assign first derivatives
    u_x = u_out[:, 0:1]
    u_y = u_out[:, 1:2]

    v_x = v_out[:, 0:1]
    v_y = v_out[:, 1:2]

    p_x = p_out[:, 0:1]
    p_y = p_out[:, 1:2]

    # assign second derivatives
    u_xx = grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
    v_xx = grad(v_x.sum(), xy, create_graph=True)[0][:, 0:1]
    v_yy = grad(v_y.sum(), xy, create_graph=True)[0][:, 1:2]

    # continuity equation
    f0 = u_x + v_y

    # navier-stokes equation
    f1 = u*u_x + v*u_y - (1/model.Re) * (u_xx + u_yy) + p_x
    f2 = u*v_x + v*v_y - (1/model.Re) * (v_xx + v_yy) + p_y

    return f0, f1, f2