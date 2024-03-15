import numpy as np
import torch
from torch.autograd import grad
from pyDOE import lhs
import torch.nn as nn
from utils.dynamic_loss_weighting import *

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class layer(nn.Module):
    def __init__(self, n_in, n_out, activation):
        super().__init__()
        self.layer = nn.Linear(n_in, n_out)
        self.activation = activation

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x

class DNN(nn.Module):
    def __init__(self, dim_in, dim_out, n_layer, n_node, ub, lb, activation=nn.Tanh()):
        super().__init__()
        self.net = nn.ModuleList()
        self.net.append(layer(dim_in, n_node, activation))
        for _ in range(n_layer):
            self.net.append(layer(n_node, n_node, activation))
        self.net.append(layer(n_node, dim_out, activation=None))
        self.ub = torch.tensor(ub, dtype=torch.float).to(device)
        self.lb = torch.tensor(lb, dtype=torch.float).to(device)
        self.net.apply(weights_init)  # xavier initialization

    def forward(self, x):
        x = (x - self.lb) / (self.ub - self.lb)  # Min-max scaling
        out = x
        for layer in self.net:
            out = layer(out)
        return out

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        torch.nn.init.zeros_(m.bias.data)

""" 
MLP_cavity is a basic direct neural network (MLP). 
It has autograd functionality to calculate the derivatives for backwards passing.
It was based of flow around a sphere problem and was adapted.
Successful training requires an initial training with ADAM optimizer before then 
training again with LBFGS.
"""

class MLP_cavity:
    def __init__(self, ub, lb, lid_velocity=100, L=1, nu=0.05) -> None:
        self.net = DNN(dim_in=2, dim_out=3, n_layer=4, n_node=40, ub=ub, lb=lb).to(device)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [], "outlet": [], "pde": [], "f0": [], "f1": [], "f2": []}
        self.iter = 0

        self.Re = lid_velocity * L/nu

        # dynamic weights
        self.dw = dyn_l_ws()

    def predict(self, xy):
        out = self.net(xy)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        return u, v, p

    def bc_loss(self, xy, uv_bnd):
        u, v = self.predict(xy)[0:2]
        mse_bc = torch.mean(torch.square(u - uv_bnd[:, 0:1])) + torch.mean(
            torch.square(v - uv_bnd[:, 1:2])
        )
        return mse_bc

    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p = self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        p_out = grad(p.sum(), xy, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]

        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        p_x = p_out[:, 0:1]
        p_y = p_out[:, 1:2]

        u_xx = grad(u_x.sum(), xy, create_graph=True)[0][:, 0:1]
        u_yy = grad(u_y.sum(), xy, create_graph=True)[0][:, 1:2]
        v_xx = grad(v_x.sum(), xy, create_graph=True)[0][:, 0:1]
        v_yy = grad(v_y.sum(), xy, create_graph=True)[0][:, 1:2]

        # continuity equation
        f0 = u_x + v_y

        # navier-stokes equation
        f1 = u*u_x + v*u_y - (1/self.Re) * (u_xx + u_yy) + p_x
        f2 = u*v_x + v*v_y - (1/self.Re) * (v_xx + v_yy) + p_y

        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))

        return mse_f0, mse_f1, mse_f2

    def closure(self, xy_bnd, xy_col, uv_bnd):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(xy_bnd, uv_bnd)
        mse_f0, mse_f1, mse_f2 = self.pde_loss(xy_col)
        mse_pde = mse_f0 + mse_f1 + mse_f2

        loss = mse_bc + mse_pde

        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.losses["f0"].append(mse_f0.detach().cpu().item())
        self.losses["f1"].append(mse_f1.detach().cpu().item())
        self.losses["f2"].append(mse_f2.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss
    
    def closure_dyn(self, xy_bnd, xy_col, uv_bnd):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        
        mse_bc = self.bc_loss(xy_bnd, uv_bnd)
        mse_f0, mse_f1, mse_f2 = self.pde_loss(xy_col)
        mse_pde = self.dw.params[1]*mse_f0 + self.dw.params[2]*mse_f1 + self.dw.params[3]*mse_f2
        loss = self.dw.params[0]*mse_bc + mse_pde

        self.dw.calculate(model = self.net, l1=mse_bc, l2=mse_f0, l3=mse_f1, l4=mse_f2)

        loss.backward(retain_graph=True)

        self.dw.renorm()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.losses["f0"].append(mse_f0.detach().cpu().item())
        self.losses["f1"].append(mse_f1.detach().cpu().item())
        self.losses["f2"].append(mse_f2.detach().cpu().item())
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss.item()