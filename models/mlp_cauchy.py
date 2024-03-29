import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn

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

class PINN_cavity:
    rho = 1
    mu = 0.01 #0.02
    
    # cavity
    lid_velocity = 82
    L = 0.1
    nu = 0.01
    Re = lid_velocity * L/nu

    def __init__(self, ub, lb) -> None:
        self.net = DNN(dim_in=2, dim_out=6, n_layer=4, n_node=40, ub=ub, lb=lb).to(device)
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            tolerance_grad=1e-7, #1e-5, 
            tolerance_change=1.0 * np.finfo(float).eps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        self.adam = torch.optim.Adam(self.net.parameters(), lr=5e-4)
        self.losses = {"bc": [], "outlet": [], "pde": [], "f0": [], "f1": [], "f2": []}
        self.iter = 0

    def predict(self, xy):
        out = self.net(xy)
        u = out[:, 0:1]
        v = out[:, 1:2]
        p = out[:, 2:3]
        sig_xx = out[:, 3:4]
        sig_xy = out[:, 4:5]
        sig_yy = out[:, 5:6]
        return u, v, p, sig_xx, sig_xy, sig_yy

    def bc_loss(self, xy, uv_bnd):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p,__,__,__ = self.predict(xy)

        # Zero Gradient Pressure Boundary
        p_out = grad(p.sum(), xy, create_graph=True)[0]

        mse_bc = torch.mean(torch.square(u - uv_bnd[:, 0:1])) + \
                torch.mean(torch.square(v - uv_bnd[:, 1:2])) + \
                torch.mean(torch.square(p_out))
        return mse_bc

    def pde_loss(self, xy):
        xy = xy.clone()
        xy.requires_grad = True
        u, v, p, sig_xx, sig_xy, sig_yy = self.predict(xy)

        u_out = grad(u.sum(), xy, create_graph=True)[0]
        v_out = grad(v.sum(), xy, create_graph=True)[0]
        sig_xx_out = grad(sig_xx.sum(), xy, create_graph=True)[0]
        sig_xy_out = grad(sig_xy.sum(), xy, create_graph=True)[0]
        sig_yy_out = grad(sig_yy.sum(), xy, create_graph=True)[0]

        u_x = u_out[:, 0:1]
        u_y = u_out[:, 1:2]

        v_x = v_out[:, 0:1]
        v_y = v_out[:, 1:2]

        sig_xx_x = sig_xx_out[:, 0:1]
        sig_xy_x = sig_xy_out[:, 0:1]
        sig_xy_y = sig_xy_out[:, 1:2]
        sig_yy_y = sig_yy_out[:, 1:2]

        # continuity equation
        f0 = u_x + v_y

        # navier-stokes equation
        f1 = self.rho * (u * u_x + v * u_y) - sig_xx_x - sig_xy_y
        f2 = self.rho * (u * v_x + v * v_y) - sig_xy_x - sig_yy_y

        # cauchy stress tensor
        f3 = -p + 2 * self.mu * u_x - sig_xx
        f4 = -p + 2 * self.mu * v_y - sig_yy
        f5 = self.mu * (u_y + v_x) - sig_xy

        mse_f0 = torch.mean(torch.square(f0))
        mse_f1 = torch.mean(torch.square(f1))
        mse_f2 = torch.mean(torch.square(f2))
        mse_f3 = torch.mean(torch.square(f3))
        mse_f4 = torch.mean(torch.square(f4))
        mse_f5 = torch.mean(torch.square(f5))
        mse_pde = mse_f0 + mse_f1 + mse_f2 + mse_f3 + mse_f4 + mse_f5

        return mse_pde, mse_f0.detach().cpu().item(), mse_f1.detach().cpu().item(), mse_f2.detach().cpu().item()

    def closure(self):
        self.lbfgs.zero_grad()
        self.adam.zero_grad()

        mse_bc = self.bc_loss(self.xy_bnd, self.uv_bnd)
        #mse_outlet = self.outlet_loss(xy_outlet)
        mse_pde, mse_f0, mse_f1, mse_f2 = self.pde_loss(self.xy_col)
        #loss = mse_bc + mse_outlet + mse_pde
        loss = mse_bc + mse_pde

        loss.backward()

        self.losses["bc"].append(mse_bc.detach().cpu().item())
        #self.losses["outlet"].append(mse_outlet.detach().cpu().item())
        self.losses["pde"].append(mse_pde.detach().cpu().item())
        self.losses["f0"].append(mse_f0)
        self.losses["f1"].append(mse_f1)
        self.losses["f2"].append(mse_f2)
        self.iter += 1
        print(
            f"\r{self.iter} Loss: {loss.item():.5e} BC: {mse_bc.item():.3e} pde: {mse_pde.item():.3e}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")
        return loss
    
    def assign_dataset(self,xy_bnd,xy_col,uv_bnd):
        self.xy_bnd = xy_bnd
        self.xy_col = xy_col
        self.uv_bnd = uv_bnd