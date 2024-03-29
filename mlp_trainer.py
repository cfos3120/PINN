import numpy as np
import torch
from pyDOE import lhs
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

from models.mlp_cauchy import PINN_cavity

torch.manual_seed(1234)
np.random.seed(1234)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def getData_cavity(N_b,N_w,N_s,N_c,N_r, x_min=0.0,x_max = 1.0,y_min = 0.0,y_max = 1.0):
    
    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])
    
    inlet_x = np.random.uniform(x_min, x_max, (N_b, 1))
    inlet_y = np.ones((N_b, 1))*y_max
    inlet_u = np.ones((N_b, 1))
    inlet_v = np.zeros((N_b, 1))
    inlet_p = np.zeros((N_b, 1))
    inlet_xy = np.concatenate([inlet_x, inlet_y], axis=1)
    inlet_uv = np.concatenate([inlet_u, inlet_v, inlet_p], axis=1)

    lwall_xy = np.random.uniform([x_min, y_min], [x_min, y_max], (N_w, 2))
    rwall_xy = np.random.uniform([x_max, y_min], [x_max, y_max], (N_w, 2))
    bwall_xy = np.random.uniform([x_min, y_min], [x_max, y_min], (N_w, 2))
    lwall_uv = np.zeros((N_w, 3))
    rwall_uv = np.zeros((N_w, 3))
    bwall_uv = np.zeros((N_w, 3))

    # all boundary except outlet
    xy_bnd = np.concatenate([inlet_xy, lwall_xy, rwall_xy, bwall_xy], axis=0)
    uv_bnd = np.concatenate([inlet_uv, lwall_uv, rwall_uv, bwall_uv], axis=0)

    # Collocation
    xy_col = lb + (ub - lb) * lhs(2, N_c)

    # concatenate all xy for collocation
    xy_col = np.concatenate((xy_col, xy_bnd), axis=0)

    # convert to tensor
    xy_bnd = torch.tensor(xy_bnd, dtype=torch.float32).to(device)
    uv_bnd = torch.tensor(uv_bnd, dtype=torch.float32).to(device)
    xy_col = torch.tensor(xy_col, dtype=torch.float32).to(device)
    return xy_col, xy_bnd, uv_bnd

def eval_model(model_checkpoint_path, x_min=0.0,x_max = 1.0,y_min = 0.0,y_max = 1.0,resolution = 1000, show_fig=False):
    
    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])

    pinn = PINN_cavity(ub=ub,lb=lb)
    pinn.net.load_state_dict(torch.load(model_checkpoint_path,map_location=torch.device('cpu')))

    x = np.arange(resolution)/(resolution-1)
    y = np.arange(resolution)/(resolution-1)
    X, Y = np.meshgrid(x, y)
    x = X.reshape(-1, 1)
    y = Y.reshape(-1, 1)

    xy = np.concatenate([x, y], axis=1)
    xy = torch.tensor(xy, dtype=torch.float32)

    with torch.no_grad():
        u, v, p = pinn.predict(xy)
        u = u.cpu().numpy().reshape(Y.shape)
        v = v.cpu().numpy().reshape(Y.shape)
        p = p.cpu().numpy().reshape(Y.shape)

    data = (u, v, p)
    labels = ["$u(x,y)$", "$v(x,y)$", "$p(x,y)$"]

    if show_fig:
        fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
        for i in range(3):
            ax = axes[i]
            im = ax.imshow(
                data[i], cmap="rainbow", extent=[x_min, x_max, y_min, y_max], origin="lower"
            )
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad="3%")
            fig.colorbar(im, cax=cax, label=labels[i])
            ax.set_title(labels[i])
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_aspect("equal")
        fig.tight_layout()

    return data

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP PINN Training Study')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--Re', type=float, default=830)
    args = parser.parse_args()

    x_min = 0.0
    x_max = 1.0
    y_min = 0.0
    y_max = 1.0

    ub = np.array([x_max, y_max])
    lb = np.array([x_min, y_min])

    ### Data Prepareation ###
    N_b = 200  # inlet & outlet
    N_w = 400  # wall
    N_s = 200  # surface
    N_c = 40000  # collocation
    N_r = 10000

    xy_col, xy_bnd, uv_bnd = getData_cavity(N_b,N_w,N_s,N_c,N_r)
    pinn = PINN_cavity(ub=ub,lb=lb)

    pinn.assign_dataset(xy_bnd, xy_col, uv_bnd)
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    torch.save(pinn.net.state_dict(), "basic_model_weights_adam.pt")
    pinn.lbfgs.step(pinn.closure)
    torch.save(pinn.net.state_dict(), "basic_model_weights_lbfgs.pt")
    #plotLoss(pinn.losses)

    np.save('basic_training_losses.npy',pinn.losses, allow_pickle=True)