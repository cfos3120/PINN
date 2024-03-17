import torch
import numpy as np
import argparse 
from models.gnot_custom import CGPTNO
from models.gnot_utils import MultipleTensors
from basic_trainer import getData_cavity

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def get_model(model_args):
    return CGPTNO(
            trunk_size          = model_args['trunk_size'],
            branch_sizes        = model_args['branch_sizes'],
            output_size         = model_args['output_size'],
            n_layers            = model_args['n_layers'],
            n_hidden            = model_args['n_hidden'],
            n_head              = model_args['n_head'],
            attn_type           = model_args['attn_type'],
            ffn_dropout         = model_args['ffn_dropout'],
            attn_dropout        = model_args['attn_dropout'],
            mlp_layers          = model_args['mlp_layers'],
            act                 = model_args['act'],
            horiz_fourier_dim   = model_args['hfourier_dim'])

def default_model_args():
    model_args = dict()
    model_args['trunk_size']        = 2
    model_args['theta_size']        = 1
    model_args['branch_sizes']      = [1]
    model_args['output_size']       = 3
    model_args['n_layers']          = 3
    model_args['n_hidden']          = 128  
    model_args['n_head']            = 1
    model_args['attn_type']         = 'linear'
    model_args['ffn_dropout']       = 0.0
    model_args['attn_dropout']      = 0.0
    model_args['mlp_layers']        = 2
    model_args['act']               = 'gelu'
    model_args['hfourier_dim']      = 0
    return model_args

def get_args():
    parser = argparse.ArgumentParser(description='GNOT Zero-Shot Autograd Training')
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--loss_criterion', type=str, default='MSE')
    args = parser.parse_args()
    return args

def get_loss_criterion(type='MSE'):
    if type == 'MSE':
        return torch.nn.MSELoss()
    elif type == 'MSE SUM':
        return torch.nn.MSELoss(reduction='sum')
    else: raise NotImplementedError(f'Loss type {type} is not implemented')

def navier_stokes_autograd(model,inputs,loss_function,Re=100):
    
    # Enable autograd:
    input, g_u = inputs
    input.requires_grad = True

    output = model(input,g_u)
    
    # Stack and Repeat Re for tensor multiplication
    Re = torch.tensor([Re]).reshape(1,1).repeat(output.shape[0],1)
    
    u = output[..., 0:1]
    v = output[..., 1:2]
    p = output[..., 2:3]

    # First Derivatives
    u_out = torch.autograd.grad(u.sum(), input, create_graph=True)[0]
    v_out = torch.autograd.grad(v.sum(), input, create_graph=True)[0]
    p_out = torch.autograd.grad(p.sum(), input, create_graph=True)[0]

    u_x = u_out[..., 0:1]
    u_y = u_out[..., 1:2]

    v_x = v_out[..., 0:1]
    v_y = v_out[..., 1:2]

    p_x = p_out[..., 0:1]
    p_y = p_out[..., 1:2]
    
    # Second Derivatives
    u_xx = torch.autograd.grad(u_x.sum(), input, create_graph=True)[0][..., 0:1]
    u_yy = torch.autograd.grad(u_y.sum(), input, create_graph=True)[0][..., 1:2]
    v_xx = torch.autograd.grad(v_x.sum(), input, create_graph=True)[0][..., 0:1]
    v_yy = torch.autograd.grad(v_y.sum(), input, create_graph=True)[0][..., 1:2]

    # Continuity equation
    f0 = u_x + v_y

    # Navier-Stokes equation
    f1 = u*u_x + v*u_y - (1/Re) * (u_xx + u_yy) + p_x
    f2 = u*v_x + v*v_y - (1/Re) * (v_xx + v_yy) + p_y

    f0_loss = loss_function(f0,torch.zeros_like(f0))
    f1_loss = loss_function(f1,torch.zeros_like(f1))
    f2_loss = loss_function(f2,torch.zeros_like(f2))
    return f0_loss, f1_loss, f2_loss



if __name__ == '__main__':
    
    # Construct Default Model
    model = get_model(default_model_args()).to(device)
    
    # Data Prepareation
    xy_col, xy_bnd, uv_bnd = getData_cavity(N_b=200,N_w=400,N_s=200,N_c=16000,N_r=10000)
    print(f'Collocation Points: {xy_col.shape[0]}\nBoundary Points:{xy_bnd.shape[0]}\nComparison Channels:{uv_bnd.shape[-1]}')
    # Input Function (lid velocity)
    lid_velocity = 82.0
    nu = 0.01
    L = 0.1
    Re = lid_velocity * L/nu
    g_u = MultipleTensors(torch.tensor([lid_velocity]).reshape(1,1,1,1))
    
    out = model(x=xy_col, inputs=g_u)

    # Training Setup
    args = get_args()
    milestones = np.linspace(0,args.epochs,6,dtype=int)[1:-1]
    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.999), lr=0.001, weight_decay=0.00005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.05)
    loss_func = get_loss_criterion(args.loss_criterion)
    print(f'Using AdamW Optimzer, With Multi-Step Scheduler on Epochs {milestones}')
    recorded_losses = {"bc": [], "outlet": [], "pde": [], "f0": [], "f1": [], "f2": []}
    # send to device
    xy_col, xy_bnd, uv_bnd = xy_col.to(device), xy_bnd.to(device), uv_bnd.to(device)
    
    # Train Model
    model.train()
    for epoch in range(args.epochs):    
        optimizer.zero_grad()

        # Soft Enforce Boundary Conditions
        output = model(xy_bnd,g_u)
        bc_loss = loss_func(output[0,...,:2],uv_bnd)

        # Evaluate PDE
        f0, f1, f2 = navier_stokes_autograd(model=model,inputs=[xy_col,g_u],loss_function=loss_func,Re=Re)
        pde_loss = (f0+f1+f2)/3
        total_loss = bc_loss+pde_loss

        # Backwards Step
        total_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1000.0)
        optimizer.step()
        scheduler.step()

        # Store Results
        recorded_losses["bc"].append(bc_loss.detach().cpu().item())
        recorded_losses["pde"].append(pde_loss.detach().cpu().item())
        recorded_losses["f0"].append(f0.detach().cpu().item())
        recorded_losses["f1"].append(f1.detach().cpu().item())
        recorded_losses["f2"].append(f2.detach().cpu().item())

    # Save Model and Losses
    torch.save(model.state_dict(), f"{args.name}_model_weights.pt")
    np.save(f"{args.name}_training_losses.pt", recorded_losses, allow_pickle=True)
