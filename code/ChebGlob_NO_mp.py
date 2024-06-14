#Hello OneDrive
"""
Implementation of global Chebyshev basis for the Darcy problem in L-shaped domain 
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import operator
from functools import reduce
from tqdm import tqdm 
# pytorch libraries
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, jaxtyped
from beartype import beartype
from torch.utils.tensorboard import SummaryWriter
# library for reading and writing data
import mat73
import pandas as pd

import sys
sys.path.append('..')
import Chebyshev_pytorch as cheb

#########################################
# parameters for save the model
#########################################
model_folder = 'test' # folder's name for the test
# description_test = "mp_ChebGlob_H1_Stupid" # name that describe the test
description_test = "test_mp_modes12_H1"
# Save the models here:
folder = model_folder + "/exp_" + description_test
name_model = model_folder + "/model_" + description_test

#########################################
# Hyperparameters
#########################################
training_properties = {
    "training_samples": 1000,# number of samples in the train set
    "test_samples": 200,     # number of samples in the test set
    "learning_rate": 0.001, # initial value for learning rate
    "scheduler": "cosineannealinglr", # "steplr", "cosineannealinglr"
    "epochs": 500,           # number of epochs for training
    "batch_size": 20,        # batch dimension
    "weight_decay": 1e-4,    # L^2 regularization
    "step_size": 200,         # step size for the scheduler
    "gamma": 0.6,            # gamma for the scheduler
    "loss": 'H1',            # 'L2', 'H1'
    "beta": 1.0,             # hyperparameter for the H^1 relative norm
    "s": 5,                  # subsample the grid
    "BC": False,             # penalty loss for Boundary Condition
    "alpha_BC": 1.0,         # coefficient for BC for training
    "cont_loss": False,   # penality loss of discontinuity on the internal boundary fo H^1 for training
    "alpha": 1.0,         # coefficient for cont_loss for training
}
gcno_architecture = {
    "RNN": False,      # if True we use the RNN architecture, otherwise the classic one
    "arc": 'Zongyi',  # 'Residual', 'Tran', 'Classic', 'Zongyi
    "dropout": 0.0,    # dropout rate
    "n_patch": 3,      # number of patches
    "d_a": 3,          # input dimension
    "width": 32,       # hidden dimension
    "d_u": 1,          # output dimension
    "n_layers": 4,     # L = depth
    "modes": 12,       # Cheb modes
    "fun_act": 'gelu', # 'relu', 'gelu', 'tanh', 'leaky_relu'
    "same_params": True, # if True we use the same parameters for all the layers
    "retrain": -1,     # we fix the seed for retrain only if retrain >= 0
}
gcno_architecture["weights_norm"] = "Xavier" if gcno_architecture["fun_act"] == 'gelu' else "Kaiming"

#########################################
# default values
#########################################
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mydevice = torch.device('cpu') # to force to use cpu
torch.set_default_device(mydevice) # default tensor device
torch.set_default_dtype(torch.float32) # default tensor dtype
TrainDataPath = 'data/Darcy_Lshape_chebyshev_grid_pc_train.mat'
TestDataPath = 'data/Darcy_Lshape_chebyshev_grid_pc_test.mat'

#########################################
# seed for extract dataset
#########################################
ntrain = training_properties["training_samples"] 
ntest = training_properties["test_samples"] 

#########################################
# hyperparameter for the neural operataor
#########################################   
#### training hyperparameter   
learning_rate = training_properties["learning_rate"]
scheduler     = training_properties["scheduler"]
epochs        = training_properties["epochs"]
batch_size    = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]

if scheduler == "steplr":
    step_size = training_properties["step_size"]
    gamma     = training_properties["gamma"]
elif scheduler == "cosineannealinglr":
    iterations    = epochs*(ntrain//batch_size)

Loss          = training_properties["loss"] # realtive L^2 or relative H^1
beta          = training_properties["beta"] # norm_{L_2} + beta * seminorm_{H_1} 
s             = training_properties["s"]    # mesh's parameter: s=3 --> 70, s=1 --> 211
BC            = training_properties["BC"]
alpha_BC      = training_properties["alpha_BC"]
cont_loss     = training_properties["cont_loss"]
alpha         = training_properties["alpha"]

#### model's hyperparameter
RNN          = gcno_architecture["RNN"]
arc          = gcno_architecture["arc"]
dropout      = gcno_architecture["dropout"]
n_patch      = gcno_architecture["n_patch"]
d_a          = gcno_architecture["d_a"] 
d_v          = gcno_architecture["width"]  
d_u          = gcno_architecture["d_u"]  
L            = gcno_architecture["n_layers"]
modes        = gcno_architecture["modes"] # k_{max,j}
fun_act      = gcno_architecture["fun_act"] # activation function, (kaiming init)
retrain_gcno = gcno_architecture["retrain"]
weights_norm = gcno_architecture["weights_norm"]
same_params  = gcno_architecture["same_params"]

#########################################
# tensorboard and plot variables
#########################################   
ep_step = 20 # save the plot on tensorboard every ep_Step epochs
idx = [0, 42, 93, 158] # casual number from 0 to n_test-1
# idx = [0]
plotting = False # if True we make the plot

#########################################
# function for reading data
#########################################
@jaxtyped(typechecker=beartype)
def MatReader(file_path:str) ->tuple[Float[Tensor, "n_samples n_patch n_x n_y"], 
                                      Float[Tensor, "n_samples n_patch n_x n_y"],
                                      Float[Tensor, "n_patch*n_x*n_y-n_x-n_y 2"]]:
    """
    Function to read .mat files version 7.3

    Parameters
    ----------
    file_path : string
        path of the .mat file to read        

    Returns
    -------
    a : tensor
        evaluations of the function a(x) of the Darcy problem 
        dimension = (n_samples)*(n_patch)*(n_x)*(n_y)
    u : tensor
        approximation of the solution u(x) obtained with a
        standard method (in our case isogeometric)
        dimension = (n_samples)*(n_patch)*(n_x)*(n_y)
    nodes : tensor
        nodes of the mesh
        dimension = (n_patch*n_x*n_y - n_x - n_y)*(2)
    """
    data = mat73.loadmat(file_path)
    
    a = data["COEFF"]
    a = torch.from_numpy(a).float() # transform np.array in torch.tensor
    
    u = data["SOL"]
    u = torch.from_numpy(u).float()
    
    nodes = data["nodes"]
    nodes = torch.from_numpy(nodes).float()
    
    return a, u, nodes

#########################################
# concatenate and set to nan for plt
#########################################
@jaxtyped(typechecker=beartype)
def helper_plot_data(x:Float[Tensor, "n_samples n_patch n_x n_y"]) -> Float[Tensor, "n_samples n_x*2-1 n_y*2-1"]:
    """ 
    Helper function to plot the data x in the L-shaped domain.

    x:  torch.tensor() 
        x is a tensor of shape (n_samples, n_patch, n_x, n_y) saved in the cpu.
        here the tensor is already flipped along y-axis.
    """    
    return torch.cat( 
        (torch.cat(
            (x[:, 1, :, :], 
            x[:, 2, 1:, :]), 
            dim = 1), 
        torch.cat(
            (x[:, 0, :, 1:], 
              torch.zeros_like(x[:, 0, 1:, :-1], device = 'cpu')*np.nan), 
            dim = 1)),
        dim = 2)

#########################################
# function for plots data
#########################################
def plot_data(data_plot:Tensor, idx:list, title:str, ep:int, plotting:bool = True):
    """ 
    Function to makes the plots of the data.
    
    data_plot: torch.tensor
        data_plot is a tensor of shape (n_samples, n_patch, n_x, n_y).
    """
    # select the data to plot
    data_plot = torch.flip(data_plot, [3])
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
        n_idx = data_plot.size(0)
    data_plot = helper_plot_data(data_plot).to('cpu')
    # plot
    fig, ax = plt.subplots(1, n_idx, figsize = (18, 4))
    fig.suptitle(title)
    ax[0].set(ylabel = 'y')
    for i in range(n_idx):
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])
        ax[i].set(xlabel = 'x')
        im = ax[i].imshow(data_plot[i])
        fig.colorbar(im, ax = ax[i])
    if plotting:
        plt.show()
    # save the plot on tensorboard
    writer.add_figure(title, fig, ep)

#########################################
# initial normalization
#########################################    
class UnitGaussianNormalizer(object):
    """ 
    Initial normalization on x, which is a tensor of 
    dimension: (n_samples)*(n_patch)*(nx)*(ny)
    normalization --> pointwise gaussian
    """
    def __init__(self, x:Tensor, eps:float=1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    @jaxtyped(typechecker=beartype)
    def encode(self, x:Float[Tensor, "n_samples n_patch n_x n_y"])->Float[Tensor, "n_samples n_patch n_x n_y"]:
        x = (x - self.mean)/(self.std + self.eps)
        return x
    
    @jaxtyped(typechecker=beartype)
    def decode(self, x:Float[Tensor, "n_samples n_patch n_x n_y"])->Float[Tensor, "n_samples n_patch n_x n_y"]:
        x = x*(self.std + self.eps) + self.mean
        return x
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
    
#########################################
# activation function
#########################################
@jaxtyped(typechecker=beartype)
def activation(x:Float[Tensor, "n_samples n_patch n_x n_y d"], activation_str:str) -> Float[Tensor, "n_samples n_patch n_x n_y d"]:
    """
    Activation function to be used within the network.
    The function is the same throughout the network.
    """
    if activation_str == 'relu':
        return F.relu(x)
    if activation_str == 'gelu':
        return F.gelu(x)
    if activation_str == 'tanh':
        return F.tanh(x)
    if activation_str == 'leaky_relu':
        return F.leaky_relu(x)

#########################################
# L2 relative loss function
#########################################
class L2relLoss():
    """ 
    Sum of relative errors in L^2 norm 
    
    x, y: torch.tensor
          x and y are tensors of shape (n_samples, n_patch, n_x, n_y)
    """ 
    def __init__(self, BC:bool = False, alpha_BC:float = 1.0, cont_loss:bool = False, alpha:float = 1.0):
        self.BC = BC
        self.alpha_BC = alpha_BC
        self.cont_loss = cont_loss
        self.alpha = alpha
    
    @jaxtyped(typechecker=beartype)
    def concatenate(self, x:Float[Tensor, "n_samples n_patch n_x n_y"]) -> Float[Tensor, "n_samples n_patch*n_x-1 n_y"]:
        """ 
        With this function we concatenate the patches without repeating points on internal bound.
        So we have to pad with zeros were we have the internal boundary repetition.

        x: torch.tensor
           x is a tensor of shape (n_samples, n_patch, n_x, n_y)        
        """
        x = torch.cat(
            (
                x[:, 0, :, :], # patch 0
                x[:, 2, :, :], # patch 2
                torch.cat(
                    (torch.zeros_like(x[:, 1, :-1, [0]]), 
                     x[:, 1, :-1, 1:]), 2) # patch 1 without repetition
            ), 1)
        return x

    @jaxtyped(typechecker=beartype)  
    def rel(self, x:Float[Tensor, "n_samples n_patch n_x n_y"], 
                  y:Float[Tensor, "n_samples n_patch n_x n_y"]) -> Float[Tensor, ""]:
        num_examples = x.size(0)
        
        ## Impose additional loss term for the boundary conditions
        if self.BC:
            BC_discont = torch.cat(
                            (x[:, 0, -1, :], # Boundary patch 0
                            x[:, 0, 0, :], 
                            x[:, 0, :, 0],
                            x[:, 1, 0, :], # Boundary patch 1
                            x[:, 1, :, -1],
                            x[:, 2, -1, :], # Boundary patch 2
                            x[:, 2, :, -1],
                            x[:, 2, :, 0]
                            ), dim = 1)
            BC_discont = torch.norm(BC_discont, 2, dim = 1) # l^2 norm
            BC_discont = torch.sum(BC_discont) # sum on the batch dim
        else:
            BC_discont = 0.0
        
        ## Impose additional loss term for the discontinuety on the internal boundary
        if self.cont_loss:
            discont =  torch.cat(
                ((x[:, 1, -1, :] - x[:, 2, 0, :]),
                 (x[:, 1, :, 0] - x[:, 0, :, -1])), 1)
            discont = torch.norm(discont, 2, dim = 1) # l^2 norm
            discont = torch.sum(discont) # sum on the batch dim
        else:
            discont = 0.0
        
        ## concatenate the patches without repeating points on internal bound
        x = self.concatenate(x)
        y = self.concatenate(y)

        ## Compute the relative L^2 norm 
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        
        ## sum all the terms
        return (torch.sum(diff_norms/y_norms) + # sum along batchsize
                self.alpha*discont + 
                self.alpha_BC*BC_discont)
    
    @jaxtyped(typechecker=beartype)
    def __call__(self,  x:Float[Tensor, "n_samples n_patch n_x n_y"], 
                        y:Float[Tensor, "n_samples n_patch n_x n_y"]) -> Float[Tensor, ""]:
        return self.rel(x, y)

#########################################
# H1 relative loss function
#########################################
class H1relLoss_cheb():
    """ 
    Relative H^1 = W^{1,2} norm with Chebyshev transform. It is calculated as the sum of the
    """
    def __init__(self, beta:float=1.0):
        self.beta = beta

    @jaxtyped(typechecker=beartype)
    def rel(self, x:Float[Tensor, "n_samples dx dy"], 
                  y:Float[Tensor, "n_samples dx dy"]) -> tuple[Float[Tensor, "n_samples"], Float[Tensor, "n_samples"]]:
        batch_size = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(batch_size, -1) - y.reshape(batch_size,-1), 2, 1)**2
        y_norms = torch.norm(y.reshape(batch_size, -1), 2, 1)**2
        
        return diff_norms, y_norms
    
    def resize(self, x:Tensor, i:int) -> Tensor:
        if i == 1:
            return x[:, :-1, 1:] # non repeting points on the internal boundary
        else:
            return x

    @jaxtyped(typechecker=beartype)
    def __call__(self, x:Float[Tensor, "n_sample n_patch n_x n_y"], 
                       y:Float[Tensor, "n_samples n_patch n_x n_y"]) -> Float[Tensor, ""]:
        n_patch = x.size(1)
        for i in range(n_patch):
            x_p, y_p = x[:, i, :, :], y[:, i, :, :]

            # relative L^2 norm
            diff, y_norm = self.rel(self.resize(x_p, i), self.resize(y_p, i))
            x_p = cheb.batched_values_to_coefficients(x_p.unsqueeze(-1)) # c=1 as last dimension (required channel dimension)
            y_p = cheb.batched_values_to_coefficients(y_p.unsqueeze(-1))

            # derivative in x
            x_p_dx = cheb.batched_coefficients_to_values(cheb.batched_differentiate(x_p, 1)).squeeze(-1)
            y_p_dx = cheb.batched_coefficients_to_values(cheb.batched_differentiate(y_p, 1)).squeeze(-1)
            diff_dx, y_p_dx = self.rel(x_p_dx, y_p_dx)

            # derivative in y
            x_p_dy = cheb.batched_coefficients_to_values(cheb.batched_differentiate(x_p, 0)).squeeze(-1)
            y_p_dy = cheb.batched_coefficients_to_values(cheb.batched_differentiate(y_p, 0)).squeeze(-1)
            diff_dy, y_p_dy = self.rel(x_p_dy, y_p_dy)

            if i == 0:
                diff_tot = diff
                diff_dx_tot = diff_dx
                diff_dy_tot = diff_dy
                y_tot = y_norm 
                y_p_dx_tot = y_p_dx
                y_p_dy_tot = y_p_dy
            else:
                diff_tot += diff 
                diff_dx_tot += diff_dx
                diff_dy_tot += diff_dy
                y_tot += y_norm 
                y_p_dx_tot += y_p_dx
                y_p_dy_tot += y_p_dy

        diff_tot = torch.sqrt(diff_tot) + self.beta*torch.sqrt(diff_dx_tot + diff_dy_tot)
        y_tot = torch.sqrt(y_tot) + self.beta*torch.sqrt(y_p_dx_tot + y_p_dy_tot)

        return torch.sum(diff_tot/y_tot)

#########################################
# chebyshev layer
#########################################
class ChebyshevLayer(nn.Module):
    def __init__(self, n_patch:int, in_channels:int, out_channels:int, modes1:int, modes2:int, 
                 M:Tensor, M_1:Tensor, weights_norm:str, fun_act:str, arc:str, same_params:bool=False):
        """
        2D Integral layer with boundary adapted Chebyshev basis 

        n_patch : int
            number of patches
            
        in_channels : int
            input dimension (d_v in the theory)
        
        out_channels : int
            output dimension (d_{v+1} in the theory)

        modes1 : int
            number of modes in the x direction
            
        modes2 : int
            number of modes in the y direction
        
        M : torch.tensor
            Matrix for the change of basis
            
        M_1 : torch.tensor
            Matrix for the inverse change of basis
            
        weights_norm : str
            string for selecting the weights normalization to use
            
        fun_act : str
            string for selecting the activation function to use, needed for the weights normalization
            
        arc : str
            string for selecting the architecture to use, needed for the boundary conditions
            
        same_params : bool
            if True we use the same parameters for all the patches
        """
        super(ChebyshevLayer, self).__init__()
        self.n_patch = n_patch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.M = M
        self.M_1 = M_1
        self.weights_norm = weights_norm
        self.fun_act = fun_act
        self.arc = arc
        self.same_params = same_params
                           
        if self.same_params: # same parameters for all the patches
            if self.weights_norm == 'Xavier':
                # Xavier normalization
                self.weights = nn.init.xavier_normal_(
                    nn.Parameter(torch.empty(self.modes1, self.modes2, self.in_channels, self.out_channels)),
                    gain = 1/(self.in_channels*self.out_channels))
            elif self.weights_norm == 'Kaiming':
                # Kaiming normalization
                self.weights = torch.nn.init.kaiming_normal_(
                    nn.Parameter(torch.empty(self.modes1, self.modes2, self.in_channels, self.out_channels)),
                    a = 0, mode = 'fan_in', nonlinearity = self.fun_act)
        else:
            if self.weights_norm == 'Xavier':
                # Xavier normalization
                self.weights = nn.init.xavier_normal_(
                    nn.Parameter(torch.empty(self.n_patch, self.modes1, self.modes2, self.in_channels, self.out_channels)),
                    gain = 1/(self.in_channels*self.out_channels))
            elif self.weights_norm == 'Kaiming':
                # Kaiming normalization
                self.weights = torch.nn.init.kaiming_normal_(
                    nn.Parameter(torch.empty(self.n_patch, self.modes1, self.modes2, self.in_channels, self.out_channels)),
                    a = 0, mode = 'fan_in', nonlinearity = self.fun_act)

    @jaxtyped(typechecker=beartype)
    def tensor_mul(self, input:Float[Tensor, "n_batch n_patch modes1 modes2 d_i"], 
                         weights:Float[Tensor, "n_patch modes1 modes2 d_i d_o"]) -> Float[Tensor, "n_batch n_patch modes1 modes2 d_o"]:
        return torch.einsum("bpxyi,pxyio->bpxyo", input, weights)

    @jaxtyped(typechecker=beartype)
    def tensor_mul_same_params(self, input:Float[Tensor, "n_batch n_patch modes1 modes2 d_i"], 
                         weights:Float[Tensor, "modes1 modes2 d_i d_o"]) -> Float[Tensor, "n_batch n_patch modes1 modes2 d_o"]:
        return torch.einsum("bpxyi,xyio->bpxyo", input, weights)

    @jaxtyped(typechecker=beartype)
    def forward(self, x:Float[Tensor, "n_batch n_patch n_x n_y d_i"]) -> Float[Tensor, "n_batch n_patch n_x n_y d_o"]:
        """ 
        input --> CFT --> boundary adapted + continuity --> parameters --> inverse boundary adapted --> ICFT --> output  
        Total computation cost is equal to O(n log(n))
        
        input: torch.tensor
            the input 'x' is a tensor of shape (n_samples, n_patch, n_x, n_y, in_channels)
        
        output: torch.tensor
            return a tensor of shape (n_samples, n_patch, n_x, n_y, out_channels)
        """
        batch_size, n_patch, n_x, n_y = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        
        # CFT
        x_ft = cheb.patched_values_to_coefficients(x)

        ### Boundary adapted transform
        # x_ft = cheb.patched_change_basis(x_ft, self.M_1)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batch_size, n_patch, n_x, n_y, self.out_channels, device = x.device)
        if self.same_params:
            out_ft[:, :, :self.modes1, :self.modes2, :] = self.tensor_mul_same_params(x_ft[:, :, :self.modes1, :self.modes2, :], self.weights)
        else:
            out_ft[:, :, :self.modes1, :self.modes2, :] = self.tensor_mul(x_ft[:, :, :self.modes1, :self.modes2, :], self.weights)

        ### Boundary adapted transform
        out_ft = cheb.patched_change_basis(out_ft, self.M_1)

        # Boundary condition
        if self.arc == 'Strong_BC':
            out_ft[:, 0, :, 0, :] = torch.zeros_like(out_ft[:, 0, :, 0, :])
            out_ft[:, 0, 0, :, :] = torch.zeros_like(out_ft[:, 0, 0, :, :])
            out_ft[:, 0, 1, :, :] = torch.zeros_like(out_ft[:, 0, 1, :, :])

            out_ft[:, 1, :, 1, :] = torch.zeros_like(out_ft[:, 1, :, 1, :])
            out_ft[:, 1, 0, :, :] = torch.zeros_like(out_ft[:, 1, 0, :, :])

            out_ft[:, 2, 1, :, :] = torch.zeros_like(out_ft[:, 2, 1, :, :])
            out_ft[:, 2, :, 0, :] = torch.zeros_like(out_ft[:, 2, :, 0, :])
            out_ft[:, 2, :, 1, :] = torch.zeros_like(out_ft[:, 2, :, 1, :])

        # continuity condition
        tmp1 = (out_ft[:, 0, :self.modes1, 1, :] + out_ft[:, 1, :self.modes1, 0, :])/2
        tmp2 = (out_ft[:, 2, 0, :self.modes2, :] + out_ft[:, 1, 1, :self.modes2, :])/2
        tmp12 = (out_ft[:, 0, 1, 1, :] + out_ft[:, 1, 1, 0, :] + out_ft[:, 2, 0, 0, :])/3
        tmp1[:, 1, :] = tmp12
        tmp2[:, 0, :] = tmp12
        out_ft[:, 0, :self.modes1, 1, :] = tmp1
        out_ft[:, 1, :self.modes1, 0, :] = tmp1
        out_ft[:, 2, 0, :self.modes2, :] = tmp2
        out_ft[:, 1, 1, :self.modes2, :] = tmp2
        # out_ft[:, 0, :self.modes1, 1, :] = out_ft[:, 1, :self.modes1, 0, :]
        # out_ft[:, 2, 0, :self.modes2, :] = out_ft[:, 1, 1, :self.modes2, :]

        # Inverse boundary adapted transform
        out_ft = cheb.patched_change_basis(out_ft, self.M)

        # ICFT
        x = cheb.patched_coefficients_to_values(out_ft)

        return x

#########################################
# MLP
#########################################
class MLP(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, mid_channels:int, fun_act:str, arc:str, dropout:float=0.0):
        """ Shallow neural network with one hidden layer """
        super(MLP, self).__init__()
        self.arc = arc
        if self.arc == 'Strong_BC':
            self.bias = False
        else:
            self.bias = True
        self.mlp1 = nn.Linear(in_channels, mid_channels, bias = self.bias)
        self.mlp2 = nn.Linear(mid_channels, out_channels, bias = self.bias)
        self.fun_act = fun_act
        self.dropout = dropout
        self.LayerDropout = nn.Dropout(p = self.dropout)

    @jaxtyped(typechecker=beartype)
    def forward(self, x:Float[Tensor, "n_batch n_patch n_x n_y d_i"]) -> Float[Tensor, "n_batch n_patch n_x n_y d_o"]:
        """ forward pass of the MLP model """
        x = self.mlp1(x) # affine transformation
        x = self.LayerDropout(x) # dropout
        x = activation(x, self.fun_act) # activation function
        x = self.mlp2(x) # affine transformation
        return x
    
#########################################
# Patch_GCNO for Darcy on L-shaped
#########################################
class GlobalCheb(nn.Module):
    def __init__(self, n_patch:int, n_x:int, n_y:int, d_a:int, d_v:int, d_u:int, L:int, modes1:int, modes2:int, 
                 fun_act:str, weights_norm:str, arc:str, same_params:bool=False, dropout:float=0.0, 
                 RNN:bool=False, retrain_gcno:int = -1):
        """         
        Global Chebyshev basis for the Darcy problem in L-shaped domain

        n_patch : int
            number of patches

        n_x : int
            number of points in the x direction

        n_y : int
            number of points in the y direction

        d_a : int
            dimension of the input space
            
        d_v : int
            dimension of the space in the integral (Fourier or Chebyshev) operator
            
        d_u : int
            dimension of the output space 

        L: int
            number of integral operators (Fourier or Chebyshev) to perform
            
        mode1 : int
            equal to k_{max, 1}
            
        mode2 : int
            equal to k_{max, 2}
            
        fun_act: str
            string for selecting the activation function to use throughout the 
            architecture

        weights_norm: str
            string for selecting the weights normalization to use throughout the
            architecture

        arc: str
            string for selecting the architecture to use.
            'Classic' for the classic and first proposed architecture
            'Strong_BC' here I impose the BC strongly here
            'Tran' for the architecture proposed by Alasdarian Tran
        
        same_params: bool
            if True we use the same parameters for all the patches

        dropout: float
            dropout rate

        RNN : bool
            if True we use the RNN architecture, otherwise the classic one

        retrain_gcno : int
            if retrain_gcno >= 0, we fix the seed for retrain
        """
        super(GlobalCheb, self).__init__()
        self.n_patch = n_patch
        self.n_x = n_x
        self.n_y = n_y
        self.d_a = d_a
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes1 = modes1
        self.modes2 = modes2
        self.fun_act = fun_act
        self.weights_norm = weights_norm
        self.same_params = same_params
        self.arc = arc
        self.dropout = dropout
        self.RNN = RNN
        self.retrain_gcno = retrain_gcno

        if self.arc == 'Strong_BC':
            self.bias = False # no bias for exact zero BC
        else:
            self.bias = True
        
        # matrix for the change of basis
        assert self.modes1 == self.modes2, "For different values of modes1 and modes2 is not implemented yet the change of basis"
        self.M = self.get_M(self.n_x) # TODO
        self.M_1 = self.get_M_1(self.n_y)

        ## fix the seed for retrain if it is greater or equal to 0
        if retrain_gcno >= 0: 
            torch.manual_seed(self.retrain_gcno) 

        ## Dropout
        self.LayerDropout = nn.Dropout(p = self.dropout)
        
        ## Lifting
        self.p = nn.Linear(self.d_a, self.d_v, bias = self.bias)

        ## Projection
        # self.q = nn.Linear(self.d_v, self.d_u, bias = self.bias) # for linear projection
        self.q = MLP(self.d_v, self.d_u, 4*self.d_u, self.fun_act, self.arc, self.dropout) # for non-linear projection
        
        ## Integral layers
        if self.arc == 'Tran':
            if self.RNN:
                self.integrals = ChebyshevLayer(self.n_patch, self.d_v, self.d_v, self.modes1, self.modes2, 
                                                self.M, self.M_1, self.weights_norm, self.fun_act, self.arc, self.same_params)
                self.ws1 = nn.Linear(self.d_v, self.d_v, bias = self.bias)
                self.ws2 = nn.Linear(self.d_v, self.d_v, bias = self.bias)
            else:
                self.integrals = nn.ModuleList([
                    ChebyshevLayer(self.n_patch, self.d_v, self.d_v, self.modes1, self.modes2, 
                                   self.M, self.M_1, self.weights_norm, self.fun_act, self.arc, self.same_params)
                    for _ in range(self.L) ])
                self.ws1 = nn.ModuleList([
                    nn.Linear(self.d_v, self.d_v, bias = self.bias) for _ in range(self.L) ])
                self.ws2 = nn.ModuleList([
                    nn.Linear(self.d_v, self.d_v, bias = self.bias) for _ in range(self.L) ])      

        elif arc == "Zongyi":
            if self.RNN:
                self.integrals = ChebyshevLayer(self.n_patch, self.d_v, self.d_v, self.modes1, self.modes2,
                                                self.M, self.M_1, self.weights_norm, self.fun_act, self.arc, self.same_params)
                self.ws = nn.Linear(self.d_v, self.d_v)
                self.mlps = MLP(self.d_v, self.d_v, self.d_v, self.fun_act, self.arc)
            else:
                self.integrals = nn.ModuleList([
                    ChebyshevLayer(self.n_patch, self.d_v, self.d_v, self.modes1, self.modes2,
                                   self.M, self.M_1, self.weights_norm, self.fun_act, self.arc, self.same_params)
                    for _ in range(self.L) ])
                self.ws = nn.ModuleList([
                    nn.Linear(self.d_v, self.d_v) for _ in range(self.L) ])
                self.mlps = nn.ModuleList([
                    MLP(self.d_v, self.d_v, self.d_v, self.fun_act, self.arc) for _ in range(self.L) ])

        elif self.arc == 'Classic' or self.arc == 'Residual': 
            if self.RNN:
                self.integrals = ChebyshevLayer(self.n_patch, self.d_v, self.d_v, self.modes1, self.modes2, 
                                                self.M, self.M_1, self.weights_norm, self.fun_act, self.arc, self.same_params)
                self.ws = nn.Linear(self.d_v, self.d_v, bias = self.bias)
            else:
                self.integrals = nn.ModuleList([
                    ChebyshevLayer(self.n_patch, self.d_v, self.d_v, self.modes1, self.modes2, 
                                   self.M, self.M_1, self.weights_norm, self.fun_act, self.arc, self.same_params)
                    for _ in range(self.L) ])
                self.ws = nn.ModuleList([
                    nn.Linear(self.d_v, self.d_v, bias = self.bias) for _ in range(self.L) ])
    
    @jaxtyped(typechecker=beartype)
    def forward(self, x:Float[Tensor, "n_batch n_patch n_x n_y"]) -> Float[Tensor, "n_batch n_patch n_x n_y"]:
        """ 
        main function to perform the forward pass of the model
        
        x: torch.tensor
            x is a tensor of shape (n_samples, n_patch, n_x, n_y)
        """
        
        # Concatenate the grid to the input
        grid = self.get_grid(x.shape)
        x = torch.cat((x.unsqueeze(-1), grid), -1) # shape = (n_samples)*(n_patch)*(n_x)*(n_y)*(d_a)

        # Apply P
        x = self.p(x) # shape = (n_samples)*(n_patch)*(n_x)*(n_y)*(d_v)
        x = self.LayerDropout(x)

        # Integral Layers
        for i in range(self.L):
            if self.arc == 'Tran':
                if self.RNN:
                    x_1 = self.integrals(x)
                    x_1 = activation(self.ws1(x_1), self.fun_act)
                    x_1 = self.ws2(x_1)
                else:
                    x_1 = self.integrals[i](x)
                    x_1 = activation(self.ws1[i](x_1), self.fun_act)
                    x_1 = self.ws2[i](x_1)
                x_1 = activation(x_1, self.fun_act) # (?)
                x = x + x_1
                    
            elif self.arc == 'Classic':
                if self.RNN:
                    x1 = self.integrals(x)
                    x2 = self.ws(x)
                else:
                    x1 = self.integrals[i](x)
                    x2 = self.ws[i](x)
                x = x1 + x2
                if i < self.L - 1:
                    x = activation(x, self.fun_act)

            elif self.arc == "Zongyi":
                if self.RNN:
                    x1 = self.integrals(x)
                    x1 = self.mlps(x1)
                    x2 = self.ws(x)
                else:
                    x1 = self.integrals[i](x)
                    x1 = self.mlps[i](x1)
                    x2 = self.ws[i](x)
                x = x1 + x2
                if i < self.L - 1:
                        x = activation(x, self.fun_act)

            elif self.arc == 'Residual':
                if self.RNN:
                    x = self.integrals(x)
                    x = self.ws(x)
                else:
                    x = self.integrals[i](x)
                    x = self.ws[i](x)
                if i < self.L - 1:
                    x = activation(x, self.fun_act)
            # Dropout
            x = self.LayerDropout(x)
        
        # Apply Q
        x = self.q(x) # shape (n_samples)*(n_patch)*(n_x)*(n_y)*(d_u)
        return x.squeeze(-1)
    
    def get_grid(self, shape:torch.Size) -> Tensor:
        """ 
        Function to get the grid for the evaluation of the model.
        shape: torch.tensor
            shape is a tensor with the shape of the input `x` of the NO.
        """
        batchsize, n_patch, size_x, size_y = shape[0], shape[1], shape[2], shape[3]

        # Initialize the grid
        grid = torch.zeros(batchsize, n_patch, size_x, size_y, 2)

        # Helper extrema of subdomain for the grid
        extrema = [ [[-1.0, -1.0], [0.0, 0.0]], # patch 0
                    [[-1.0,  0.0], [0.0, 1.0]], # patch 1
                    [[ 0.0,  0.0], [1.0, 1.0]]  # patch 2
                    ]

        for i in range(n_patch):
            tmp_grid = cheb.Chebyshev_grid_2d(size_x, size_y, extrema[i][0], extrema[i][1])
            gridx, gridy = tmp_grid[:, :, 0], tmp_grid[:, :, 1]
            gridx = gridx.reshape(1, size_x, size_y).repeat([batchsize, 1, 1])
            gridy = gridy.reshape(1, size_x, size_y).repeat([batchsize, 1, 1])
            grid[:, i, :, :, 0] = gridx
            grid[:, i, :, :, 1] = gridy

        return grid
    
    def get_M(self, n:int) -> Float[Tensor, "n n"]:
        """ 
        Function to get the matrix for the change of basis, B = M*T.
        n: int
            n is the dimension of the matrix.
        """
        M = torch.eye(n)
        M[0,0] = 1/2
        M[0, 1] = -1/2
        M[1, 0] = 1/2
        M[1, 1] = 1/2
        for i in range(2, n):
            if i % 2 == 0:
                M[i, 0] = -1
            else:
                M[i, 1] = -1
        return M

    def get_M_1(self, n:int) -> Float[Tensor, "n n"]:
        """ 
        Function to get the inverse of the matrix for the change of basis, T = M_1*B.
        n: int
            n is the dimension of the matrix.
        """
        M_1 = torch.eye(n)
        M_1[0, 1] = 1
        M_1[1, 0] = -1
        for i in range(2, n):
            if i % 2 == 0:
                M_1[i, 0] = 1
                M_1[i, 1] = 1
            else:
                M_1[i, 0] = -1
                M_1[i, 1] = 1
        return M_1

if __name__ == '__main__':
    # select folder for tensorboard
    writer = SummaryWriter(log_dir = folder)

    print('Device disponibile:', mydevice)
    
    #### save hyper-parameter in txt files
    df = pd.DataFrame.from_dict([training_properties]).T
    df.to_csv(folder + '/training_properties.txt', 
              header = False, index = True, mode = 'w')
    df = pd.DataFrame.from_dict([gcno_architecture]).T
    df.to_csv(folder + '/net_architecture.txt',
              header = False, index = True, mode = 'w')
    
    #########################################
    # Read data and perform initial normalization
    ######################################### 
    # Training data
    a_train, u_train, _ = MatReader(TrainDataPath)
    idx_train = list( [i for i in range(0, a_train.size(0))] )
    random.Random(1).shuffle(idx_train)
    idx_train = idx_train[:ntrain]
    a_train, u_train = a_train[idx_train, :, ::s, ::s], u_train[idx_train, :, ::s, ::s] # shape: (n_samples)*(n_patch)*(n_x)*(n_y)
    n_x, n_y = a_train.size(2), a_train.size(3)
    
    # Test data
    a_test, u_test, _ = MatReader(TestDataPath)
    idx_test = list( [i for i in range(0, a_test.size(0))] )
    random.Random(1).shuffle(idx_test)
    idx_test = idx_test[:ntest]
    a_test, u_test = a_test[idx_test, :, ::s, ::s], u_test[idx_test, :, ::s, ::s]

    # Gaussian pointwise normalization
    a_normalizer = UnitGaussianNormalizer(a_train) # compute mean e std
    a_train = a_normalizer.encode(a_train) # apply normalization
    a_test = a_normalizer.encode(a_test) # apply normalization with mean and std of training data
    
    u_normalizer = UnitGaussianNormalizer(u_train) # compute mean e std
    u_normalizer.cuda()

    # Data batch subdivisions
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_train, u_train),
                                                batch_size = batch_size, shuffle=True, generator=torch.Generator(device='cuda'))
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(a_test, u_test),
                                              batch_size = batch_size, shuffle=False)    

    print('Data loaded')  
    
    ################################################################
    # training, evaluation e plot
    ################################################################
    # Initialize the model
    model = GlobalCheb(n_patch, n_x, n_y, d_a, d_v, d_u, L, modes, modes,
                       fun_act, weights_norm, arc, same_params, dropout, RNN, retrain_gcno)

    # Move the model to the device
    model.to(mydevice)
    
    # total number of trainable parameters
    par_tot = 0
    for p in model.parameters():
        # print(p.shape)
        par_tot += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    print("total number of trainable parameters is: ", par_tot)
    writer.add_text("Parameters", 'total number of trainable parameters is: ' + str(par_tot), 0)
    
    # Optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = 1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    # Scheduler
    if scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)
    elif scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    # Funzione da minimizzare
    if Loss == 'L2':
        myloss = L2relLoss(BC, alpha_BC, cont_loss, alpha)
    elif Loss == 'H1':
        myloss = H1relLoss_cheb(beta)
    
    for ep in range(epochs + 1):
        with tqdm(desc = f"Epoch {ep}", bar_format = "{desc}: [{elapsed_s:.2f}{postfix}]") as tepoch:
            model.train()
            train_loss = 0
            for step, (a, u) in enumerate(train_loader):            
                a, u = a.to(mydevice), u.to(mydevice)
                
                optimizer.zero_grad() # reset the gradient
                out = model.forward(a) 
                out = u_normalizer.decode(out)

                loss = myloss(out, u)
                train_loss += loss.item()
                
                loss.backward()
        
                optimizer.step()
                
                # set the postfix for print
                tepoch.set_postfix( {'Batch': step + 1,
                    'Train loss (in progress)': train_loss/(batch_size*(step+1))} )
                
                if scheduler == "cosineannealinglr":
                    scheduler.step()
                
            if scheduler == "steplr": 
                scheduler.step()
                
            model.eval()
            test_l2 = 0.0
            test_h1 = 0.0
            with torch.no_grad(): # efficency
                for a, u in test_loader:
                    a, u = a.to(mydevice), u.to(mydevice)
        
                    out = model.forward(a)   
                    out = u_normalizer.decode(out)
                    test_l2 += L2relLoss()(out, u).item()
                    test_h1 += H1relLoss_cheb()(out, u).item()
                    
            train_loss /= ntrain
            test_l2 /= ntest
            test_h1 /= ntest
                    
            # set the postfix for print
            tepoch.set_postfix({'Train_loss:': train_loss,
                                'Test_loss_l2:': test_l2,
                                'Test_loss_h1:': test_h1})
            tepoch.close()
            
            writer.add_scalars('Global Chebyshev', {'Train_loss': train_loss,
                                                   'Test_loss_l2': test_l2,
                                                   'Test_loss_h1': test_h1}, ep)

            #########################################
            # PLOTS
            #########################################
            if ep == 0:
                # Initial data
                esempio_test = a_test[idx]
                plot_data(a_normalizer.decode(a_test), idx, "Coefficients a(x)", ep, plotting)
                
                # Exact solution
                soluzione_test = u_test[idx] 
                plot_data(u_test, idx, "Exact solution u(x)", ep, plotting)
                    
            # Approximate solution with NO and absolute difference
            if ep % ep_step == 0:
                with torch.no_grad(): # no grad for efficiency
                    out_test = model.forward(esempio_test.to(mydevice))
                    out_test = u_normalizer.decode(out_test)
                    out_test = out_test.to('cpu')
                    diff = torch.abs(out_test - soluzione_test)
                
                # Plot approximate solution
                plot_data(out_test, [], "Approximate solution with NO", ep, plotting)                
                
                # Plot the difference
                plot_data(diff, [], "Absolute difference", ep, plotting)    

    with open(folder + '/errors.txt', 'w') as file:
                file.write("Training Error: " + str(train_loss) + "\n")
                file.write("Test relative L^2 error: " + str(test_l2) + "\n")
                file.write("Test relative H^1 error: " + str(test_h1) + "\n")
                file.write("Current Epoch: " + str(ep) + "\n")
                file.write("Params: " + str(par_tot) + "\n")
    
    writer.flush() # to make sure that all pending events have been written
    writer.close() # close tensorboard's writer
    
    torch.save(model, name_model)   