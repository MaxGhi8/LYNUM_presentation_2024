"""
Implementation of FNO continuatoin for Darcy problem on L-shaped domain.
"""
import matplotlib.pyplot as plt
import numpy as np
import operator
from functools import reduce
from tqdm import tqdm 
# pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
# library for reading and writing data
import mat73
import pandas as pd

#########################################
# nomi file da salvare
#########################################
#### Nomi file per tenosrboard e modello 
model_folder = 'test' # folder's name for the test
description_test = "test_FNOcontinuation_H1" # name that describe the test
# description_test = "test"
# Save the models here:
folder = model_folder + "/exp_" + description_test
name_model = model_folder + "/model_" + description_test

#########################################
# Hyperparameters
#########################################
training_properties = {
    "learning_rate": 0.001,
    "scheduler": "steplr", # "steplr", "cosineannealinglr"
    "epochs": 500,
    "batch_size": 20,
    "weight_decay": 1e-4, # L^2 regularization
    "loss": 'H1', # 'L2', 'H1'
    "beta": 1, # hyperparameter for the H^1 relative norm
    "BC": False, # Boundary Conditions
    "alpha_BC": 1, # Coefficient for boundary condition
    "training_samples": 1000,
    "test_samples": 200,
    "s": 5, # subsample the grid
}
fno_architecture = {
    "RNN": False,
    "arc": 'Classic', # 'Residual', 'Tran', 'Classic'
    "d_a": 1, # input dimension (2 dimension for the grid is added in the initialization of the model)
    "width": 32, # hidden dimension
    "d_u": 1, # output dimension
    "modes": 12, # Fourier modes
    "fft_norm": None, # None or 'ortho'
    "fun_act": 'relu', # 'relu', 'gelu', 'tanh', 'leaky_relu'
    "n_layers": 4,  # L = depth
    "padding": 0, # number of points for padding
    "retrain": -1, # seed for retraining
}
fno_architecture["weights_norm"] = "Xavier" if fno_architecture["fun_act"] == 'gelu' else "Kaiming"
#########################################
# default values
#########################################
mydevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# mydevice = torch.device('cpu') # to force to use cpu
torch.set_default_device(mydevice) # default tensor device
torch.set_default_tensor_type(torch.FloatTensor) # default tensor dtype
TrainDataPath = './data/Darcy_Lshape_uniform_grid_pc_train.mat'
TestDataPath = './data/Darcy_Lshape_uniform_grid_pc_test.mat'

#########################################
# seed for extract dataset
#########################################
g = torch.Generator().manual_seed(1) # fisso seed
ntrain = training_properties["training_samples"] 
ntest = training_properties["test_samples"] 

#########################################
# hyperparameter for the neural operataor
#########################################   
#### training hyperparameter   
learning_rate = training_properties["learning_rate"]
scheduler = training_properties["scheduler"]
epochs = training_properties["epochs"]
batch_size = training_properties["batch_size"]
weight_decay = training_properties["weight_decay"]
iterations = epochs*(ntrain//batch_size)
Loss = training_properties["loss"] # realtive L^2 or relative H^1
beta = training_properties["beta"] # norm_{L_2} + beta * seminorm_{H_1} 
BC = training_properties["BC"]
alpha_BC = training_properties["alpha_BC"]
s = training_properties["s"] # mesh's parameter: s=3 --> 70, s=1 --> 211

#### model's hyperparameter
RNN = fno_architecture["RNN"]
arc = fno_architecture["arc"]
d_a = fno_architecture["d_a"] 
d_v = fno_architecture["width"]  
d_u = fno_architecture["d_u"]  
L = fno_architecture["n_layers"]
modes = fno_architecture["modes"] # k_{max,j}
FFTnorm = fno_architecture["fft_norm"]
fun_act = fno_architecture["fun_act"] # activation function, (kaiming init)
weights_norm = fno_architecture["weights_norm"]
padding = fno_architecture["padding"]

#########################################
# tensorboard and plot variables
#########################################   
ep_step = 20 # save the plot on tensorboard every ep_Step epochs
idx = [7, 42, 93, 13] # casual number from 0 to n_test-1
# idx = [0]
n_idx = len(idx)
plotting = False # if True we make the plot

#########################################
# reading data
#########################################
def MatReader(file_path):
    """
    Funzione per leggere i file di estensione .mat version 7.3

    Parameters
    ----------
    file_path : string
        path del file .mat da leggere        

    Returns
    -------
    a : tensor
        valutazioni della funzione a(x) del problema di Darcy 
        dimension = (n_samples)*(n_patch)*(nx)*(ny)
    u : tensor
        risultato dell'approssimazione della soluzione u(x) ottenuta con un
        metodo standard (nel nostro isogeometrico)
        dimension = (n_samples)*(n_patch)*(nx)*(ny)

    """
    data = mat73.loadmat(file_path)
    
    a = data["COEFF"]
    a = torch.from_numpy(a).float() # trasforma np.array in torch.tensor
    
    u = data["SOL"]
    u = torch.from_numpy(u).float()
    
    nodes = data["nodes"]
    nodes = torch.from_numpy(nodes).float()
    
    a, u, nodes = a.to('cpu'), u.to('cpu'), nodes.to('cpu')
    return a, u, nodes

#########################################
# nan for plt
#########################################
def set_nan(x):
    """
    helper function for improve visualization

    Parameters
    ----------
    x : torch.tensor
        x is a tensor with the last two components that is nx e ny respectively
        x can be of dimension 2,3,4.

    Returns
    -------
    x : torch.tensor
        torch.tensor with the components in the square in right-down corner
        set to NaN (with nx and ny even)

    """
    nx, ny = x.size()[-2], x.size()[-1]
    assert nx == ny, "Different dimension"
    if nx % 2 == 0: # even points per direction
        if len(x.size()) == 2:
            x[nx//2:, ny//2:] = np.nan 
        elif len(x.size()) == 3:
            x[:, nx//2:, ny//2:] = np.nan 
        elif len(x.size()) == 4:
            x[:, :, nx//2:, ny//2:] = np.nan 
    else: # odd points per direction
        if len(x.size()) == 2:
            x[nx//2 + 1:, ny//2 + 1:] = np.nan 
        elif len(x.size()) == 3:
            x[:, nx//2 + 1:, ny//2 + 1:] = np.nan 
        elif len(x.size()) == 4:
            x[:, :, nx//2 + 1:, ny//2 + 1:] = np.nan
    return x

#########################################
# function for plots data
#########################################
def plot_data(data_plot, idx, title, ep, plotting = True):
    """ 
    Function to makes the plots of the data.
    
    data_plot: torch.tensor
        data_plot is a tensor of shape (n_samples, n_patch, n_x, n_y).
    """  
    # select the data to plot
    if idx != []:
        data_plot = data_plot[idx]
        n_idx = len(idx)
    else:
        n_idx = data_plot.size(0)
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
# activation function
#########################################
def activation(x, activation_str):
    """
    Activation function che si vuole utilizzare all'interno della rete.
    La funzione è la stessa in tutta la rete.
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
# loss function
#########################################
class L2relLoss(): 
    """ somma degli errori relativi in norma L2 """      
    def __init__(self, BC = False, alpha_BC = 1):
        self.BC = BC
        self.alpha_BC = alpha_BC
    
    def rel(self, x, y):
        num_examples, n_x, n_y = x.size()
        
        assert n_x == n_y, "Different dimension"
        if n_x % 2 == 0: # even points per direction
            x = torch.cat((x[:, :, :n_y//2], x[:, :n_x//2, n_y//2:]), dim = 1)
            y = torch.cat((y[:, :, :n_y//2], y[:, :n_x//2, n_y//2:]), dim = 1)
        else: # odd points per direction
            x = torch.cat( (x[:, :, :n_y//2 + 1], 
                                torch.cat((x[:, :n_x//2 + 1, n_y//2 + 1:],
                                          torch.zeros_like(x[:, :n_x//2 + 1, [0]], device = x.device)),
                                          dim = 2) ),
                           dim = 1)
            y = torch.cat( (y[:, :, :n_y//2 + 1], 
                                torch.cat((y[:, :n_x//2 + 1, n_y//2 + 1:],
                                          torch.zeros_like(y[:, :n_x//2 + 1, [0]], device = y.device)),
                                          dim = 2) ),
                           dim = 1)
            
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), 2, 1)
        
        if self.BC:
            if n_x % 2 == 0:
                BC_discont =  torch.cat ( (x[:, 0, :], 
                                           x[:, :, 0],
                                           x[:, 1:n_x//2, -1], 
                                           x[:, n_x//2-1, n_y//2-1:-1],
                                           x[:, n_x//2:, n_y//2-1],
                                           x[:, -1, n_y//2:-1]), 1)
            else:
                BC_discont =  torch.cat ( (x[:, 0, :],
                                           x[:, :, 0],
                                           x[:, 1:n_x//2+1, -1],
                                           x[:, n_x//2, n_y//2:-1],
                                           x[:, n_x//2+1:, n_y//2], 
                                           x[:, -1, n_y//2+1:-1]), 1)   
                
            BC_discont = torch.norm(BC_discont, 2, dim = 1) # l^2 norm
            BC_discont = torch.sum(BC_discont) # sum on the batch dim
        else:
            BC_discont = 0
        
        return torch.sum(diff_norms/y_norms) + self.alpha_BC*BC_discont # sum along batchsize
    
    def __call__(self, x, y):
        return self.rel(x, y)
    
class H1relLoss():
    """ Norma H^1 = W^{1,2} relativa, approssimata con la trasformata di Fourier """
    def __init__(self, BC = False, alpha_BC = 1, beta = 1):
        self.BC = BC
        self.alpha_BC = alpha_BC
        self.beta = beta
        
    def rel(self, x, y):
        num_examples = x.size()[0]
        
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), 2, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), 2, 1)
        
        return diff_norms, y_norms
    
    def computeH1_patch(self, x_i, y_i):
        nx, ny = x_i.size()[1], x_i.size()[2]
        
        x_i = x_i.view(x_i.shape[0], nx, ny, -1)
        y_i = y_i.view(y_i.shape[0], nx, ny, -1)
        
        k_x = torch.cat(
            (torch.arange(start=0, end=nx//2, step=1),
             torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat(
            (torch.arange(start=0, end=ny//2, step=1),
             torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x_i.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x_i.device)
        
        x_i = torch.fft.fftn(x_i, dim=[1, 2])
        y_i = torch.fft.fftn(y_i, dim=[1, 2])
        
        weight = 1 + self.beta*(k_x**2 + k_y**2)
        weight = torch.sqrt(weight)
        return self.rel(x_i*weight, y_i*weight)

    def __call__(self, x, y):
        _, n_x, n_y = x.size()
        if n_x % 2 == 0: # even points per direction
            x1 = x[:, :n_x//2, n_y//2:] # lower-left
            x2 = x[:, :n_x//2, :n_y//2] # upper-left
            x3 = x[:, n_x//2:, :n_y//2] # upper-right
            y1 = y[:, :n_x//2, n_y//2:] # lower-left
            y2 = y[:, :n_x//2, :n_y//2] # upper-left
            y3 = y[:, n_x//2:, :n_y//2] # upper-right
        else: # odd points per direction
            x1 = x[:, :n_x//2 + 1, n_y//2 + 1:] # lower-left
            x2 = x[:, :n_x//2 + 1, :n_y//2 + 1] # upper-left
            x3 = x[:, n_x//2 + 1:, :n_y//2 + 1] # upper-right
            y1 = y[:, :n_x//2 + 1, n_y//2 + 1:] # lower-left
            y2 = y[:, :n_x//2 + 1, :n_y//2 + 1] # upper-left
            y3 = y[:, n_x//2 + 1:, :n_y//2 + 1] # upper-right
            
        if self.BC:
            if n_x % 2 == 0:
                BC_discont =  torch.cat ( (x[:, 0, :], 
                                           x[:, :, 0],
                                           x[:, 1:n_x//2, -1], 
                                           x[:, n_x//2-1, n_y//2-1:-1],
                                           x[:, n_x//2:, n_y//2-1],
                                           x[:, -1, n_y//2:-1]), 1)
            else:
                BC_discont =  torch.cat ( (x[:, 0, :],
                                           x[:, :, 0],
                                           x[:, 1:n_x//2+1, -1],
                                           x[:, n_x//2, n_y//2:-1],
                                           x[:, n_x//2+1:, n_y//2], 
                                           x[:, -1, n_y//2+1:-1]), 1)   
                
            BC_discont = torch.norm(BC_discont, 2, dim = 1) # l^2 norm
            BC_discont = torch.sum(BC_discont) # sum on the batch dim
        else:
            BC_discont = 0
            
        norm_diff_1, norm_y_1 = self.computeH1_patch(x1, y1)
        norm_diff_2, norm_y_2 = self.computeH1_patch(x2, y2)
        norm_diff_3, norm_y_3 = self.computeH1_patch(x3, y3)
        
        norm_diff = norm_diff_1 + norm_diff_2 + norm_diff_3 
        norm_y = norm_y_1 + norm_y_2 + norm_y_3
    
        return torch.sum(norm_diff/norm_y) + self.alpha_BC*BC_discont
    
#########################################
# initial normalization
#########################################    
class UnitGaussianNormalizer(object):
    """ 
    Initial normalization on x, which is a tensor of 
    dimension: (n_samples)*()*(nx)*(ny)
    normalization --> pointwise gaussian
    """
    def __init__(self, x, eps:float=1e-5):
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean)/(self.std + self.eps)
        return x
    
    def decode(self, x):
        x = x*(self.std + self.eps) + self.mean
        return x
    
    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

#########################################
# fourier layer
#########################################
class FourierLayer(nn.Module):
    """
    2D Fourier layer 
    input --> FFT --> linear transform --> IFFT --> output    
    """
    def __init__(self, in_channels, out_channels, modes1, modes2, FFTnorm):
        super(FourierLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply
        self.modes2 = modes2
        self.FFTnorm = FFTnorm
                           
        if weights_norm == 'Xavier':
            # Xavier normalization
            self.weights1 = nn.init.xavier_normal_(
                nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
                gain = 1/(self.in_channels*self.out_channels))
            self.weights2 = nn.init.xavier_normal_(
                nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
                gain = 1/(self.in_channels*self.out_channels)) 
        elif weights_norm == 'Kaiming':
            # Kaiming normalization
            self.weights1 = torch.nn.init.kaiming_normal_(
                nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
                a = 0, mode = 'fan_in', nonlinearity = fun_act)
            self.weights2 = torch.nn.init.kaiming_normal_(
                nn.Parameter(torch.empty(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat)),
                a = 0, mode = 'fan_in', nonlinearity = fun_act) 
            

    def compl_mul2d(self, input, weights):
        """ Moltiplicazione tra numeri complessi """
        # (batch, in_channel, x, y), (in_channel, out_channel, x, y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        
        # Trasformata di Fourier 2D per segnali reali
        x_ft = torch.fft.rfft2(x, norm=self.FFTnorm)

        #### Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2),
                             x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        # angolo in alto a sx
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        # angolo in basso a sx
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Trasformata inversa di Fourier 2D per segnali reali
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm=self.FFTnorm)
        return x
    
#########################################
# MLP
#########################################
class MLP(nn.Module):
    """ Rete neurale con un hidden layer (shallow neural network) """
    def __init__(self, in_channels, out_channels, mid_channels, fun_act):
        super(MLP, self).__init__()
        self.mlp1 = torch.nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = torch.nn.Conv2d(mid_channels, out_channels, 1)
        self.fun_act = fun_act

    def forward(self, x):
        x = self.mlp1(x) # affine transformation
        x = activation(x, self.fun_act) # activation function
        x = self.mlp2(x) # affine transformation
        return x
    
#########################################
# FourierContinuationFNO for Darcy on L-shaped
#########################################
class FourierContinuationFNO_DarcyL(nn.Module):
    """ 
    Fourier Neural Operator per il problema di Darcy in dimensione due    
    """
    def __init__(self, d_a, d_v, d_u, L, modes1, modes2, fun_act, FFTnorm, padding):
        """ 
        L: int
            numero di Fourier operator da fare
            
        d_a : int
            pari alla dimensione dello spazio in input
            
        d_v : int
            pari alla dimensione dello spazio nell'operatore di Fourier
            
        d_u : int
            pari alla dimensione dello spazio in output 
            
        mode1 : int
            pari a k_{max, 1}
            
        mode2 : int
            pari a k_{max, 2}
            
        fun_act: str
            string for selecting the activation function to use along all the 
            architecture
            
        """
        super(FourierContinuationFNO_DarcyL, self).__init__()
        self.d_a = d_a + 2
        self.d_v = d_v
        self.d_u = d_u
        self.L = L
        self.modes1 = modes1
        self.modes2 = modes2
        self.fun_act = fun_act
        self.FFTnorm = FFTnorm
        self.padding = padding
        self.retrain_fno = fno_architecture["retrain"] # seed for retraining
        self.RNN = RNN
        
        if self.retrain_fno > 0:
            torch.manual_seed(self.retrain_fno) 
        
        ## Lifting
        self.p = torch.nn.Conv2d(self.d_a, self.d_v, 1) # input features is d_a=3: (a(x, y), x, y)
        
        if arc == 'Tran':
            if self.RNN:
                self.fouriers = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                self.ws1 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
                self.ws2 = torch.nn.Conv2d(self.d_v, self.d_v, 1)
            else:
                self.fouriers = nn.ModuleList([
                    FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                    for _ in range(self.L) ])
                self.ws1 = nn.ModuleList([
                    torch.nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L) ])
                self.ws2 = nn.ModuleList([
                    torch.nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L) ])
                    
        elif arc == 'Classic' or arc == 'Residual': 
            if self.RNN:
                self.fouriers = FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                self.ws = torch.nn.Conv2d(self.d_v, self.d_v, 1)
            else:
                self.fouriers = nn.ModuleList([
                    FourierLayer(self.d_v, self.d_v, self.modes1, self.modes2, self.FFTnorm)
                    for _ in range(self.L) ])
                self.ws = nn.ModuleList([
                    torch.nn.Conv2d(self.d_v, self.d_v, 1) for _ in range(self.L) ])
            
        ## Projection
        self.q = torch.nn.Conv2d(self.d_v, self.d_u, 1)
        # self.q = MLP(self.d_v, self.d_u, 4*self.d_u, self.fun_act) # output features is d_u: u(x, y)
        
        self.to(mydevice)
        
    def forward(self, x):
        grid = self.get_grid(x.shape, mydevice)
        x = torch.cat((x, grid), dim = -1) # concatenate last dimension --> (n_samples)*(n_x)*(n_y)*(d_a+2)
        
        x = x.permute(0, 3, 1, 2) # (n_samples)*(3)*(n_x)*(n_y)
                
        ## Lifting P
        x = self.p(x) # shape = (n_samples)*(d_v)*(nx)*(ny)
        
        ## Padding
        if self.padding > 0:
            x = F.pad(x, [0, self.padding, 0, self.padding])
        
        ## Fourier Layers
        for i in range(self.L):
            if arc == 'Tran':
                if self.RNN:
                    x_1 = self.fouriers(x)
                    x_1 = activation(self.ws1(x_1), self.fun_act)
                    x_1 = self.ws2(x_1)
                    x_1 = activation(x_1, self.fun_act) # (?)
                    x = x + x_1
                else:
                    x_1 = self.fouriers[i](x)
                    x_1 = activation(self.ws1[i](x_1), self.fun_act)
                    x_1 = self.ws2[i](x_1)
                    x_1 = activation(x_1, self.fun_act) # (?)
                    x = x + x_1
                    
            elif arc == 'Classic':
                if self.RNN:
                    x1 = self.fouriers(x)
                    x2 = self.ws(x)
                    x = x1 + x2
                    if i < self.L - 1:
                        x = activation(x, self.fun_act)
                else:
                    x1 = self.fouriers[i](x)
                    x2 = self.ws[i](x)
                    x = x1 + x2
                    if i < self.L - 1:
                        x = activation(x, self.fun_act)
                    
            elif arc == 'Residual':
                if self.RNN:
                    x = self.fouriers(x)
                    x = self.ws(x)
                    if i < self.L - 1:
                        x = activation(x, self.fun_act)
                else:
                    x = self.fouriers[i](x)
                    x = self.ws[i](x)
                    if i < self.L - 1:
                        x = activation(x, self.fun_act)
                    
        ## Padding
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        ## Projection Q
        x = self.q(x) # shape (n_samples)*(d_u)*(nx)*(ny)
        
        x = x.permute(0, 2, 3, 1)
        
        return x.squeeze(-1)
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        # grid x
        gridx = torch.tensor(np.linspace(-1, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, 1, size_x, 1).repeat([batchsize, size_y, 1, 1])
        # idem for grid y
        gridy = torch.tensor(np.linspace(-1, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, size_y, 1, 1).repeat([batchsize, 1, size_x, 1])
        grid = torch.cat((gridy, gridx), dim=-1).to(mydevice) # concatenate along the last dimension

        return grid

if __name__ == '__main__':
    # select folder for tensorboard
    writer = SummaryWriter(log_dir = folder)
    # 'cuda' if GPU is available, 'cpu' otherwise
    print('Device disponibile:', mydevice)
    
    #### save hyper-parameter in txt files
    df = pd.DataFrame.from_dict([training_properties]).T
    df.to_csv(folder + '/training_properties.txt', 
              header = False, index = True, mode = 'w')
    df = pd.DataFrame.from_dict([fno_architecture]).T
    df.to_csv(folder + '/net_architecture.txt',
              header = False, index = True, mode = 'w')
    
    #########################################
    # lettura dati e initial normalization
    ######################################### 
    #### Training data
    a_train, u_train, nodes = MatReader(TrainDataPath)
    idx_train = torch.randperm(a_train.size()[0], device = 'cpu', generator = g)[:ntrain]
    a_train, u_train = a_train[idx_train, :, :, :], u_train[idx_train, :, :, :]   
    a_train = torch.flip(a_train, [3])
    u_train = torch.flip(u_train, [3])
    # Gaussian pointwise normalization
    a_normalizer = UnitGaussianNormalizer(a_train) # compute mean e std
    a_train = a_normalizer.encode(a_train) # normalize
    
    #### Test data
    a_test, u_test, _ = MatReader(TestDataPath)
    idx_test = torch.randperm(a_test.size()[0], device = 'cpu', generator = g)[:ntest]
    a_test, u_test = a_test[idx_test, :, :, :], u_test[idx_test, :, :, :]
    a_test = torch.flip(a_test, [3])
    u_test = torch.flip(u_test, [3])
    # Gaussian pointwise normalization
    a_test = a_normalizer.encode(a_test) # normalize
    
    def concatenate_data(x):
        return torch.cat( 
            (torch.cat(
                (x[:, 1, :, :-1], 
                x[:, 2, 1:, :-1]), 
                dim = 1), 
            torch.cat(
                (x[:, 0, :, :], 
                 torch.zeros_like(x[:, 0, 1:, :], device = 'cpu')), 
                dim = 1)),
            dim = 2)
    
    # extraction training data
    a_train, u_train = a_train[:, :, ::s, ::s], u_train[:, :, ::s, ::s]
    a_train, u_train = concatenate_data(a_train), concatenate_data(u_train)
    a_train = a_train.unsqueeze(-1)
    
    # extraction test data
    a_test, u_test = a_test[:, :, ::s, ::s], u_test[:, :, ::s, ::s]
    a_test, u_test = concatenate_data(a_test), concatenate_data(u_test)
    a_test = a_test.unsqueeze(-1)

    u_normalizer = UnitGaussianNormalizer(u_train) # compute mean and std
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
    model = FourierContinuationFNO_DarcyL(d_a, d_v, d_u, L, modes, modes, fun_act, FFTnorm, padding)
    # Move the model to the device
    model.to(mydevice)
    
    # total number of trainable parameters
    par_tot = 0
    for p in model.parameters():
        # print(p.shape)
        par_tot += reduce(operator.mul, list(p.shape + (2,) if p.is_complex() else p.shape))
    print("total number of trainable parameters is: ", par_tot)
    writer.add_text("Parameters", 'total number of trainable parameters is: ' + str(par_tot), 0)
    
    # Adam optimizer
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-7)
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate, weight_decay=weight_decay)
    
    # Cosine Annealing Scheduler
    if scheduler == "steplr":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 100, gamma = 0.50)
    elif scheduler == "cosineannealinglr":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    
    # Training loss function
    if Loss == 'L2':
        myloss = L2relLoss(BC, alpha_BC)
    elif Loss == 'H1':
        myloss = H1relLoss(BC, alpha_BC, beta)
    
    for ep in range(epochs + 1):
        with tqdm(desc = f"Epoch {ep}", bar_format = "{desc}: [{elapsed_s:.2f}{postfix}]") as tepoch:
            model.train()
            train_loss = 0
            for step, (a, u) in enumerate(train_loader):            
                a, u = a.to(mydevice), u.to(mydevice)
                
                optimizer.zero_grad()
                out = model.forward(a) 
                # out = u_normalizer.decode(out)
                if Loss == 'L2':
                    loss = myloss(out, u)
                elif Loss == 'H1':
                    loss = myloss(out, u)
                loss.backward()
        
                optimizer.step()
                train_loss += loss.item()
                
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
            with torch.no_grad():
                for a, u in test_loader:
                    a, u = a.to(mydevice), u.to(mydevice)
        
                    out = model.forward(a)   
                    # out = u_normalizer.decode(out)
                    test_l2 += L2relLoss()(out, u).item()
                    test_h1 += H1relLoss()(out, u).item()
                    
            train_loss /= ntrain
            test_l2 /= ntest
            test_h1 /= ntest
                    
            # set the postfix for print
            tepoch.set_postfix({'Train_loss:': train_loss,
                                'Test_loss_l2:': test_l2, 
                                'Test_loss_h1:': test_h1})
            tepoch.close()
            
            writer.add_scalars('FNOplus_darcy2D', {'Train_loss': train_loss,
                                                   'Test_loss_l2': test_l2,
                                                   'Test_loss_h1': test_h1}, ep)
            
            with open(folder + '/errors.txt', 'w') as file:
                file.write("Training Error: " + str(train_loss) + "\n")
                file.write("Test relative L^2 error: " + str(test_l2) + "\n")
                file.write("Test relative H^1 error: " + str(test_h1) + "\n")
                file.write("Current Epoch: " + str(ep) + "\n")
                file.write("Params: " + str(par_tot) + "\n")

            #########################################
            # plot dei dati alla fine ogni ep_step epoche
            #########################################
            if ep == 0:
                # Initial data
                esempio_test = a_test[idx]
                esempio_test_nan = a_test[idx]
                esempio_test_nan = set_nan(esempio_test_nan.squeeze(-1)).to('cpu')  
                plot_data(esempio_test_nan, [], "Coefficients a(x)", ep, plotting)
                  
                
                # Exact solution
                soluzione_test = u_test[idx]
                soluzione_test_nan = u_test[idx]
                soluzione_test_nan = set_nan(soluzione_test_nan).to('cpu')
                plot_data(soluzione_test_nan, [], "Exact solution u(x)", ep, plotting)             
                    
            # Approximate solution with NO and absolute difference
            if ep % ep_step == 0:
                with torch.no_grad():
                    out_test = model.forward(esempio_test.to(mydevice))
                    # out_test = u_normalizer.decode(out_test)
                    out_test = out_test.to('cpu')
                    diff = torch.abs(out_test - soluzione_test)
                    
                # Plot approximate solution
                plot_data(set_nan(out_test), [], "Approximate solution with NO", ep, plotting)  
                
                # Plot absolute difference
                plot_data(set_nan(diff), [], "Absolute difference", ep, plotting)
    
    writer.flush() # per salvare i dati finali
    writer.close() # chiusura writer tensorboard
    
    torch.save(model, name_model)   
