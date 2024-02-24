# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 20:51:16 2023

@author: Luca
"""


import matplotlib.pyplot as plt
from neurodiffeq.solvers import BundleSolver1D
from neurodiffeq.conditions import BundleIVP
from neurodiffeq.callbacks import ActionCallback
from neurodiffeq import diff  # the differentiation operation
import torch
from neurodiffeq.generators import Generator1D
import numpy as np
import torch.nn as nn
from neurodiffeq.networks import FCNN
import torch.nn as nn
import torch.nn.functional as F


class CustomNN(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))

        # Learnable parameters mu and sigma for the first layer
        self.mu = torch.linspace(0,1, hidden_units[0])#nn.Parameter(torch.linspace(0,1, hidden_units[0]))
        self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.1)

        # Remaining hidden layers
        for i in range(len(hidden_units) - 1):
            self.layers.append(actv())
            self.layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))

        # Output layer
        self.layers.append(actv())
        self.fc_out = nn.Linear(hidden_units[-1], n_output_units)

    def forward(self, x):

        inputx = x[:,0].reshape(-1,1)
        #print(inputx.shape)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            #print(x.shape)
            # Apply the custom operation after the first layer
            if i == 0:
                x = x * torch.exp(- (inputx - self.mu) ** 2 / self.sigma ** 2)

        # Output layer transformation
        x = self.fc_out(x)
        return x
    
    
# Set a fixed random seed:
    
torch.manual_seed(42)


# Set the parameters of the problem

Om_r_0=5.38*10**(-5)
Om_m_0=0.272
a_eq=Om_r_0/Om_m_0
Om_L_0=1-Om_m_0-Om_r_0
alpha=a_eq**3*Om_L_0/Om_m_0


# Set the range of the independent variable:

a_0 = 10**(-3)
a_f = 1

N_0 = np.log(a_0)
N_f = np.log(a_f)

n_0=np.abs(np.log(a_0))

N_p_0 = N_0/n_0
N_p_f = N_f/n_0



# Define the differential equation:
    
def ODE_LCDM(x, x_prime, N_p):
    
    a_eq=Om_r_0/Om_m_0
    Om_L_0=1-Om_m_0-Om_r_0
    alpha=a_eq**3*Om_L_0/Om_m_0
    N=n_0*N_p
    
    res1 = diff(x, N_p) - x_prime
    res2 = diff(x_prime, N_p) + (x_prime)**2 - (3*torch.exp(N)/(2*a_eq*(1+(torch.exp(N)/a_eq)+alpha*(torch.exp(N)/a_eq)**4)))*n_0**2 + n_0*((1+4*alpha*(torch.exp(N)/a_eq)**3)/(2*(1+(a_eq/torch.exp(N))+alpha*(torch.exp(N)/a_eq)**3)))*x_prime
    
    return [res1 , res2]

# Define the initial condition:

condition = [BundleIVP(N_p_0, -n_0),
             BundleIVP(N_p_0, n_0)]

# Define a custom loss function:

def weighted_loss_LCDM(res, x, t):
    
    #PARA IMPLEMENTAR LA DIF CON EL NUMERICO, CALCULARLA EN ODE Y SUMAR RES3
    
    N = t[0]
    w = 0

    loss = (res ** 2) * torch.exp(-w * (N - N_p_0))
    
    return loss.mean()

# Define the optimizer (this is commented in the solver)

#nets = [FCNN(n_input_units=1,  hidden_units=(64,64,)) for _ in range(2)]

nets = [CustomNN(n_input_units=1, hidden_units= (256, 128, 64), actv = nn.SiLU, n_output_units = 1) for _ in range(2)]
#nn.SiLU
#nets = torch.load('nets_LCDM.ph')


adam = torch.optim.Adam(set([p for net in nets for p in net.parameters()]),
                        lr=1e-3)


tgz = Generator1D(32, t_min=N_p_0, t_max=N_p_f)#, method='log-spaced-noisy')

vgz = Generator1D(32, t_min=N_p_0, t_max=N_p_f)#, method='log-spaced')

#tg0 = Generator1D(64, t_min=Om_m_0_min, t_max=Om_m_0_max)#, method='log-spaced-noisy')

#vg0 = Generator1D(64, t_min=Om_m_0_min, t_max=Om_m_0_max)#, method='log-spaced')

train_gen = tgz 

valid_gen = vgz 


# Define the ANN based solver:
    
solver = BundleSolver1D(ode_system=ODE_LCDM,
                        nets=nets,
                        conditions=condition,
                        t_min=N_p_0, t_max=N_p_f,
                        optimizer=adam,
                        train_generator=train_gen,
                        #valid_generator=valid_gen,
                        n_batches_valid=0,
                        loss_fn=weighted_loss_LCDM,
                        )

# Set the amount of interations to train the solver:
iterations = 100000

# Start training:
solver.fit(iterations)

# Plot the loss during training, and save it:
loss = solver.metrics_history['train_loss']
plt.plot(loss, label='training loss')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
plt.xlabel('iterations')
plt.ylabel('Loss')
plt.yscale('log')
plt.legend()
plt.suptitle('Total loss during training')
plt.savefig('loss_LCDM_silu_custom_nb.png')

# Save the neural network:
torch.save(solver._get_internal_variables()['best_nets'], 'nets_LCDM_silu_custom_nb.ph')