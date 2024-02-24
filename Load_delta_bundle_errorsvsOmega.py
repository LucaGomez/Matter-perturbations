# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 18:42:20 2023

@author: Luca
"""


import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomNN(nn.Module):
    def __init__(self, n_input_units, hidden_units, actv, n_output_units):
        super(CustomNN, self).__init__()

        # Layers list to hold all layers
        self.layers = nn.ModuleList()

        # First hidden layer with special behavior
        self.layers.append(nn.Linear(n_input_units, hidden_units[0]))

        # Learnable parameters mu and sigma for the firs layer
        self.mu = torch.linspace(0,1, hidden_units[0])#nn.Parameter(torch.linspace(0,1, hidden_units[0])) #torch.linspace(0,1, hidden_units[0])#
        self.sigma = nn.Parameter(torch.ones(hidden_units[0])*0.2)

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
    
    
# Load the networks:
nets = torch.load('nets_LCDM_silu_custom.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )
                  
Om_r=5.38*10**(-5)
a_0 = 10**(-3)
a_f = 1
N_0 = np.log(a_0)
N_f= np.log(a_f)
n_0=np.abs(np.log(a_0))
N_p_0=-1
N_p_f=0

Om_m_min=0.1
Om_m_max=0.7


condition = [BundleIVP(N_p_0, -n_0),
             BundleIVP(N_p_0, n_0)]


sol = BundleSolution1D(nets, condition)


# The Hubble parameter as a function of the dependent variables of the system:

def x(N,Om_m_0):
    xs = sol(N, Om_m_0, to_numpy=True)[0]
    return xs

def x_pnn(N, Om_m_0):
    xs_p = sol(N, Om_m_0, to_numpy=True)[1]
    return xs_p


N_vec = np.linspace(N_p_0, N_p_f,200)
Om_vec = np.linspace(Om_m_min,Om_m_max,100)

err=[]
err_p=[]

for i in range(len(Om_vec)):
    
    Om_m=Om_vec[i]
    Om_m_0_vec=Om_m*np.ones(len(N_vec))

    x_nn=x(N_vec,Om_m_0_vec)
    x_p_nn=x_pnn(N_vec,Om_m_0_vec)

    Om_L=1-Om_r-Om_m
    a_eq=Om_r/Om_m #Equality between matter and radiation
    alpha=a_eq**3*Om_L/Om_m
    
    def F(N_p,X):
        
        N=n_0*N_p
        
        f1=X[1] 

        term1=(3*np.exp(N)/(2*a_eq*(1+(np.exp(N)/a_eq)+alpha*(np.exp(N)/a_eq)**4)))*n_0**2
        
        term2=-((1+4*alpha*(np.exp(N)/a_eq)**3)/(2*(1+(a_eq/np.exp(N))+alpha*(np.exp(N)/a_eq)**3)))*X[1]*n_0
      
        term3=-X[1]**2
      
        f2=term1+term2+term3
        
        return np.array([f1,f2])


    atol, rtol = 1e-15, 1e-12
    #Perform the backwards-in-time integration
    out2 = solve_ivp(fun = F, t_span = [N_p_0,N_p_f], y0 = np.array([-n_0,n_0]),
                    t_eval = N_vec, method = 'RK45')

    x_num=out2.y[0]
    x_p_num=out2.y[1]
    
    delta_p_nn=np.exp(x_nn)*x_p_nn/n_0
    delta_p_num=np.exp(x_num)*x_p_num/n_0
    delta_nn=np.exp(x_nn)
    delta_num=np.exp(x_num)
    N=N_vec*n_0
    
    dif_rel=[]
    dif_rel_p=[]
    for i in range(len(N)):
        dif_rel.append(100*np.abs(delta_nn[i]-delta_num[i])/np.abs(delta_num[i]))
        dif_rel_p.append(100*np.abs(delta_p_nn[i]-delta_p_num[i])/np.abs(delta_p_num[i]))
    
    err.append(np.max(dif_rel))
    err_p.append(dif_rel_p[-1])
    
plt.scatter(Om_vec,err,label=r'err $\delta(a=1)$')
plt.scatter(Om_vec,err_p,label=r'err $\frac{d\delta}{da}$ max')
plt.xlabel(r'$\Omega_{m0}$')
plt.ylabel('err%')
plt.legend()
plt.savefig('errvsOmsilucustom.png')

#%%