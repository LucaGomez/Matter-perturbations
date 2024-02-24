# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 17:35:10 2024

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
nets = torch.load('nets_LCDM_mg.ph',
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

g_a_min=-4
g_a_max=4

n=2


condition = [BundleIVP(N_p_0, -n_0),
             BundleIVP(N_p_0, n_0)]


sol = BundleSolution1D(nets, condition)


# The Hubble parameter as a function of the dependent variables of the system:

def x(N,Om_m_0,g_a):
    xs = sol(N, Om_m_0, g_a, to_numpy=True)[0]
    return xs

def x_pnn(N, Om_m_0, g_a):
    xs_p = sol(N, Om_m_0, g_a, to_numpy=True)[1]
    return xs_p


N_vec = np.linspace(N_p_0, N_p_f,200)
Om_vec = np.linspace(Om_m_min,Om_m_max,100)
ga_vec = np.linspace(g_a_min,g_a_max,100)


err=np.zeros((len(ga_vec),len(Om_vec)))
err_p=np.zeros((len(ga_vec),len(Om_vec)))

for i in range(len(Om_vec)):
    fila=[]
    fila_p=[]
    for j in range(len(ga_vec)):
        
        g_a=ga_vec[j]
        g_a_eval=ga_vec[j]*np.ones(len(N_vec))
        
        Om_m=Om_vec[i]
        Om_m_eval=Om_m*np.ones(len(N_vec))
    

        x_nn=x(N_vec,Om_m_eval,g_a_eval)
        x_p_nn=x_pnn(N_vec,Om_m_eval,g_a_eval)

        Om_L=1-Om_r-Om_m
        a_eq=Om_r/Om_m #Equality between matter and radiation
        alpha=a_eq**3*Om_L/Om_m
    
        def F(N_p,X):
        
            N=n_0*N_p
        
            f1=X[1] 
        
            g_eff= 1 + g_a*(1-np.exp(N))**n - g_a*(1-np.exp(N))**(2*n)

            term1=(3*np.exp(N)/(2*a_eq*(1+(np.exp(N)/a_eq)+alpha*(np.exp(N)/a_eq)**4)))*g_eff*n_0**2
        
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
        for k in range(len(N)):
            dif_rel.append(100*np.abs(delta_nn[k]-delta_num[k])/np.abs(delta_num[k]))
            dif_rel_p.append(100*np.abs(delta_p_nn[k]-delta_p_num[k])/np.abs(delta_p_num[k]))
    
        fila.append(np.max(dif_rel))
        fila_p.append(dif_rel_p[-1])
    
    fila=np.array(fila)
    fila_p=np.array(fila_p)
    
    err[i,:]=fila
    err_p[i,:]=fila_p
    
    
plt.figure()
plt.pcolormesh(ga_vec, Om_vec, err, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'err% hoy (a=1)')
plt.xlabel(r'$g_a$')
plt.ylabel(r'$\Omega_{m0}$')
#plt.title(r'err% $\delta_m$')
plt.savefig('heatmap_delta_mg.png')
   

plt.figure()
plt.pcolormesh(ga_vec, Om_vec, err_p, cmap='viridis')  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'Max err%')
plt.xlabel(r'$g_a$')
plt.ylabel(r'$\Omega_{m0}$')
#plt.title(r'err% $\frac{d\delta_m}{da}$')
plt.savefig('heatmap_delta_p_mg.png')
