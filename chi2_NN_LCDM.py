# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 18:34:37 2024

@author: Luca
"""

"""
Created on Sat Dec 23 12:20:47 2023

@author: Luca
"""




import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import ticker, cm
import emcee                                          # Library for implementing the MCMC method
import corner                                         # Library for plotting figures with contours and piramids.
from neurodiffeq.solvers import BundleSolution1D
from neurodiffeq.conditions import BundleIVP
import torch
from scipy.integrate import trapz, simps
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
    


nets = torch.load('nets_LCDM_silu_custom.ph',
                  map_location=torch.device('cpu')  # Needed if trained on GPU but this sciprt is executed on CPU
                  )
Om_r=5.38*10**(-5)
a_0 = 10**(-3)
a_f = 1
N_0 = np.log(a_0)
N_f= np.log(a_f)
H_0=70

n_0=np.abs(N_0)

condition = [BundleIVP(-1, -n_0),
             BundleIVP(-1, n_0)]

sol = BundleSolution1D(nets, condition)


# The Hubble parameter as a function of the dependent variables of the system:


def x(params, N):
    Om_m_0, s8 = params
    Om_m_vec=Om_m_0*np.ones(len(N))
    xs = sol(N, Om_m_vec, to_numpy=True)[0]
    return xs

def x_pnn(params, N):
    Om_m_0, s8 = params
    Om_m_vec=Om_m_0*np.ones(len(N))
    xs_p = sol(N, Om_m_vec, to_numpy=True)[1]
    return xs_p

def delta(params, N):
    N_p=N/n_0
    x_nn = x(params, N_p)
    delta_nn=np.exp(x_nn)
    return delta_nn

def delta_pann(params, N):
    N_p=N/n_0
    x_nn = x(params, N_p)
    x_p_nn=x_pnn(params, N_p)
    delta_p_nn=np.exp(x_nn)*x_p_nn/n_0
    return delta_p_nn

def fs8(params, a): #fs8
    Om_m_0, s8 = params
    N=np.log(a)
    delta_today=delta(params, np.array([0]))
    return s8*delta_pann(params, N)/delta_today



def Hh(params, a):
    Om_m_0, s8=params
    Om_L=1-Om_m_0-Om_r
    return np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)

def Hh_p(params, a):
    Om_m_0, s8=params
    Om_L = 1-Om_m_0-Om_r
    num = (3*Om_m_0/a**4+4*Om_r/a**5)
    den = 2*np.sqrt(Om_L+Om_m_0/a**3+Om_r/a**4)
    return -num/den



def Integrando(params):
    #print(params)
    Om_m_0, s8 = params
    return lambda a: 1/((a**2)*Hh(params,a))

def dL(params,a):
    #print('dL'+str(params))
    Om_m_0, s8=params    
    x = np.linspace(a, 1, 500)
    y = Integrando((Om_m_0, s8))(x)
    return simps(y, x)


z = [0.02, 0.02, 0.02, 0.10, 0.15, 0.17, 0.18, 0.38, 0.25, 0.37, 0.32, 0.59, 0.44, 0.60, 0.73, 0.60, 0.86, 1.40]
fs8_data = [0.428, 0.398, 0.314, 0.370, 0.490, 0.510, 0.360, 0.440, 0.3512, 0.4602, 0.384, 0.488, 0.413, 0.390, 0.437, 0.550, 0.400, 0.482]
err = [0.0465, 0.065, 0.048, 0.130, 0.145, 0.060, 0.090, 0.060, 0.0583, 0.0378, 0.095, 0.060, 0.080, 0.063, 0.072, 0.120, 0.110, 0.116]
fid_Om_m=[0.3,0.3,0.266,0.3,0.31,0.3,0.27,0.27,0.25,0.25,0.274,0.307115,0.27,0.27,0.27,0.3,0.3,0.270]
a_prueba=np.array(z)
a_prueba=1/(1+a_prueba)
z_prueba=np.array(z)
'''
DEFINITION OF THE SCALE FACTOR DATA VECTOR
'''

a=[]
for i in range(len(z)):
    a.append(1/(1+z[i]))
a=sorted(a)

def ratio(params):
    #print(params)
    Om_m_0,s8=params
    rat=[]
    for i in range(len(a)):
        params_fid=fid_Om_m[i],0.8
        rat.append((Hh(params,a[i])*dL(params,a[i]))/(Hh(params_fid,a[i])*dL(params_fid,a[i])))
    return np.array(rat)

'''
HERE WE DEFINE DE COVARIANCE MATRIX
'''

sigmas1=[]
for i in range(11):
    sigmas1.append(1/err[i]**2)
sigmas1=np.array(sigmas1)
    
sigmas2=[]
for i in range(15,18):
    sigmas2.append(1/err[i]**2)
sigmas2=np.array(sigmas2)    

# Números en la diagonal para las filas de 0 a 12
diag_values_1 = sigmas1

# Matriz de 3x3 no diagonal para las filas 13, 14 y 15
non_diag_values = np.linalg.inv(10**(-3)*np.array([[6.400,2.570,0.000],[2.570,3.969,2.540],[0.00,2.540,5.184]]))

# Números en la diagonal para las filas 16, 17 y 18
diag_values_2 = sigmas2

# Crear la matriz de covarianza
cov_matrix = np.zeros((18, 18))

# Asignar los valores en la diagonal
np.fill_diagonal(cov_matrix, np.concatenate([diag_values_1, diag_values_2]))

# Asignar la submatriz de 3x3 no diagonal
cov_matrix[12:15, 12:15] = non_diag_values

'''
LIKELIHOOD AND POSTERIOR
'''

chi=[]
# define the log likelihood function  #Om_m,s8 - N_data - fs8_data - err fs8 
def log_likelihood(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8  = params
    #print(params)
    fs8_teo=fs8(params, a_data)
    rati=ratio(params)
    V=fs8_data-rati*fs8_teo
    chi2=V@cov_matrix@V
    #chi2 = np.sum( ( fs8_data - fs8_teo )**2 / err**2 )
    loglike = -0.5 * chi2
    #print("Chi2 = ", chi2)
    #chi.append(chi2)
    #print("loglike Chi2 = ", loglike)
    return chi2



# define the log prior function
def log_posterior(params, a_data, fs8_data, fs8_err):
    Om_m_0, s8  = params
    if 0.1 < Om_m_0 < 0.9 and 0.5 < s8 < 1.3 :
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    #print("logpost = ", logpost)
    return logpost

Omegam,sigma8=0.272,0.802
params=Omegam,sigma8
fsigma8_prueba=np.array(2000)
fsigma8_prueba=fs8(params, a)
fsigma8_prueba=sorted(fsigma8_prueba)
like=log_likelihood(params, a_prueba, fs8_data, err)
print(like)

plt.figure()
plt.plot(z_prueba,fsigma8_prueba,'r.')
plt.errorbar(z_prueba,fs8_data,err)
plt.xlabel('z')
plt.ylabel('fs8')
plt.ylim(0.1,0.8)
#plt.savefig('pruebafs8.png',format='png')
plt.show()

#%%
N=50
Om=np.linspace(0.2,0.4,N)
s8=np.linspace(0.7,0.9,N)

val=np.inf
M=np.zeros((N,N))
for i in range(len(Om)):
    fila=[]
    for j in range(len(s8)):
        params=Om[i],s8[j]
        like=log_likelihood(params, a_prueba, fs8_data, err)
        fila.append(like)
        if like < val:
            val=like
            vals=params
    fila=np.array(fila)
    M[i,:]=fila

Om_b,s8_b=vals

Om_label = "{:.3f}".format(Om_b)
s8_label = "{:.3f}".format(s8_b)

plt.figure(figsize=(10, 6))
plt.pcolormesh(s8, Om, M, cmap='viridis', vmax=20)  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar(label=r'$\chi^2$')
plt.ylabel(r'$\Omega_{m0}$')
plt.xlabel(r'$\sigma_8$')
plt.axvline(s8_b,linestyle='dotted',label=r'$\sigma_8=$'+str(s8_label))
plt.axhline(Om_b,linestyle='dotted',label=r'$\Omega_{m0}=$'+str(Om_label))
plt.legend()
#plt.title(r'$\chi^2$')
plt.savefig('chi2NNLCDM.png')