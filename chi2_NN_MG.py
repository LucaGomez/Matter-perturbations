# -*- coding: utf-8 -*-
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
    


nets = torch.load('nets_LCDM_mg.ph',
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


def x(N,Om_m_0,g_a):
    xs = sol(N, Om_m_0, g_a, to_numpy=True)[0]
    return xs

def x_pnn(N, Om_m_0, g_a):
    xs_p = sol(N, Om_m_0, g_a, to_numpy=True)[1]
    return xs_p

def delta(N, Om_m_0, g_a):
    N_p=N/n_0
    Om_m_vec=Om_m_0*np.ones(len(N))
    g_a_vec=g_a*np.ones(len(N))
    x_nn = x(N_p,Om_m_vec,g_a_vec)
    delta_nn=np.exp(x_nn)
    return delta_nn

def delta_pann(N, Om_m_0, g_a):
    N_p=N/n_0
    Om_m_vec=Om_m_0*np.ones(len(N))
    g_a_vec=g_a*np.ones(len(N))
    x_nn = x(N_p,Om_m_vec,g_a_vec)
    x_p_nn=x_pnn(N_p,Om_m_vec,g_a_vec)
    delta_p_nn=np.exp(x_nn)*x_p_nn/n_0
    return delta_p_nn

def fs8(a, Om_m_0, s8, g_a): #fs8
    N=np.log(a)
    delta_today=delta(np.array([0]),Om_m_0,g_a)
    return s8*delta_pann(N,Om_m_0,g_a)/delta_today



def Hh(a,Om_m):
    Om_L=1-Om_m-Om_r
    return np.sqrt(Om_L+Om_m/a**3+Om_r/a**4)

def Hh_p(a,Om_m):
    Om_L = 1-Om_m-Om_r
    num = (3*Om_m/a**4+4*Om_r/a**5)
    den = 2*np.sqrt(Om_L+Om_m/a**3+Om_r/a**4)
    return -num/den



def Integrando(a,Om_m):
    return 1/((a**2)*Hh(a,Om_m))

def dL(a,Om_m):    
    x = np.linspace(a, 1, 500)
    y = Integrando(x,Om_m)
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

def ratio(Om_m):
    rat=[]
    for i in range(len(a)):
        rat.append((Hh(a[i],Om_m)*dL(a[i],Om_m))/(Hh(a[i],fid_Om_m[i])*dL(a[i],fid_Om_m[i])))
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
    Om_m_0, s8, g_a  = params
    #print(params)
    fs8_teo=fs8(a_data,Om_m_0,s8,g_a)
    rati=ratio(Om_m_0)
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
    Om_m_0, s8, g_a  = params
    if 0.1 < Om_m_0 < 0.9 and 0.5 < s8 < 1.3 and -4 < g_a < 2 :
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    #print("logpost = ", logpost)
    return logpost

Omegam,sigma8,ga=0.250,0.917,-1.167
params=Omegam,sigma8,ga
fsigma8_prueba=np.array(2000)
fsigma8_prueba=fs8(a, Omegam, sigma8, ga)
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
N=35
Om=np.linspace(0.1,0.5,N)
s8=np.linspace(0.6,1,N)
ga=np.linspace(-3,0,N)

val=np.inf

for i in range(len(Om)):
    print(i)
    for j in range(len(s8)):
        for k in range(len(ga)):
            params=Om[i],s8[j],ga[k]
            like=log_likelihood(params, a_prueba, fs8_data, err)
            if like < val:
                val=like
                vals=params
            

Om_b,s8_b,ga_b=vals

Om_label = "{:.3f}".format(Om_b)
s8_label = "{:.3f}".format(s8_b)
ga_label = "{:.3f}".format(ga_b)

print('chi2 = '+str(val))
print('Omega_matter = '+str(Om_b))
print('sigma_8 = '+str(s8_b))
print('ga = '+str(ga_b))

#%%
# set up the problem
ndim     = 2                                 # number of parameters
nwalkers = 10                               # number of walkers
nsteps   = 7000                           # number of steps per walker
init0    = 0.3                         # initial value for log_mu_phi
init1    = 0.8                             # initial value for log_g_X

p0 = np.array([init0, init1])
p0 = p0 + np.zeros( (nwalkers, ndim) )
p0[:,0] = p0[:,0] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )
p0[:,1] = p0[:,1] + np.random.uniform( low=-0.1, high=0.1, size=nwalkers )

backend   = emcee.backends.HDFBackend('chain_num_N.h5')
backend.reset(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(a, fs8_data, err),backend=backend)

max_n = nsteps

# We'll track how the average autocorrelation time estimate changes
index = 0
autocorr = np.empty(max_n)

# This will be useful to testing convergence
old_tau = np.inf


# Now we'll sample for up to max_n steps
for sample in sampler.sample(p0, iterations=max_n, progress=True):
    # Only check convergence every 100 steps
    if sampler.iteration % 100:
        continue

    # Compute the autocorrelation time so far
    # Using tol=0 means that we'll always get an estimate even
    # if it isn't trustworthy
    tau = sampler.get_autocorr_time(tol=0)
    autocorr[index] = np.mean(tau)
    index += 1

    # Check convergence
    converged = np.all(tau * 100 < sampler.iteration)
    converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
    if converged:
        break
    old_tau = tau

# save the data from teh chain of parameter values

witness_file ='numerical_N.txt'
textfile_witness = open(witness_file,'w+')
textfile_witness.close()


# plot the evolution of the chains
from matplotlib.ticker import MaxNLocator
plt.figure()
fig, ax = plt.subplots( ndim, 1, sharex=True, figsize=(8,9) )
ax0 = ax[0]
ax1 = ax[1]

ax0.plot( sampler.chain[:, :, 0].T, color="k", alpha=0.4 )
ax0.yaxis.set_major_locator(MaxNLocator(5))
ax0.axhline(init0, color="#888888", lw=2)
ax0.set_ylabel("$\Omega_{m0}$")

ax1.plot(sampler.chain[:, :, 1].T, color="k", alpha=0.4)
ax1.yaxis.set_major_locator(MaxNLocator(5))
ax1.axhline(init1, color="#888888", lw=2)
ax1.set_ylabel("$\sigma_8$")


fig.tight_layout()
fig.savefig('chains_num_N.png')
#plt.show()

# get the chain of parameter values and calculate the posterior probabilities
samples = sampler.chain[:, :, :].reshape( (-1, ndim) )
post_probs = np.exp( sampler.flatlnprobability - np.max(sampler.flatlnprobability) )

# find the best fit parameters using the maximum a posteriori (MAP) method
best_fit_params = samples[ np.argmax(post_probs), : ]

# print the results
print( r'Best fit parameters: Om={:.3f}, s8={:.3f}'.format(*best_fit_params) )

# mean adn std
meann_bfit = np.mean(samples, axis=0)
std_bfit   = np.std( samples, axis=0)

# make the triangle plot
fig = corner.corner( samples, labels=[ "$\Omega_{m0}$", "$\sigma_8$"], truths=[init0, init1], \
                              quantiles=[0.16, 0.50], bins=40, plot_datapoints = True, \
                              scale_hist=True )
#plt.show()
fig.savefig('triangplot_num_N.png')
plt.close()

print(np.min(chi))
#%%
#Sanity check

Om=np.linspace(0.05,0.7,50)
s8=np.linspace(0.5,1.3,50)

M=np.zeros((50,50))
for i in range(len(Om)):
    chi=[]
    for j in range(len(s8)):
        params=Om[i],s8[j]
        chi.append(log_likelihood(params, a_prueba, fs8_data, err))
        #if 0.48<Om[i]<0.52 and 0.58<s8[j]<0.62:
         #   print('chi_luca', chi2L(params, N_data, fs8_data))
        #if 0.18<Om[i]<0.22 and 0.94<s8[j]<0.96:
         #   print('chi_ness', chi2L(params, N_data, fs8_data))            
    M[i,:]=np.array(chi)
    


val=np.inf
for i in range(len(Om)):
    for j in range(len(s8)):

        if M[i][j]<val:
            val=M[i][j]
            eti=i
            etj=j
print(val,eti,etj)

Om_minc=Om[eti]
s8_mic=s8[etj]


Om_label = "{:.2f}".format(Om_minc)
s8_label = "{:.2f}".format(s8_mic)


plt.figure(figsize=(10, 6))
plt.pcolormesh(s8,Om, M, cmap='viridis', vmax=30)  # 'viridis' es un mapa de colores, puedes elegir otro
plt.colorbar()
plt.ylabel(r'$\Omega_{m0}$')
plt.xlabel(r'$\sigma_8$')
plt.axhline(Om_minc,linestyle='dotted',label=r'$\Omega_{m0}$='+str(Om_label))
plt.axvline(s8_mic,linestyle='dotted',label=r'$\sigma_8$='+str(s8_label))
plt.title(r'$\chi^2$')
plt.legend()

# Muestra el mapa de calor
plt.show()



