# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 12:20:47 2023

@author: Luca
"""


import numpy as np
import matplotlib.pyplot as plt
import emcee                                          # Library for implementing the MCMC method
import corner                                         # Library for plotting figures with contours and piramids.
from scipy.integrate import simps
from scipy.integrate import solve_ivp

'''
METHODS, DATA AND CONSTANTS
'''

Om_r=5.38*10**(-5)
a_0=10**(-3)
a_f=1

def Hh(a,Om_m):
    Om_L=1-Om_m-Om_r
    return np.sqrt(Om_L+Om_m/a**3+Om_r/a**4)

def Hh_p(a,Om_m):
    Om_L = 1-Om_m-Om_r
    num = (3*Om_m/a**4+4*Om_r/a**5)
    den = 2*np.sqrt(Om_L+Om_m/a**3+Om_r/a**4)
    return -num/den



    
def fs8(a, Om_m_0, s8): #fs8  
    #print(Om_m_0, s8)
    a=np.array(a)    
    def F(a,X):
        
        f1=X[1] 

        term1=X[0]*3*Om_m_0/(2*(Hh(a,Om_m_0)**2)*(a**5))
        
        term2=-X[1]*((3/a)+(Hh_p(a,Om_m_0)/Hh(a,Om_m_0)))
        
        f2=term1+term2
        
        return np.array([f1,f2])
    
    a_vec=np.linspace(a_0,a_f,2000)
    
    atol, rtol = 1e-15, 1e-12
    #Perform the backwards-in-time integration
    out2 = solve_ivp(fun = F, t_span = [a_0,a_f], y0 = np.array([a_0,1]),
                    t_eval = a_vec, method = 'RK45')

    delta_num=out2.y[0]
    delta_p_num=out2.y[1]
    
    delta_today=delta_num[-1]
    
    fs8_teo=[]
    for i in range(len(a)):
        a_val=a[i]
        indice = np.argmin(np.abs(np.array(a_vec) - a_val))
        fs8_teo.append(s8*a[i]*delta_p_num[indice]/delta_today)
    
    return fs8_teo

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
    Om_m_0, s8  = params
    #print(params)
    fs8_teo=fs8(a_data,Om_m_0,s8)
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
    Om_m_0, s8  = params
    if 0.1 < Om_m_0 < 0.9 and 0.5 < s8 < 1.3 :
        logpost = log_likelihood(params, a_data, fs8_data, fs8_err)
    else:
        logpost = -np.inf
    #print("logpost = ", logpost)
    return logpost

Omegam,sigma8=0.272,0.78623199
params=Omegam,sigma8
fsigma8_prueba=np.array(2000)
fsigma8_prueba=fs8(a, Omegam, sigma8)
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
plt.savefig('chi2numLCDM.png')




