# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 18:16:01 2024

@author: Luca
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

Om_m=0.272

Om_r=5.38*10**(-5)
a_0 = 10**(-3)
a_f = 1
N_0 = np.log(a_0)
N_f= np.log(a_f)
n_0=np.abs(np.log(a_0))
N_p_0=-1
N_p_f=0
Om_L=1-Om_r-Om_m
a_eq=Om_r/Om_m #Equality between matter and radiation
alpha=a_eq**3*Om_L/Om_m

N_vec = np.linspace(N_p_0, N_p_f,200)

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

delta_p_num=np.exp(x_num)*x_p_num/n_0
delta_num=np.exp(x_num)
N=N_vec*n_0
a=np.exp(N)


plt.plot(a, delta_num, label=r'$\delta$ Num')
plt.plot(a, delta_p_num/a,label=r'$\delta_a$ Num')
plt.xlabel('a')
plt.ylabel(r'$\delta , \delta_a$')
plt.xscale('log')
plt.savefig('Numerical'+str(Om_m)+'.png')
plt.legend()

#%%


import numpy as np
import matplotlib.pyplot as plt                                 # Library for plotting figures with contours and piramids.
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

a_vec=np.linspace(a_0,a_f,2000)

fs88=fs8(a_vec,0.272,0.8)

plt.errorbar(a, fs8_data, yerr=err, fmt='o', capsize=5, label='Datos')
plt.plot(a_vec,fs88,label='Numérico')
plt.xlabel('a')
plt.ylabel(r'$f_{\sigma8}$')
plt.xlim(0.3,1)
plt.legend()
plt.savefig('fs8numdata.png')

#%%
z_vec=1/a_vec-1


plt.errorbar(z, fs8_data, yerr=err, fmt='o', capsize=5, label='Datos')
plt.plot(z_vec,fs88,label='Numérico')
plt.xlabel('z')
plt.ylabel(r'$f_{\sigma8}$')
plt.xlim(0,1.5)
plt.legend()
plt.savefig('fs8numdataz.png')
