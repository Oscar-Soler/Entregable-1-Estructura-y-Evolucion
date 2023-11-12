# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 08:45:38 2023

@author: osole
"""

#We load the relevant packages
from scipy . integrate import solve_ivp
import matplotlib . pyplot as plt
import numpy as np
import matplotlib
import pandas as pd
import scipy as sp

matplotlib.rcParams['figure.figsize'] = (10, 8)
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['figure.titlesize'] = 26
matplotlib.rcParams['axes.labelsize'] = 16
matplotlib.rcParams['legend.fontsize'] = 16

def lane_emden (xi , th , n) :
    return ([th[1], -th[0]**n - 2/xi * th[1]])

def self_destruct(xi, th, n):
    return(th[0])

self_destruct.terminal = True
self_destruct.direction = -1

#%%
#constants
M =  1.9891E30  # kg
R = 6.957E8     # m
rho_av = M/(4/3*np.pi*R**3) #kg/m^3
G = 6.6743E-11  # SI
m_H = 1.660538921E-27 # kg
mu_sol = 0.61
k_B = 1.3806488E-23 #SI

#initial and final integration values xi
xi0 = .1
xi_max = 10

#boundary condtitions
th0 = 1
dth0 = 0

#values of n
n=np.arange(0,5.5,.5)

# Figure containing different results of Lane-Emden equation
plt.figure(1)
plt.clf()
fig, ax = plt.subplots(num=1)
#colors for the different n values
colors = plt.cm.viridis(np.linspace(0, 1, len(n)))

#list that will contain the solutions of the ODEs of each n value
solutions = []
#parameters calculated for each n value
xi1, Dn, Mn, Bn, rrho_c, Pp_c = [], [], [], [], [], []

for i in range(len(n)):
    solution = solve_ivp(lane_emden, [xi0, xi_max], [th0,dth0],\
                         args = (n[i],), events = (self_destruct),\
                         max_step=.01)
    solutions.append(solution)
    
    xi1.append(solution.t[-1])
    Dn.append(-(3/solution.t[-1]*solution.y[1][-1])**(-1))
    Mn.append(-solution.t[-1]**2*solution.y[1][-1])
    Bn.append(((3*Dn[-1])**((3-n[i])/(3*n[i])))/((n[i]+1)*Mn[-1]**((n[i]-1)/n[i])*xi1[-1]**((3-n[i])/n[i])))
    rrho_c.append(Dn[-1]*rho_av)
    Pp_c.append((4*np.pi)**(1/3)*Bn[-1]*G*M**(2/3)*rrho_c[-1]**(4/3))
    ax.plot(solution.t, solution.y[0,:], color=colors[i], label = '%.1f'%n[i])

ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\theta(\\xi)$')
ax.set_title('Solutions of the Lane-Emden equation')
ax.set_xlim(0, xi_max)
ax.set_ylim(0)
ax.grid()
plt.legend()

# alfas = R/np.array(xi1)

#%% Polytropic constants
#Returning results in CSV format to import to latex table
print('n,Rn,Dn,Mn,Bn')
for i in range(len(n)):
    print('%.1f,%.2f,%.3f,%.2f,%.3f'%(n[i],xi1[i],Dn[i],Mn[i],Bn[i]))
    
#%% Analytic results
def ana0(x): return -x**2/6+1
def ana1(x): return np.sin(x)/x
plt.figure(3)
plt.clf()
fig, ax = plt.subplots(num=3)
colors = ['tab:blue', 'tab:red']
for i in range(2):
    ax.plot(solutions[2*i].t, solutions[2*i].y[0,:], color=colors[i], linestyle = '--', label = 'Numeric n=%.1f'%n[2*i])

ax.plot(solutions[0].t, ana0(solutions[0].t),color = 'b', label = '$-\\frac{\\xi^2}{6}+1$')
ax.plot(solutions[2].t, ana1(solutions[2].t),color = 'r', label = '$\\frac{\\sin \\xi}{\\xi}$')

ax.set_xlabel('$\\xi$')
ax.set_ylabel('$\\theta(\\xi)$')
ax.set_title('Comparison of analytic solutions for n=0 and n=1')

ax.set_xlim(0, solutions[2].t[-1])
ax.set_ylim(0)
ax.grid()
plt.legend()


#%% For comparing with the model of the sun
#function to extract log rho, M, log P and log T from the polytropic model
def poly(n_ind, ns = n, s=solutions, x=xi1, R_sun=R, M_sun = M, r_c = rrho_c, G=G,\
         Pc = Pp_c, mH = m_H, mu = mu_sol, kB = k_B, M_n = Mn, Rn = xi1):
    
    index = np.where(ns == n_ind)[0][0] #for locating the values corresponding to the n requested
    alfas = R_sun/np.array(x)
    
    # calculation of mass(xi)
    integrando_n = s[index].t**2*s[index].y[0]**ns[index]
    M_xi = np.zeros(len(s[index].t))
    for i in range(len(M_xi)-1):
        M_xi[i+1] = sp.integrate.simpson(integrando_n[:i+1], s[index].t[:i+1])
    M_xi_full = 4*np.pi*alfas[index]**3*r_c[index]*M_xi
    MM_xi = M_xi_full/M_sun
    
    # calculation of radius
    RR = alfas[index]*s[index].t/R_sun
    
    # calculation of density
    rrho = r_c[index]*s[index].y[0]**ns[index]
    log_rho_n = np.log10(rrho)
    

    # calculation of pressure
    integrando_P = M_xi*rrho/(s[index].t)**2 
    P_xi = np.zeros(len(s[index].t))
    for i in range(len(P_xi)-1):
        P_xi[i+1] = sp.integrate.simpson(integrando_P[:i+1], s[index].t[:i+1])
    int_P = -G/alfas[index]*P_xi
    P_escala = np.abs(Pc[index]/int_P[-1])
    P_xi = Pc[index]-G/(alfas[index])*P_xi*P_escala
    log_P_n = np.log10(P_xi)
    
    # calculation of temperature
    T = mH*mu*P_xi/(kB*rrho)
    
    return RR, log_rho_n, MM_xi, log_P_n, np.log10(T) 

# specific calculations for n=2.5, 3 and 3.5
RR_3, log_rho_3, MM_xi3, log_P_3, log_T3 = poly(3)
RR_25, log_rho_25, MM_xi25, log_P_25, log_T25= poly(2.5)
RR_35, log_rho_35, MM_xi35, log_P_35, log_T35 = poly(3.5)
#%% Model of the sun

# reading of data
df = pd.read_csv('C:/Users/osole/OneDrive/Documentos/1_Astro/Estructura_evolucion/Entregable_1/model_sun.txt',\
                 sep = '  ')

plt.figure(2)
plt.clf()
fig, axs = plt.subplots(2,2, num=2, sharex = True)

#extraction of data from model and con version to IS of units
rr = df[df.keys()[1]]
mm = df[df.keys()[0]]
log_T = np.log10(df[df.keys()[2]])
log_rho = np.log10(df[df.keys()[3]])+3 #to turn into IS
log_P = np.log10(df[df.keys()[4]])-1 #to turn into IS

lst = [log_rho, mm, log_P, log_T]
lst_names = ['$\\log \\rho$', '$M/M_\\odot$', '$\\log P$', '$\\log T$']
axs_lst = [axs[0,0], axs[0,1], axs[1,0], axs[1,1]]

#plotting of model results
i=0
for y in lst:
    axs_lst[i].plot(rr, y, color = 'k', linestyle = '--', label = 'Bahcall') # lst_names[i])
    axs_lst[i].set_title(lst_names[i])
    if i>1:
        axs_lst[i].set_xlabel('$R/R_\\odot$')
    i+=1

# plotting of polytropic models
colors = plt.cm.cool(np.linspace(0, 1, 3))
def plot_n(RR, log_rho, MM, log_P, log_T, n, k, c=colors):
    axs_lst[0].plot(RR, log_rho, color = c[k], label = '%.1f'%n)
    axs_lst[0].legend()
    axs_lst[1].plot(RR, MM, color = c[k], label = '%.1f'%n)
    axs_lst[1].legend()
    axs_lst[2].plot(RR, log_P, color = c[k], label = '%.1f'%n)
    axs_lst[2].legend()
    axs_lst[3].plot(RR[np.where(log_T!=-np.inf)[0][:-4]], log_T[np.where(log_T!=-np.inf)[0][:-4]], color = c[k], label = '%.1f'%n)
    axs_lst[3].legend()
plot_n(RR_25, log_rho_25, MM_xi25, log_P_25, log_T25, 2.5, 0)
plot_n(RR_3, log_rho_3, MM_xi3, log_P_3, log_T3, 3, 1)
plot_n(RR_35, log_rho_35, MM_xi35, log_P_35, log_T35, 3.5, 2)

axs_lst[0].set_ylim(-1,6)
axs_lst[2].set_ylim(7.5)

fig.suptitle('Comparison of different polytropic models for the Sun $vs$ \n Sophisticated solar model by Bahcall et al. (2005)')

fig.tight_layout()


