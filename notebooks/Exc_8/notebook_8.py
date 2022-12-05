# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: intro-to-comp-neuro
#     language: python
#     name: intro-to-comp-neuro
# ---

# # Rate Networks

# +
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
import scipy

# %matplotlib inline

# Settings for the figures
plt.style.use(plt.style.available[20])
plt.style.use("https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/plots_style.txt")

# -

# \begin{equation}
# F(x;\alpha,\theta) = \left(1+\text{e}^{-\frac{x-\theta}{\alpha}}\right)^{-1} - \left(1+\text{e}^{\frac{\theta}{\alpha}}\right)^{-1}
# \end{equation}

# +
def nonlinearity_f(x, alpha, theta):    
    return (1 + np.exp(-alpha*(x - theta)))**-1 - (1 + np.exp(alpha*theta))**-1

fig, ax = plt.subplots(figsize = (4,4))
X = np.linspace(0,10,1000)
ax.plot(X, nonlinearity_f(X, alpha=1.2, theta=2.8), color = '#2171B5')
ax.plot(X, nonlinearity_f(X, alpha=1.0, theta=4.0), c='#CB181D')
ax.set(
    title = 'F-I curve',
    xlabel = 'Input current $I$',
    ylabel = 'Output firing rate $F$',
);


def interactive_f(alpha=0.2, theta=40):
    
    x_max = 80
    fig, ax = plt.subplots(figsize = (14,8))
    X = np.linspace(-0.1,x_max,100)
    ax.plot(X, nonlinearity_f(X, alpha=alpha, theta=theta), color='#6baed6', linewidth = 3)
    
    y_theta = nonlinearity_f(theta, alpha=alpha, theta=theta)
    
    ax.plot([theta,theta],[0,y_theta],
           linestyle = '--', c = '#e34a33', linewidth=1, label = r'threshold $\theta$')
    
    # we want to visualize the gain of the 
    
    diff = 8
    
    ax.plot([theta-diff,theta+diff],[y_theta-diff*alpha/4, y_theta+diff*alpha/4],
           linestyle = '--', c = 'k', linewidth=1, label = r'linear function with gain $\alpha/4$')
    
    ax.set(
        title = 'F-I curve',
        xlabel = 'Input current',
        ylabel = 'Output rate',
        ylim = [-0.05,1.05],
        xlim = [-0.05*x_max,x_max*1.05]
    );
    ax.legend()
    
widgets.interactive(interactive_f, alpha = (0.01,1,0.01), theta=(0,60,10))

# +
r_0 = 0.3
I_ext = 20

fig, ax = plt.subplots()

nonlinearity_f

r_list = [r_0]
t_list = [0]
dt = 0.1
tau = 20
for ii in range(1500):
    t_list.append(t_list[-1]+dt)
    r_list.append(r_list[-1] + dt*(-r_list[-1]+nonlinearity_f(I_ext,alpha=0.5,theta=10))/tau)

ax.plot(t_list, r_list, color='k')
ax.set_xticks([0,75,150])
ax.set_xticklabels([0,75,150],fontsize=14)
ax.set_yticks([0,0.5,1])
ax.set_yticklabels([0,0.5,1],fontsize=14)
ax.set_xlabel('time in ms', fontsize=16)
ax.set_ylabel('firing rate $r$', fontsize=16)


# -

# \begin{align}
# \tau_E \frac{dr_E}{dt} &= -r_E + F(w_{EE}r_E -w_{EI}r_I + I_E;\alpha_E,\theta_E) \\
# \tau_I \frac{dr_I}{dt} &= -r_I + F(w_{IE}r_E -w_{II}r_I + I_I;\alpha_I,\theta_I) \qquad (1)
# \end{align}
#
# \begin{align}
# \tau_E \frac{dr_E}{dt} &= -r_E + F_E(w_{EE}r_E -w_{EI}r_I + I_E) \\
# \tau_I \frac{dr_I}{dt} &= -r_I + F_I(w_{IE}r_E -w_{II}r_I + I_I) \qquad (1)
# \end{align}

# +
def compute_drdt(r, I_ext, w, alpha, theta, tau, **other_pars):
  """Given parameters, compute dr/dt as a function of r.

  Args:
    r (1D array) : Average firing rate of the excitatory population
    I_ext, w, a, theta, tau (numbers): Simulation parameters to use
    other_pars : Other simulation parameters are unused by this function

  Returns
    drdt function for each value of r
  """
  # Calculate drdt
  drdt = (-r + nonlinearity_f(w * r + I_ext, alpha, theta)) / tau

  return drdt


# Define a vector of r values and the simulation parameters
r = np.linspace(0, 1, 1000)

# Compute dr/dt
drdt = compute_drdt(r, I_ext = 0.5, w = 5, alpha = 1.2, theta = 2.8, tau = 20)

fig, ax = plt.subplots()
ax.plot(r, drdt)
ax.plot([0,1],[0,0],linestyle = '--', color = 'k')

x_ticks = [0,0.5,1]
y_ticks = [-0.003, 0, 0.003]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('$r$', fontsize=16, fontweight='bold')
ax.set_ylabel(r'$\frac{dr}{dt}$', fontsize=22, rotation=0, fontweight='bold')

# +


fig, ax = plt.subplots()



r_0 = 0.2
I_ext = 0.5
w=5
r_list = [r_0]
t_list = [0]
dt = 0.1
tau = 20
for ii in range(1500):
    t_list.append(t_list[-1]+dt)
    r_list.append(r_list[-1] + dt*(-r_list[-1] + nonlinearity_f(w*r_list[-1]+I_ext,alpha=1.2,theta=2.8))/tau)

ax.plot(t_list, r_list, color='#984ea3')

r_0 = 0.6
r_list = [r_0]
t_list = [0]
for ii in range(1500):
    t_list.append(t_list[-1]+dt)
    r_list.append(r_list[-1] + dt*(-r_list[-1] + nonlinearity_f(w*r_list[-1]+I_ext,alpha=1.2,theta=2.8))/tau)

ax.plot(t_list, r_list, color='#ff7f00')



ax.set_xticks([0,75,150])
ax.set_xticklabels([0,75,150],fontsize=14)
ax.set_yticks([0,0.5,1])
ax.set_yticklabels([0,0.5,1],fontsize=14)
ax.set_xlabel('time in ms', fontsize=16)
ax.set_ylabel('firing rate $r$', fontsize=16)

# +
fig, ax = plt.subplots()



for r_0 in [0, 0.1, 0.2, 0.3, 0.4, 0.44]:
    I_ext = 0.5
    w=5
    r_list = [r_0]
    t_list = [0]
    dt = 0.1
    tau = 20
    for ii in range(2200):
        t_list.append(t_list[-1]+dt)
        r_list.append(r_list[-1] + dt*(-r_list[-1] + nonlinearity_f(w*r_list[-1]+I_ext,alpha=1.2,theta=2.8))/tau)

    ax.plot(t_list, r_list, color='#984ea3', linewidth = 0.8)

for r_0 in [0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
    r_list = [r_0]
    t_list = [0]
    for ii in range(2200):
        t_list.append(t_list[-1]+dt)
        r_list.append(r_list[-1] + dt*(-r_list[-1] + nonlinearity_f(w*r_list[-1]+I_ext,alpha=1.2,theta=2.8))/tau)

    ax.plot(t_list, r_list, color='#ff7f00', linewidth = 0.8)



ax.set_xticks([0,100, 200])
ax.set_xticklabels([0,100,200],fontsize=14)
ax.set_yticks([0,0.5,1])
ax.set_yticklabels([0,0.5,1],fontsize=14)
ax.set_xlabel('time in ms', fontsize=16)
ax.set_ylabel('firing rate $r$', fontsize=16)

# +
I_E = 0
I_I = 0

w_EE = 9
w_EI = 4
w_II = 13
w_IE = 11

tau_E = 20
tau_I = 10

alpha_E = 1.2
theta_E = 2.8

alpha_I = 1.0
theta_I = 4.0

dt = 1
    
def run_E_I_simulation(r_E_0, r_I_0, timesteps=220):    
    
    r_E_list = [r_E_0]
    r_I_list = [r_I_0]
    t_list = [0]
    for ii in range(timesteps):
        t_list.append(t_list[-1]+dt)

        r_E = r_E_list[-1]
        r_I = r_I_list[-1]

        input_current_E = w_EE*r_E- w_EI*r_I+I_E

        dr_E_dt = dt*(-r_E + nonlinearity_f(input_current_E,alpha=alpha_E,theta=theta_E))/tau_E
        r_E_list.append(r_E + dr_E_dt)

        input_current_I = w_IE*r_E - w_II*r_I+I_I

        dr_I_dt = dt*(-r_I + nonlinearity_f(input_current_I,alpha=alpha_I,theta=theta_I))/tau_I
        r_I_list.append(r_I + dr_I_dt)
        
        
    return t_list, r_E_list, r_I_list


fig, ax = plt.subplots()
fig_2, ax_2 = plt.subplots()

t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.32, r_I_0 = 0.3)

ax.plot(t_list, r_E_list, color='#2171B5', linewidth = 1.5)
ax.plot(t_list, r_I_list, color='#CB181D', linewidth = 1.5)
ax_2.plot(r_E_list, r_I_list, linewidth = 1.2, color = '#984ea3')

t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.33, r_I_0 = 0.3)

ax.plot(t_list, r_E_list, color='#2171B5', linewidth = 1.5, linestyle = '--')
ax.plot(t_list, r_I_list, color='#CB181D', linewidth = 1.5, linestyle = '--')
ax_2.plot(r_E_list, r_I_list, linewidth = 1.2, color = '#ff7f00')


x_ticks = [0,100, 200]
y_ticks = [0,0.5,1]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('time in ms', fontsize=16)
ax.set_ylabel('firing rate $r$', fontsize=16)
ax.set(
    ylim = [-0.1,1.1])


x_ticks = [0,0.5, 1]
y_ticks = [0,0.25,0.5]
ax_2.set_xticks(x_ticks)
ax_2.set_xticklabels(x_ticks,fontsize=14)
ax_2.set_yticks(y_ticks)
ax_2.set_yticklabels(y_ticks,fontsize=14)
ax_2.set_xlabel('Excitatory rate $r_E$', fontsize=16)
ax_2.set_ylabel('Inhibitory rate $r_I$', fontsize=16)




# +
# to_remove solution
def F_inv(x, a, theta):
  """
  Args:
    x         : the population input
    a         : the gain of the function
    theta     : the threshold of the function

  Returns:
    F_inverse : value of the inverse function
  """

  # Calculate Finverse (ln(x) can be calculated as np.log(x))
  F_inverse = -1/a * np.log((x + (1 + np.exp(a * theta))**-1)**-1 - 1) + theta

  return F_inverse


x = np.linspace(1e-6, 1, 100)
# plot_FI_inverse(x, a=1, theta=3)

# to_remove solution
def get_E_nullcline(rE, a_E, theta_E, wEE, wEI, I_ext_E, **other_pars):
  """
  Solve for rI along the rE from drE/dt = 0.

  Args:
    rE    : response of excitatory population
    a_E, theta_E, wEE, wEI, I_ext_E : Wilson-Cowan excitatory parameters
    Other parameters are ignored

  Returns:
    rI    : values of inhibitory population along the nullcline on the rE
  """
  # calculate rI for E nullclines on rI
  rI = 1 / wEI * (wEE * rE - F_inv(rE, a_E, theta_E) + I_ext_E)

  return rI


def get_I_nullcline(rI, a_I, theta_I, wIE, wII, I_ext_I, **other_pars):
  """
  Solve for E along the rI from dI/dt = 0.

  Args:
    rI    : response of inhibitory population
    a_I, theta_I, wIE, wII, I_ext_I : Wilson-Cowan inhibitory parameters
    Other parameters are ignored

  Returns:
    rE    : values of the excitatory population along the nullcline on the rI
  """
  # calculate rE for I nullclines on rI
  rE = 1 / wIE * (wII * rI + F_inv(rI, a_I, theta_I) - I_ext_I)

  return rE


# Set parameters
Exc_null_rE = np.linspace(-0.01, 0.96, 100)
Inh_null_rI = np.linspace(-.01, 0.8, 100)

# Compute nullclines
Exc_null_rI = get_E_nullcline(Exc_null_rE, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI=w_EI, I_ext_E=I_E)
Inh_null_rE = get_I_nullcline(Inh_null_rI, a_I = alpha_I, theta_I = theta_I, wII = w_II, wIE=w_IE, I_ext_I=I_I)

fig, ax = plt.subplots()

ax.plot(Exc_null_rE,Exc_null_rI, color='#2171B5', linewidth = 1.5, label = 'E-nullcline')
ax.plot(Inh_null_rE,Inh_null_rI, color='#CB181D', linewidth = 1.5, label = 'I-nullcline')
x_ticks = [0,0.5, 1]
y_ticks = [0,0.5,1]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('Excitatory rate $r_E$', fontsize=16)
ax.set_ylabel('Inhibitory rate $r_I$', fontsize=16)
ax.set_xlim([np.min(Inh_null_rE)*1.3,1])


def EIderivs(rE, rI,
             tau_E, a_E, theta_E, wEE, wEI, I_ext_E,
             tau_I, a_I, theta_I, wIE, wII, I_ext_I,
             **other_pars):
  """Time derivatives for E/I variables (dE/dt, dI/dt)."""

  # Compute the derivative of rE
  drEdt = (-rE + nonlinearity_f(wEE * rE - wEI * rI + I_ext_E, a_E, theta_E)) / tau_E

  # Compute the derivative of rI
  drIdt = (-rI + nonlinearity_f(wIE * rE - wII * rI + I_ext_I, a_I, theta_I)) / tau_I

  return drEdt, drIdt


EI_grid = np.linspace(-0.2, 1., 20)
rE, rI = np.meshgrid(EI_grid, EI_grid)
drEdt, drIdt = EIderivs(rE, rI, tau_E = tau_E, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI = w_EI, I_ext_E = I_E,
                                tau_I = tau_I, a_I = alpha_I, theta_I = theta_I, wIE = w_IE, wII = w_II, I_ext_I = I_I)
n_skip = 2
plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
         drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
         angles='xy', scale_units='xy', scale=1, facecolor='k', width = 0.003)


# +

# Set parameters
Exc_null_rE = np.linspace(-0.01, 0.96, 100)
Inh_null_rI = np.linspace(-.01, 0.8, 100)

# Compute nullclines
Exc_null_rI = get_E_nullcline(Exc_null_rE, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI=w_EI, I_ext_E=I_E)
Inh_null_rE = get_I_nullcline(Inh_null_rI, a_I = alpha_I, theta_I = theta_I, wII = w_II, wIE=w_IE, I_ext_I=I_I)

fig, ax = plt.subplots()


x_ticks = [0,0.5, 1]
y_ticks = [0,0.5,1]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('Excitatory rate $r_E$', fontsize=16)
ax.set_ylabel('Inhibitory rate $r_I$', fontsize=16)
ax.set_xlim([np.min(Inh_null_rE)*1.3,1])

EI_grid = np.linspace(-0.2, 1., 20)
rE, rI = np.meshgrid(EI_grid, EI_grid)
drEdt, drIdt = EIderivs(rE, rI, tau_E = tau_E, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI = w_EI, I_ext_E = I_E,
                                tau_I = tau_I, a_I = alpha_I, theta_I = theta_I, wIE = w_IE, wII = w_II, I_ext_I = I_I)
n_skip = 2
plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
         drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
         angles='xy', scale_units='xy', scale=1, facecolor='k', width = 0.003, alpha = 0.3)

ax.plot(Exc_null_rE,Exc_null_rI, color='#2171B5', linewidth = 1.5, label = 'E-nullcline', alpha = 0.3)
ax.plot(Inh_null_rE,Inh_null_rI, color='#CB181D', linewidth = 1.5, label = 'I-nullcline', alpha = 0.3)


t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.32, r_I_0 = 0.3)

ax.plot(r_E_list, r_I_list, linewidth = 2.2, color = '#984ea3')

t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.33, r_I_0 = 0.3)

ax.plot(r_E_list, r_I_list, linewidth = 2.2, color = '#ff7f00')


# +
def make(w_EE = 9, w_EI = 4, w_II = 13, w_IE = 11,tau_E = 20, tau_I = 20,
        alpha_E = 1.2,
    theta_E = 2.8, I_E = 0,
    I_I = 0):

    
    



    

    alpha_I = 1.0
    theta_I = 4.0

    # Set parameters
    Exc_null_rE = np.linspace(-0.01, 0.96, 100)
    Inh_null_rI = np.linspace(-.01, 1, 100)

    # Compute nullclines
    Exc_null_rI = get_E_nullcline(Exc_null_rE, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI=w_EI, I_ext_E=I_E)
    Inh_null_rE = get_I_nullcline(Inh_null_rI, a_I = alpha_I, theta_I = theta_I, wII = w_II, wIE=w_IE, I_ext_I=I_I)

    fig, ax = plt.subplots()

    ax.plot(Exc_null_rE,Exc_null_rI, color='#2171B5', linewidth = 1.5, label = 'E-nullcline')
    ax.plot(Inh_null_rE,Inh_null_rI, color='#CB181D', linewidth = 1.5, label = 'I-nullcline')
    x_ticks = [0,0.5, 1]
    y_ticks = [0,0.5,1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks,fontsize=14)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks,fontsize=14)
    ax.set_xlabel('Excitatory rate $r_E$', fontsize=16)
    ax.set_ylabel('Inhibitory rate $r_I$', fontsize=16)
#     ax.set_xlim([np.min(Inh_null_rE)*1.3,1])
    EI_grid = np.linspace(-0.2, 1., 20)
    rE, rI = np.meshgrid(EI_grid, EI_grid)
    drEdt, drIdt = EIderivs(rE, rI, tau_E = tau_E, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI = w_EI, I_ext_E = I_E,
                                    tau_I = tau_I, a_I = alpha_I, theta_I = theta_I, wIE = w_IE, wII = w_II, I_ext_I = I_I)
    n_skip = 2
    plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
             drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
             angles='xy', scale_units='xy', scale=1, facecolor='k', width = 0.003)

widgets.interactive(make)

# +
w_EE = 9 
w_EI = 6
w_II = 7
w_IE = 14
tau_E = 10 
tau_I = 10
alpha_E = 1.2
theta_E = 2.8
I_E = 1
I_I = 0


# Set parameters
Exc_null_rE = np.linspace(-0.01, 0.96, 100)
Inh_null_rI = np.linspace(-.01, 0.8, 100)

# Compute nullclines
Exc_null_rI = get_E_nullcline(Exc_null_rE, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI=w_EI, I_ext_E=I_E)
Inh_null_rE = get_I_nullcline(Inh_null_rI, a_I = alpha_I, theta_I = theta_I, wII = w_II, wIE=w_IE, I_ext_I=I_I)

fig, ax = plt.subplots()


x_ticks = [0,0.5, 1]
y_ticks = [0,0.5,1]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('Excitatory rate $r_E$', fontsize=16)
ax.set_ylabel('Inhibitory rate $r_I$', fontsize=16)
ax.set_xlim([np.min(Inh_null_rE)*1.3,1])

EI_grid = np.linspace(-0.2, 1., 20)
rE, rI = np.meshgrid(EI_grid, EI_grid)
drEdt, drIdt = EIderivs(rE, rI, tau_E = tau_E, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI = w_EI, I_ext_E = I_E,
                                tau_I = tau_I, a_I = alpha_I, theta_I = theta_I, wIE = w_IE, wII = w_II, I_ext_I = I_I)
n_skip = 2
plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
         drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
         angles='xy', scale_units='xy', scale=1, facecolor='k', width = 0.003, alpha = 0.3)

ax.plot(Exc_null_rE,Exc_null_rI, color='#2171B5', linewidth = 1.5, label = 'E-nullcline', alpha = 0.3)
ax.plot(Inh_null_rE,Inh_null_rI, color='#CB181D', linewidth = 1.5, label = 'I-nullcline', alpha = 0.3)


t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.8, r_I_0 = 0.7)

ax.plot(r_E_list, r_I_list, linewidth = 2.2, color = '#548235')


fig, ax = plt.subplots()

ax.plot(t_list, r_E_list, color='#2171B5', linewidth = 1.5)
ax.plot(t_list, r_I_list, color='#CB181D', linewidth = 1.5)




x_ticks = [0,100, 200]
y_ticks = [0,0.5,1]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('time in ms', fontsize=16)
ax.set_ylabel('firing rate $r$', fontsize=16)
ax.set(
    ylim = [-0.1,1.1])


# +
# 'tau_E': 1.0, 'a_E': 1.2, 'theta_E': 2.8, 
# 'tau_I': 2.0, 'a_I': 1.0, 'theta_I': 4.0, 
# 'wEE': 6.4, 'wEI': 4.8, 'wIE': 6.0, 
# 'wII': 1.2, 'I_ext_E': 0.8, 'I_ext_I': 0.0, 
# 'T': 100.0, 'dt': 0.1, 'rE_init': 0.25, 'rI_init': 0.25
                
w_EE = 6.4 
w_EI = 4.8
w_II = 1.2
w_IE = 6
tau_E = 10 
tau_I = 20
alpha_E = 1.2
theta_E = 2.8
alpha_I = 1
theta_I = 4.0
I_E = 0.8
I_I = 0


# Set parameters
Exc_null_rE = np.linspace(-0.01, 0.96, 100)
Inh_null_rI = np.linspace(-.01, 0.8, 100)

# Compute nullclines
Exc_null_rI = get_E_nullcline(Exc_null_rE, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI=w_EI, I_ext_E=I_E)
Inh_null_rE = get_I_nullcline(Inh_null_rI, a_I = alpha_I, theta_I = theta_I, wII = w_II, wIE=w_IE, I_ext_I=I_I)

fig, ax = plt.subplots()


x_ticks = [0,0.5, 1]
y_ticks = [0,0.5,1]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=14)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=14)
ax.set_xlabel('Excitatory rate $r_E$', fontsize=16)
ax.set_ylabel('Inhibitory rate $r_I$', fontsize=16)
ax.set_xlim([np.min(Inh_null_rE)*1.3,1])

EI_grid = np.linspace(-0.2, 1., 20)
rE, rI = np.meshgrid(EI_grid, EI_grid)
drEdt, drIdt = EIderivs(rE, rI, tau_E = tau_E, a_E = alpha_E, theta_E = theta_E, wEE = w_EE, wEI = w_EI, I_ext_E = I_E,
                                tau_I = tau_I, a_I = alpha_I, theta_I = theta_I, wIE = w_IE, wII = w_II, I_ext_I = I_I)
n_skip = 2
plt.quiver(rE[::n_skip, ::n_skip], rI[::n_skip, ::n_skip],
         drEdt[::n_skip, ::n_skip], drIdt[::n_skip, ::n_skip],
         angles='xy', scale_units='xy', scale=1, facecolor='k', width = 0.003, alpha = 0.3)

ax.plot(Exc_null_rE,Exc_null_rI, color='#2171B5', linewidth = 1.5, label = 'E-nullcline', alpha = 0.3)
ax.plot(Inh_null_rE,Inh_null_rI, color='#CB181D', linewidth = 1.5, label = 'I-nullcline', alpha = 0.3)


t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.8, r_I_0 = 0.75, timesteps = 500)

ax.plot(r_E_list, r_I_list, linewidth = 2.2, color = '#548235')


fig, ax_2 = plt.subplots()

ax_2.plot(t_list, r_E_list, color='#2171B5', linewidth = 1.5)
ax_2.plot(t_list, r_I_list, color='#CB181D', linewidth = 1.5)


t_list, r_E_list, r_I_list = run_E_I_simulation(r_E_0 = 0.5, r_I_0 = 0.25, timesteps = 500)

ax.plot(r_E_list, r_I_list, linewidth = 2.2, color = '#548235', linestyle = '--')



ax_2.plot(t_list, r_E_list, color='#2171B5', linewidth = 1.5, linestyle = '--')
ax_2.plot(t_list, r_I_list, color='#CB181D', linewidth = 1.5, linestyle = '--')




x_ticks = [0,100, 200]
y_ticks = [0,0.5,1]
ax_2.set_xticks(x_ticks)
ax_2.set_xticklabels(x_ticks,fontsize=14)
ax_2.set_yticks(y_ticks)
ax_2.set_yticklabels(y_ticks,fontsize=14)
ax_2.set_xlabel('time in ms', fontsize=16)
ax_2.set_ylabel('firing rate $r$', fontsize=16)
ax_2.set(
    ylim = [-0.1,1.1])
# -


