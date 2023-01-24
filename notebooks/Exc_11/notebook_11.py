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

# +
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import ConnectionPatch
from matplotlib import animation
import ipywidgets as widgets
from scipy import optimize as opt
from urllib.request import urlopen
from PIL import Image
from io import BytesIO
import scipy.stats as stats
from matplotlib.animation import FuncAnimation
from scipy import special
from scipy.optimize import minimize


# Settings for the figures
plt.style.use(plt.style.available[20])
plt.style.use("https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/plots_style.txt")

# +
np.random.seed(121)

# Let's set some parameters
theta = 1.2
n_samples = 30

# Draw x and then calculate y
x = 10 * np.random.rand(n_samples)  # sample from a uniform distribution over [0,10)
noise = np.random.randn(n_samples)  # sample from a standard normal distribution
y = theta * x + noise

# Plot the results
fig, ax = plt.subplots()
ax.scatter(x, y)  # produces a scatter plot
ax.set(xlabel='x', ylabel='y');


# +
def mse(x, y, theta_hat):
    """Compute the mean squared error
    Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding
      measurement values to the inputs.
    theta_hat (float): An estimate of the slope parameter
    Returns:
    float: The mean squared error of the data with the estimated parameter.
    """

    # Compute the estimated y
    y_hat = theta_hat * x

    # Compute mean squared error
    mse = np.mean((y - y_hat)**2)

    return mse


theta_hats = [0.75, 1.0, 1.5]
for theta_hat in theta_hats:
    print(f"theta_hat of {theta_hat} has an MSE of {mse(x, y, theta_hat):.2f}")

# +
fig, axes = plt.subplots(ncols=3, figsize=(18, 4))
for theta_hat, ax in zip(theta_hats, axes):

    # True data
    ax.scatter(x, y, label='Observed')  # our data scatter plot

    # Compute and plot predictions
    y_hat = theta_hat * x
    ax.plot(x, y_hat, color='r', label='Fit')  # our estimated model

    ax.set(
      title= fr'$\hat{{\theta}}$= {theta_hat}, MSE = {np.mean((y - y_hat)**2):.2f}',
      xlabel='x',
      ylabel='y'
    );

axes[0].legend()

# +
# Loop over different thetas, compute MSE for each
theta_hat_grid = np.linspace(-2.0, 4.0,41)
errors = np.zeros(len(theta_hat_grid))
for i, theta_hat in enumerate(theta_hat_grid):
    errors[i] = mse(x, y, theta_hat)

# Find theta that results in lowest error
best_error = np.min(errors)
theta_hat = theta_hat_grid[np.argmin(errors)]

print (theta_hat_grid)

# Plot results
fig, ax = plt.subplots()
ax.plot(theta_hat_grid, errors, '-o', label='MSE', c='C1',linewidth=0, markeredgecolor='k', markeredgewidth=1.2)
ax.axvline(theta, color='g', ls='--', label=r"$\theta_{True}$")
ax.axvline(theta_hat, color='r', ls='-', label=r"$\theta^*$")
ax.set(
  title=fr"Best fit: $\hat{{\theta}}$ = {theta_hat:.2f}, MSE = {best_error:.2f}",
  xlabel=r"$\hat{{\theta}}$",
  ylabel='MSE')
ax.legend(fontsize=18);
x_ticks = [-2,-1,0,1,2,3,4]
y_ticks = [0,100,200,300]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=20)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=20)

ax.set_xlabel(r'$\theta$', fontsize=25, fontweight='bold')
ax.set_ylabel('MSE', fontsize=25, fontweight='bold', labelpad=19)
plt.savefig('theta_agains_MSE.pdf', dpi=300, format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None,
       )

# +
theta_hat_grid = np.linspace(-2.0, 4.0,41)
errors = np.zeros(len(theta_hat_grid))
for i, theta_hat in enumerate(theta_hat_grid):
    errors[i] = mse(x, y, theta_hat)

# Find theta that results in lowest error
best_error = np.min(errors)
theta_hat = theta_hat_grid[np.argmin(errors)]

print (theta_hat_grid)

# Plot results

fig, (ax_0, ax) = plt.subplots(1,2, figsize = (10,5))
ax.plot(theta_hat_grid, errors, '-o', label='MSE', c='C1',linewidth=0, markeredgecolor='#31a354',
            markerfacecolor=(0,0,0,0),markeredgewidth=1.2)

theta_scatter = ax.scatter(theta_hat_grid[0],errors[0],zorder=-10,facecolor='#fec44f',edgecolor='#3182bd',linewidth=1, alpha=0.7, s=100)
ax.axvline(theta, color='k', ls='--', label=r"$\theta_{True}$")
# ax.axvline(theta_hat, color='r', ls='-', label=r"$\theta^*$")
ax.set(
#   title=fr"Best fit: $\hat{{\theta}}$ = {theta_hat:.2f}, MSE = {best_error:.2f}",
  xlabel=r"$\hat{{\theta}}$",
  ylabel='MSE')
ax.legend(fontsize=18);
x_ticks = [-2,-1,0,1,2,3,4]
y_ticks = [0,100,200,300]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=20)
ax.set_yticks(y_ticks)
ax.set_yticklabels(y_ticks,fontsize=20)

ax.set_xlabel(r'$\theta$', fontsize=15, fontweight='bold')
ax.set_ylabel('MSE', fontsize=15, fontweight='bold', labelpad=19)

# True data
ax_0.scatter(x, y, label='Observed', edgecolor='#0570B0', facecolor=(0,0,0,0), linewidth=1.2)  # our data scatter plot

# Compute and plot predictions
X = np.array([0,12])
y_hat = theta_hat * X
line_fit, = ax_0.plot(X, y_hat, color='r', label='Fit')  # our estimated model
x_ticks = [0,5,10]
y_ticks = [0,5,10]
ax_0.set_xticks(x_ticks)
ax_0.set_xticklabels(x_ticks,fontsize=20)
ax_0.set_yticks(y_ticks)
ax_0.set_yticklabels(y_ticks,fontsize=20)

ax_0.set_xlabel('Stimulus intensity in a.u.', fontsize=15, fontweight='bold')
ax_0.set_ylabel('Neural response in Hz', fontsize=15, fontweight='bold', labelpad=19)


plt.tight_layout()

def update(t):

    tt = (1-2*(t//41)) * (t%41) + ((t//41)*40)
     
    theta = theta_hat_grid[tt]
    err = errors[tt]
    theta_scatter.set_offsets(np.array([theta,err]))
    
    X = np.array([0,12])
    y_hat = theta * X
    line_fit.set_ydata(y_hat)
    
    

frames = [t for t in range(82)]


# Create and render the animation
anim = animation.FuncAnimation(fig, func = update,  frames = frames )
# save it to file
anim.save(f"iterate_through_theta.gif",
          fps = 4, writer='imagemagick',dpi=300)

# -

def plot_observed_vs_predicted(x, y, y_hat, theta_hat):
    """ Plot observed vs predicted data

    Args:
    x (ndarray): observed x values
    y (ndarray): observed y values
    y_hat (ndarray): predicted y values
    theta_hat (ndarray):
    """
    fig, ax = plt.subplots()
    ax.scatter(x, y, label='Observed')  # our data scatter plot
    ax.plot(x, y_hat, color='r', label='Fit')  # our estimated model
    # plot residuals
    ymin = np.minimum(y, y_hat)
    ymax = np.maximum(y, y_hat)
#     ax.vlines(x, ymin, ymax, 'g', alpha=0.5, label='Residuals')
    ax.set(
      title=fr"$\hat{{\theta}}$ = {theta_hat:0.2f}, MSE = {np.mean((y - y_hat)**2):.2f}",
      xlabel='x',
      ylabel='y'
    )
    ax.legend()


# +
def solve_normal_eqn(x, y):
    """Solve the normal equations to produce the value of theta_hat that minimizes
    MSE.
    Args:
    x (ndarray): An array of shape (samples,) that contains the input values.
    y (ndarray): An array of shape (samples,) that contains the corresponding
      measurement values to the inputs.
    Returns:
    float: the value for theta_hat arrived from minimizing MSE
    """

    # Compute theta_hat analytically
    theta_hat = (x.T @ y) / (x.T @ x)

    return theta_hat


theta_hat = solve_normal_eqn(x, y)
y_hat = theta_hat * x

plot_observed_vs_predicted(x, y, y_hat, theta_hat)

plt.savefig('ISI_inputs.pdf', dpi=300, format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None,
       )
# -

# # Fit with MSLE 
#
#

np.random.seed(121)
theta = 1.2
n_samples = 30
x = 10 * np.random.rand(n_samples) # sample from a uniform distribution over [0,10)
noise = np.random.randn(n_samples) # sample from a standard normal distribution
y = theta * x + noise


# +
def likelihood(theta_hat, x, y):
    """The likelihood function for a linear model with noise sampled from a
        Gaussian distribution with zero mean and unit variance.
        Args:
        theta_hat (float): An estimate of the slope parameter.
        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.
        Returns:
        float: the likelihood value for the theta_hat estimate
    """
    sigma = 1

    # Compute Gaussian likelihood
    pdf = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(y - theta_hat * x)**2 / (2 * sigma**2))

    return pdf


print(likelihood(1.0, x[1], y[1]))


# -

def plot_density_image(x, y, theta, sigma=1, ax=None):
    """ Plots probability distribution of y given x, theta, and sigma

        Args:

        x (ndarray): An array of shape (samples,) that contains the input values.
        y (ndarray): An array of shape (samples,) that contains the corresponding
          measurement values to the inputs.
        theta (float): Slope parameter
        sigma (float): standard deviation of Gaussian noise

    """

  # plot the probability density of p(y|x,theta)
    if ax is None:
        fig, ax = plt.subplots()

    xmin, xmax = np.floor(np.min(x)), np.ceil(np.max(x))
    ymin, ymax = np.floor(np.min(y)), np.ceil(np.max(y))
    xx = np.linspace(xmin, xmax, 200)
    yy = np.linspace(ymin, ymax, 200)

    surface = np.zeros((len(yy), len(xx)))
    for i, x_i in enumerate(xx):
        surface[:, i] = stats.norm(theta * x_i, sigma).pdf(yy)

    ax.set(xlabel='x', ylabel='y')

    return ax.imshow(surface, origin='lower', aspect='auto', 
            cmap=plt.get_cmap('RdPu'),
            extent=[xmin, xmax, ymin, ymax],vmin=0,vmax=np.max(surface)*1.2)


# +
fig, ax1 = plt.subplots()

# Invokes helper function to generate density image plots from data and parameters
im = plot_density_image(x, y, 1.3, ax=ax1)
cbar = plt.colorbar(im, ax=ax1)
cbar.ax.set_ylabel(r'p(y|x=, $\theta$)')
# ax1.axvline(8, color='k')
ax1.scatter(x, y, label='Observed', edgecolor='#0570B0', facecolor='w', linewidth=1.2)
x_ticks = [0,5,10]
y_ticks = [0,5,10]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks,fontsize=20)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_ticks,fontsize=20)
ax1.set_xlabel('Stimulus intensity in a.u.', fontsize=15, fontweight='bold')
ax1.set_ylabel('Neural response in Hz', fontsize=15, fontweight='bold', labelpad=19)



# +
fig, ax1 = plt.subplots()
X = np.linspace(0.0000000000001,1,1000)
Y = np.log(X)
ax1.plot(X,Y)
x_ticks = [0,0.5,1]
y_ticks = [0,-15,-30]
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(x_ticks,fontsize=20)
ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_ticks,fontsize=20)
ax1.set_xlabel('x', fontsize=15, fontweight='bold')
ax1.set_ylabel('log(x)', fontsize=15, fontweight='bold', labelpad=19)

print (np.log(1E-15))
# -

theta_hats = [0.5, 1.0, 2.2]
fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
for theta_hat, ax1 in zip(theta_hats, axes):
    im = plot_density_image(x, y, theta_hat, ax=ax1)
    cbar = plt.colorbar(im, ax=ax1)
    cbar.ax.set_ylabel(r'p(y|x=, $\theta$)')
    # ax1.axvline(8, color='k')
    ax1.scatter(x, y, label='Observed', edgecolor='#0570B0', facecolor='w', linewidth=1.2)
    x_ticks = [0,5,10]
    y_ticks = [0,5,10]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_ticks,fontsize=20)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_ticks,fontsize=20)
    ax1.set_xlabel('Stimulus intensity in a.u.', fontsize=15, fontweight='bold')
    ax1.set_ylabel('Neural response in Hz', fontsize=15, fontweight='bold', labelpad=19)
    ll = np.sum(np.log(likelihood(theta_hat, x, y)))
    ax1.set(title=fr'$\hat{{\theta}}$ = {theta_hat}, log likelihood: {ll:.2f}')
    
    plt.tight_layout()

# +
theta_hats = [0.5, 1.0, 2.2]
fig, axes = plt.subplots(ncols=3, figsize=(16, 4))
for theta_hat, ax in zip(theta_hats, axes):
    ll = np.sum(np.log(likelihood(theta_hat, x, y)))  # log likelihood
    im = plot_density_image(x, y, theta_hat, ax=ax)
    ax.scatter(x, y)
    ax.set(title=fr'$\hat{{\theta}}$ = {theta_hat}, log likelihood: {ll:.2f}')
    plt.colorbar(im, ax=ax,);




# -








def p(x,sigma, mean):
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(1/(2*sigma**2))*(x-mean)**2)


fig, ax = plt.subplots()
X = np.linspace(-5,5,501)
sigma = 1.5
mean = 0
ax.plot(X,p(X,sigma, mean))
plt.savefig('gaussian.pdf', dpi=300, format=None, metadata=None,
        bbox_inches=None, pad_inches=0.1,
        facecolor='auto', edgecolor='auto',
        backend=None,
       )

# +
fig, (ax_0, ax) = plt.subplots(1,2, figsize=(12,6),sharey=True, gridspec_kw={'width_ratios': [3, 1]})
np.random.seed(12)
X = np.sort(np.random.rand(80)*10)
theta = 1.3
Y = theta * X

line_fit, = ax_0.plot(X, Y, color='r', label='Model')

scatter_data = ax_0.scatter(X, Y, label='Observed', edgecolor='#0570B0', facecolor=(0,0,0,0), linewidth=1.2)  # our data scatter plot

# Compute and plot predictions


  # our estimated model
x_ticks = [0,5,10]
y_ticks = [-5,0,5,10,15]
ax_0.set_xticks(x_ticks)
ax_0.set_xticklabels(x_ticks,fontsize=20)
ax_0.set_yticks(y_ticks)
ax_0.set_yticklabels(y_ticks,fontsize=20)

ax_0.set_xlabel('Stimulus intensity in a.u.', fontsize=15, fontweight='bold')
ax_0.set_ylabel('Neural response in Hz', fontsize=15, fontweight='bold', labelpad=19)
ax_0.set_ylim([-6,18])





XX = np.linspace(-5+mean,5+mean,501)
sigma = 1.5
mean = 0
gaussian_plot, = ax.plot(p(XX,sigma, mean),XX)

noise_scatter = ax.scatter([],[],color='grey',s=60,alpha = 0.7)
noise_scatter_highlight = ax.scatter([],[],color='#fe9929',s=80,alpha=0.8)
x_ticks = [0,0.1,0.2]
ax.set_xticks(x_ticks)
ax.set_xticklabels(x_ticks,fontsize=20)
ax.set_xlabel(r'$p(y|\sigma,\mu)$', fontsize=15)
plt.tight_layout()


def update(t):

    if t >= len(X):
        return 
    
    x = X[t]
    y = np.random.normal(loc=x*theta, scale=sigma)
    
    
    
    offsets = scatter_data.get_offsets()
    offsets[t,:] = np.array([x,y])
    scatter_data.set_offsets(offsets)

    
    
    mean = x*theta
    XX = np.linspace(-5+mean,5+mean,501)
    gaussian_plot.set_xdata(p(XX,sigma, mean))
    gaussian_plot.set_ydata(XX)
    
    random_x = np.random.random()*p(y,sigma,mean)
    
    
    if t > 0:
        offsets = noise_scatter.get_offsets()
        offsets[:,1] += (X[t] - X[t-1])*theta # shift the offsets to fit the gaussian
        offsets = np.vstack([offsets,[random_x,y]])

    else:
        offsets = [random_x,y]
        
    noise_scatter.set_offsets(offsets)
    
    noise_scatter_highlight.set_offsets([random_x,y])
    
#     p(theta*x,sigma=sigma, mean)
     
#     theta = theta_hat_grid[tt]
#     err = errors[tt]
#     theta_scatter.set_offsets(np.array([theta,err]))
    
#     X = np.array([0,12])
#     y_hat = theta * X
#     line_fit.set_ydata(y_hat)
    
    

frames = [t for t in range(len(X) + 40)]


# Create and render the animation
anim = animation.FuncAnimation(fig, func = update,  frames = frames )
# save it to file
anim.save(f"generate_noisy_data.gif",
          fps = 4, writer='imagemagick',dpi=300)

# -
# ### Linear Regression - Ordinary Least Squares (OLS) solution 
#
# Following the lecture, we start with a linear regression on 1D data. First we generate the data with a true theta. 
#
# (as in the lecture, we do not have an offset ($\theta_0=0$))

# +
np.random.seed(121)

# Let's set some parameters
theta = 1.2
n_samples = 30

# Draw x and then calculate y
x = 10 * np.random.rand(n_samples)[:,None]  # sample from a uniform distribution over [0,10)
noise = np.random.randn(n_samples)[:,None]  # sample from a standard normal distribution
y = theta * x + noise

# Plot the results
fig, ax = plt.subplots()
ax.scatter(x, y)  # produces a scatter plot
ax.set(xlabel='x', ylabel='y', title = 'Data');


# -

# The solution for this problem is the same, no matter if we follow the idea of the mean square error (MSE) or the maximum likelihod estimate (MLE):
#
# \begin{aligned}
# \theta^*=\left(\mathbf{X}^{\top} \mathbf{X}\right)^{-1} \mathbf{X}^{\top} \boldsymbol{y}
# \end{aligned}

def solve_normal_eqn(X, y):
    # Compute theta_hat using OLS
    theta_star = np.linalg.inv(X.T @ X) @ X.T @ y

    return theta_star[:,0]


# +
def solve_normal_equ_and_plot(X,y):

    theta_star = solve_normal_eqn(X, y)
    print (theta_star.shape)
    
    x_plot = X
    if x_plot.shape[1] > 1:
        x_plot = X[:,1]
        
    y_star = np.sum(X*theta_star,axis=1) # we do the sum here so that we can use the same function later on 
    
    
    fig, ax = plt.subplots()
    ax.scatter(x_plot, y, label='Observed')  # our data scatter plot
    ax.plot(x_plot, y_star, color='r', label='Fit')  # our estimated model
    ax.set(
      title=fr"$\theta^*$ ={np.round(theta_star,2)}, MSE = {np.round(np.mean((y - y_hat)**2),2)}",
      xlabel='x',
      ylabel='y'
    )
    ax.legend()
    

# -

solve_normal_equ_and_plot(x,y)

# great! that woorks

# ## Multiple linear regression - model evaluation
#
#
# As a first intermediate step towards multiple linear regression, we now introduce an offset and see how the previous model will fail to capture the data and how we can fix this by adjusting the design matrix
#
#
# Let's generate new data

# +
np.random.seed(121)

# Let's set some parameters
theta = 1.2
n_samples = 30

theta_0 = 4

# Draw x and then calculate y
x_offset = 10 * np.random.rand(n_samples)[:,None]  # sample from a uniform distribution over [0,10)
noise = np.random.randn(n_samples)[:,None]  # sample from a standard normal distribution
y_offset = theta * x_offset + noise + theta_0

# Plot the results
fig, ax = plt.subplots()
ax.scatter(x_offset, y_offset)  # produces a scatter plot
ax.set(xlabel='x', ylabel='y', title = 'Data');
# -

solve_normal_equ_and_plot(x_offset,y_offset)

X_design_offset = np.hstack([np.ones_like(x_offset),x_offset])
solve_normal_equ_and_plot(X_design_offset,y_offset)

# Now moving towards polynomal regression, we can add another dimension:
#
#

# +
# Set parameters
theta = np.array([4, -1, 5])
n_samples = 40

x0 = np.ones((n_samples, 1))
x1 = np.random.uniform(-2, 2, (n_samples, 1))
x2 = np.random.uniform(-2, 2, (n_samples, 1))
X_3d = np.hstack([x0, x1, x2])
noise = np.random.randn(n_samples)
y_3d = X_3d @ theta + noise


ax = plt.subplot(projection='3d')
ax.plot(X_3d[:,1], X_3d[:,2], y_3d, '.')

ax.set(
    xlabel='$\mathbf{x_1}$: Orientation',
    ylabel='$\mathbf{x_2}$: Contrast',
    zlabel='y: Neural Response'
)
plt.tight_layout()

# +
theta_star = solve_normal_eqn(X_3d, y_3d[:,None])
xx, yy = np.mgrid[-2:2:50j, -2:2:50j]
y_star_grid = np.array([xx.flatten(), yy.flatten()]).T @ theta_star[1:] + theta_star[0]
y_star_grid = y_star_grid.reshape((50, 50))

ax = plt.subplot(projection='3d')
ax.plot(X_3d[:, 1], X_3d[:, 2], y_3d, '.')
ax.plot_surface(xx, yy, y_star_grid, linewidth=0, alpha=0.5, color='C1',
                cmap=plt.get_cmap('coolwarm'))
ax.set(
    xlabel='$\mathbf{x_1}$: Orientation',
    ylabel='$\mathbf{x_2}$: Contrast',
    zlabel='y: Neural Response',
    title=fr"$\theta^*$ ={np.round(theta_star,2)}",
);
# -

# ## Polynomal Regression
#
# We now look at 2D data again but with polynomal regression. We also use this to investigate the model evaluation
#
# First we generate data and in order to evaluate whether our model captures the data well, we will randomly hold back some data as our test-set. The data we use to fit our model will be our train-set.

# +
np.random.seed(240)
n_samples = 70
x = np.random.uniform(-2, 2.5, n_samples)  
y = 0.1*x**3+x**2 - x + 3   # computing the outputs

output_noise = np.random.randn(n_samples) * 0.1
y += output_noise  # adding some output noise

input_noise = np.random.randn(n_samples) * 0.3
x += input_noise  # adding some input noise


n_test = 25
all_ids = np.linspace(0,n_samples-1,n_samples).astype(int)
test_set_ids = np.random.choice(all_ids, size=n_test, replace=False)

x_test = x[test_set_ids] 
y_test = y[test_set_ids]

train_set_ids = [ii for ii in all_ids if ii not in test_set_ids]

x_train = x[train_set_ids]
y_train = y[train_set_ids]

fig, ax = plt.subplots()
ax.scatter(x_train, y_train,label='train set')
ax.scatter(x_test, y_test, label='test set',marker='x')
ax.set(
    xlabel='x', 
    ylabel='y');
ax.legend();


# -

# Now we have the basic idea of polynomial regression and some noisy data, let's begin! The key difference between fitting a linear regression model and a polynomial regression model lies in how we structure the input variables.  
#
# Let's go back to one feature for each data point. For linear regression, we used $\mathbf{X} = \mathbf{x}$ as the input data, where $\mathbf{x}$ is a vector where each element is the input for a single data point. To add a constant bias (a y-intercept in a 2-D plot), we use $\mathbf{X} = \big[ \boldsymbol 1, \mathbf{x} \big]$, where $\boldsymbol 1$ is a column of ones.  When fitting, we learn a weight for each column of this matrix. So we learn a weight that multiples with column 1 - in this case that column is all ones so we gain the bias parameter ($+ \theta_0$). 
#
# This matrix $\mathbf{X}$ that we use for our inputs is known as a **design matrix**. We want to create our design matrix so we learn weights for $\mathbf{x}^2, \mathbf{x}^3,$ etc. Thus, we want to build our design matrix $X$ for polynomial regression of order $k$ as:
#
# \begin{equation}
# \mathbf{X} = \big[ \boldsymbol 1 , \mathbf{x}^1, \mathbf{x}^2 , \ldots , \mathbf{x}^k \big],
# \end{equation}
#
# where $\boldsymbol{1}$ is the vector the same length as $\mathbf{x}$ consisting of of all ones, and $\mathbf{x}^p$ is the vector $\mathbf{x}$ with all elements raised to the power $p$. Note that $\boldsymbol{1} = \mathbf{x}^0$ and $\mathbf{x}^1 = \mathbf{x}$.  
#
# If we have inputs with more than one feature, we can use a similar design matrix but include all features raised to each power. Imagine that we have two features per data point: $\mathbf{x}_m$ is a vector of one feature per data point and  $\mathbf{x}_n$ is another.  Our design matrix for a polynomial regression would be:
#
# \begin{equation}
# \mathbf{X} = \big[ \boldsymbol 1 , \mathbf{x}_m^1, \mathbf{x}_n^1, \mathbf{x}_m^2 , \mathbf{x}_n^2\ldots , \mathbf{x}_m^k , \mathbf{x}_n^k \big],
# \end{equation}


def make_design_matrix(x, order):
    """Create the design matrix of inputs for use in polynomial regression
    Args:
    x (ndarray): input vector of shape (samples,)
    order (scalar): polynomial regression order
    Returns:
    ndarray: design matrix for polynomial regression of shape (samples, order+1)
    """

    # Broadcast to shape (n x 1) so dimensions work
    if x.ndim == 1:
        x = x[:, None]

    #if x has more than one feature, we don't want multiple columns of ones so we assign
    # x^0 here
    design_matrix = np.ones((x.shape[0], 1))

    # Loop through rest of degrees and stack columns (hint: np.hstack)
    for degree in range(1, order + 1):
        design_matrix = np.hstack((design_matrix, x**degree))

    return design_matrix


# Now that we have the inputs structured correctly in our design matrix, fitting a polynomial regression is the same as fitting a linear regression model! All of the polynomial structure we need to learn is contained in how the inputs are structured in the design matrix. We can use the same least squares solution we computed in previous exercises. 

# Here, we will fit polynomial regression models to find the regression coefficients ($\theta_0, \theta_1, \theta_2,$ ...) by solving the least squares problem. Create a function `solve_poly_reg` that loops over different order polynomials (up to `max_order`), fits that model, and saves out the weights for each. You may invoke the `ordinary_least_squares` function. 
#
# We will then qualitatively inspect the quality of our fits for each order by plotting the fitted polynomials on top of the data. In order to see smooth curves, we evaluate the fitted polynomials on a grid of $x$ values (ranging between the largest and smallest of the inputs present in the dataset).

# +
def solve_poly_reg(x, y, max_order):
    """Fit a polynomial regression model for each order 0 through max_order.
    Args:
    x (ndarray): input vector of shape (n_samples)
    y (ndarray): vector of measurements of shape (n_samples)
    max_order (scalar): max order for polynomial fits
    Returns:
    dict: fitted weights for each polynomial model (dict key is order)
    """

    # Create a dictionary with polynomial order as keys,
    # and np array of theta_hat (weights) as the values
    theta_stars = {}

    # Loop over polynomial orders from 0 through max_order
    for order in range(max_order + 1):

        # Create design matrix
        X_design = make_design_matrix(x, order)

        # Fit polynomial model
        this_theta = solve_normal_eqn(X_design, y[:,None])

        theta_stars[order] = this_theta

    return theta_stars


list_of_orders = [0,1,2,3,4,5,15]
theta_stars = solve_poly_reg(x_train, y_train, np.max(list_of_orders))


x_grid = np.linspace(x.min() - .5, x.max() + .5)
fig, ax = plt.subplots()

for order in list_of_orders:
    X_design = make_design_matrix(x_grid, order)
    ax.plot(x_grid, X_design @ theta_stars[order], label=f'order: {order}');

ax.set(
    xlabel ='x',
    ylabel = 'y',
    title = 'polynomial fits',
    ylim = [0,10]
)
ax.scatter(x_train, y_train,label='train set')

ax.legend()

# -

# As with linear regression, we can compute mean squared error (MSE) to get a sense of how well the model fits the data. 
#
# We compute MSE as:
#
# \begin{equation}
# \mathrm{MSE} = \frac 1 N ||\mathbf{y} - \mathbf{y}^*||^2 = \frac 1 N \sum_{i=1}^N (y_i -  y_i^*)^2 
# \end{equation}
#
# where the predicted values for each model are given by $\mathbf{y}^* = \mathbf{X}\boldsymbol{\theta}^*$.
#
# Now we can do this for both, the _train_set_ that we used to train the model, and with the _test_set_ that we hold back and that was not part of the training procedure 
#
# *Which model (i.e. which polynomial order) do you think will have the best MSE for the test set, which one will have the best for the train set?*

# +


mse_list_test = []
mse_list_train = []


for order in list_of_orders:

    X_design_train = make_design_matrix(x_train, order)
    X_design_test = make_design_matrix(x_test, order)

    # Get prediction for the polynomial regression model of this order
    y_star_train = X_design_train @ theta_stars[order]
    y_star_test = X_design_test @ theta_stars[order]
    
    # Compute the residuals
    residuals_train = y_train - y_star_train
    
    
    residuals_test = y_test - y_star_test

    # Compute the MSE
    mse = np.mean(residuals_train ** 2)
    mse_list_train.append(mse)
    
    mse = np.mean(residuals_test ** 2)
    mse_list_test.append(mse)



fig, ax = plt.subplots()
width = .35

ax.bar(np.array(list_of_orders) - width / 2, mse_list_train, width, label="train set")
ax.bar(np.array(list_of_orders) + width / 2, mse_list_test , width, label="test set")
ax.legend()
ax.set(
    xlabel = 'Order of polynomal fit',
    ylabel = 'MSE',)
# -

# We see that the MSE on the train set gets smaller, the more paramter we use. 
# However the MSE on the test sets first gets smaller but then increases again. 
#
# This is an example of the bias-variance trade-off.
#
# A model with few parameters has a high bias: In our example, we put in a very strong assumption into the model with only one parameter, namely that y is independent of x. This is not the case, therefore the training error is high. 
#
# The more paramter we include, the better the model can be fit to the data, or in other words, the better it can capture the variance of the data. 
# However, this can lead to over-fitting as we can see for the models with many parameters. Capturing to much of the variance of the data therefore prevents the model from making good predictions about future data. 
#



# ### Generate a model
#
# 2D makes spikes
#

def show_visual_space(pattern, ax = None, style='filter',alpha=1):
    
    if ax == None:
        fig, ax = plt.subplots()
    
    if style == 'filter':
        max_ = np.max(np.abs(pattern))
        print (max_)
        ax.imshow(pattern,cmap='bwr', alpha=alpha, vmin = -max_, vmax=max_)
    else:
        ax.imshow(pattern,cmap='binary', alpha=alpha)
    ax.grid(False)
    
    ax.spines[['right', 'top']].set_visible(True)

    ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
    
    ax.tick_params(
    axis='y',          # changes apply to the y-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the left edge are off
    right=False,         # ticks along the righ edge are off
    labelleft=False) # labels along the left edge are off


def gaussian_2d(size, mu_x, mu_y, sig_x,sig_y):
    """ 
    Make an eliptic gaussian kernel in a square.

    size is the length of a side of the square
    mu_x and mu_y are the mean values of x and y
    sig_x and sig_y define the spread of the guassian
    """

    x = np.arange(0, size, 1, float)      
    y = x[:,np.newaxis]

    return np.exp(-4*np.log(2) * (((x-mu_x)/sig_x)**2 + ((y-mu_y)/ sig_y)**2))



# +
def generator_model(input_pattern, size = 20, ax=None):
    
    
    
    def poisson(y,lam):
        return (lam)**y/special.factorial(y) * np.exp(-lam)
    
    
    
    pos = gaussian_2d(size=size, mu_x=10, mu_y=8, sig_x = 6,sig_y = 8)
    neg = gaussian_2d(size=size, mu_x=8, mu_y=13, sig_x = 4,sig_y = 7)
    filter_matrix = pos-neg

    if ax != None:
        ax.imshow(filter_matrix, cmap='bwr') 

        
    dim_red = np.sum(input_pattern * filter_matrix)


    y = np.linspace(0,80,81).astype(int)
    prob = np.cumsum(poisson(lam=np.exp(dim_red),y = y))
    threshold = np.random.rand()
    n_spikes = np.argmax(prob>threshold)
    
    return n_spikes
        
    


# -
# ### Generate sample images
#
# We use the function _create_random_image_ to gerate the sample data
#
# ...

# +
def create_random_image(size=20):
    
    random_dots = np.random.rand(size,size)*1
    random_dots[np.random.rand(size,size)<0.92] = 0
    
    return random_dots

def get_random_gaussian_stim(size=20):
    
    mu_x = np.random.rand()*size
    mu_y = np.random.rand()*size
    
    sig_x = (0.5+np.random.rand())*6
    sig_y = (0.5+np.random.rand())*6
    
    pattern = gaussian_2d(size=size, mu_x=mu_x, mu_y=mu_y, sig_x = sig_x,sig_y = sig_y)
    scaled_pattern = pattern * 0.2
    return scaled_pattern


# +
def get_input_image(size=20):
    
    return create_random_image(size=size)
#     return get_random_gaussian_stim(size=20)


size=20

np.random.seed(14)
fig, axes = plt.subplots(4,5) 
axes_flat = axes.flatten()
for ii in range(20):
    
    image = get_input_image(size=size)
    spikes = generator_model(image, size=size)
    
    show_visual_space(image, style='random',ax=axes_flat[ii])
    axes_flat[ii].set_title(f'{spikes} spikes')

fig.suptitle('Show example input images')
fig.tight_layout()

    

# -




# +
def neg_log_lik_lnp(theta, X, y):
    """Return -loglike for the Poisson GLM model.
    Args:
        theta (1D array): Parameter vector.
        X (2D array): Full design matrix.
        y (1D array): Data values.
    Returns:
        number: Negative log likelihood.
    """
    # Compute the Poisson log likelihood
    rate = np.exp(X @ theta)
    log_lik = y.T @ np.log(rate) - rate.sum()
    return -log_lik


def fit_lnp(stim, spikes):
    """Obtain MLE parameters for the Poisson GLM.
    Args:
        stim (1D array): Stimulus values at each timepoint
        spikes (1D array): Spike counts measured at each timepoint
        d (number): Number of time lags to use.
    Returns:
        1D array: MLE parameters
    """

    # Build the design matrix
    y = spikes
    constant = np.ones_like(y)
    
    print (constant.shape, stim.shape)
    X = np.hstack([constant, stim])

    # Use a random vector of weights to start (mean 0, sd .2)
    x0 = np.random.normal(0, .2, X.shape[1])

    # Find parameters that minmize the negative log likelihood function
    res = minimize(neg_log_lik_lnp, x0, args=(X, y))

    return res["x"]


# Fit LNP model

np.random.seed(100)


for ii in range(1000):
    
    image = get_input_image()
    spikes = generator_model(image)
    
    if ii == 0:
        stim = np.array(image.flatten())
        all_spikes = np.array(spikes)
    
    
    
    stim = np.vstack([stim,image.flatten()])
    all_spikes = np.vstack([all_spikes,spikes])




theta_lnp = fit_lnp(stim, all_spikes)
show_visual_space(theta_lnp[0]+theta_lnp[1:].reshape((20,20)), style='filter')



# -



# +
more on GLMs

https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Tutorial1.html

https://compneuro.neuromatch.io/tutorials/W1D3_GeneralizedLinearModels/student/W1D3_Tutorial2.html
    
    
    
# -






