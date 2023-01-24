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


list_of_orders = [0,1,2,3,4,5]
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
# This is summarized in the following figure (taken from [here](http://scott.fortmann-roe.com/docs/BiasVariance.html))
# <div>
# <img src="https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/notebooks/Exc_11/static/biasvariance.png" width="550"/>
# </div>
#

# # GLMs 
#
# We now look at the 

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


for ii in range(2000):
    
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






