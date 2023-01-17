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

# # Dimensionality Reduction

# +
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import ipywidgets as widgets
import scipy
from scipy import special
import time
import gif
from IPython.display import HTML
from sklearn.datasets import fetch_openml
import matplotlib.patches as mpatches


# Settings for the figures
plt.style.use(plt.style.available[20])
plt.style.use("https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/plots_style.txt")
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["animation.embed_limit"] = 400
# -

# ## The notebook
#
# In this notebook we want to look at the most abundant dimensionality reduction technique: Principle component analysis. Even much more sophisticated methods as a pre-processing step use PCA to get a first reduction of dimensionality. 
# Therefore it is beneficial to get a firm understanding of the approach of PCA. We go through the same steps that have been introduced in the lecture in order to perform PCA on the MNIST dataset.

# ## PCA Step 0 - The data
#
# We look at the MNIST dataset, which is a very classical dataset for machine learning where each sample is a 784 dimensional vector that represents handwritten digits from 0 to 9. It is rather outdated for modern approaches since the task of classifying the 10 digits is rather simple for the current technology but it will do perfectly well for our purposes.
#
# First, we need to load the dataset and store the data in an array

mnist = fetch_openml(name='mnist_784', as_frame = False)

X = mnist.data

# First, we look at the data:

print (np.max(X), np.min(X))

# The maximum number is 255, the minumum number is 0. Each feature reflects a pixel and each pixel can take any integer value between these two values.

print (X.shape)

# We see the 784 feautures and another dimension that gives us the number of samples: 70.000
# Now we want to look at example samples.
#
# To do so, we nee to reshape the 784 dimensions so that we get a 2D image again:
#

# +
example_number = X[0,:]

example_number_reshaped = example_number.reshape((28,28))

# now we can visualize the number

fig, ax = plt.subplots()
ax.imshow(example_number_reshaped, cmap='binary')


# -

# ### Task 1 - look at some examples
#
# Change the selected sample above to look at some of the samples. 

# We now create a function that allows us to look at any sample

# +
def visualize_number(number, ax=None, title='', style='regular'):
    if ax==None:
        fig, ax = plt.subplots()
    
    number_reshaped = number.reshape((28,28))
    
    if style == 'regular':
        cmap = 'binary'
        vmin = 0
        vmax = 255
    if style == 'mean centered':
        cmap = 'bwr'
        vmin = -255
        vmax = 255
        
    if style == 'eigenvectors':
        cmap = 'PRGn'
        max_val = np.max([np.abs(np.max(number)),np.abs(np.min(number))])
        vmin = - max_val
        vmax = max_val
        
    
    ax.imshow(number_reshaped, cmap=cmap, vmin=vmin, vmax=vmax)
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
    
    ax.set(
    xlim = [-0.5,27.5],
    ylim = [27.5,-0.5],
    title=title)
    
    ax.grid(False)
    
visualize_number(example_number, ax=None);
# -

# We can now look at many examples of the dataset at the same time

# +
I, J = 4, 8
fig, ax = plt.subplots(I,J)

for ii in range(I):
    for jj in range(J):
        random_integer = int(np.random.rand()*70_000)
        random_sample = X[random_integer]
        visualize_number(random_sample, ax=ax[ii,jj], title=f'{random_integer}')
fig.tight_layout()

# -

# We now want to perform PCA on the dataset by following the steps as discussed in the lecture

# ## PCA Step 1 - Mean Center
#
# Mean-centering is a crucial preprocessing step for PCA. However, first we need to make sure our data is organized in the way we want it to be.

# ### Task 2 - Create the feature-sample matrix
#
# Create the feature - sample matrix as discussed in the lecture, please assing it to the variable 
# ```python
# design_matrix
# ```
# Then, mean-center this matrix as discussed in the lecture. You can use
# ```python
# np.mean( A, axis = n)
# ```
# where A is the matrix you are investigating and n gives the dimension along which you want to take the mean.
# Please call the new matrix
# ```python
# design_matrix_centered
# ```

X_mean = np.mean(X, axis=0)
design_matrix = X
design_matrix_centered = design_matrix - X_mean
visualize_number(X_mean, title='Mean')

# ### Solution 2
#
# The matrix X we have for our data is already in the correct shape (samples, features)
# therefore we can calculate the mean and substract it while using the original matrix X.
#
# We need to take the mean along the first axis (axis=0) because we need the mean for every pixel across all samples.
#
# ```python
# design_matrix = X
# design_matrix_mean = np.mean(design_matrix, axis=0)
# design_matrix_centered = design_matrix - design_matrix_mean
# visualize_number(design_matrix_mean, title='Mean')
# ```
#
#
#

# We can now look at mean-centered samples

# +
I, J = 4, 8
fig, ax = plt.subplots(I,J)

for ii in range(I):
    for jj in range(J):
        random_integer = int(np.random.rand()*70_000)
        random_sample = design_matrix_centered[random_integer]
        visualize_number(random_sample, ax=ax[ii,jj], title=f'{random_integer}', style='mean centered')
fig.tight_layout()


# -

# ## PCA Step 2 - Calculate the Covariance Matrix
#
#
# ### Task 3
#
# Now you should calculate the covariance matrix for the design matrix. Follow the lecture slides to do so.
#
# save your result in 
# ```python
# cov_matrix
# ```
#
# you can use 
# ```python
# example_matrix.T
# ```
# to create the transpose of a matrix and 
# ```python
# np.matmul(matrix_1, matrix_2)
# ```
# to multiply two matrices
#

cov_matrix = 1 / (design_matrix_centered.shape[0]) * np.matmul(design_matrix_centered.T, design_matrix_centered)


# ### Solution 3
#
# it is important to take the mean centered matrix. 
# However, it is fine to either use N or N-1
# ```python
# cov_matrix = 1 / (design_matrix_centered.shape[0]-1) * np.matmul(design_matrix_centered.T, design_matrix_centered)
# ``` 
#

# We can look how two pixels of the image visually are correlated across the dataset. You can change the two selected pixels down below

# +
def select_pixel(ii_1=10, jj_1=10, ii_2=12,jj_2=12):
    
    
    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(8,4))
    visualize_number(design_matrix_mean, title='Mean',ax=ax1)


    
    rect= mpatches.Rectangle((ii_1-0.5,jj_1-0.5),1,1, 
                            fill=False,
                            color='#fd8d3c',
                           linewidth=1)
                           #facecolor="red")
    ax1.add_patch(rect)
    
    rect= mpatches.Rectangle((ii_2-0.5,jj_2-0.5),1,1, 
                            fill=False,
                            color='#e7298a',
                           linewidth=1)
                           #facecolor="red")
    ax1.add_patch(rect)
    
    x1 = 28 * jj_1 + ii_1
    x2 = 28 * jj_2 + ii_2
    
    # we only plot a subset (2000) of all samples to make it faster
    ax2.scatter(design_matrix_centered[:2000,x1],design_matrix_centered[:2000,x2])
    ylim = ax2.get_ylim()
    xlim = ax2.get_xlim()
    ax2.plot([xlim[0],xlim[0]],[ylim[0],ylim[1]], color='#e7298a', linewidth=5)
    ax2.plot([xlim[0],xlim[1]],[ylim[0],ylim[0]], color='#fd8d3c', linewidth=5)
    ax2.set(
        xlim = xlim,
        ylim = ylim,
        xlabel = f'Pixel {x1}',
        ylabel = f'Pixel {x2}',
        title = f'covariance = {cov_matrix[x1,x2]}')
    fig.tight_layout()

    
widgets.interactive(select_pixel, ii_1=(0,27,1),jj_1=(0,27,1),ii_2=(0,27,1),jj_2=(0,27,1))

# -

# ## PCA Step 3 - Find Eigenvectors and -Values
#
# To find the eigenvectors and eigenvalues we use a scipy library.
# We then have to sort these eigenvectors

# +
evals, evectors = np.linalg.eigh(cov_matrix)
index = np.argsort(np.abs(evals))[::-1]
evals = evals[index]
evectors = evectors[:, index]


# -

#  We can now visually look at these eigenvectors
#  
#  Be aware how the eigenvectors are selected. As an index we take the column value (look at the lecture how the eigenvector matrix is organized)
#  
#  ### Task 4 
#  
#  Scroll through the eigenvectors and see how the structure becomes more and more noise - and especially irrelevant noise that is outside of the main part of the image - when going to smaller eigenvalues

def show_eigenvector(nn):
    visualize_number(evectors[:,nn], title=f'Eigenvector {nn} - Eigenvalue = {np.round(evals[nn])}', style='eigenvectors')
widgets.interactive(show_eigenvector, nn=(0,783,1))

# ### Task 5
#
# You learned four different methods to select the number of components in the lecture. Can you find a solution for all for of them for this dataset?



# ### Solution 5
#
# ```python
# # 1 - looking at the cumulative explained variance
#
# variance_expalained = 0.98
#
# fig, ax = plt.subplots()
# csum = np.cumsum(evals)
# # Normalize by the sum of eigenvalues
# variance_explained = csum / np.sum(evals)
# print (np.argmax(variance_explained > variance_expalained))
# ax.plot(np.arange(1, len(variance_explained) + 1), variance_explained,
#        '--k')
# ax.set(
#     xlabel ='Number of components',
#     ylabel = 'Explaeined variance')
#
# # 2 - if we want to visualize the data we select the top 1,2 or 3 eigenvalues
#
# # 3 - all eigenvalues that are bigger than 1 
#     
# print (np.argmax(np.abs(evals) < 1))
#     
# # 4 - look at the scree plot
# fig, ax = plt.subplots()
# ax.plot(np.arange(1, len(evals) + 1), evals, 'o-k')
# ax.set(
#     xlabel ='Component',
#     ylabel ='Eigenvalue',
#     title = 'Scree plot',
#     # change xlim to zoom in
#     xlim=[0,784],)
# ```
#
#
#

# ## Step 4 - project into the new basis
#
# to project into the new basis we multiply the normalized matrix with the weight matrix (the eigenvectors), but we include only as many components as we like

# +
n_components = 4
score = np.matmul(design_matrix_centered, evectors[:,:n_components])

fig, ax = plt.subplots()
# we only show a subset of all samples
ax.scatter(score[:2000,0],score[:2000,1])
ax.set(
    xlabel='PC 1',
    ylabel='PC 2')


# -

# ## Step 5 - project back into the original space
#
#

# +
def project_back(n_components = 3, chosen_sample=0):
    
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)
    score = np.matmul(design_matrix_centered, evectors[:,:n_components])
    
    
    
    project_back = np.matmul(score,evectors[:,:n_components].T) + design_matrix_mean 
    
    visualize_number(number=design_matrix_centered[chosen_sample], title='original number', ax=ax1)
    visualize_number(number=project_back[chosen_sample], title='projection', ax=ax2)
    
    strongest_eigenvector = np.argmax(np.abs(score[chosen_sample]))
    visualize_number(number=evectors[:,strongest_eigenvector], title=f'Most contributing eigenvector {strongest_eigenvector+1}', ax=ax3, style ='eigenvectors' )


widgets.interactive(project_back, n_components=(1,784,1), chosen_sample=(0,69000,1))
# -


