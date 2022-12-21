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

# # Artificial networks

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

# %matplotlib inline

# Settings for the figures
plt.style.use(plt.style.available[20])
plt.style.use("https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/plots_style.txt")

# from pylab import *
from sklearn.datasets import fetch_openml


# -

# # Perceptron

# ### Task X
#
# Can you implement the perceptron algorithm ?
#
#
# <div>
# <img src="https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/notebooks/Exc_9/static/learning_algorithm_perceptron.png" width="400"/>
# </div>

# +
def perceptron(x, w):  
    return np.dot(w, x) >= 0

starting_weights_seed = 44
np.random.seed(10) 

n_samples = 40 # number of samples

X = np.random.rand(2,n_samples)*1.1 # create random samples X
labels = X[0,:] + X[1,:]  >=1 # create the corresponding labels


X = np.vstack((np.ones(n_samples), X)) # include x_0 that is always 1 - the bias


np.random.seed(starting_weights_seed)
w = np.random.rand(3) # initialize a random connectivity matrix to start with (includes w_0)
np.random.seed(10)


all_w = w # this is an array 
selected_points = [None] # this is a list

''' put the algorithm here '''
# be sure to include the following two lines after every step
# then the viusalization works in the end
all_w = np.vstack((all_w, w))
selected_points.append(x)


            



def scroll_through_weights(nn):


    fig,ax = plt.subplots(figsize = (8,8))
    labels = perceptron(X, all_w[nn,:]) # get the classification results from the perceptron

    data_true = ax.scatter(*X[1:,labels], color = 'b', label = 'classified as True', s=90)
    data_false = ax.scatter(*X[1:,labels == False], color = 'r', label = 'classified as False', s=90)
    ax.scatter(*selected_points[nn+1][1:], color = (0,0,0,0), edgecolor='k', linewidth=2, s=90 )
    
    ax.set(
        xlabel = r'$x_1$',
        ylabel = r'$x_2$')

    x_ticks = [0,0.5,1]
    y_ticks = [0,0.5,1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticks,fontsize=20)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks,fontsize=20)
    ax.set_xlabel('$x_1$', fontsize=32, fontweight='bold')
    ax.set_ylabel('$x_2$', fontsize=32, fontweight='bold')

    
widgets.interactive(scroll_through_weights, nn = (0,all_w.shape[0],1))



# -

# ### Solution 1
#
# 78a5a35c0791e03abff0b965447c82b9
#

mnist = fetch_openml(name='mnist_784', as_frame = False)


# +
# print(mnist.keys())
# print (mnist.target)

def to_mat(image_array):
    return image_array.reshape(28,28)

def show_example_plot(image_array, label=None):
    fig, ax = plt.subplots()
    ax.imshow(to_mat(image_array),cmap='Greys')
    if label != None:
        ax.set(
            title = f'Number {label}',
        )
    ax.axis('off')


# -

example_id = 4
show_example_plot(mnist.data[example_id],mnist.target[example_id])

# +

# Take two rows
patterns = mnist.data
labels = mnist.target

# We need only the sign (transform to binary input)
patterns = np.sign(patterns/255.0 - 0.5)

# Set the number of patterns (two in out case)
n_patterns = 2

# Number of units of the network
# n = img_side*img_side
# -

def show_example_plot_binary(example_ids = [0,1]):
    sqrt = np.sqrt(len(example_ids))
    rows = int(np.floor(sqrt))
    cols = int(np.ceil(sqrt))
    
    if rows * cols < len(example_ids):
        rows += 1
    
    fig, axes = plt.subplots(rows, cols,figsize=(12,12))
    ax = axes.ravel()
    for ii, ex_id in enumerate(example_ids):
        ax[ii].imshow(to_mat(patterns[ex_id]),cmap='Greys')
        ax[ii].set(
            title = f'Number {labels[ex_id]}',
        )
    for this_ax in ax:
        this_ax.axis('off')
    plt.tight_layout()


show_example_plot_binary([0,1,2,3,4,5,6,7])

# +
import numpy as np
from PIL import Image
import requests
from io import BytesIO





# +
# The matplotlib object to do animations
from matplotlib import animation
import numpy as np


class HopfieldNetwork(object):
    """docstring for HopfieldNetwork

    patterns: np.array with shape (n_of_patterns, dim_of_patterns)
    """

    def __init__(self, 
        training_patterns,
        training_labels,
        store_overlap_with_training_data=False):
        super(HopfieldNetwork, self).__init__()

        self.training_patterns = training_patterns
        self.training_labels = training_labels

        self.n_training_patterns = self.training_patterns.shape[0]
        self.dim_patterns = self.training_patterns.shape[1]
        self.init_network()

        self.current_target_pattern = self.training_patterns[0]
        self.current_target_label = self.training_labels[0]

        self.store_overlap_with_training_data = store_overlap_with_training_data

    def init_network(self):
        # Initialize weights to zero values
        self.W = np.zeros([self.dim_patterns, self.dim_patterns])

    def train(self):
        # Accumulate outer products
        for pattern in self.training_patterns:
            self.W += np.outer(pattern, pattern)

        # Divide times the number of patterns
        self.W /= float(self.n_training_patterns)

        # Exclude the autoconnections
        self.W *= 1.0 - np.eye(self.dim_patterns)

    def run_simuation(
        self,
        noise=0.2,  # 0 = no noise, 1 = only noise
        sim_time=5500,  # timesteps
        frames_to_save=100,
        target_pattern = np.array([]),
        target_label = None,
        save_simulation = True,
        synchrounous_update = False,
    ):
        if target_pattern.size != 0:
            self.current_target_pattern = target_pattern
            self.current_target_label = target_label



        sample_interval = sim_time // frames_to_save

        self.store_images = np.zeros([self.dim_patterns, frames_to_save])
        self.store_energy = np.zeros(frames_to_save)

        x = self.current_target_pattern.copy()

        # We randomly perturb the initial image by swapping the values
        mask = np.sign(np.random.random(self.dim_patterns) - noise)
        random_array = np.sign(np.random.random(self.dim_patterns)-0.5)
        x[mask == -1] = random_array[mask == -1]

        # During the iterations we ranomly select a unit to update
        x_indices = np.arange(self.dim_patterns)
        np.random.shuffle(x_indices)


        # the iterations
        for tt in range(sim_time):
            
            # Store current activations
            if tt % sample_interval == 0:
                # Energy of the current state of the network
                self.store_energy[tt // sample_interval] = -0.5 * np.dot(x, np.dot(self.W, x))

                # array containing frames_to_save of network activation
                self.store_images[:, tt // sample_interval] = x


                if self.store_overlap_with_training_data:
                    print (np.sum(self.training_patterns == x,axis=1)/self.training_patterns.shape[1])
                    # self.overlap_with_training_data[tt//sample_interval] = a
                    

            if synchrounous_update:
                x = np.sign(np.dot(self.W,x))
            else:
                # get a random index 
                current_x = x_indices[tt % self.dim_patterns]
                # Activation of a unit
                x[current_x] = np.sign(np.dot(self.W[current_x, :], x))


            


        print ('simulation finished')

        if save_simulation:
            self.save_simulation()

    def init_figure(self):

        fig, ax = plt.subplots(2,3, figsize=(15,10))

        # Plot 1 - showing the target digit
        # Create subplot
        ax1 = ax[0,0]
        ax1.set_title("Start")
        # Create the imshow and save the handler
        self.display_image(ax1, self.store_images[:,0]) 
        
        
        # Plot 2 - plot the state of the network

        # Create subplot
        ax2 = ax[0,1]
        ax2.set_title("Recalling")

        # Create the imshow and save the handler
        im_activation = self.display_image(ax2, self.store_images[:,0]) 
        
        ax6 = ax[0,2]
        ax6.set_title("Target")
        # Create the imshow and save the handler
        im_target = self.display_image(ax6, self.current_target_pattern) 


        # Plot 3 - plot the history of the energy
        # Create subplot
        ax3 = ax[1,1]

        ax3.set_title("Energy")

        # Create the line plot and save the handler
        im_energy, = ax3.plot(self.store_energy) # the comma after im_energy is important (line plots are returned in lists)

        # style
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        ax3.set_xticks([])
        ax3.set_yticks([])   


        ax4 = ax[1,0]
        ax4.set_title("Errors")

        # Create the imshow and save the handler
        im_errors = self.display_image(ax4, self.store_images[:,0]+ self.current_target_pattern * -1, cmap='bwr') 
        
        # return plot handlers
        return fig, im_target, im_activation, im_energy, im_errors


    def save_simulation(self):

    
        fig, im_target, im_activation, im_energy, im_errors = self.init_figure()
        
        frames = [t for t in range(self.store_images.shape[1])]

        def update(t,
            im_activation=im_activation, 
            im_energy=im_energy,
            im_errors=im_errors,) :
            
            
            A = np.squeeze(self.store_images[:,t])
            im_activation.set_array(self.to_mat(A))
            im_errors.set_array(self.to_mat(A + self.current_target_pattern*-1)) 
            im_energy.set_data(np.arange(t), self.store_energy[:t]) 


        # Create and render the animation
        anim = animation.FuncAnimation(fig, func = update,  frames = frames )
        # save it to file
        anim.save(f"mnist-hopfield_{self.current_target_label}.gif",
                  fps = 10, writer='imagemagick',dpi=50)
        
    def to_mat(self, pattern):
        img_dim = int(np.sqrt(self.dim_patterns))
        return pattern.reshape(img_dim,img_dim)
    
    def display_image(self, ax, img_array,cmap='binary'):
        im = ax.imshow(self.to_mat(img_array), 
                    interpolation = 'none', 
                    aspect = 'auto',
                    cmap = cmap) 
        ax.axis('off')
        return im






# -

test_network = HopfieldNetwork(
    training_patterns = patterns[[0,1]],
    training_labels = labels[[0,1]])


test_network.train()

# +
print (patterns[1].shape)
print (labels[1])
test_network.run_simuation(
    noise=0.1,
    target_pattern=patterns[1],
    target_label=labels[1])



# +
def load_binary_images_and_labels(
    labels = ['homer', 'tintin', 'pikachu', 'hello_kitty','super_mario', 'lab_logo',
              'lucky_luke','obelix','scrooge_duck','winnie_pooh'],
    show_images = False):
    
    images = np.array([])
    
    
    
    for name in labels:
    
        url = f'https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/notebooks/Exc_9/static/pixel_images_{name}.png'
        response = requests.get(url)
        img = np.array(Image.open(BytesIO(response.content)))[:,:,0] # just use one channel

        binary_image = -1 * np.ones_like(img)
        binary_image[img < 100] = 1
   
        if images.size == 0:
            images = binary_image.flatten()[None,:]
        else:
            images = np.vstack((images, binary_image.flatten()[None,:]))

            
            
    if show_images:
        
        fig, axes = plt.subplots(2,5, figsize = (19,9))
        
        for img, ll, ax in zip(images, labels, axes.flatten()):
            ax.imshow(img.reshape(64,64),cmap='binary',interpolation = 'none', 
                    aspect = 'auto')
            ax.set_title(ll)
            ax.axis('off')

    return images, labels


load_binary_images_and_labels(show_images=True);

# +
images, labels = load_binary_images_and_labels()

test_network = HopfieldNetwork(
    training_patterns = images,
    training_labels = labels)
# -

test_network.train()

test_network.run_simuation(
    noise=0.0,
    target_pattern= images[0],
    target_label=labels[0],
    synchrounous_update = False,
    sim_time=5500,)


# +
def pca(X):
  """
  Performs PCA on multivariate data. Eigenvalues are sorted in decreasing order

  Args:
     X (numpy array of floats) :   Data matrix each column corresponds to a
                                   different random variable

  Returns:
    (numpy array of floats)    : Data projected onto the new basis
    (numpy array of floats)    : Corresponding matrix of eigenvectors
    (numpy array of floats)    : Vector of eigenvalues

  """

  X = X - np.mean(X, 0)
  cov_matrix = get_sample_cov_matrix(X)
  evals, evectors = np.linalg.eigh(cov_matrix)
  evals, evectors = sort_evals_descending(evals, evectors)
  score = change_of_basis(X, evectors)

  return score, evectors, evals

def get_sample_cov_matrix(X):
  """
  Returns the sample covariance matrix of data X.

  Args:
    X (numpy array of floats) : Data matrix each column corresponds to a
                                different random variable

  Returns:
    (numpy array of floats)   : Covariance matrix
"""

  X = X - np.mean(X, 0)
  cov_matrix = 1 / X.shape[0] * np.matmul(X.T, X)
  return cov_matrix

def sort_evals_descending(evals, evectors):
  """
  Sorts eigenvalues and eigenvectors in decreasing order. Also aligns first two
  eigenvectors to be in first two quadrants (if 2D).

  Args:
    evals (numpy array of floats)    :   Vector of eigenvalues
    evectors (numpy array of floats) :   Corresponding matrix of eigenvectors
                                         each column corresponds to a different
                                         eigenvalue

  Returns:
    (numpy array of floats)          : Vector of eigenvalues after sorting
    (numpy array of floats)          : Matrix of eigenvectors after sorting
  """

  index = np.flip(np.argsort(evals))
  evals = evals[index]
  evectors = evectors[:, index]
  if evals.shape[0] == 2:
    if np.arccos(np.matmul(evectors[:, 0],
                           1 / np.sqrt(2) * np.array([1, 1]))) > np.pi / 2:
      evectors[:, 0] = -evectors[:, 0]
    if np.arccos(np.matmul(evectors[:, 1],
                           1 / np.sqrt(2)*np.array([-1, 1]))) > np.pi / 2:
      evectors[:, 1] = -evectors[:, 1]

  return evals, evectors

def change_of_basis(X, W):
  """
  Projects data onto a new basis.

  Args:
    X (numpy array of floats) : Data matrix each column corresponding to a
                                different random variable
    W (numpy array of floats) : new orthonormal basis columns correspond to
                                basis vectors

  Returns:
    (numpy array of floats)   : Data matrix expressed in new basis
  """

  Y = np.matmul(X, W)

  return Y

# +


training_data = []
for ii in range(1,10):
    numbers = patterns[labels==f'{ii}']
    print (numbers.shape)
    score, evectors, evals = pca(numbers)
    
    X = evectors[:, 0]
    Y = np.sign(X-np.mean(X))
    if np.sum(Y) > 0:
        Y *= -1
    show_example_plot(Y)
    training_data.append(Y)
# -

test_network = HopfieldNetwork(
    training_patterns = np.array(training_data),
    training_labels = range(1,10),
    store_overlap_with_training_data=True,
)
test_network.train()

test_network.run_simuation(
    noise=0.2,
    target_pattern=patterns[2],
    target_label=labels[2],
    synchrounous_update=False)


