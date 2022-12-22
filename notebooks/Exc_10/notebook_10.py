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

# Settings for the figures
plt.style.use(plt.style.available[20])
plt.style.use("https://github.com/comp-neural-circuits/intro-to-comp-neuro/raw/dev/plots_style.txt")
plt.rcParams["animation.html"] = "jshtml"

# +
np.random.seed(12)

X = -0.9 + np.random.rand(45)*1.8
Y = 0.42*X + (-0.5 + np.random.rand(45))

fig, ax = plt.subplots()
ax.scatter(X,Y, c = '#1f78b4', s=30)
ax.set(
    xlim = [-1.15,1.15],
    ylim = [-1.15,1.15],
    xticklabels = [],
    yticklabels = [],
    xlabel = 'Wine Darkness',
    ylabel = 'Strength');
ax.yaxis.label.set_size(22)
ax.xaxis.label.set_size(22)
ax.set_box_aspect(1)


x = 2


theta  = np.pi*10/360.
alpha = np.tan(theta)
x_line = np.linspace(-1.4,1.4)


line_plot, = ax.plot(x_line,alpha*x_line, linewidth = 2.5, color = 'k', zorder=-10)
line_plots = []
scatter_plots = []
for xx,yy in zip(X,Y):
        v = np.array([1, alpha])
        projection = np.array([xx,yy])@v.T/(v.T@v)*v
        dummy, = ax.plot([xx,projection[0]],[yy,projection[1]], c='#EE0E1D',linewidth=1, linestyle = '--' )
        line_plots.append(dummy)
        scatter_plots.append(ax.scatter(*projection, c='#EE0E1D',s=30))
        





def animate(i):
    
    
    alpha = 2*np.pi*i/360.
    a = np.tan(alpha)
    line_plot.set_ydata(a*x_line)
    
    
    v = np.array([1, a])
    
    
    for jj, (xx,yy) in enumerate(zip(X,Y)):
        
        projection = np.array([xx,yy])@v.T/(v.T@v)*v
        line_plots[jj].set_ydata([yy,projection[1]])
        line_plots[jj].set_xdata([xx,projection[0]])
        scatter_plots[jj].set_offsets(projection)
        
    
    return line_plots

# Init only required for blitting to give a clean slate.
def init():
    line_plot.set_ydata(np.ma.array(x_line, mask=True))
    return line_plot,


ani = FuncAnimation(fig, animate, frames=np.linspace(0, 179,360), init_func=init,
                              interval=20, blit=False)
# ani.save('animation.gif', writer='Pillow', fps=24)
from IPython.display import HTML
HTML(ani.to_jshtml(fps=20,embed_frames=False))

# +
# project data into a new space

np.random.seed(12)

X = -0.9 + np.random.rand(45)*1.8
Y = 0.42*X + (-0.5 + np.random.rand(45))
# vec = np.array([[1,0.25],[0.5,1],[1,1]]).T
vec = np.vstack([X,Y])

print (vec.shape)




def interact(i):
    
    fig, (ax, new_ax) = plt.subplots(1,2, figsize=(12,6))
    ax.scatter(*vec, c = '#1f78b4', s=50)
    ax.set_box_aspect(1)


#     i = 120 # in degree


    # calculate
    alpha = 2*np.pi*i/360.
    a = np.tan(alpha)
    
    
    x_0 = 1
    if 90 < i <= 270:
        x_0 = -1
        a = -a


    vec_x = np.array([x_0,a])
    vec_y = np.array([-a,x_0])

    norm_x = vec_x/np.sqrt(np.sum(vec_x**2))
    norm_y = vec_y/np.sqrt(np.sum(vec_y**2))
    
    project_x = np.dot(norm_x, vec)
    project_y = np.dot(norm_y, vec)
      
                       
    # PLOT onto axes
    ax.quiver(0,0,*norm_x, scale = 2, color='#984ea3')
    ax.quiver(0,0,*norm_x*-1, scale = 2, color='#984ea3', headlength=0, headaxislength=0)
    ax.quiver(0,0,*norm_y, scale = 2, color='#ff7f00')
    ax.quiver(0,0,*norm_y*-1, scale = 2, color='#ff7f00', headlength=0, headaxislength=0)
    
    for bb, cc in zip([(norm_x * np.vstack([project_x,project_x]).T),(norm_y * np.vstack([project_y,project_y]).T)],( '#984ea3','#ff7f00')):
        
        ax.scatter(*bb.T, color = cc) 
        for a,b in zip(bb,vec.T):
            ax.plot([a[0],b[0]],[a[1],b[1]], linestyle = '--', color = cc)
            
                       
                       
    new_ax.scatter([project_x],[project_y], c = '#1f78b4', s=50)


    
        
    # Style of the axes
    for dummy_ax in [ax,new_ax]:
        
        dummy_ax.set(
            xlim = [-1.5,1.5],
            ylim = [-1.5,1.5])

        for axis in ['top','bottom','left','right']:
            dummy_ax.spines[axis].set_linewidth(2)
        dummy_ax.tick_params(width=2)

        x_ticks = [-1,0,1]
        y_ticks = [-1,0,1]
        dummy_ax.set_xticks(x_ticks)
        dummy_ax.set_xticklabels(x_ticks,fontsize=20)
        dummy_ax.set_yticks(y_ticks)
        dummy_ax.set_yticklabels(y_ticks,fontsize=20)
        
    ax.set_xlabel('$x_1$', fontsize=25, fontweight='bold')
    ax.set_ylabel('$x_2$', fontsize=25, rotation=0, fontweight='bold', labelpad=19)
    
    new_ax.set_xlabel('$c_1$', fontsize=25, fontweight='bold')
    new_ax.set_ylabel('$c_2$', fontsize=25, rotation=0, fontweight='bold', labelpad=19)
    
    new_ax.spines['left'].set_color('#ff7f00')
    new_ax.spines['bottom'].set_color('#984ea3')    
    
    
    plt.tight_layout()
widgets.interactive(interact, i=(0,359,1))


# -

def calculations(i, vec):
    # calculations
    alpha = 2*np.pi*i/360.
    a = np.tan(alpha)

    x_0 = 1
    if 90 < i <= 270:
        x_0 = -1
        a = -a


    vec_x = np.array([x_0,a])
    vec_y = np.array([-a,x_0])

    norm_x = vec_x/np.sqrt(np.sum(vec_x**2))
    norm_y = vec_y/np.sqrt(np.sum(vec_y**2))

    project_x = np.dot(norm_x, vec)
    project_y = np.dot(norm_y, vec)
    
    
    return vec_x, vec_y, norm_x, norm_y, project_x, project_y

# +
# project data into a new space
show_projection_line_plots = True



np.random.seed(12)


# create vector
X = -0.9 + np.random.rand(45)*1.8
Y = 0.42*X + (-0.5 + np.random.rand(45))
vec = np.vstack([X,Y])

# vec = np.array([[1,0.25],[0.5,1],[1,1]]).T








# calculations
vec_x, vec_y, norm_x, norm_y, project_x, project_y = calculations(i=0, vec=vec)


# create figure
fig, (ax, new_ax) = plt.subplots(1,2, figsize=(12,6))


# PLOT onto axes
qiver_x_1 = ax.quiver(0,0,*norm_x, scale = 2, color='#984ea3')
qiver_x_2 = ax.quiver(0,0,*norm_x*-1, scale = 2, color='#984ea3', headlength=0, headaxislength=0)
qiver_y_1 = ax.quiver(0,0,*norm_y, scale = 2, color='#ff7f00')
qiver_y_2 = ax.quiver(0,0,*norm_y*-1, scale = 2, color='#ff7f00', headlength=0, headaxislength=0)


projection_scatter_plots = []
projection_line_plots = []
for bb, cc in zip([(norm_x * np.vstack([project_x,project_x]).T),(norm_y * np.vstack([project_y,project_y]).T)],( '#984ea3','#ff7f00')):

    projection_scatter_plots.append(ax.scatter(*bb.T, color = cc)) 
    for a,b in zip(bb,vec.T):
        line_plot, = ax.plot([a[0],b[0]],[a[1],b[1]], linestyle = '--', color = cc) # the , is important
        projection_line_plots.append(line_plot)
        
ax.scatter(*vec, c = '#1f78b4', s=50) # this remains constant



new_ax_scatter_plot = new_ax.scatter([project_x],[project_y], c = '#1f78b4', s=50)




# Style of the axes
for dummy_ax in [ax,new_ax]:
    dummy_ax.set_box_aspect(1)
    dummy_ax.set(
        xlim = [-1.5,1.5],
        ylim = [-1.5,1.5])

    for axis in ['top','bottom','left','right']:
        dummy_ax.spines[axis].set_linewidth(2)
    dummy_ax.tick_params(width=2)

    x_ticks = [-1,0,1]
    y_ticks = [-1,0,1]
    dummy_ax.set_xticks(x_ticks)
    dummy_ax.set_xticklabels(x_ticks,fontsize=20)
    dummy_ax.set_yticks(y_ticks)
    dummy_ax.set_yticklabels(y_ticks,fontsize=20)

ax.set_xlabel('$x_1$', fontsize=25, fontweight='bold')
ax.set_ylabel('$x_2$', fontsize=25, rotation=0, fontweight='bold', labelpad=19)

new_ax.set_xlabel('$c_1$', fontsize=25, fontweight='bold')
new_ax.set_ylabel('$c_2$', fontsize=25, rotation=0, fontweight='bold', labelpad=19)

new_ax.spines['left'].set_color('#ff7f00')
new_ax.spines['bottom'].set_color('#984ea3')    
plt.tight_layout()



def animate(i):
    
    
    vec_x, vec_y, norm_x, norm_y, project_x, project_y = calculations(i, vec=vec)
    
    
#     # PLOT onto axes
#     qiver_x_1 = ax.quiver(0,0,*norm_x, scale = 2, color='#984ea3')
#     qiver_x_2 = ax.quiver(0,0,*norm_x*-1, scale = 2, color='#984ea3', headlength=0, headaxislength=0)
#     qiver_y_1 = ax.quiver(0,0,*norm_y, scale = 2, color='#ff7f00')
#     qiver_y_2 = ax.quiver(0,0,*norm_y*-1, scale = 2, color='#ff7f00', headlength=0, headaxislength=0)


    
    for ii, (bb, cc) in enumerate(zip([(norm_x * np.vstack([project_x,project_x]).T),(norm_y * np.vstack([project_y,project_y]).T)],( '#984ea3','#ff7f00'))):

        projection_scatter_plots[ii].set_offsets(bb.T) 
#         for a,b in zip(bb,vec.T):
#             line_plot, = ax.plot([a[0],b[0]],[a[1],b[1]], linestyle = '--', color = cc) # the , is important
#             projection_line_plots.append(line_plot)
#     new_ax_scatter_plot = new_ax.scatter([project_x],[project_y], c = '#1f78b4', s=50)
    
    
    
#     for jj, (xx,yy) in enumerate(zip(X,Y)):
        
#         projection = np.array([xx,yy])@v.T/(v.T@v)*v
#         line_plots[jj].set_ydata([yy,projection[1]])
#         line_plots[jj].set_xdata([xx,projection[0]])
        
        
#         new_ax_scatter_plot.set_offsets(projection)
        
        
#         line_plots.append(dummy)
    
    return projection_scatter_plots

# Init only required for blitting to give a clean slate.
# def init():
#     line_plot.set_ydata(np.ma.array(x_line, mask=True))
#     return line_plot,


ani = FuncAnimation(fig, animate, np.linspace(0, 20,21), init_func=init,
                              interval=25, blit=True)
# ani.save('rotate_coordinate_system.gif', writer='Pillow', fps=24)
plt.show()

# +
import matplotlib.pyplot as plt

# Coordinates of the starting and ending points of the arrow
x1, y1 = 0, 0
x2, y2 = 3, 4

# Displacements in the x and y directions from the starting point to the ending point
u = x2 - x1
v = y2 - y1

# Plot the arrow using quiver
plt.quiver(x1, y1, u, v, color='r', scale=1)

# Add some labels and show the plot
plt.xlabel('x')
plt.ylabel('y')
plt.title('Arrow between two points')
plt.xlim([0,5])
plt.ylim([0,5])
plt.show()


# +
def AAA(alpha=70):
    np.random.seed(15)

    X = -0.5 + np.random.rand(45)
    Y = X + (-0.5 + np.random.rand(45))*0.5

    fig, ax = plt.subplots()
    ax.scatter(X,Y)
    ax.set(
        xlim = [-1,1],
        ylim = [-1,1],
#         xticklabels = [],
#         yticklabels = [],
        xlabel = 'Wine Darkness',
        ylabel = 'Strength');


    theta  = 2*np.pi*alpha/360.
    m = np.tan(theta)
    x_line = np.linspace(-1,1,1000)


#     line_plot, = ax.plot(x_line,alpha*x_line)
    line_plots = []
    scatter_plots = []
    
    def normalized(a, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
        l2[l2==0] = 1
        return (a / np.expand_dims(l2, axis))[0]
    
    v = np.array([1, m])
    v = normalized(v)
    
    ax.scatter([0],[0],color='k')
    
    
    def circle(x, minus=1):
        return minus * np.sqrt(1 - x**2)
    
    xxx = np.linspace(-1,1,1000)
    ax.plot(xxx,circle(xxx),'k')
    ax.plot(xxx,circle(xxx,-1),'k')

    ax.set_box_aspect(1)
    for xx,yy in zip(X,Y):         
            
            projection = np.dot(np.array([xx,yy]),v)/np.dot(v,v)*v
            dummy, = ax.plot([xx,projection[0]],[yy,projection[1]], 'r',linewidth=0.5, linestyle = '--' )
            line_plots.append(dummy)
            scatter_plots.append(ax.scatter([projection[1]],[projection[0]], c='r',s=24))

        



widgets.interactive(AAA,alpha=(0,179,10))

# -


# \begin{equation}
# U = m \cdot K + b
# \end{equation}
#
# Die Äquivalenz können wir nun sehen, wenn wir uns die Gleichung des Neurons genauer anschauen:
#
# \begin{equation}
# z = K \cdot w_1 + U \cdot w_2
# \end{equation}
#
# Nun wissen wir, dass das Neuron einen Punkt der Klasse $A$ zuordnet, wenn $z<s$ gilt, und der Klasse $B$, wenn $z>s$ gilt. 
# Das bedeutet, dass wir bei $z=s$ eine Grenze zwischen den beiden Klassen ziehen können.
# Wenn wir $z=s$ einsetzen und nach $U$ umformen, bekommen wir:
#
# \begin{equation}
# z = K \cdot w_1 + U \cdot w_2 = s \qquad \Leftrightarrow \qquad U \cdot w_2 = - K \cdot w_1 + s \qquad \Leftrightarrow \qquad U = \left(-\frac{w_1}{w_2}\right) \cdot K + \frac{s}{w_2}
# \end{equation}
#
# Hier sehen wir die selbe Geradengleichung wie zu Beginn, wenn wir $m=\left(-\frac{w_1}{w_2}\right)$ und $b=\frac{s}{w_2}$ setzen.
#     
#

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
# -


