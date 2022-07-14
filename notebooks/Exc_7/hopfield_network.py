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
        self.W = zeros([self.dim_patterns, self.dim_patterns])

    def train(self):
        # Accumulate outer products
        for pattern in self.training_patterns:
            self.W += np.outer(pattern, pattern)

        # Divide times the number of patterns
        self.W /= float(self.n_training_patterns)

        # Exclude the autoconnections
        self.W *= 1.0 - eye(self.dim_patterns)

    def run_simuation(
        self,
        noise=0.2,  # 0 = no noise, 1 = only noise
        sim_time=1500,  # timesteps
        frames_to_save=100,
        target_pattern = np.array([]),
        target_label = None,
        save_simulation = True,
        synchrounous_update = False,
    ):
        if target_pattern.size != 0:
            self.current_target_pattern = target_pattern
            self.current_target_label = target_label



        # store data at each sampling interval
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

            if synchrounous_update:
                x = sign(np.dot(self.W,x))
            else:
                # get a random index 
                current_x = x_indices[tt % self.dim_patterns]
                # Activation of a unit
                x[current_x] = sign(np.dot(self.W[current_x, :], x))


            # Store current activations
            if sim_time % sample_interval == 0:
                # Energy of the current state of the network
                self.store_energy[tt // sample_interval] = -0.5 * np.dot(x, np.dot(self.W, x))

                # array containing frames_to_save of network activation
                self.store_images[:, tt // sample_interval] = x


                if self.store_overlap_with_training_data:
                    print (np.sum(self.training_patterns == x,axis=1)/self.training_patterns.shape[1])
                    # self.overlap_with_training_data[tt//sample_interval] = a


        print ('simulation finished')

        if save_simulation:
            self.save_simulation()

    def init_figure(self):

        fig, ax = plt.subplots(2,2, figsize=(10,10))

        # Plot 1 - showing the target digit
        # Create subplot
        ax1 = ax[0,0]
        ax1.set_title("Target")
        # Create the imshow and save the handler
        im_target = display_image(ax1, self.current_target_pattern) 

        # Plot 2 - plot the state of the network

        # Create subplot
        ax2 = ax[0,1]
        ax2.set_title("Recalling")

        # Create the imshow and save the handler
        im_activation = display_image(ax2, self.store_images[:,0]) 


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
        im_errors = display_image(ax4, self.store_images[:,0]+ self.current_target_pattern * -1, cmap='bwr') 
        
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
            im_activation.set_array(to_mat(A))
            im_errors.set_array(to_mat(A + self.current_target_pattern*-1)) 
            im_energy.set_data(np.arange(t), self.store_energy[:t]) 


        # Create and render the animation
        anim = animation.FuncAnimation(fig, func = update,  frames = frames )
        # save it to file
        anim.save(f"mnist-hopfield_{self.current_target_label}.gif",
                  fps = 10, writer='imagemagick',dpi=50)



def to_mat(pattern):
    return pattern.reshape(28,28)

def display_image(ax, img_array,cmap=cm.binary):
    im = ax.imshow(to_mat(img_array), 
                interpolation = 'none', 
                aspect = 'auto',
                cmap = cmap) 
    ax.axis('off')
    return im