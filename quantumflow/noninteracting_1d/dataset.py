import tensorflow as tf
import numpy as np
import os

import quantumflow
import quantumflow.noninteracting_1d

@tf.function
def generate_potentials(return_x=False,
                        return_h=False,
                        dataset_size=100, 
                        discretisation_points=500, 
                        n_gauss=3, 
                        interval_length=1.0,
                        a_minmax=(0.0, 3*10.0), 
                        b_minmax=(0.4, 0.6), 
                        c_minmax=(0.03, 0.1), 
                        n_method='sum',
                        dtype='float64',
                        **kwargs):
    
    if dtype == 'double' or dtype == 'float64':
        dtype = tf.float64
    elif dtype == 'float' or dtype == 'float32':
        dtype = tf.float32
    else:
        raise ValueError(f"unknown dtype {dtype}")

    x = tf.linspace(tf.constant(0.0, dtype=dtype), interval_length, discretisation_points, name="x")

    a = tf.random.uniform((dataset_size, 1, n_gauss), minval=a_minmax[0], maxval=a_minmax[1], dtype=dtype, name="a")
    b = tf.random.uniform((dataset_size, 1, n_gauss), minval=b_minmax[0]*interval_length, maxval=b_minmax[1]*interval_length, dtype=dtype, name="b")
    c = tf.random.uniform((dataset_size, 1, n_gauss), minval=c_minmax[0]*interval_length, maxval=c_minmax[1]*interval_length, dtype=dtype, name="c")

    curves = -tf.square(tf.expand_dims(tf.expand_dims(x, 0), 2) - b)/(2*tf.square(c))
    curves = -a*tf.exp(curves)

    if n_method == 'sum':
        potentials = tf.reduce_sum(curves, -1, name="potentials")
    elif n_method == 'mean':
        potentials = tf.reduce_mean(curves, -1, name="potentials")
    else:
        raise NotImplementedError(f"Method {n_method} is not implemented.")

    returns = [potentials]

    if return_x:
        returns += [x]
    
    if return_h:
        h = tf.cast(interval_length/(discretisation_points-1), dtype=dtype) # discretisation interval
        returns += [h]
   
    return returns

def save_dataset(directory, filename, x, h, potential, wavefunctions, energies):
        if filename.endswith('.pickle') or filename.endswith('.pkl'):
            import pickle
            with open(os.path.join(directory, filename), 'wb') as f:
                pickle.dump({'x': x, 'h': h, 'potential': potential, 'wavefunctions': wavefunctions, 'energies': energies}, f)
            
        elif filename.endswith('.hdf5') or filename.endswith('.h5'):
            import h5py
            with h5py.File(os.path.join(directory, filename), "w") as f:
                f.attrs['x'] = x
                f.attrs['h'] = h
                f.create_dataset('potential', data=potential, compression="gzip")
                f.create_dataset('wavefunctions', data=wavefunctions, compression="gzip")
                f.create_dataset('energies', data=energies, compression="gzip")
        else:
            raise KeyError(f"Unknown format {filename} to save dataset.")


def load_dataset(directory, filename):
    if filename.endswith('.pickle') or filename.endswith('.pkl'):
        import pickle
        with open(os.path.join(directory, filename), 'rb') as f:
            x, h, potential, wavefunctions, energies = pickle.load(f).values()

    elif filename.endswith('.hdf5') or filename.endswith('.h5'):
        import h5py
        with h5py.File(os.path.join(directory, filename), 'r') as f:
            x = f.attrs['x']
            h = f.attrs['h']
            potential = f['potential'][()]
            wavefunctions = f['wavefunctions'][()]
            energies = f['energies'][()]

    return x, h, potential, wavefunctions, energies


class PotentialDataset(quantumflow.QFDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.kwargs = kwargs

    def build(self):
        self.potential, self.x, self.h = generate_potentials(
            return_x=True, 
            return_h=True,
            **self.kwargs)


class Non1D_QFDataset(quantumflow.QFDataset):
    def __init__(self, filename, seed, dataset_size, generate_batch_size, N, numerov_init_slope, dtype, potentials, **kwargs):
        super().__init__(**kwargs)
        self.filename = filename
        self.seed = seed
        self.dataset_size = dataset_size
        self.generate_batch_size = generate_batch_size
        self.N = N
        self.h = None
        self.numerov_init_slope = numerov_init_slope
        self.dtype = dtype

        self.potential_generator = quantumflow.instantiate(potentials, dataset_size=self.dataset_size, dtype=self.dtype, **kwargs)
    
        
    def build(self):
        if not os.path.isfile(os.path.join(self.run_dir, self.filename)):

            tf.keras.backend.clear_session()
            tf.random.set_seed(self.seed)
            self.potential_generator.build()

            potential, x, h = self.potential_generator.potential, self.potential_generator.x, self.potential_generator.h 

            self.x = x.numpy()
            self.h = h.numpy()
            self.potential = potential.numpy()

            energies_batches = []
            wavefunctions_batches = []

            for i in range(self.dataset_size//self.generate_batch_size):
                energies, wavefunctions = quantumflow.noninteracting_1d.solve_schroedinger(
                    potential[i*self.generate_batch_size:(i+1)*self.generate_batch_size], 
                    self.N, 
                    h, 
                    self.dtype, 
                    self.numerov_init_slope
                    )

                energies_batches.append(energies.numpy())
                wavefunctions_batches.append(wavefunctions.numpy())

            if self.dataset_size % self.generate_batch_size:
                energies, wavefunctions = quantumflow.noninteracting_1d.solve_schroedinger(
                    potential[(self.dataset_size//self.generate_batch_size)*self.generate_batch_size:], 
                    self.N, 
                    h, 
                    self.dtype, 
                    self.numerov_init_slope
                    )

                energies_batches.append(energies.numpy())
                wavefunctions_batches.append(wavefunctions.numpy())

            self.energies = np.concatenate(energies_batches, axis=0)
            self.wavefunctions = np.concatenate(wavefunctions_batches, axis=0)

            save_dataset(self.run_dir, self.filename, self.x, self.h, self.potential, self.wavefunctions, self.energies)
            
            print(f"dataset {self.filename} saved to {self.run_dir}")
        else:
            self.x, self.h, self.potential, self.wavefunctions, self.energies = load_dataset(self.run_dir, self.filename)

    def visualize(self, preview=5, figsize=(20, 3), dpi=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.x, np.transpose(self.potential)[:, :preview]) # only plot first potentials
        plt.title("Randomly Generated Potentials")
        plt.xlabel("x / bohr")
        plt.ylabel("Energy / hartree")
        plt.grid(which='major', axis='y', linestyle='--')
        plt.show()


        #----------------------------
        plt.figure(figsize=figsize, dpi=dpi)
        for i, plot in enumerate(self.wavefunctions[:preview]):
            plt.plot(self.x, plot, 'C' + str(i%10))
        plt.xlim(self.x[[0, -1]])
        plt.title('Wavefunctions - Numerov Solutions')
        plt.show()

        #----------------------------
        fig, axs = plt.subplots(1, preview, figsize=(figsize[0], figsize[1]*2), dpi=dpi)

        for i, plot in enumerate(self.wavefunctions[:preview]**2 + self.energies[:preview, np.newaxis, :]):
            for n, plot_single in enumerate(plot.transpose()):
                axs[i].plot(self.x, self.potential[i], 'k')
                axs[i].plot(self.x, np.ones(self.x.shape)*self.energies[i, n], ':k')
                axs[i].plot(self.x, plot_single, 'C' + str(i%10))
                axs[i].set_ylim([np.min(self.potential[:preview]), max(np.max(self.energies[:preview]*1.1), 0.5)])
                if i == 0: 
                    axs[i].set_ylabel('energies, potential / hartree')
                    axs[i].set_xlabel("x / bohr")
                else:
                    axs[i].get_yaxis().set_visible(False)
        fig.suptitle('Energies and Densities')
        plt.show()


        #-------------------------------

        plt.figure(figsize=figsize, dpi=dpi)
        bins = np.linspace(min(self.energies.flatten()), max(self.energies.flatten()), 100)

        for i in range(self.energies.shape[1]):
            color = i/(self.energies.shape[1] - 1)
            plt.hist(self.energies[:, i], bins, alpha=0.7, color=[0.8*color**2, 0.8*(1-color)**2, 1.6*color*(1-color)], label= f"{i}-th solution")

        plt.xlabel('split energy / hartree')
        plt.ylabel('count')
        plt.legend(loc='upper right')
        plt.title("Energies")
        plt.show()
        

