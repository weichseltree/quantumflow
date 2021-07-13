import tensorflow as tf
import numpy as np
import os

import quantumflow


def weizsaecker_pseudo_integrand(pseudo, h):
    return 1/2*tf.square(quantumflow.utils.derivative_five_point(pseudo, h))


def weizsaecker_pseudo_functional(pseudo, h):
    return quantumflow.utils.integrate(weizsaecker_pseudo_integrand(pseudo, h), h)


def weizsaecker_pseudo_functional_derivative(pseudo, h):
    return -quantumflow.utils.laplace_five_point(pseudo, h)


def calculate_density_and_energies(potential, wavefunctions, energies, N, h):
    assert(N <= wavefunctions.shape[2])
    density = np.sum(np.square(wavefunctions)[:, :, :N], axis=-1)

    kinetic_energy_densities = 0.5*quantumflow.utils.derivative_five_point(wavefunctions, h)**2 # == -0.5*wavefunctions*laplace_five_point(wavefunctions)
    
    potential_energy_densities = np.expand_dims(potential, axis=2)*wavefunctions**2
    
    potential_energies = quantumflow.utils.np_integrate(potential_energy_densities, h)
    kinetic_energies = quantumflow.utils.np_integrate(kinetic_energy_densities, h)

    energy = np.sum(energies[:, :N], axis=-1)
    potential_energy = np.sum(potential_energies[:, :N], axis=-1)
    kinetic_energy = np.sum(kinetic_energies[:, :N], axis=-1)
    
    kinetic_energy_density = np.sum(kinetic_energy_densities[:, :, :N], axis=-1)
    potential_energy_density = np.sum(potential_energy_densities[:, :, :N], axis=-1)

    return density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density


def calculate_system_properties(potential, wavefunctions, energies, N, h):
    assert(N <= wavefunctions.shape[2])

    density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density = calculate_density_and_energies(potential, wavefunctions, energies, N, h)
    derivative = np.expand_dims(energy/N, axis=1) - potential

    pseudo = np.sqrt(density)
    vW_kinetic_energy_density = weizsaecker_pseudo_integrand(pseudo, h).numpy()
    vW_kinetic_energy = weizsaecker_pseudo_functional(pseudo, h).numpy()
    vW_pseudo_derivative = weizsaecker_pseudo_functional_derivative(pseudo, h).numpy()
    
    vW_derivative = vW_pseudo_derivative[:, 1:-1]/(2*pseudo[:, 1:-1])
    vW_derivative = np.concatenate([vW_derivative[:, 0:1], vW_derivative, vW_derivative[:, -1:]], axis=1)

    return density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_kinetic_energy_density, vW_derivative


class DensityKineticEnergyDataset(quantumflow.QFDataset):

    def __init__(self, experiment, run_name, N, dtype, features, targets, subtract_von_weizsaecker, **kwargs):
        super().__init__(**kwargs)
        self.experiment = experiment
        self.run_name = run_name

        self.N = N
        self.dtype = dtype
        self.features_names = features
        self.targets_names = targets
        self.subtract_von_weizsaecker = subtract_von_weizsaecker

    def build(self):

        dataset_base_dir = os.path.abspath(os.path.join(self.run_dir, '../..', self.experiment))
        dataset_params = quantumflow.utils.load_yaml(os.path.join(dataset_base_dir, 'hyperparams.yaml'))[self.run_name]
        dataset_run_dir = os.path.join(dataset_base_dir, self.run_name)

        dataset = quantumflow.instantiate(dataset_params, run_dir=dataset_run_dir)
        dataset.build()

        potential = dataset.potential
        wavefunctions = dataset.wavefunctions
        energies = dataset.energies
        x = dataset.x
        h = dataset.h

        del dataset

        if self.N == 'all':
            all_data = [calculate_system_properties(potential, wavefunctions, energies, self.N, h) for N in range(1, wavefunctions.shape[2]+1)]
            density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_kinetic_energy_density, vW_derivative = \
                [np.concatenate([all_data[i][j] for i in range(len(all_data))], axis=0) for j in range(len(all_data[0]))]
        else:
            density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_kinetic_energy_density, vW_derivative = \
                calculate_system_properties(potential, wavefunctions, energies, self.N, h)
            
        self.dataset_size, self.discretisation_points = density.shape

        if self.subtract_von_weizsaecker:
            kinetic_energy -= vW_kinetic_energy
            kinetic_energy_density -= vW_kinetic_energy_density
            derivative -= vW_derivative
            
        if self.dtype == 'double' or self.dtype == 'float64':
            if potential.dtype == np.float32:
                raise ImportError("requested dtype={}, but dataset is saved with dtype={}, which is less precise.".format(params['dtype'], potential.dtype))
            self.dtype = np.float64
        elif self.dtype == 'float' or self.dtype == 'float32':
            self.dtype = np.float32
        else:
            raise ValueError(f"unknown dtype {self.dtype}")

        self.x = x.astype(self.dtype)
        self.h = h.astype(self.dtype)
        self.potential = potential.astype(self.dtype)
        self.density = density.astype(self.dtype)
        self.pseudo = np.sqrt(density).astype(self.dtype)
        self.energy = energy.astype(self.dtype)
        self.potential_energy = potential_energy.astype(self.dtype)
        self.kinetic_energy = kinetic_energy.astype(self.dtype)
        self.potential_energy_density = potential_energy_density.astype(self.dtype)
        self.kinetic_energy_density = kinetic_energy_density.astype(self.dtype)
        self.derivative = derivative.astype(self.dtype)
        self.vW_kinetic_energy = vW_kinetic_energy.astype(self.dtype)
        self.vW_kinetic_energy_density = vW_kinetic_energy_density.astype(self.dtype)
        self.vW_derivative = vW_derivative.astype(self.dtype)

        self.features = {}
        self.targets = {}

        def add_by_name(dictionary, name):
            if hasattr(self, name):
                dictionary[name] = getattr(self, name)
            else:
                raise KeyError(f"feature/target {name} does not exist or is not implemented.")

        for feature in self.features_names:
            add_by_name(self.features, feature)

        for target in self.targets_names:
            add_by_name(self.targets, target)

    def visualize(self, preview=5, figsize=(20, 3), dpi=None):
        import matplotlib.pyplot as plt

        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.x, np.transpose(self.density)[:, :preview]) # only plot first potentials
        plt.title("Input Density")
        plt.xlabel("x / bohr")
        plt.ylabel("Energy / hartree")
        plt.grid(which='major', axis='y', linestyle='--')
        plt.show()

        #----------------------------
       
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.x, np.transpose(self.kinetic_energy_density)[:, :preview]) # only plot first potentials
        plt.title("Output Kinetic Energy Density")
        plt.xlabel("x / bohr")
        plt.ylabel("Energy / hartree")
        plt.grid(which='major', axis='y', linestyle='--')
        plt.show()

        
        #----------------------------
       
        plt.figure(figsize=figsize, dpi=dpi)
        plt.plot(self.x, np.transpose(self.derivative)[:, :preview]) # only plot first potentials
        plt.title("Output Kinetic Energy Functional Derivative")
        plt.xlabel("x / bohr")
        plt.ylabel("Energy / hartree")
        plt.grid(which='major', axis='y', linestyle='--')
        plt.show()

        import pandas as pd
        from IPython.display import display

        df = pd.DataFrame(data={
            'energy': self.energy, 
            'potential_energy': self.potential_energy,
            'kinetic_energy': self.kinetic_energy,
            'vW_kinetic_energy': self.kinetic_energy,
            f'diff ({self.dtype})': self.energy - self.potential_energy - self.kinetic_energy
            })

        display(df)

