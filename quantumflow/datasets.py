import quantumflow


class QFDataset():
    def __init__(self, dataset_file, params):
        extension = dataset_file.split('.')[-1]
        
        if extension in ['pkl', 'pickle']:
            import pickle
            with open(dataset_file, 'rb') as f:
                x, h, potential, wavefunctions, energies = pickle.load(f).values()

        elif extension in ['hdf5', 'h5']:
            import h5py
            with h5py.File(dataset_file, 'r') as f:
                x = f.attrs['x']
                h = f.attrs['h']
                potential = f['potential'][()]
                wavefunctions = f['wavefunctions'][()]
                energies = f['energies'][()]
        else:
            raise NotImplementedError('File extension missing or not supported.')  

        if params['N'] == 'all':
            all_data = [calculate_system_properties(potential, wavefunctions, energies, N, h) for N in range(1, wavefunctions.shape[2]+1)]
            density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_kinetic_energy_density, vW_derivative = \
                [np.concatenate([all_data[i][j] for i in range(len(all_data))], axis=0) for j in range(len(all_data[0]))]
        else:
            density, energy, potential_energy, kinetic_energy, potential_energy_density, kinetic_energy_density, derivative, vW_kinetic_energy, vW_kinetic_energy_density, vW_derivative = \
                calculate_system_properties(potential, wavefunctions, energies, params['N'], h)
            
        self.dataset_size, self.discretisation_points = density.shape

        if params.get('subtract_von_weizsaecker', False):
            kinetic_energy -= vW_kinetic_energy
            kinetic_energy_density -= vW_kinetic_energy_density
            derivative -= vW_derivative
            
        if params['dtype'] == 'double' or params['dtype'] == 'float64':
            if potential.dtype == np.float32:
                raise ImportError("requested dtype={}, but dataset is saved with dtype={}, which is less precise.".format(params['dtype'], potential.dtype))
            self.dtype = np.float64
        elif params['dtype'] == 'float' or params['dtype'] == 'float32':
            self.dtype = np.float32
        else:
            raise ValueError('unknown dtype {}'.format(params['dtype']))

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

        if not 'features' in params or not 'targets' in params: 
            return

        self.features = {}
        self.targets = {}

        def add_by_name(dictionary, name):
            if hasattr(self, name):
                dictionary[name] = getattr(self, name)
            else:
                raise KeyError('feature/target {} does not exist or is not implemented.'.format(name))

        for feature in params['features']:
            add_by_name(self.features, feature)

        for target in params['targets']:
            add_by_name(self.targets, target)

    def get_params(self, shapes=True, h=True, mean=False, full=False):
        import numpy as np

        params = {}
        if h:
            params['h'] = self.h

        if shapes:
            params['features_shape'] = {name:feature.shape[1:] for name, feature in self.features.items()}
            params['targets_shape'] = {name:target.shape[1:] for name, target in self.targets.items()}

        if mean:
            params['features_mean'] = {name:np.mean(feature, axis=0) for name, feature in self.features.items()}
            params['targets_mean'] = {name:np.mean(target, axis=0) for name, target in self.targets.items()}

        if full:
            params['features'] = {name:feature for name, feature in self.features.items()}
            params['targets'] = {name:target for name, target in self.targets.items()}

        return params