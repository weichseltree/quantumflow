import h5py

def save_hdf5(filename, datasets, attrs=None, compression="gzip"):

    with h5py.File(filename, "w") as f:
        if attrs is not None:
            for attr_name, attr_data in attrs.items():
                f.attrs[attr_name] = attr_data

        for array_name, array_data in datasets.items():
            f.create_dataset(array_name, data=array_data, compression=compression)

def load_hdf5(filename):
    with h5py.File(filename, 'r') as f:
        attrs = dict(f.attrs)
        array_dict = {key:f[key][()] for key in f.keys()}

    if len(attrs) == 0:
        return array_dict
    else:
        return array_dict, attrs
