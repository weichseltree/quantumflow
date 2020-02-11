# QuantumFlow

## Installation
This project can be run locally or on the cloud in Google Colab. To install, download the repository to your computer or into Google Drive. The default folder on Google Drive is **Projects/QuantumFlow**. If you want to save if in any other folder you will have to change the first code cells in the notebooks to use your path.

The notebooks are best set to use a GPU instance in Google Colab. Note that every notebook will need create own VM to run in. You should close unused Sessions via Runtime->Manage Sessions.

The required python packages are:

```
tensorboard>=2.0.0
matplotlib
ruamel.yaml
pandas
scikit-learn
```

Informations about some packages are available here:
https://www.tensorflow.org/install
https://ipywidgets.readthedocs.io/en/latest/user_install.htm

If you use conda, you should be able to install the required packages like this:

`conda install -c conda-forge tensorboard`

`conda install -c conda-forge matplotlib`

`conda install -c conda-forge ruamel.yaml`

`conda install -c conda-forge pandas`

`conda install -c conda-forge scikit-learn`

## Usage

As a general rule, the notebooks should be run in the order of the numbers at the beginning of the filename. However, after generating the datasets and the shared files, any other notebook should be able to run without having to run the others.

```
0_create_shared_project_files.ipynb
1a_generate_datasets.ipynb
1b_recreate_dataset.ipynb
1c_analyze_dataset.ipynb
2_kernel_ridge_regession.ipynb
...
```
The notebook `0_create_shared_project_files.ipynb` will write the shared `.py` files to a folder called `quantumflow` - if they dont exist already - that will be imported by the rest of the notebooks like a python module.

All the files assume to be run with the current working directory being the folder they are in. Please make sure this is the case if you want to run the notebooks somewhere else.

