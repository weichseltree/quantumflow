# QuantumFlow

## Installation
This project can be run locally or on the cloud in Google Colab. To install, download the repository to your computer or into Google Drive. The default folder inside Google Drive is **Colab Projects/QuantumFlow**. If you want to save it in any other folder you will have to change the variable `project_path` in the first code cell in every notebooks to your path.

Some notebooks are best set to use a GPU instance in Google Colab. Every notebook will need create its own VM to run in, so make sure to close unused sessions via Runtime->Manage Sessions.

The required python packages are:

```
tensorboard>=2.0.0
matplotlib
ruamel.yaml
pandas
```

## Clean Notebooks

In order to keep notebooks in this project small and clean, there is a script to remove output data from the .ipynb files. Add this filter by running the following command inside the repository:

```
git config filter.clean_notebook.clean $PWD/clean_notebook.py
```
