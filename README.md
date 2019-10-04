# QuantumFlow

## Installation
This project can be run locally or on the cloud in Google Colab. To install, download the repository to your computer or into Google Drive. The default folder on Google Drive is **Projects/QuantumFlow**. If you want to save if in any other folder you will have to change the first code cells in the notebooks to use your path.

Since TensorFlow and Google Colab are constantly changing, long-term compatibility can not be guaranteed. It's best to use tensorflow==1.13.1 which can be installed in Google Colab with:

```
!pip install -q tensorflow-gpu==1.13.1
```
or locally without the '!'. 

The notebooks are best set to use a GPU instance in Google Colab, the neural network training code can also use a TPU instance.

## Usage

As a general rule, the notebooks should be run in the order of the numbers at the beginning of the filename. However, after generating the datasets and the shared files using all the files beginning with a 0 any other notebook should be able to run without having to run the others.

```
0_define_helper_functions.ipynb
1a_generate_datasets.ipynb
1b_recreate_dataset.ipynb
1c_accuracy_checks_testing.ipynb
2_kernel_ridge_regession.ipynb
...
```
The notebook `0_define_helper_functions.ipynb` will also write the shared `.py` files to a folder called `quantumflow` - if they dont exist already - that will be imported by the rest of the notebooks like a python module.

All the files assume to be run with the current working directory being the folder they are in. Please make sure this is the case if you want to run the notebooks somewhere else.

#### Changes:
The variable `dataset` can be changed to run the same notebook for different datasets. All the hyperparameters for the Machine Learning parts can also be changed. 
