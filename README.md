# QuantumFlow

## Installation
This project can be run locally or on the cloud in Google Colab. To install, download the repository to your computer or into Google Drive. The default folder on Google Drive is **Projects/QuantumFlow**. If you want to save if in any other folder you will have to change the first code cells in the notebooks to use your path.

Since TensorFlow and Google Colab are constantly changing, long-term compatibility can not be guaranteed. It's best to use tensorflow==1.12.2 which can be installed in Google Colab with:

```
!pip install -q tensorflow-gpu==1.12.2
```
or locally without the '!'. As one can see, the notebooks are set to use a GPU instance in Google Colab.

## Usage

As a general rule, the notebooks should be run in the order of the numbers at the beginning of the filename. However, after generating the datasets and the shared files using all the files beginning with a 0 any other notebook should be able to run without having to run the others.

```
0_generate_datasets.ipynb
0b_recreate_dataset.ipynb
1_accuracy_checks_testing.ipynb
2_kernel_ridge_regession.ipynb
...
```
The notebook `0_generate_datasets.ipynb` will also write the shared `.py` files to a folder called `quantumflow` that will be imported by the rest of the notebooks like a python module.

All the files assume to be run with the current working directory being the folder they are in. Please make sure this is the case if you want to run the notebooks somewhere else.

#### Changes:
The variable `dataset` can be changed to run the same notebook for different datasets. All the hyperparameters for the Machine Learning parts can also be changed. 
