import os
import sys
import importlib


def instantiate(params, *args, **kwargs):
    class_ = get_class(params['class'])
    params = dict(params)
    for key, param in params.items():
        if isinstance(param, dict) and 'class' in param and param.get('instantiate', False):
            del param['instantiate']
            params[key] = instantiate(param)
    params.update(kwargs)
    del params['class']
    
    try:
        return class_(*args, **params)
    except TypeError as e:
        reraise(type(e), type(e)(str(e) + f"\n           class: {class_.__name__}"), sys.exc_info()[2])


def get_class(class_):
    try:
        module_name, class_name = class_.rsplit('.', 1)
        return getattr(importlib.import_module(module_name), class_name)
    except ValueError as e:
        reraise(type(e), type(e)(str(e) + f" | class = {class_}"), sys.exc_info()[2])

        
def reraise(exc_type, exc_value, exc_traceback=None):
    if exc_value is None:
        exc_value = exc_type()
    if exc_value.__traceback__ is not exc_traceback:
        raise exc_value.with_traceback(exc_traceback) from None
    raise exc_value from None

    
def check_dir_exists(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError("The provided directory '{}' does not exist.".format(directory))


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return file_path

    
from . import utils
from . import layers
from .definitions import Dataset, Model
from .__main__ import build_dataset, train_model

def get_experiment_dir_and_params(experiment, run_name):
    project_path = os.path.abspath(os.path.join(os.path.dirname(utils.__file__), '../../'))
    run_dir = os.path.join(project_path, "experiments", experiment, run_name)
    params_file = os.path.join(project_path, 'experiments', experiment, f'{experiment}.yaml')
    all_params = utils.load_yaml(params_file)
    assert run_name in all_params, f"Couldn't find entry {run_name} in {params_file}. \npossible entries are: {list(all_params.keys())}."
    return run_dir, all_params[run_name]


def get_dataset_dir_and_params(dataset, run_name):
    project_path = os.path.abspath(os.path.join(os.path.dirname(utils.__file__), '../../'))
    run_dir = os.path.join(project_path, "datasets", dataset, run_name)
    params_file = os.path.join(project_path, 'datasets', dataset, f'{dataset}.yaml')
    all_params = utils.load_yaml(params_file)
    assert run_name in all_params, f"Couldn't find entry {run_name} in {params_file}. \npossible entries are: {list(all_params.keys())}."
    return run_dir, all_params[run_name]
