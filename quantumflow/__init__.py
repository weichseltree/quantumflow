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

    
class Dataset():
    def __init__(self, run_dir, **kwargs):
        self.run_dir = run_dir

    def build(self, force=False):
        pass

    def visualize(self):
        pass

    def iter(self):
        raise NotImplementedError()


from . import utils