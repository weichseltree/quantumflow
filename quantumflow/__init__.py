from quantumflow import utils
from quantumflow.dataset import QFDataset

def instantiate(params, *args, **kwargs):
    class_ = get_class(params['class'])
    params = dict(params)
    params.update(kwargs)
    del params['class']
    return class_(*args, **params)

def reraise(exc_type, exc_value, exc_traceback=None):
    if exc_value is None:
        exc_value = exc_type()
    if exc_value.__traceback__ is not exc_traceback:
        raise exc_value.with_traceback(exc_traceback) from None
    raise exc_value from None

def get_class(class_):
    import importlib
    try:
        module_name, class_name = class_.rsplit('.', 1)
        return getattr(importlib.import_module(module_name), class_name)
    except ValueError as e:
        import sys
        reraise(type(e), type(e)(str(e) + f" | class = {class_}"), sys.exc_info()[2])

