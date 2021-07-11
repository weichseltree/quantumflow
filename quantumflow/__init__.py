from quantumflow import utils


def instantiate(params, *args, **kwargs):
    class_ = get_class(params['class'])
    params = dict(params)
    params.update(kwargs)
    del params['class']
    return class_(*args, **params)


def get_class(class_):
    import importlib
    try:
        module_name, class_name = class_.rsplit('.', 1)
        return getattr(importlib.import_module(module_name), class_name)
    except ValueError as e:
        agent42.reraise(type(e), type(e)(str(e) + f" | class = {class_}"), sys.exc_info()[2])

