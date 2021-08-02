import ruamel.yaml
from ruamel.yaml import YAML
import os
import sys

def load_yaml(filename):
    with open(filename, 'r') as f:
        return YAML(typ='safe').load(f)
    return yaml_dict


def save_yaml(filename, python_object):
    success = False
    while not success:
        try:
            with open(filename, 'w') as f:
                yaml = YAML()
                yaml.indent(mapping=2, sequence=4, offset=2)
                yaml.dump(python_object, f)
                success = True
        
        except OSError as e:
            print(repr(e), file=sys.stderr)
            time.sleep(1)
