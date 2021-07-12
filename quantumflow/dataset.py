import os
import sys

import quantumflow

class QFDataset():
    def __init__(self, run_dir, **kwargs):
        self.run_dir = run_dir

    def build(self):
        pass

    def visualize(self):
        pass

    def iter(self):
        raise NotImplementedError()
    