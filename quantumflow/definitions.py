   
class Dataset():
    def __init__(self, run_dir, **kwargs):
        self.run_dir = run_dir

    def build(self, force=False):
        return self

    def visualize(self):
        pass

    def iter(self):
        raise NotImplementedError()
