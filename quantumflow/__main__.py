import sys


def build_dataset(dataset, run_name, force=False):
    import quantumflow
    run_dir, params = quantumflow.get_dataset_dir_and_params(dataset, run_name)
    dataset = quantumflow.instantiate(params, run_dir=run_dir)
    dataset.build(force=force)
    return dataset
    
    
if __name__ == '__main__':
    if len(sys.argv) == 1:
        print('USAGE: build_dataset dataset run_name')
        sys.exit(1)
    
    if len(sys.argv) == 4:
        command = sys.argv[1]
        if command == 'build_dataset':
            build_dataset(sys.argv[2], sys.argv[3], force=True)
