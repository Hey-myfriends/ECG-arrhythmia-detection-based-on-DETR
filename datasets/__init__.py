from .MIT_BIH_dataset import MIT_BIH_dataset, collate_fn

def build_dataset(rootpath, samples):
    return MIT_BIH_dataset(rootpath, samples)