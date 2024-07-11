from TAGLAS import get_dataset

def get_pretrain_dataset(name, root, **kwargs):
    # Can be expanded in the future for advanced dataset loading.
    return get_dataset(name, root, **kwargs)
