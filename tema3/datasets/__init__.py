# datasets/__init__.py
from .mnist import get_mnist_data_loader
from .cifar10 import get_cifar10_data_loader
from .cifar100 import get_cifar100_data_loader


def get_data_loaders(dataset_config):
    dataset_name = dataset_config['name']
    batch_size = dataset_config['batch_size']
    augment_config = dataset_config['augmentations']
    augment_config['dataset'] = dataset_name
    cache = dataset_config['cache']

    if dataset_name == 'MNIST':
        return get_mnist_data_loader(batch_size, augment_config, cache)
    elif dataset_name == 'CIFAR10':
        return get_cifar10_data_loader(batch_size, augment_config, cache)
    elif dataset_name == 'CIFAR100':
        return get_cifar100_data_loader(batch_size, augment_config, cache)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
