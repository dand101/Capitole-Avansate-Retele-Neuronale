# datasets/cifar10.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from .cached_dataset import CachedDataset
from .transform_utils import create_transform


def get_cifar10_data_loader(batch_size, augment_config, cache=True):
    transform = create_transform(augment_config)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)

    if cache:
        train_dataset = CachedDataset(train_dataset, transform=transform)
    else:
        train_dataset.transform = transform

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.25, 0.25, 0.25])
    ])
    test_dataset = CachedDataset(test_dataset, transform=test_transform) if cache else test_dataset
    test_dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4,
                              persistent_workers=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4,
                             persistent_workers=True, prefetch_factor=2)

    return train_loader, test_loader
