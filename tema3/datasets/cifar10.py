# datasets/cifar10.py
from torchvision import datasets, transforms  # Import transforms explicitly
from torch.utils.data import DataLoader
from .transform_utils import create_transform


def get_cifar10_data_loader(batch_size, augment_config, cache=True):
    transform = create_transform(augment_config)

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=cache)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=cache)

    return train_loader, test_loader
