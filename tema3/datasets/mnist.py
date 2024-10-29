# datasets/mnist.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .transform_utils import create_transform


def get_mnist_data_loader(batch_size, augment_config, cache=True):
    transform = create_transform(augment_config)

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=cache)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=cache)

    return train_loader, test_loader
