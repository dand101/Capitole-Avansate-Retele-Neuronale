# datasets/transform_utils.py
from torchvision import transforms


def create_transform(augment_config):
    transform_list = []

    if augment_config.get('random_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip())

    if augment_config.get('random_crop', False):
        transform_list.append(transforms.RandomCrop(32, padding=4))

    if augment_config.get('random_rotation', False):
        transform_list.append(transforms.RandomRotation(15))

    if augment_config.get('color_jitter', False):
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))

    transform_list.append(transforms.ToTensor())

    if augment_config.get('normalization', True):
        if augment_config['dataset'] in ['CIFAR10', 'CIFAR100']:
            transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        elif augment_config['dataset'] == 'MNIST':
            transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))

    return transforms.Compose(transform_list)
