# config/config.py
import os
import yaml
import argparse


def load_config(config_path='config/default_config.yaml'):
    parser = argparse.ArgumentParser(description="Training Configuration")

    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--dataset', type=str, help='Dataset to use (MNIST, CIFAR10, CIFAR100)')
    parser.add_argument('--batch_size', type=int, help='Batch size for DataLoader')
    parser.add_argument('--cache', type=bool, help='Enable caching for DataLoader')
    parser.add_argument('--augment_random_flip', type=bool, help='Enable random flip augmentation')
    parser.add_argument('--augment_random_crop', type=bool, help='Enable random crop augmentation')
    parser.add_argument('--augment_random_rotation', type=bool, help='Enable random rotation augmentation')
    parser.add_argument('--augment_color_jitter', type=bool, help='Enable color jitter augmentation')
    parser.add_argument('--normalize', type=bool, help='Enable normalization')

    args = parser.parse_args()

    config_file = args.config if args.config else config_path
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    config['dataset'] = args.dataset or os.getenv('DATASET', config.get('dataset', 'CIFAR10'))
    config['batch_size'] = args.batch_size or int(os.getenv('BATCH_SIZE', config.get('batch_size', 64)))
    config['cache'] = args.cache if args.cache is not None else bool(os.getenv('CACHE', config.get('cache', False)))

    if 'augmentations' not in config:
        config['augmentations'] = {}

    config['augmentations']['random_flip'] = args.augment_random_flip if args.augment_random_flip is not None else bool(
        os.getenv('AUGMENT_RANDOM_FLIP', config['augmentations'].get('random_flip', False)))
    config['augmentations']['random_crop'] = args.augment_random_crop if args.augment_random_crop is not None else bool(
        os.getenv('AUGMENT_RANDOM_CROP', config['augmentations'].get('random_crop', False)))
    config['augmentations'][
        'random_rotation'] = args.augment_random_rotation if args.augment_random_rotation is not None else bool(
        os.getenv('AUGMENT_RANDOM_ROTATION', config['augmentations'].get('random_rotation', False)))
    config['augmentations'][
        'color_jitter'] = args.augment_color_jitter if args.augment_color_jitter is not None else bool(
        os.getenv('AUGMENT_COLOR_JITTER', config['augmentations'].get('color_jitter', False)))
    config['augmentations']['normalization'] = args.normalize if args.normalize is not None else bool(
        os.getenv('NORMALIZE', config['augmentations'].get('normalization', True)))

    return config
