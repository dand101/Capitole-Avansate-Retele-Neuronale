import yaml
import argparse


def load_yaml(config_path='config/default_config.yaml'):
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def configure_dataset(config, args):
    """Configure dataset settings based on YAML and command-line overrides."""
    dataset_config = config.get('dataset', {})
    dataset_config['name'] = args.dataset if args.dataset is not None else dataset_config.get('name')
    dataset_config['batch_size'] = args.batch_size if args.batch_size is not None else dataset_config.get('batch_size',
                                                                                                          64)
    dataset_config['cache'] = args.cache if args.cache is not None else dataset_config.get('cache', True)

    augment_config = dataset_config.get('augmentations', {})
    augment_config[
        'random_flip'] = args.augment_random_flip if args.augment_random_flip is not None else augment_config.get(
        'random_flip', False)
    augment_config[
        'random_crop'] = args.augment_random_crop if args.augment_random_crop is not None else augment_config.get(
        'random_crop', False)
    augment_config[
        'random_rotation'] = args.augment_random_rotation if args.augment_random_rotation is not None else augment_config.get(
        'random_rotation', False)
    augment_config[
        'color_jitter'] = args.augment_color_jitter if args.augment_color_jitter is not None else augment_config.get(
        'color_jitter', False)
    augment_config['normalization'] = args.normalize if args.normalize is not None else augment_config.get(
        'normalization', True)
    dataset_config['augmentations'] = augment_config

    return dataset_config


def configure_model(config, args):
    """Configure model settings based on YAML and command-line overrides."""
    model_config = config.get('model', {})
    model_config['name'] = args.model if args.model is not None else model_config.get('name', 'resnet18')
    model_config['num_classes'] = args.num_classes if args.num_classes is not None else model_config.get('num_classes',
                                                                                                         10)
    model_config['pretrained'] = args.pretrained if args.pretrained is not None else model_config.get('pretrained',
                                                                                                      True)
    return model_config


def configure_optimizer(config, args):
    """Configure optimizer settings based on YAML and command-line overrides."""
    optimizer_config = config.get('optimizer', {})
    optimizer_config['type'] = args.optimizer if args.optimizer is not None else optimizer_config.get('type', 'SGD')
    optimizer_config['learning_rate'] = args.learning_rate if args.learning_rate is not None else optimizer_config.get(
        'learning_rate', 0.001)
    optimizer_config['momentum'] = args.momentum if args.momentum is not None else optimizer_config.get('momentum', 0.9)
    optimizer_config['weight_decay'] = args.weight_decay if args.weight_decay is not None else optimizer_config.get(
        'weight_decay', 0.0001)
    optimizer_config['nesterov'] = args.nesterov if args.nesterov is not None else optimizer_config.get('nesterov',
                                                                                                        False)
    return optimizer_config


def configure_scheduler(config, args):
    """Configure scheduler settings based on YAML and command-line overrides."""
    scheduler_config = config.get('scheduler', {})
    scheduler_config['type'] = args.scheduler if args.scheduler is not None else scheduler_config.get('type', 'StepLR')
    scheduler_config['step_size'] = args.step_size if args.step_size is not None else scheduler_config.get('step_size',
                                                                                                           10)
    scheduler_config['gamma'] = args.gamma if args.gamma is not None else scheduler_config.get('gamma', 0.1)
    scheduler_config['patience'] = args.patience if args.patience is not None else scheduler_config.get('patience', 5)
    return scheduler_config


def configure_training(config, args):
    """Configure training settings, including epochs and early stopping."""
    training_config = config.get('training', {})
    training_config['epochs'] = args.epochs if args.epochs is not None else training_config.get('epochs', 20)

    early_stopping_config = training_config.get('early_stopping', {})
    early_stopping_config[
        'enabled'] = args.early_stopping if args.early_stopping is not None else early_stopping_config.get('enabled',
                                                                                                           False)
    early_stopping_config[
        'patience'] = args.early_stopping_patience if args.early_stopping_patience is not None else early_stopping_config.get(
        'patience', 5)
    early_stopping_config[
        'min_delta'] = args.early_stopping_min_delta if args.early_stopping_min_delta is not None else early_stopping_config.get(
        'min_delta', 0.01)
    training_config['early_stopping'] = early_stopping_config

    return training_config


def configure_sweep(config):
    """Configure the sweep settings if sweep is enabled."""
    sweep_config = config.get('sweep', {})
    sweep_config['enabled'] = sweep_config.get('enabled', False)

    if sweep_config['enabled']:
        sweep_config['method'] = sweep_config.get('method', 'grid')
        sweep_config['metric'] = sweep_config.get('metric', {
            'name': 'Validation Accuracy',
            'goal': 'maximize'
        })
        sweep_config['parameters'] = sweep_config.get('parameters', {})
    return sweep_config


def load_config():
    """Main configuration loader that combines dataset, model, optimizer, and scheduler configurations."""
    parser = argparse.ArgumentParser(description="Training Configuration")

    parser.add_argument('--dataset', type=str, help='Dataset to use (MNIST, CIFAR10, CIFAR100)')
    parser.add_argument('--batch_size', type=int, help='Batch size for DataLoader')
    parser.add_argument('--cache', type=bool, help='Enable caching for DataLoader')
    parser.add_argument('--augment_random_flip', type=bool, help='Enable random flip augmentation')
    parser.add_argument('--augment_random_crop', type=bool, help='Enable random crop augmentation')
    parser.add_argument('--augment_random_rotation', type=bool, help='Enable random rotation augmentation')
    parser.add_argument('--augment_color_jitter', type=bool, help='Enable color jitter augmentation')
    parser.add_argument('--normalize', type=bool, help='Enable normalization')

    parser.add_argument('--model', type=str, help='Model name (e.g., resnet18)')
    parser.add_argument('--num_classes', type=int, help='Number of output classes')
    parser.add_argument('--pretrained', type=bool, help='Use pretrained model weights')

    parser.add_argument('--optimizer', type=str, help='Optimizer type (e.g., SGD, Adam)')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--momentum', type=float, help='Momentum for SGD')
    parser.add_argument('--weight_decay', type=float, help='Weight decay for regularization')
    parser.add_argument('--nesterov', type=bool, help='Use Nesterov momentum (only for SGD)')

    parser.add_argument('--scheduler', type=str, help='Scheduler type (e.g., StepLR, ReduceLROnPlateau)')
    parser.add_argument('--step_size', type=int, help='Step size for StepLR')
    parser.add_argument('--gamma', type=float, help='Learning rate decay factor')
    parser.add_argument('--patience', type=int, help='Patience for ReduceLROnPlateau')

    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--early_stopping', type=bool, help='Enable early stopping')
    parser.add_argument('--early_stopping_patience', type=int, help='Patience for early stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, help='Minimum delta for early stopping')

    args = parser.parse_args()

    config = load_yaml()

    config['dataset'] = configure_dataset(config, args)
    config['model'] = configure_model(config, args)
    config['optimizer'] = configure_optimizer(config, args)
    config['scheduler'] = configure_scheduler(config, args)
    config['training'] = configure_training(config, args)
    config['sweep'] = configure_sweep(config)

    return config
