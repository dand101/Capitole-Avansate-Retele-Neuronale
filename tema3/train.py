import torch
from torch import GradScaler, nn
from tqdm import tqdm
import wandb
from config import load_config
from config.optimizers import get_optimizer
from config.schedulers import get_scheduler
from datasets import get_data_loaders
from models import get_model
from torchvision.transforms import v2
from torch.backends import cudnn

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
cudnn.benchmark = True


def set_dataset_sweep(is_enabled, name, batch_size, cache):
    return {
        'name': name,
        'batch_size': batch_size,
        'cache': cache,
        'augmentations': {
            'random_flip': is_enabled,
            'random_crop': is_enabled,
            'random_rotation': is_enabled,
            'color_jitter': is_enabled,
        }}


def initialize_training(config, sweep_enabled=False):
    """Initialize model, optimizer, scheduler, data loaders, and device based on config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if sweep_enabled is True:
        model_name = config.get('model.name')
        pretrained = config.get('model_pretrained', False)
        num_classes = config.get('num_classes', 100)

        optimizer_type = config.get('optimizer.type')
        optimizer_learning_rate = config.get('optimizer.learning_rate')
        optimizer_weight_decay = config.get('optimizer.weight_decay', 0.0001)
        optimizer_momentum = config.get('optimizer.momentum', 0.01)
        optimizer_nesterov = config.get('optimizer.nesterov', False)

        scheduler_type = config.get('scheduler.type')
        scheduler_step_size = config.get('scheduler.step_size', 10)
        scheduler_gamma = config.get('scheduler.gamma', 0.1)
        scheduler_patience = config.get('scheduler.patience', 5)
        scheduler_momentum = config.get('scheduler.momentum', 0.01)
        scheduler_nesterov = config.get('scheduler.nesterov', False)
        scheduler_T_max = config.get('scheduler.T_max', 50)

    else:
        model_name = config['model']['name']
        pretrained = config['model'].get('pretrained', True)
        num_classes = config['model'].get('num_classes', 100)

        optimizer_type = config['optimizer']['type']
        optimizer_learning_rate = config['optimizer']['learning_rate']
        optimizer_weight_decay = config['optimizer'].get('weight_decay', 0.0001)
        optimizer_momentum = config['optimizer'].get('momentum', 0.01)
        optimizer_nesterov = config['optimizer'].get('nesterov', False)

        scheduler_type = config['scheduler']['type']
        scheduler_step_size = config['scheduler'].get('step_size', 10)
        scheduler_gamma = config['scheduler'].get('gamma', 0.1)
        scheduler_patience = config['scheduler'].get('patience', 5)
        scheduler_momentum = config['scheduler'].get('momentum', 0.01)
        scheduler_nesterov = config['scheduler'].get('nesterov', False)
        scheduler_T_max = config['scheduler'].get('T_max', 50)

    model_config = {"name": model_name, "pretrained": pretrained, "num_classes": num_classes}
    model = get_model(model_config).to(device)

    optimizer_config = {"optimizer": {
        "type": optimizer_type,
        "learning_rate": optimizer_learning_rate,
        "weight_decay": optimizer_weight_decay,
        "momentum": optimizer_momentum,
        "nesterov": optimizer_nesterov

    }}

    optimizer = get_optimizer(model, optimizer_config)

    scheduler_config = {"scheduler": {
        "type": scheduler_type,
        "step_size": scheduler_step_size,
        "gamma": scheduler_gamma,
        "patience": scheduler_patience,
        "momentum": scheduler_momentum,
        "nesterov": scheduler_nesterov,
        "T_max": scheduler_T_max
    }}

    scheduler = get_scheduler(optimizer, scheduler_config)

    if sweep_enabled:
        train_loader, test_loader = get_data_loaders(
            set_dataset_sweep(config['augmentations_enabled'], config['dataset.name'], config['dataset.batch_size'],
                              config['dataset.cache']))
    else:
        train_loader, test_loader = get_data_loaders(config['dataset'])

    return model, optimizer, scheduler, train_loader, test_loader, device


cutMix = v2.CutMix(num_classes=100)
mixUp = v2.MixUp(num_classes=100)
rand_choice = v2.RandomChoice([cutMix, mixUp])
criterion = nn.CrossEntropyLoss()


def train_one_epoch(model, train_loader, optimizer, scaler, device, other_augmentation=False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(train_loader, desc="Training", unit="batch") as tbar:
        for inputs, labels in tbar:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            if other_augmentation:
                inputs, labels = rand_choice(inputs, labels)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * inputs.size(0)

            if labels.dim() > 1:
                predicted = outputs.argmax(1)
                true_labels = labels.argmax(1)
                correct += (predicted == true_labels).sum().item()
            else:
                predicted = outputs.argmax(1)
                correct += (predicted == labels).sum().item()

            total += labels.size(0)
            tbar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(f"Average Training Loss: {avg_loss:.4f}, Training Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def evaluate(model, test_loader, device):
    """Evaluate the model on the test set and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
        with torch.no_grad():
            for inputs, labels in tbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tbar.set_postfix(loss=loss.item(), accuracy=100.0 * correct / total)

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    print(f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy


def run_training(config, sweep_enabled=False):
    """Run a single training session."""
    model, optimizer, scheduler, train_loader, test_loader, device = initialize_training(config, sweep_enabled)
    wandb.init(project="YourProjectName", config=config)

    if sweep_enabled:
        num_epochs = config.get('training.epochs')
        early_stopping = config.get('training.early_stopping.enabled')
        patience = config.get('training.early_stopping.patience')
        min_delta = config.get('training.early_stopping.min_delta')
        other_augmentation = config.get('dataset.other_augmentation')
    else:
        num_epochs = config['training']['epochs']
        early_stopping = config['training']['early_stopping']['enabled']
        patience = config['training']['early_stopping']['patience']
        min_delta = config['training']['early_stopping']['min_delta']
        other_augmentation = config['dataset']['other_augmentations']

    best_accuracy = 0.0
    epochs_no_improve = 0
    scaler = GradScaler()
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, scaler, device, other_augmentation)
        val_loss, val_accuracy = evaluate(model, test_loader, device)

        wandb.log({
            "epoch": epoch,
            "Training Loss": train_loss,
            "Training Accuracy": train_accuracy,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy
        })

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if early_stopping:
            if val_accuracy > best_accuracy + min_delta:
                best_accuracy = val_accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
                    break


def main():
    config = load_config()

    if config['sweep']['enabled']:
        sweep_id = wandb.sweep(sweep=config['sweep'], project="YourProjectName")

        def sweep_train():
            wandb.init()
            run_training(wandb.config, config['sweep']['enabled'])

        wandb.agent(sweep_id, function=sweep_train)
    else:
        run_training(config, config['sweep']['enabled'])


if __name__ == '__main__':
    main()

# python train.py --model lenet --dataset MNIST --batch_size 256 --epochs 10 --learning_rate 0.001
