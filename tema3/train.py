import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
from config import load_config
from config.optimizers import get_optimizer
from config.schedulers import get_scheduler
from datasets import get_data_loaders
from models import get_model
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def initialize_training(config):
    """Initialize model, optimizer, scheduler, data loaders, and device based on config."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(config['model']).to(device)
    print("Loaded model:", config['model']['name'])
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    train_loader, test_loader = get_data_loaders(config['dataset'])
    return model, optimizer, scheduler, train_loader, test_loader, device


def train_one_epoch(model, train_loader, optimizer, device):
    """Train the model for one epoch and return the average loss."""
    model.train()
    total_loss = 0.0

    with tqdm(train_loader, desc="Training", unit="batch") as tbar:
        for batch in tbar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            tbar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"Average Training Loss: {avg_loss:.4f}")
    return avg_loss


def evaluate(model, test_loader, device):
    """Evaluate the model on the test set and return accuracy."""
    model.eval()
    correct, total = 0, 0

    with tqdm(test_loader, desc="Evaluating", unit="batch") as tbar:
        with torch.no_grad():
            for inputs, labels in tbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                tbar.set_postfix(accuracy=100 * correct / total)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")
    return accuracy


def run_training(config):
    """Run a single training session."""
    model, optimizer, scheduler, train_loader, test_loader, device = initialize_training(config)
    writer = SummaryWriter(log_dir="runs/experiment")
    wandb.init(project="YourProjectName", config=config)

    num_epochs = config['training']['epochs']
    early_stopping = config['training']['early_stopping']['enabled']
    patience = config['training']['early_stopping']['patience']
    min_delta = config['training']['early_stopping']['min_delta']

    best_accuracy = 0.0
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        avg_loss = train_one_epoch(model, train_loader, optimizer, device)
        writer.add_scalar("Training Loss", avg_loss, epoch)
        wandb.log({"Training Loss": avg_loss, "epoch": epoch})

        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        accuracy = evaluate(model, test_loader, device)
        writer.add_scalar("Validation Accuracy", accuracy, epoch)
        wandb.log({"Validation Accuracy": accuracy, "epoch": epoch})

        if early_stopping:
            if accuracy > best_accuracy + min_delta:
                best_accuracy = accuracy
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
                    break

    writer.close()


def main():
    config = load_config()

    if config['sweep']['enabled']:
        sweep_id = wandb.sweep(sweep=config['sweep'], project="YourProjectName")

        def sweep_train():
            wandb.init()
            run_training(wandb.config)

        wandb.agent(sweep_id, function=sweep_train)
    else:
        run_training(config)


if __name__ == '__main__':
    main()
