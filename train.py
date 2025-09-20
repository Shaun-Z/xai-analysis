#!/usr/bin/env python3
"""
Training script for XAI analysis models.
Supports flexible model and dataset selection with Wandb logging.
"""

import argparse
import importlib
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import wandb

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train XAI models')
    
    # Model and dataset selection
    parser.add_argument('--model', type=str, required=True,
                       help='Model name (Python module in models/ folder)')
    parser.add_argument('--dataset', type=str, required=True,
                       help='Dataset name (Python module in datasets/ folder)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                       help='Weight decay (L2 regularization)')
    
    # Optimizer and scheduler
    parser.add_argument('--optimizer', type=str, default='Adam',
                       choices=['Adam', 'SGD', 'RMSprop'],
                       help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'],
                       help='Learning rate scheduler')
    
    # Checkpoint and logging
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume training from checkpoint path')
    
    # Hardware and misc
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    # Wandb logging
    parser.add_argument('--wandb-project', type=str, default='xai-analysis',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity (username or team)')
    parser.add_argument('--no-wandb', action='store_true',
                       help='Disable Wandb logging')
    
    return parser.parse_args()


def get_device(device_arg):
    """Determine the best available device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    else:
        return torch.device(device_arg)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)


def load_model(model_name, device):
    """Dynamically load model from models/ folder."""
    try:
        model_module = importlib.import_module(f'models.{model_name}')
        # Look for common model creation functions/classes
        if hasattr(model_module, 'create_model'):
            model = model_module.create_model()
        elif hasattr(model_module, 'Model'):
            model = model_module.Model()
        elif hasattr(model_module, 'Net'):
            model = model_module.Net()
        else:
            # Try to find the first class that inherits from nn.Module
            for attr_name in dir(model_module):
                attr = getattr(model_module, attr_name)
                if (isinstance(attr, type) and 
                    issubclass(attr, nn.Module) and 
                    attr != nn.Module):
                    model = attr()
                    break
            else:
                raise ValueError(f"No suitable model class found in {model_name}")
        
        return model.to(device)
    except ImportError:
        raise ImportError(f"Could not import model '{model_name}' from models/{model_name}.py")


def load_dataset(dataset_name, batch_size, num_workers):
    """Dynamically load dataset from datasets/ folder."""
    try:
        dataset_module = importlib.import_module(f'datasets.{dataset_name}')
        
        # Look for common dataset creation functions
        if hasattr(dataset_module, 'get_dataloaders'):
            train_loader, val_loader = dataset_module.get_dataloaders(
                batch_size=batch_size, num_workers=num_workers
            )
        elif hasattr(dataset_module, 'create_dataloaders'):
            train_loader, val_loader = dataset_module.create_dataloaders(
                batch_size=batch_size, num_workers=num_workers
            )
        else:
            raise ValueError(f"No suitable dataloader function found in {dataset_name}")
        
        return train_loader, val_loader
    except ImportError:
        raise ImportError(f"Could not import dataset '{dataset_name}' from datasets/{dataset_name}.py")


def create_optimizer(model, optimizer_name, lr, weight_decay, momentum=0.9):
    """Create optimizer based on name."""
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_scheduler(optimizer, scheduler_name, epochs):
    """Create learning rate scheduler based on name."""
    if scheduler_name is None:
        return None
    elif scheduler_name == 'StepLR':
        return optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    elif scheduler_name == 'CosineAnnealingLR':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_name == 'ReduceLROnPlateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    else:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        # Log every 100 batches
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc


def save_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, experiment_name):
    """
    Save model checkpoint in experiment-specific subfolder.
    
    Creates a folder structure: checkpoint_dir/experiment_name/epoch_X.pt
    This keeps different experiments organized and prevents mixing checkpoints.
    """
    # Create experiment-specific subfolder
    experiment_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time()
    }
    
    checkpoint_path = os.path.join(
        experiment_checkpoint_dir, 
        f'epoch_{epoch}.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')
    
    return checkpoint_path


def save_best_checkpoint(model, optimizer, scheduler, epoch, loss, checkpoint_dir, experiment_name):
    """
    Save best model checkpoint with special naming.
    
    Saves as: checkpoint_dir/experiment_name/best_model.pt
    """
    # Create experiment-specific subfolder
    experiment_checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
    os.makedirs(experiment_checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss,
        'timestamp': time.time()
    }
    
    checkpoint_path = os.path.join(
        experiment_checkpoint_dir, 
        'best_model.pt'
    )
    torch.save(checkpoint, checkpoint_path)
    print(f'Best model checkpoint saved: {checkpoint_path}')
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    return start_epoch


def main():
    args = parse_args()
    
    # Set up experiment name
    if args.experiment_name is None:
        args.experiment_name = f"{args.model}_{args.dataset}_{int(time.time())}"
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Determine device
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    # Initialize Wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.experiment_name,
            config=vars(args)
        )
    
    try:
        # Load model and dataset
        print(f"Loading model: {args.model}")
        model = load_model(args.model, device)
        
        print(f"Loading dataset: {args.dataset}")
        train_loader, val_loader = load_dataset(args.dataset, args.batch_size, args.num_workers)
        
        # Create optimizer and scheduler
        optimizer = create_optimizer(
            model, args.optimizer, args.lr, args.weight_decay, args.momentum
        )
        scheduler = create_scheduler(optimizer, args.scheduler, args.epochs)
        
        # Define loss function
        criterion = nn.CrossEntropyLoss()
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        
        print(f"Starting training for {args.epochs} epochs...")
        print(f"Model: {args.model}, Dataset: {args.dataset}")
        print(f"Batch size: {args.batch_size}, Learning rate: {args.lr}")
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(start_epoch, args.epochs):
            print(f"\nEpoch {epoch}/{args.epochs-1}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
            
            # Update scheduler
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss)
                else:
                    scheduler.step()
            
            # Print epoch results
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Learning Rate: {current_lr:.6f}')
            
            # Log to Wandb
            if not args.no_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss,
                    args.checkpoint_dir, args.experiment_name
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss,
                    args.checkpoint_dir, args.experiment_name
                )
                print(f'New best model saved with val_loss: {val_loss:.4f}')
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {e}")
        raise
    finally:
        if not args.no_wandb:
            wandb.finish()


if __name__ == '__main__':
    main()