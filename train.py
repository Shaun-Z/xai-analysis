"""
Training script for XAI analysis models
"""

import argparse
import os
import time
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from models.cifar_models import get_model
from datasets import get_dataset_loaders


def setup_logging(log_dir):
    """Setup logging configuration"""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir, filename=None):
    """Save model checkpoint"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss'], checkpoint['accuracy']


def evaluate_model(model, testloader, criterion, device):
    """Evaluate model on test set"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(testloader)
    
    return avg_loss, accuracy


def train_model(args):
    """Main training function"""
    # Setup logging
    logger = setup_logging(args.log_dir)
    logger.info(f"Starting training with arguments: {args}")
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup tensorboard
    if args.tensorboard:
        writer = SummaryWriter(log_dir=args.tensorboard_dir)
    
    # Load dataset
    logger.info(f"Loading {args.dataset} dataset...")
    trainloader, testloader, classes, num_classes = get_dataset_loaders(
        args.dataset, args.data_root, args.batch_size, args.num_workers, args.augmented
    )
    logger.info(f"Dataset loaded. Number of classes: {num_classes}")
    
    # Create model
    logger.info(f"Creating {args.model} model...")
    model = get_model(args.model, num_classes).to(device)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Setup optimizer and loss function
    criterion = nn.CrossEntropyLoss()
    
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, 
                             weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    
    # Setup learning rate scheduler
    if args.scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        start_epoch, _, _ = load_checkpoint(args.resume, model, optimizer)
        start_epoch += 1
    
    # Training loop
    logger.info("Starting training...")
    best_accuracy = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % args.print_freq == (args.print_freq - 1):
                avg_loss = running_loss / args.print_freq
                train_acc = 100 * correct / total
                logger.info(f'Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(trainloader)}], '
                           f'Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%')
                running_loss = 0.0
        
        # Update learning rate
        if scheduler is not None:
            scheduler.step()
        
        # Evaluate on test set
        test_loss, test_accuracy = evaluate_model(model, testloader, criterion, device)
        
        epoch_time = time.time() - start_time
        train_accuracy = 100 * correct / total
        
        logger.info(f'Epoch [{epoch+1}/{args.epochs}] completed in {epoch_time:.2f}s')
        logger.info(f'Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
        logger.info(f'Test Loss: {test_loss:.4f}')
        
        # Tensorboard logging
        if args.tensorboard:
            writer.add_scalar('Loss/Train', running_loss, epoch)
            writer.add_scalar('Loss/Test', test_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_accuracy, epoch)
            writer.add_scalar('Accuracy/Test', test_accuracy, epoch)
            if scheduler is not None:
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = save_checkpoint(
                model, optimizer, epoch, test_loss, test_accuracy, args.checkpoint_dir
            )
            logger.info(f'Checkpoint saved: {checkpoint_path}')
        
        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_path = save_checkpoint(
                model, optimizer, epoch, test_loss, test_accuracy, 
                args.checkpoint_dir, 'best_model.pth'
            )
            logger.info(f'New best model saved: {best_model_path} (Accuracy: {best_accuracy:.2f}%)')
    
    # Save final model
    final_model_path = save_checkpoint(
        model, optimizer, args.epochs-1, test_loss, test_accuracy, 
        args.checkpoint_dir, 'final_model.pth'
    )
    logger.info(f'Final model saved: {final_model_path}')
    
    # Close tensorboard writer
    if args.tensorboard:
        writer.close()
    
    logger.info(f'Training completed! Best accuracy: {best_accuracy:.2f}%')


def main():
    parser = argparse.ArgumentParser(description='Train models for XAI analysis')
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, default='net', 
                       choices=['net', 'resnet18', 'vgg16'],
                       help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'cifar100'],
                       help='Dataset to use')
    parser.add_argument('--augmented', action='store_true',
                       help='Use data augmentation')
    
    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='Momentum for SGD optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='sgd',
                       choices=['sgd', 'adam', 'adamw'],
                       help='Optimizer to use')
    
    # Learning rate scheduler
    parser.add_argument('--scheduler', type=str, default=None,
                       choices=['step', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--step-size', type=int, default=10,
                       help='Step size for StepLR scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                       help='Gamma for StepLR scheduler')
    
    # Data and I/O
    parser.add_argument('--data-root', type=str, default='./data',
                       help='Root directory for data')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                       help='Directory to save logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    # Other options
    parser.add_argument('--num-workers', type=int, default=2,
                       help='Number of workers for data loading')
    parser.add_argument('--print-freq', type=int, default=100,
                       help='Print frequency during training')
    parser.add_argument('--save-freq', type=int, default=5,
                       help='Save checkpoint frequency (epochs)')
    parser.add_argument('--tensorboard', action='store_true',
                       help='Use tensorboard for logging')
    parser.add_argument('--tensorboard-dir', type=str, default='./runs',
                       help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Start training
    train_model(args)


if __name__ == '__main__':
    main()