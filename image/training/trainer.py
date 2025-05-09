import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class Trainer:
    """Trainer class for handling model training and evaluation"""
    
    def __init__(self, config):
        """Initialize the trainer with given configuration"""
        self.config = config
        self.device = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        # Create SummaryWriter for TensorBoard
        self.writer = SummaryWriter(log_dir=config['log_dir'])
        
        # Ensure checkpoint directory exists
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    def train(self, model, train_dataloader, val_dataloader=None):
        """Train the model"""
        # Set device if not already set
        if self.device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("Using MPS (Apple Silicon GPU)")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("Using CUDA GPU")
            else:
                self.device = torch.device("cpu")
                logger.info("Using CPU")
        
        # Move model to device
        model = model.to(self.device)
        
        # Set up loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config['learning_rate'],
            steps_per_epoch=len(train_dataloader),
            epochs=self.config['num_epochs']
        )
        
        # Initialize best validation loss for model saving
        best_val_loss = float('inf')
        
        # Training loop
        for epoch in range(self.config['num_epochs']):
            # Training
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            # Use tqdm for progress bar
            pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch_idx, (images, targets) in enumerate(pbar):
                # Move data to device
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                if self.config['grad_clip'] > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), self.config['grad_clip'])
                
                optimizer.step()
                scheduler.step()
                
                # Accumulate statistics
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': train_loss / (batch_idx + 1),
                    'acc': 100. * correct / total
                })
                
                # Log to TensorBoard
                global_step = epoch * len(train_dataloader) + batch_idx
                self.writer.add_scalar('Loss/train_step', loss.item(), global_step)
                self.writer.add_scalar('LR', scheduler.get_last_lr()[0], global_step)
                
                # Save model checkpoint at regular intervals
                if (batch_idx + 1) % self.config['save_steps'] == 0:
                    checkpoint = {
                        'epoch': epoch,
                        'batch_idx': batch_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                        'config': self.config
                    }
                    torch.save(
                        checkpoint, 
                        os.path.join(self.config['checkpoint_dir'], f'checkpoint_e{epoch}_b{batch_idx}.pt')
                    )
            
            # Calculate epoch statistics
            train_loss = train_loss / len(train_dataloader)
            train_acc = 100. * correct / total
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Log epoch metrics to TensorBoard
            self.writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            self.writer.add_scalar('Accuracy/train', train_acc, epoch)
            
            logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            
            # Validation
            if val_dataloader is not None:
                val_loss, val_acc = self.evaluate(model, val_dataloader, criterion)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                
                # Log validation metrics to TensorBoard
                self.writer.add_scalar('Loss/val', val_loss, epoch)
                self.writer.add_scalar('Accuracy/val', val_acc, epoch)
                
                logger.info(f"Epoch {epoch+1}/{self.config['num_epochs']} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': self.config
                    }
                    torch.save(
                        checkpoint, 
                        os.path.join(self.config['checkpoint_dir'], 'best_model.pt')
                    )
                    logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
        
        # Plot training curves
        self._plot_training_curves()
        
        # Close TensorBoard writer
        self.writer.close()
        
        # Save final model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config
        }
        torch.save(
            checkpoint, 
            os.path.join(self.config['checkpoint_dir'], 'final_model.pt')
        )
        logger.info("Training completed. Final model saved.")
        
        return model, optimizer
    
    def evaluate(self, model, dataloader, criterion):
        """Evaluate the model on the given dataloader"""
        model.eval()
        loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in dataloader:
                # Move data to device
                images, targets = images.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = model(images)
                batch_loss = criterion(outputs, targets)
                
                # Accumulate statistics
                loss += batch_loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        # Calculate metrics
        avg_loss = loss / len(dataloader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def _plot_training_curves(self):
        """Plot and save training and validation curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.train_losses, label='Train Loss')
        if self.val_losses:
            ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Plot accuracy
        ax2.plot(self.train_accs, label='Train Accuracy')
        if self.val_accs:
            ax2.plot(self.val_accs, label='Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['log_dir'], 'training_curves.png'))
        plt.close() 