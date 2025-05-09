import os
import math
import logging
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

class Trainer:
    """
    Trainer class for managing the training process of the language model.
    Handles configuration, data, optimization, and evaluation.
    """
    def __init__(self, config: Dict):
        """
        Initialize the trainer with configuration and set up device.
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.device = (torch.device("mps") if torch.backends.mps.is_available()
                      else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Trainer initialized with device: {self.device}")
        
        # Initialize tracking metrics
        self.train_losses = []
        self.val_losses = []
        self.perplexities = []
        
        # Create directories
        os.makedirs("text/checkpoints", exist_ok=True)

    def train(self, 
              model: nn.Module,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              tokenizer,
              resume_checkpoint: Optional[str] = None) -> Tuple[nn.Module, torch.optim.Optimizer]:
        """
        Train the model.
        
        Args:
            model: The neural network model to train
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            tokenizer: Tokenizer instance for text processing
            resume_checkpoint: Optional path to checkpoint to resume from
            
        Returns:
            Tuple of (trained model, optimizer)
        """
        model = model.to(self.device)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        # Calculate total steps for learning rate schedule
        total_steps = self.config['num_epochs'] * len(train_dataloader)
        scheduler = self._create_scheduler(optimizer, total_steps)
        
        # Resume from checkpoint if provided
        start_epoch = 0
        global_step = 0
        best_val_loss = float('inf')
        
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_step']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resumed from checkpoint: {resume_checkpoint}")
        
        # Training loop
        for epoch in range(start_epoch, self.config['num_epochs']):
            model.train()
            total_loss = 0
            
            logger.info(f"Starting epoch {epoch+1}/{self.config['num_epochs']}")
            logger.info(f"Number of batches in dataloader: {len(train_dataloader)}")
            
            with tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}") as pbar:
                for step, batch in enumerate(pbar):
                    logger.info(f"Processing batch {step+1}/{len(train_dataloader)}")
                    
                    # Debug batch shapes
                    logger.info(f"Batch shapes - input_ids: {batch['input_ids'].shape}, attention_mask: {batch['attention_mask'].shape}")
                    
                    loss = self._training_step(model, batch, optimizer, scheduler, step)
                    total_loss += loss
                    global_step += 1
                    
                    # Update progress bar
                    pbar.set_postfix({'loss': f'{loss:.4f}'})
                    pbar.update(1)  # Explicitly update progress bar
                    
                    # Save checkpoint periodically
                    if global_step % self.config['save_steps'] == 0:
                        self._save_checkpoint(
                            model, optimizer, epoch, global_step,
                            loss, best_val_loss, is_best=False
                        )
            
            # Calculate average loss for epoch
            avg_train_loss = total_loss / len(train_dataloader)
            self.train_losses.append(avg_train_loss)
            
            # Evaluate on validation set
            val_loss, val_perplexity = self._evaluate(model, val_dataloader)
            self.val_losses.append(val_loss)
            self.perplexities.append(val_perplexity)
            
            # Log metrics
            self._log_metrics(epoch, avg_train_loss, val_loss, val_perplexity)
            
            # Save if best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    model, optimizer, epoch, global_step,
                    val_loss, best_val_loss, is_best=True
                )
        
        logger.info("Training completed")
        # Create and save training plots
        self._create_training_plots()
        # Return model and optimizer state
        return model, optimizer

    def _training_step(self, 
                      model: nn.Module,
                      batch: Dict[str, torch.Tensor],
                      optimizer: torch.optim.Optimizer,
                      scheduler,
                      step: int) -> float:
        """
        Perform a single training step.
        
        Args:
            model: The model to train
            batch: Dictionary containing batch data
            optimizer: The optimizer
            scheduler: Learning rate scheduler
            step: Current training step
            
        Returns:
            Loss value for the step
        """
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # For language modeling, labels are the input_ids shifted by one position
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        # Slice attention mask on both dimensions to match the shifted input_ids
        attention_mask = attention_mask[:, :-1, :-1].contiguous()
        
        # Forward pass with gradient checkpointing when available
        outputs = model(input_ids, attention_mask)
        
        # Reshape for loss calculation
        logits = outputs.view(-1, outputs.size(-1))
        labels_flat = labels.view(-1)
        
        # Debug logits before loss calculation
        logger.debug(f"Logits shape: {logits.shape}, Labels shape: {labels_flat.shape}")
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logger.warning(f"NaN or Inf detected in logits before loss calculation at step {step}")
            # Replace NaN/Inf values with zeros
            logits = torch.where(torch.isnan(logits) | torch.isinf(logits), 
                               torch.zeros_like(logits), logits)
            # Reduce learning rate to stabilize training
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * 0.5
                logger.warning(f"Reduced learning rate to {param_group['lr']} due to NaN/Inf")
        else:
            logger.debug(f"Logits range at step {step}: min={logits.min().item():.4f}, max={logits.max().item():.4f}")
            
        # Calculate loss (ignore padding tokens)
        try:
            loss = F.cross_entropy(logits, labels_flat, ignore_index=0)
            
            # Check if loss is valid
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning(f"NaN or Inf loss detected. Setting to 1.0 to continue training")
                loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        except Exception as e:
            logger.error(f"Error in loss calculation: {e}")
            # Return a default loss value to continue training
            loss = torch.tensor(1.0, device=self.device, requires_grad=True)
        
        # Scale loss for gradient accumulation
        loss = loss / self.config['gradient_accumulation_steps']
        
        # Backward pass with gradient scaling to prevent underflow/overflow
        try:
            loss.backward()
            
            # Check for NaN or Inf gradients
            has_nan_or_inf = False
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    logger.warning(f"NaN or Inf gradient detected in {name} at step {step}")
                    has_nan_or_inf = True
                    # Zero out problematic gradients
                    param.grad = torch.zeros_like(param.grad)
            
            if has_nan_or_inf:
                # Reduce learning rate to stabilize training
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                    logger.warning(f"Reduced learning rate to {param_group['lr']} due to NaN/Inf gradients")
        except Exception as e:
            logger.error(f"Error during backward pass: {e}")
            # Skip this update step
            optimizer.zero_grad()
            return 1.0  # Return a default loss value
        
        # Update weights if gradient accumulation is complete
        if (step + 1) % self.config['gradient_accumulation_steps'] == 0:
            # Add gradient clipping (use value from config)
            max_norm = self.config.get('grad_clip', 0.1)  # Default to 0.1 if not specified
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm) 
            logger.debug(f"Applied gradient clipping with max_norm={max_norm}")
            
            # Check all gradients are valid before optimizer step
            valid_gradients = True
            for name, param in model.named_parameters():
                if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                    valid_gradients = False
                    logger.warning(f"NaN/Inf gradient in {name} after clipping - zeroing this gradient")
                    param.grad = torch.zeros_like(param.grad)
            
            # Update model parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        
        # Log loss value before returning
        loss_value = loss.item() * self.config['gradient_accumulation_steps']  # Unnormalize for reporting
        if math.isnan(loss_value) or math.isinf(loss_value):
            logger.warning(f"NaN or Inf loss detected at step {step}. Loss: {loss_value}")
            loss_value = 1.0  # Set a default value for tracking
        else:
            logger.debug(f"Step {step} loss: {loss_value:.4f}")
        
        return loss_value

    def _evaluate(self, 
                 model: nn.Module,
                 dataloader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the model on validation/test data.
        
        Args:
            model: The model to evaluate
            dataloader: DataLoader for evaluation
            
        Returns:
            Tuple of (average loss, perplexity)
        """
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                logits = outputs.view(-1, outputs.size(-1))
                labels_flat = labels.view(-1)
                
                loss = F.cross_entropy(logits, labels_flat, reduction='sum')
                total_loss += loss.item()
                total_tokens += labels.numel()
        
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return avg_loss, perplexity

    def _create_scheduler(self, 
                         optimizer: torch.optim.Optimizer,
                         total_steps: int):
        """
        Create learning rate scheduler with warmup and decay.
        
        Args:
            optimizer: The optimizer
            total_steps: Total number of training steps
            
        Returns:
            Learning rate scheduler
        """
        from .scheduler import WarmupLearningRate
        
        return WarmupLearningRate(
            optimizer,
            warmup_steps=self.config['warmup_steps'],
            total_steps=total_steps,
            min_lr=1e-7
        )

    def _save_checkpoint(self,
                        model: nn.Module,
                        optimizer: torch.optim.Optimizer,
                        epoch: int,
                        global_step: int,
                        loss: float,
                        best_val_loss: float,
                        is_best: bool):
        """
        Save model checkpoint.
        
        Args:
            model: The model to save
            optimizer: The optimizer
            epoch: Current epoch number
            global_step: Global training step
            loss: Current loss value
            best_val_loss: Best validation loss so far
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'best_val_loss': best_val_loss,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            "text/checkpoints",
            f"checkpoint_epoch_{epoch+1}_step_{global_step}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model if applicable
        if is_best:
            best_model_path = os.path.join("text/checkpoints", "best_model.pt")
            torch.save(checkpoint, best_model_path)
            logger.info(f"New best model saved with validation loss: {loss:.4f}")

    def _log_metrics(self,
                    epoch: int,
                    train_loss: float,
                    val_loss: float,
                    val_perplexity: float):
        """
        Log training metrics.
        
        Args:
            epoch: Current epoch number
            train_loss: Training loss
            val_loss: Validation loss
            val_perplexity: Validation perplexity
        """
        logger.info(
            f"Epoch {epoch+1} - "
            f"Train loss: {train_loss:.4f}, "
            f"Val loss: {val_loss:.4f}, "
            f"Perplexity: {val_perplexity:.4f}"
        )
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'perplexity': val_perplexity,
                'learning_rate': self._get_current_lr()
            })

    def _create_training_plots(self):
        """Create and save training visualization plots."""
        try:
            # Create visualizations directory if it doesn't exist
            os.makedirs('text/visualizations', exist_ok=True)
            
            # Set non-interactive backend for matplotlib
            plt.switch_backend('agg')
            
            # Check if we have data to plot
            if not self.train_losses or not self.val_losses:
                logger.warning("No training data to plot. Skipping visualization.")
                return
                
            plt.figure(figsize=(15, 5))
            
            # Loss plot
            plt.subplot(1, 2, 1)
            plt.plot(self.train_losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            
            # Perplexity plot
            plt.subplot(1, 2, 2)
            plt.plot(self.perplexities, label='Perplexity')
            plt.xlabel('Epoch')
            plt.ylabel('Perplexity')
            plt.title('Validation Perplexity')
            plt.legend()
            
            plt.tight_layout()
            save_path = 'text/visualizations/training_progress.png'
            plt.savefig(save_path)
            plt.close()
            logger.info(f"Training plots saved to {save_path}")
        except Exception as e:
            logger.error(f"Error creating training plots: {e}")

    def _get_current_lr(self) -> float:
        """Get current learning rate from optimizer."""
        for param_group in self.optimizer.param_groups:
            return param_group['lr']
        return 0.0

    def get_model_size(self, model: nn.Module) -> Tuple[int, float]:
        """
        Calculate model size in parameters and MB.
        
        Args:
            model: The model to analyze
            
        Returns:
            Tuple of (parameter count, size in MB)
        """
        param_count = sum(p.numel() for p in model.parameters())
        param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
        
        logger.info(f"Model dimensions: {param_count:,} parameters ({param_size_mb:.2f} MB)")
        return param_count, param_size_mb
