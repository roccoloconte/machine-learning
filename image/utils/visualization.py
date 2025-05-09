import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import os
import logging

logger = logging.getLogger(__name__)

def visualize_predictions(model, dataloader, device, num_images=10, save_dir='./image/visualizations'):
    """Visualize model predictions on sample images
    
    Args:
        model: Trained model
        dataloader: DataLoader containing images
        device: Device to run inference on
        num_images: Number of images to visualize
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Get a batch of images
    model.eval()
    images, labels = next(iter(dataloader))
    
    # Limit to specified number of images
    images = images[:num_images]
    labels = labels[:num_images]
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, predictions = torch.max(outputs, 1)
    
    # Convert predictions and labels to CPU
    predictions = predictions.cpu().numpy()
    labels = labels.numpy()
    
    # Denormalize images for visualization
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    images_np = images.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < num_images:
            # Display image
            img = np.transpose(images_np[i], (1, 2, 0))
            img = img * std.numpy() + mean.numpy()
            img = np.squeeze(img)  # Remove channel dimension for grayscale
            
            ax.imshow(img, cmap='gray')
            ax.set_title(f'Pred: {predictions[i]}, True: {labels[i]}')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'))
    logger.info(f"Saved prediction visualization to {save_dir}/sample_predictions.png")
    
    # Close the figure to free memory
    plt.close(fig)

def visualize_attention(model, dataloader, device, layer_idx=5, head_idx=0, num_images=5, save_dir='./image/visualizations/attention'):
    """Visualize attention maps from the model
    
    Args:
        model: Trained Vision Transformer model
        dataloader: DataLoader containing images
        device: Device to run inference on
        layer_idx: Index of transformer layer to visualize (default: last layer)
        head_idx: Index of attention head to visualize
        num_images: Number of images to visualize
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Put model in eval mode
    model.eval()
    
    # For simplicity, since we can't easily access attention maps without modifying the model,
    # let's just visualize the patch embeddings instead
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)
    labels = labels[:num_images].cpu().numpy()
    
    # Get the patch embeddings
    with torch.no_grad():
        # Forward pass through patch embedding only
        patch_embeddings = model.patch_embedding(images).detach().cpu()
        
        # Also get full predictions for comparison
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
        predictions = predictions.cpu().numpy()
    
    # Visualize embeddings for each image
    for img_idx in range(num_images):
        # Get the original image
        img = images[img_idx].cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = img * 0.3081 + 0.1307  # Denormalize
        img = np.squeeze(img)  # Remove channel dimension for grayscale
        
        # Get the patch embedding for this image (excluding CLS token)
        # Shape is [num_patches + 1, hidden_dim] where index 0 is CLS token
        img_embedding = patch_embeddings[img_idx, 1:, :].numpy()
        
        # Calculate the average activation across hidden dimensions
        patch_activations = np.mean(np.abs(img_embedding), axis=1)
        
        # Reshape to match the image spatial dimensions
        num_patches_1d = int(np.sqrt(patch_activations.shape[0]))
        activations_map = patch_activations.reshape(num_patches_1d, num_patches_1d)
        
        # Create a figure with the original image and the activation map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot original image
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f'Original (Label: {labels[img_idx]}, Pred: {predictions[img_idx]})')
        ax1.axis('off')
        
        # Plot activation map
        im = ax2.imshow(activations_map, cmap='viridis')
        ax2.set_title(f'Patch embedding activations')
        ax2.axis('off')
        fig.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'patch_activations_img{img_idx}.png'))
        plt.close(fig)
    
    logger.info(f"Saved patch embedding visualizations to {save_dir}")
    logger.info("Note: These visualize patch embedding activations rather than attention maps")
    logger.info("To get actual attention maps, model code would need to be modified to return attention weights") 