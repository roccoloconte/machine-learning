import logging
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Dict

logger = logging.getLogger(__name__)

class Visualizer:
    """
    Handles visualization of training metrics and model attention patterns
    """
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = save_dir
        import os
        os.makedirs(save_dir, exist_ok=True)

    def plot_training_history(self,
                            metrics: Dict[str, List[float]],
                            title: str = 'Training History'):
        """
        Plot training metrics over time
        """
        plt.figure(figsize=(10, 6))
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
        
        plt.title(title)
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        save_path = f"{self.save_dir}/training_history.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved training history plot to {save_path}")

    def plot_attention_patterns(self,
                              attention_weights: torch.Tensor,
                              tokens: List[str],
                              layer: int,
                              head: int):
        """
        Plot attention patterns for a specific layer and head
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention_weights[layer, head].cpu().numpy(),
            xticklabels=tokens,
            yticklabels=tokens,
            cmap='viridis'
        )
        
        plt.title(f'Attention Pattern (Layer {layer}, Head {head})')
        plt.xlabel('Key Tokens')
        plt.ylabel('Query Tokens')
        
        save_path = f"{self.save_dir}/attention_l{layer}_h{head}.png"
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved attention pattern plot to {save_path}")

    def plot_loss_landscape(self,
                          model: torch.nn.Module,
                          loss_fn: callable,
                          data_point: torch.Tensor,
                          param_range: float = 1.0,
                          resolution: int = 50):
        """
        Plot loss landscape around current model parameters
        """
        # Get two random directions in parameter space
        params = [p for p in model.parameters()]
        direction1 = [torch.randn_like(p) for p in params]
        direction2 = [torch.randn_like(p) for p in params]
        
        # Normalize directions
        norm1 = torch.sqrt(sum((d * d).sum() for d in direction1))
        norm2 = torch.sqrt(sum((d * d).sum() for d in direction2))
        direction1 = [d / norm1 for d in direction1]
        direction2 = [d / norm2 for d in direction2]
        
        # Create grid of points
        x = np.linspace(-param_range, param_range, resolution)
        y = np.linspace(-param_range, param_range, resolution)
        X, Y = np.meshgrid(x, y)
        loss_surface = np.zeros_like(X)
        
        # Compute loss at each point
        original_params = [p.clone() for p in params]
        for i in range(resolution):
            for j in range(resolution):
                # Update parameters
                for p, d1, d2, p_orig in zip(params, direction1, direction2, original_params):
                    p.data = p_orig + X[i,j] * d1 + Y[i,j] * d2
                
                # Compute loss
                with torch.no_grad():
                    loss = loss_fn(model(data_point))
                loss_surface[i,j] = loss.item()
        
        # Restore original parameters
        for p, p_orig in zip(params, original_params):
            p.data = p_orig
        
        # Plot surface
        plt.figure(figsize=(10, 8))
        plt.contour(X, Y, loss_surface, levels=20)
        plt.colorbar(label='Loss')
        plt.title('Loss Landscape')
        plt.xlabel('Direction 1')
        plt.ylabel('Direction 2')
        
        save_path = f"{self.save_dir}/loss_landscape.png"
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Saved loss landscape plot to {save_path}")
