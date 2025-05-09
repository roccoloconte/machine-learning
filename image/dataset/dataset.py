import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import logging
import os
import urllib.request
import ssl

logger = logging.getLogger(__name__)

class MNISTDataset:
    """Helper class for MNIST dataset loading and preprocessing"""
    
    @staticmethod
    def get_transforms():
        """Get training and validation transforms for MNIST"""
        # Basic transforms for MNIST
        # We'll normalize with mean=0.1307 and std=0.3081 which are the MNIST dataset statistics
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # For data augmentation in training
        train_transform = transforms.Compose([
            transforms.RandomRotation(10),  # Random rotation of +/- 10 degrees
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random shift by up to 10%
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        return train_transform, transform
    
    @staticmethod
    def create_dataloaders(batch_size=64, val_split=0.1, num_workers=4):
        """Create train and validation dataloaders for MNIST"""
        # Get transforms
        train_transform, val_transform = MNISTDataset.get_transforms()
        
        # Download MNIST dataset
        # Create data directory if it doesn't exist
        os.makedirs('./image/data', exist_ok=True)
        
        # Disable SSL certificate verification directly in torchvision
        # Note: This approach is less secure but simpler
        ssl._create_default_https_context = ssl._create_unverified_context
        
        try:
            logger.info("Downloading MNIST dataset")
            full_train_dataset = torchvision.datasets.MNIST(
                root='./image/data',
                train=True,
                download=True,
                transform=train_transform
            )
            
            # Split into train and validation
            val_size = int(len(full_train_dataset) * val_split)
            train_size = len(full_train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_train_dataset, [train_size, val_size]
            )
            
            # Override transform for validation set
            val_dataset.dataset.transform = val_transform
            
            # Test dataset
            test_dataset = torchvision.datasets.MNIST(
                root='./image/data',
                train=False,
                download=True,
                transform=val_transform
            )
        finally:
            # Restore the default SSL context
            ssl._create_default_https_context = ssl.create_default_context
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader 