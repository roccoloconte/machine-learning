import logging
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """
    Dataset class for handling text data
    """
    def __init__(self, 
                 texts: List[str],
                 tokenizer,
                 max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        logger.info(f"Created dataset with {len(texts)} samples")
        
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize text
        tokens = self.tokenizer.tokenize(text)
        
        # Convert tokens to IDs
        token_ids = [self.tokenizer.vocab.get(token, 0) for token in tokens]  # Use 0 as default for unknown tokens
        
        # Truncate if necessary
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
            
        # Create attention mask (causal mask for language modeling)
        attention_mask = torch.tril(torch.ones(len(token_ids), len(token_ids)))
        
        # Convert to tensor
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        
        # For language modeling, labels are the same as input_ids
        # but shifted by one position (next token prediction)
        labels = input_ids.clone()
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function to pad sequences to the same length within a batch
        """
        # Find max length in this batch
        max_len = max(item['input_ids'].size(0) for item in batch)
        
        # Pad sequences
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = []
        
        for item in batch:
            # Pad input_ids
            padding_length = max_len - item['input_ids'].size(0)
            padded_input_ids.append(
                torch.cat([
                    item['input_ids'],
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            )
            
            # Pad attention mask
            padding_length = max_len - item['attention_mask'].size(0)
            padded_attention_mask.append(
                torch.cat([
                    item['attention_mask'],
                    torch.zeros(padding_length, item['attention_mask'].size(1))
                ])
            )
            
            # Also pad the other dimension
            padded_attention_mask[-1] = torch.cat([
                padded_attention_mask[-1],
                torch.zeros(padded_attention_mask[-1].size(0), padding_length)
            ], dim=1)
            
            # Pad labels
            padded_labels.append(
                torch.cat([
                    item['labels'],
                    torch.zeros(padding_length, dtype=torch.long)
                ])
            )
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'labels': torch.stack(padded_labels)
        }

    def create_dataloaders(
        train_texts: List[str],
        val_texts: List[str],
        tokenizer,
        batch_size: int,
        max_length: int = 512,
        num_workers: int = 4
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Create train and validation dataloaders
        """
        train_dataset = TextDataset(train_texts, tokenizer, max_length)
        val_dataset = TextDataset(val_texts, tokenizer, max_length)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=TextDataset.collate_fn
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=TextDataset.collate_fn
        )
        
        return train_loader, val_loader
