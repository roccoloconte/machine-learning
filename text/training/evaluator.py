import logging
from tqdm import tqdm
from typing import Dict, List
from collections import defaultdict
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class Evaluator:
    """
    Handles model evaluation and metrics calculation
    """
    def __init__(self, model: nn.Module, tokenizer, device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.metrics = defaultdict(list)

    def evaluate(self, dataloader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate model on the given dataloader
        """
        self.model.eval()
        total_loss = 0
        total_perplexity = 0
        total_accuracy = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate metrics
                loss = self._calculate_loss(outputs, input_ids)
                perplexity = torch.exp(loss)
                accuracy = self._calculate_accuracy(outputs, input_ids)
                
                total_loss += loss.item()
                total_perplexity += perplexity.item()
                total_accuracy += accuracy

        # Calculate average metrics
        metrics = {
            'loss': total_loss / num_batches,
            'perplexity': total_perplexity / num_batches,
            'accuracy': total_accuracy / num_batches
        }

        logger.info(f"Evaluation metrics: {metrics}")
        return metrics

    def generate_samples(self, 
                        prompts: List[str], 
                        max_length: int = 100,
                        temperature: float = 0.7,
                        top_p: float = 0.9) -> List[str]:
        """
        Generate text samples from the model
        """
        self.model.eval()
        generated_texts = []

        for prompt in tqdm(prompts, desc="Generating samples"):
            input_ids = self.tokenizer.tokenize(prompt)
            input_ids = torch.tensor(input_ids).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output_sequence = self._generate_sequence(
                    input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p
                )
                
            generated_text = self.tokenizer.detokenize(output_sequence[0].cpu().numpy())
            generated_texts.append(generated_text)

        return generated_texts

    def _calculate_loss(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate cross-entropy loss"""
        return nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            targets.view(-1),
            ignore_index=self.tokenizer.pad_token_id
        )

    def _calculate_accuracy(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Calculate token prediction accuracy"""
        predictions = outputs.argmax(dim=-1)
        mask = targets != self.tokenizer.pad_token_id
        correct = (predictions == targets) & mask
        return correct.float().mean().item()

    def _generate_sequence(self, 
                         input_ids: torch.Tensor,
                         max_length: int,
                         temperature: float,
                         top_p: float) -> torch.Tensor:
        """Generate sequence using nucleus (top-p) sampling"""
        for _ in range(max_length):
            outputs = self.model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply top-p sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Use sorted_indices to map back to original indices
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = float('-inf')
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
                
        return input_ids
