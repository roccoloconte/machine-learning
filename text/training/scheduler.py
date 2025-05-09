import torch
from torch.optim.lr_scheduler import _LRScheduler
import math

class WarmupLearningRate(_LRScheduler):
    """
    Learning rate scheduler with warmup and cosine decay
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 min_lr: float = 1e-7):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self) -> float:
        step = self._step_count
        
        # Warmup phase
        if step < self.warmup_steps:
            return [base_lr * (step / self.warmup_steps) 
                    for base_lr in self.base_lrs]
        
        # Cosine decay phase
        progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        
        return [self.min_lr + (base_lr - self.min_lr) * cosine_decay 
                for base_lr in self.base_lrs]

class CyclicLearningRate(_LRScheduler):
    """
    Cyclic learning rate scheduler with triangular policy
    """
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 base_lr: float,
                 max_lr: float,
                 step_size: int,
                 mode: str = 'triangular'):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        super().__init__(optimizer)

    def get_lr(self) -> float:
        cycle = math.floor(1 + self._step_count / (2 * self.step_size))
        x = abs(self._step_count / self.step_size - 2 * cycle + 1)
        
        if self.mode == 'triangular2':
            x = x / (2 ** (cycle - 1))
        
        return [base_lr + (self.max_lr - base_lr) * max(0, (1 - x))
                for base_lr in self.base_lrs]
