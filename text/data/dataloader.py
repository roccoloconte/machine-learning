import queue
import threading
import torch
from torch.utils.data import DataLoader

class PrefetchDataLoader:
    """
    Custom DataLoader that prefetches data to GPU in a background thread
    to optimize training speed.
    """
    def __init__(self, dataloader: DataLoader, device: torch.device, queue_size: int = 3):
        self.dataloader = dataloader
        self.device = device
        self.queue_size = queue_size
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.thread.start()

    def _prefetch_loop(self):
        """Background thread that prefetches data to GPU"""
        try:
            for batch in self.dataloader:
                if self.stop_event.is_set():
                    break
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                self.queue.put(batch)
        finally:
            self.queue.put(None)  # Signal the end of the dataset

    def __iter__(self):
        return self

    def __next__(self):
        batch = self.queue.get()
        if batch is None:
            self.stop_event.set()
            raise StopIteration
        return batch

    def __len__(self):
        return len(self.dataloader)
