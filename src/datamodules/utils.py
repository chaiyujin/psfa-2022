import random
from typing import Callable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2 ** 32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)


def build_dataloader(
    dataset: Dataset,
    batch_size: Optional[int] = 1,
    shuffle: bool = False,
    sampler: Optional[Sampler[int]] = None,
    batch_sampler: Optional[Sampler[Sequence[int]]] = None,
    num_workers: int = 0,
    collate_fn: Callable = None,
    pin_memory: bool = False,
    drop_last: bool = False,
):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        # worker_init_fn=seed_worker,
    )
