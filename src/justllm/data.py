from pathlib import Path

import numpy as np
import torch

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def get_batch(split: str, batch_size: int, block_size: int, device: str):
    """Sample a random batch of (input, target) token windows from a .bin file.

    Targets are the inputs shifted by one token: at every position the model
    learns to predict the next token.

    Returns:
        x, y: int64 tensors of shape (batch_size, block_size) on `device`.
    """
    # Re-opening the memmap on every call avoids a known memory leak with long-lived memmaps.
    data = np.memmap(DATA_DIR / f"{split}.bin", dtype=np.uint16, mode="r")

    # Highest start index leaves room for block_size tokens plus the one-token shift.
    starts = torch.randint(len(data) - block_size - 1, (batch_size,)).tolist()

    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in starts])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in starts])

    return x.to(device), y.to(device)
