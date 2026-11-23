import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from justllm.data import get_batch
from justllm.model import GPT, GPTConfig

SMOKE_TEST = False

if SMOKE_TEST:
    CFG = GPTConfig(n_layer=4, n_head=4, d_model=256, d_ff=704, block_size=256)
    BATCH_SIZE, ACCUM_STEPS = 16, 1
    MAX_STEPS, WARMUP_STEPS = 500, 20
    EVAL_INTERVAL, EVAL_ITERS = 100, 20
else:
    CFG = GPTConfig()
    BATCH_SIZE, ACCUM_STEPS = 4, 128  # 4 * 1024 * 128 ~ 0.5M tokens per optimizer step
    MAX_STEPS, WARMUP_STEPS = 4800, 200  # placeholder: set after measuring tokens/s
    EVAL_INTERVAL, EVAL_ITERS = 100, 50

MAX_LR, MIN_LR, WEIGHT_DECAY = 6e-4, 6e-5, 0.1
LOG_INTERVAL = 5
DEVICE = "cuda"
CKPT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CKPT_PATH = CKPT_DIR / "ckpt.pt"
RESUME = True

def get_lr(step: int, max_lr: float, min_lr: float, warmup_steps: int, max_steps: int) -> float:
    """Linear warmup followed by cosine decay from max_lr to min_lr."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)  # goes 0 -> 1
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))            # goes 1 -> 0
    return min_lr + cosine * (max_lr - min_lr)


def configure_optimizer(model: torch.nn.Module, weight_decay: float, lr: float) -> torch.optim.Optimizer:
    """AdamW with weight decay only on matrices (>= 2D); norm scales are not decayed."""
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), fused=torch.cuda.is_available())

@torch.no_grad()
def estimate_loss(model: torch.nn.Module, split: str, eval_iters: int,
                  batch_size: int, block_size: int, device: str) -> float:
    """Average loss over eval_iters random batches of the given split."""
    model.eval()
    losses = torch.zeros(eval_iters)
    for i in range(eval_iters):
        x, y = get_batch(split, batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        losses[i] = loss.item()
    model.train()
    return losses.mean().item()

from justllm.model import GPT, GPTConfig


def train_step(model: torch.nn.Module, optimizer: torch.optim.Optimizer, lr: float,
               accum_steps: int, batch_size: int, block_size: int, device: str):
    """One optimizer step over accum_steps micro-batches. Returns (loss, grad_norm)."""
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)

    loss_total = 0.0
    for _ in range(accum_steps):
        x, y = get_batch("train", batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / accum_steps  # average, not sum, over micro-batches
        loss.backward()
        loss_total += loss.item()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss_total, grad_norm.item()


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int) -> None:
    """Atomically save model, optimizer and the number of completed steps."""
    tmp_path = CKPT_PATH.with_suffix(".tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
         "step": step, "config": asdict(CFG)},
        tmp_path,
    )
    tmp_path.replace(CKPT_PATH)  # a crash mid-save cannot corrupt the old checkpoint


def main():
    torch.manual_seed(1337)
    CKPT_DIR.mkdir(exist_ok=True)

    model = GPT(CFG).to(DEVICE)
    optimizer = configure_optimizer(model, WEIGHT_DECAY, MAX_LR)
    tokens_per_step = BATCH_SIZE * ACCUM_STEPS * CFG.block_size

    start_step = 0
    if RESUME and CKPT_PATH.exists():
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"resumed from step {start_step}")

    step = start_step
    try:
        for step in range(start_step, MAX_STEPS):
            t0 = time.time()
            lr = get_lr(step, MAX_LR, MIN_LR, WARMUP_STEPS, MAX_STEPS)
            loss, grad_norm = train_step(model, optimizer, lr, ACCUM_STEPS, BATCH_SIZE, CFG.block_size, DEVICE)
            torch.cuda.synchronize()
            dt = time.time() - t0

            if step % LOG_INTERVAL == 0:
                print(f"step {step} | loss {loss:.4f} | lr {lr:.2e} | norm {grad_norm:.2f} | "
                      f"{tokens_per_step / dt:,.0f} tok/s | mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")

            if (step + 1) % EVAL_INTERVAL == 0 or step == MAX_STEPS - 1:
                val_loss = estimate_loss(model, "val", EVAL_ITERS, BATCH_SIZE, CFG.block_size, DEVICE)
                train_loss = estimate_loss(model, "train", EVAL_ITERS, BATCH_SIZE, CFG.block_size, DEVICE)
                print(f"--- step {step + 1} | train {train_loss:.4f} | val {val_loss:.4f}")
                save_checkpoint(model, optimizer, step + 1)
    except KeyboardInterrupt:
        # Weights are only updated at the end of a step, so `step` steps are fully completed.
        print(f"interrupted, saving checkpoint at step {step}")
        save_checkpoint(model, optimizer, step)

if __name__ == "__main__":
    main()


