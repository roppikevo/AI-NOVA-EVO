import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    seed: int = 1001

    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    max_steps: int = 1000

    grad_clip: float = 1.0

    eval_every: int = 100
    log_every: int = 10

    device: str = "cuda"

    # None = legacy behavior; 0 = ignore PAD tokens for task-learning.
    pad_id: int | None = None


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: TrainConfig):
    if config.device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def causal_loss(logits, targets, pad_id=None):
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=pad_id if pad_id is not None else -100,
    )


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None, pad_id=None):
    model.eval()

    total_loss = 0.0
    total_tokens = 0

    for batch_index, (input_ids, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        output = model(input_ids)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = causal_loss(logits, targets, pad_id=pad_id)

        if pad_id is None:
            tokens = targets.numel()
        else:
            tokens = (targets != pad_id).sum().item()

        if tokens == 0:
            continue

        total_loss += loss.item() * tokens
        total_tokens += tokens

    mean_loss = total_loss / total_tokens

    return {
        "loss": mean_loss,
        "perplexity": math.exp(mean_loss),
    }


def train(
    model,
    train_dataset,
    val_dataset,
    config: TrainConfig,
):
    set_seed(config.seed)

    device = get_device(config)
    model = model.to(device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    model.train()

    train_iter = iter(train_loader)

    history = []

    for step in range(1, config.max_steps + 1):

        try:
            input_ids, targets = next(train_iter)

        except StopIteration:
            train_iter = iter(train_loader)
            input_ids, targets = next(train_iter)

        input_ids = input_ids.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        output = model(input_ids)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        loss = causal_loss(
            logits,
            targets,
            pad_id=config.pad_id,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.grad_clip,
        )

        optimizer.step()

        if step == 1 or step % config.log_every == 0:
            print(
                f"step={step:5d} "
                f"train_loss={loss.item():.6f}"
            )

        if step == 1 or step % config.eval_every == 0:
            metrics = evaluate(
                model,
                val_loader,
                device,
                pad_id=config.pad_id,
            )

            print(
                f"          "
                f"val_loss={metrics['loss']:.6f} "
                f"ppl={metrics['perplexity']:.4f}"
            )

            history.append(
                {
                    "step": step,
                    "train_loss": loss.item(),
                    **metrics,
                }
            )

            model.train()

    return model, history
