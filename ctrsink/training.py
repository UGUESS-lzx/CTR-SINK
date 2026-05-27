import json
import math
import os
import random
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from .config import TrainConfig
from .data import TextCtrCollator, TextCtrDataset, read_table
from .modeling.model import CtrSinkModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg: TrainConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = TextCtrDataset(read_table(cfg.train_file), cfg)
    valid_ds = TextCtrDataset(read_table(cfg.valid_file), cfg)
    collate = TextCtrCollator()
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, valid_loader


def autocast_dtype(cfg: TrainConfig):
    if not torch.cuda.is_available():
        return None
    if cfg.precision == "bf16":
        return torch.bfloat16
    if cfg.precision in {"fp16", "16"}:
        return torch.float16
    return None


def train_one_epoch(
    model: CtrSinkModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: TrainConfig,
    epoch_idx: int,
    device: torch.device,
) -> float:
    model.train()
    dtype = autocast_dtype(cfg)
    stage = cfg.stage_for_epoch(epoch_idx)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"train epoch={epoch_idx + 1} stage={stage}")

    for step, batch in enumerate(pbar):
        labels = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            logits = model(batch, stage=stage)
            loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
            loss = loss / cfg.gradient_accumulation_steps
        loss.backward()

        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * cfg.gradient_accumulation_steps
        pbar.set_postfix(loss=total_loss / max(1, step + 1))
    return total_loss / max(1, len(loader))


@torch.no_grad()
def evaluate(model: CtrSinkModel, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    labels_all, scores_all = [], []
    for batch in tqdm(loader, desc="valid"):
        labels = batch["labels"].to(device)
        logits = model(batch, stage="all_tokens")
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
        total_loss += loss.item()
        labels_all.extend(labels.detach().cpu().float().tolist())
        scores_all.extend(torch.sigmoid(logits).detach().cpu().float().tolist())

    metrics = {"valid_loss": total_loss / max(1, len(loader))}
    if len(set(int(x) for x in labels_all)) > 1:
        metrics["valid_auc"] = float(roc_auc_score(labels_all, scores_all))
    else:
        metrics["valid_auc"] = float("nan")
    return metrics


def save_checkpoint(model: CtrSinkModel, cfg: TrainConfig, path: str, metrics: Dict[str, float]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": cfg.to_dict(),
            "metrics": metrics,
        },
        path,
    )


def train(cfg: TrainConfig) -> Dict[str, float]:
    cfg.validate()
    set_seed(cfg.random_seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    with open(os.path.join(cfg.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader = build_loaders(cfg)
    model = CtrSinkModel(cfg).to(device)

    steps_per_epoch = math.ceil(len(train_loader) / cfg.gradient_accumulation_steps)
    total_steps = steps_per_epoch * cfg.total_epochs()
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_auc = -1.0
    best_metrics: Dict[str, float] = {}
    last_metrics: Dict[str, float] = {}
    for epoch_idx in range(cfg.total_epochs()):
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, cfg, epoch_idx, device)
        metrics = evaluate(model, valid_loader, device)
        metrics["train_loss"] = train_loss
        metrics["epoch"] = epoch_idx + 1
        last_metrics = metrics
        print(
            "epoch={epoch} stage={stage} train_loss={train_loss:.6f} "
            "valid_loss={valid_loss:.6f} valid_auc={valid_auc}".format(
                epoch=epoch_idx + 1,
                stage=cfg.stage_for_epoch(epoch_idx),
                train_loss=train_loss,
                valid_loss=metrics["valid_loss"],
                valid_auc=metrics["valid_auc"],
            )
        )

        if cfg.save_every_epoch:
            save_checkpoint(model, cfg, os.path.join(cfg.output_dir, f"epoch_{epoch_idx + 1}.pt"), metrics)
        auc = metrics.get("valid_auc", float("nan"))
        if not math.isnan(auc) and auc > best_auc:
            best_auc = auc
            best_metrics = metrics
            save_checkpoint(model, cfg, os.path.join(cfg.output_dir, "best.pt"), metrics)

    save_checkpoint(model, cfg, os.path.join(cfg.output_dir, "last.pt"), last_metrics)
    with open(os.path.join(cfg.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"best": best_metrics, "last": last_metrics}, f, indent=2)
    return best_metrics or last_metrics
