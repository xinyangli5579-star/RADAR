"""Rolling-window training of RADAR."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, fields

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import CHOICES, Config
from .data import CrossSectionalDataset, load_market_data, stock_histories
from .model import RADAR


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rolling_windows(n_days: int, cfg: Config):
    """(train_start, train_end, test_start, test_end) index tuples."""
    year = cfg.trading_days_per_year
    train, test, step = cfg.train_years * year, cfg.test_years * year, cfg.step_years * year
    n_windows = max((n_days - train - test) // step + 1, 0)
    return [(i * step, i * step + train, i * step + train, min(i * step + train + test, n_days))
            for i in range(n_windows)]


def make_loader(dataset, cfg: Config, shuffle: bool):
    return DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=cfg.num_workers,
                      pin_memory=cfg.device.startswith("cuda"), persistent_workers=cfg.num_workers > 0)


def run_epoch(model, loader, cfg: Config, optimizer=None) -> float:
    """Mean total loss over one pass; trains when an optimizer is given."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(cfg.device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            loss, _, _, _ = model.compute_loss(batch["return_seq"], batch["news_seq"], batch["news_mask"],
                                               batch["target"], current_t=batch["t"])
            if not torch.isfinite(loss):
                continue
            if training:
                optimizer.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                if not torch.isfinite(grad_norm):
                    continue
                optimizer.step()
            total += loss.item()
            n += 1
    return total / max(n, 1)


def train_window(k: int, window, data, cfg: Config, run_dir: str) -> dict:
    """Train one rolling window with early stopping on the validation loss and save the best model."""
    set_seed(cfg.seed)
    train_start, train_end, test_start, test_end = window
    dates = data.dates
    val_start = train_start + int((train_end - train_start) * (1 - cfg.val_ratio))
    print(f"\n[window {k}] train {dates[train_start]} ~ {dates[train_end - 1]} | "
          f"test {dates[test_start]} ~ {dates[test_end - 1]}")

    train_loader = make_loader(CrossSectionalDataset(data, train_start, val_start, cfg), cfg, shuffle=True)
    val_loader = make_loader(CrossSectionalDataset(data, val_start, train_end, cfg), cfg, shuffle=False)

    model = RADAR(cfg).to(cfg.device)
    history = stock_histories(data, train_start, train_end, cfg.news_dim) if model.uses_bank else None
    if history is not None:
        model.build_context_bank(*history, start_idx=train_start)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    best_loss, best_state, patience = math.inf, None, 0
    for epoch in range(1, cfg.epochs + 1):
        if history is not None and cfg.bank_refresh_freq > 0 and epoch > 1 \
                and (epoch - 1) % cfg.bank_refresh_freq == 0:
            model.build_context_bank(*history, start_idx=train_start)
        train_loss = run_epoch(model, train_loader, cfg, optimizer)
        val_loss = run_epoch(model, val_loader, cfg)
        scheduler.step()
        print(f"  epoch {epoch:3d} | train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_loss:
            best_loss, patience = val_loss, 0
            best_state = {name: t.detach().clone() for name, t in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"  early stop at epoch {epoch} (best val {best_loss:.6f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    train_period = [dates[train_start], dates[train_end - 1]]
    test_period = [dates[test_start], dates[test_end - 1]]
    checkpoint = f"window_{k}.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "train_period": train_period,
        "test_period": test_period,
        "best_val_loss": best_loss,
    }, os.path.join(run_dir, checkpoint))
    return {"window": k, "train_period": train_period, "test_period": test_period,
            "checkpoint": checkpoint, "best_val_loss": best_loss}


def run_training(cfg: Config, single_window: int | None = None) -> str:
    """Train all (or one) rolling windows and write rolling_results.json."""
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        cfg.device = "cpu"
    run_dir = os.path.join(cfg.save_dir, cfg.default_run_name, f"seed_{cfg.seed}")
    os.makedirs(run_dir, exist_ok=True)

    data = load_market_data(cfg.returns_path, cfg.news_path)
    windows = rolling_windows(len(data.dates), cfg)
    print(f"{len(data.dates)} days x {len(data.tickers)} stocks, {len(windows)} windows -> {run_dir}")
    selected = range(1, len(windows) + 1) if single_window is None else [single_window]

    results_path = os.path.join(run_dir, "rolling_results.json")
    results = {}
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = {w["window"]: w for w in json.load(f)["windows"]}
    for k in selected:
        results[k] = train_window(k, windows[k - 1], data, cfg, run_dir)
        with open(results_path, "w") as f:
            json.dump({"config": asdict(cfg), "windows": [results[i] for i in sorted(results)]}, f, indent=2)
    return results_path


def parse_config(argv=None):
    parser = argparse.ArgumentParser(description="Train RADAR over rolling windows.")
    for f in fields(Config):
        if isinstance(f.default, bool):
            continue
        parser.add_argument(f"--{f.name}", type=type(f.default), default=f.default, choices=CHOICES.get(f.name))
    parser.add_argument("--window", type=int, default=None, help="train only this window (1-indexed)")
    args = vars(parser.parse_args(argv))
    window = args.pop("window")
    return Config(**args), window


def main():
    cfg, window = parse_config()
    print(json.dumps(asdict(cfg), indent=2))
    path = run_training(cfg, window)
    print(f"\nresults: {path}")


if __name__ == "__main__":
    main()
