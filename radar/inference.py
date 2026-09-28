"""Daily portfolio weights from trained RADAR checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch

from .config import Config
from .data import date_range, load_market_data, news_window, return_window, stock_histories
from .model import RADAR
from .train import set_seed


@torch.no_grad()
def window_weights(model, data, start: int, end: int, cfg: Config, seed: int):
    """Weights for test days [start, end), rebalanced every `horizon` days."""
    t0 = max(start, cfg.seq_len)
    n_days, n_stocks = end - t0, len(data.tickers)
    weights = np.zeros((n_days, n_stocks), dtype=np.float32)
    current = None
    for offset in range(0, n_days, cfg.horizon):
        t = t0 + offset
        stocks = torch.where(~torch.isnan(data.log_returns[t]))[0]
        if len(stocks) >= max(1, n_stocks // 2):
            news, mask = news_window(data, stocks.tolist(), t - cfg.seq_len, t, cfg.news_dim)
            torch.manual_seed(seed + t)
            z = model(return_window(data, stocks, t - cfg.seq_len, t).to(model.device),
                      news.to(model.device), mask.to(model.device), current_t=t)
            w, _ = model.portfolio_head(z)
            current = np.zeros(n_stocks, dtype=np.float32)
            current[stocks.numpy()] = w.cpu().numpy()
        if current is not None:
            weights[offset:offset + cfg.horizon] = current
    return data.dates[t0:end], weights


def generate_weights(results_path: str, data_dir: str, output_dir: str, device: str = "cuda", seed: int = 42):
    """Write window_<k>.npz weight files and run_meta.json for every trained window."""
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    with open(results_path) as f:
        results = json.load(f)
    run_dir = os.path.dirname(os.path.abspath(results_path))
    data = load_market_data(os.path.join(data_dir, "log_returns.npz"),
                            os.path.join(data_dir, "news_finbert_sparse.pt"))
    os.makedirs(output_dir, exist_ok=True)

    windows_meta, horizon = [], None
    for entry in results["windows"]:
        k = int(entry["window"])
        ckpt = torch.load(os.path.join(run_dir, entry["checkpoint"]), map_location=device, weights_only=True)
        cfg = Config(**{**ckpt["config"], "device": device})
        horizon = cfg.horizon
        model = RADAR(cfg).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        train_start, train_end = date_range(data.dates, *entry["train_period"])
        test_start, test_end = date_range(data.dates, *entry["test_period"])
        if model.uses_bank:
            set_seed(seed)
            model.build_context_bank(*stock_histories(data, train_start, train_end, cfg.news_dim),
                                     start_idx=train_start)

        dates, weights = window_weights(model, data, test_start, test_end, cfg, seed + 100000 * k)
        path = os.path.join(output_dir, f"window_{k}.npz")
        np.savez_compressed(
            path,
            dates=np.array(dates, dtype=object),
            tickers=np.array(data.tickers, dtype=object),
            weights=weights,
            train_period=np.array(entry["train_period"], dtype=object),
            test_period=np.array(entry["test_period"], dtype=object),
        )
        print(f"[window {k}] {len(dates)} days ({dates[0]} ~ {dates[-1]}) -> {path}")
        windows_meta.append({"window": k, "train_period": entry["train_period"],
                             "test_period": entry["test_period"], "weights_path": path})

    meta = {"results": os.path.abspath(results_path), "horizon": horizon, "seed": seed,
            "created_at": datetime.now().isoformat(timespec="seconds"), "windows": windows_meta}
    with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate portfolio weights from RADAR checkpoints.")
    parser.add_argument("--results", default="checkpoints/radar/seed_42/rolling_results.json")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output", default="checkpoints/weights/radar")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_weights(args.results, args.data_dir, args.output, args.device, args.seed)


if __name__ == "__main__":
    main()
