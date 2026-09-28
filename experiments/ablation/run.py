"""Train an ablated RADAR variant and generate its portfolio weights."""

from __future__ import annotations

import argparse
import os

from radar.config import ABLATIONS, Config
from radar.inference import generate_weights
from radar.train import run_training


def main():
    parser = argparse.ArgumentParser(description="Run one RADAR ablation.")
    parser.add_argument("--ablation", required=True, choices=[a for a in ABLATIONS if a != "none"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save_dir", default="checkpoints")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--stage", choices=("train", "weights", "all"), default="all")
    args = parser.parse_args()

    cfg = Config(ablation=args.ablation, seed=args.seed, device=args.device, save_dir=args.save_dir,
                 returns_path=os.path.join(args.data_dir, "log_returns.npz"),
                 news_path=os.path.join(args.data_dir, "news_finbert_sparse.pt"))
    results = os.path.join(cfg.save_dir, cfg.default_run_name, f"seed_{cfg.seed}", "rolling_results.json")
    if args.stage in ("train", "all"):
        results = run_training(cfg)
    if args.stage in ("weights", "all"):
        generate_weights(results, args.data_dir, os.path.join(cfg.save_dir, "weights", args.ablation),
                         args.device, args.seed)


if __name__ == "__main__":
    main()
