"""Step 4: merge FinBERT shards into the single news file used for training."""

from __future__ import annotations

import argparse
import glob

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news_sparse_pt", default="data/processed/news_finbert_sparse.part*.pt",
                    help="shard path or glob pattern")
    ap.add_argument("--out", default="data/processed/news_finbert_sparse.pt")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.news_sparse_pt))
    if not paths:
        raise FileNotFoundError(f"no files match {args.news_sparse_pt}")
    merged, header, n_cells = {}, {}, 0
    for path in paths:
        shard = torch.load(path, map_location="cpu", weights_only=False)
        header = {k: shard[k] for k in ("model_name", "dim", "dtype") if k in shard}
        for ticker, by_date in shard["data"].items():
            merged.setdefault(ticker, {}).update(by_date)
            n_cells += len(by_date)
    torch.save({**header, "data": merged}, args.out)
    print(f"[OK] merged {len(paths)} file(s) into {args.out}: {len(merged)} tickers, {n_cells} cells")


if __name__ == "__main__":
    main()
