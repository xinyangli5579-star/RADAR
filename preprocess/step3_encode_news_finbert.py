"""Step 3: encode each (ticker, day) news text with FinBERT and store the sparse [CLS] embeddings."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.common import ensure_dir, iter_news_rows, iter_news_rows_long_format, load_universe, parse_date


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--news_csv", default="data/raw/All_external.csv")
    ap.add_argument("--news_format", choices=["wide", "long"], default="long")
    ap.add_argument("--universe", default="data/processed/universe.json")
    ap.add_argument("--start", default=None, help="defaults to the first date of log_returns.npz")
    ap.add_argument("--end", default=None, help="defaults to the last date of log_returns.npz")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--model_name", default="ProsusAI/finbert")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--fp16_store", action="store_true", help="store embeddings as float16")
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--progress_every", type=int, default=200, help="dates between progress logs (0 = off)")
    ap.add_argument("--shard_size_cells", type=int, default=0,
                    help="write a shard every N encoded cells (0 = single file); merge shards with step 4")
    ap.add_argument("--shard_prefix", default="news_finbert_sparse")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    tickers = load_universe(args.universe)
    returns_path = os.path.join(args.out_dir, "log_returns.npz")
    if (args.start is None or args.end is None) and os.path.exists(returns_path):
        dates = np.load(returns_path)["dates"].astype(str).tolist()
        args.start = args.start or dates[0]
        args.end = args.end or dates[-1]
    args.start = args.start or "2011-06-01"
    args.end = args.end or "2020-06-30"
    start, end = parse_date(args.start), parse_date(args.end)

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModel.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model.eval().to(torch.device(args.device))
    print(f"[init] model={args.model_name} device={args.device} batch={args.batch_size} "
          f"max_length={args.max_length} tickers={len(tickers)} range=[{args.start},{args.end}]")

    dtype = np.float16 if args.fp16_store else np.float32
    data: Dict[str, Dict[str, np.ndarray]] = {}
    buf_texts: List[str] = []
    buf_keys: List[Tuple[str, str]] = []
    written: List[str] = []
    shard_idx = cells_in_shard = n_cells = n_rows = 0
    t0 = time.time()

    def flush() -> None:
        nonlocal buf_texts, buf_keys
        if not buf_texts:
            return
        with torch.no_grad():
            enc = tokenizer(buf_texts, padding=True, truncation=True, max_length=args.max_length,
                            return_tensors="pt")
            enc = {k: v.to(args.device) for k, v in enc.items()}
            out = model(**enc).last_hidden_state[:, 0, :].cpu().numpy()
        for (t, ds), emb in zip(buf_keys, out):
            data.setdefault(t, {})[ds] = emb.astype(dtype, copy=False)
        buf_texts, buf_keys = [], []

    def save(final: bool) -> None:
        nonlocal data, shard_idx, cells_in_shard
        if not data or (args.shard_size_cells <= 0 and not final):
            return
        suffix = f".part{shard_idx:04d}" if args.shard_size_cells > 0 else ""
        path = os.path.join(args.out_dir, f"{args.shard_prefix}{suffix}.pt")
        torch.save({"model_name": args.model_name, "dim": 768, "dtype": np.dtype(dtype).name, "data": data,
                    "meta": {"start": args.start, "end": args.end, "encoded_cells": n_cells,
                             "shard_idx": shard_idx, "cells_in_shard": cells_in_shard, "final": final}}, path)
        written.append(path)
        print(f"[save] {path} (tickers={len(data)}, cells={cells_in_shard})")
        data, cells_in_shard, shard_idx = {}, 0, shard_idx + 1

    rows = (iter_news_rows(args.news_csv, tickers, start, end) if args.news_format == "wide"
            else iter_news_rows_long_format(args.news_csv, tickers, start, end))
    for ds, texts in rows:
        n_rows += 1
        for t, text in texts.items():
            n_cells += 1
            cells_in_shard += 1
            buf_texts.append(text)
            buf_keys.append((t, ds))
            if len(buf_texts) >= args.batch_size:
                flush()
        if args.progress_every > 0 and n_rows % args.progress_every == 0:
            dt = time.time() - t0
            print(f"[progress] dates={n_rows} cells={n_cells} cells/sec={n_cells / max(dt, 1e-9):.2f} "
                  f"elapsed={dt / 60:.1f}m")
        if args.shard_size_cells > 0 and cells_in_shard >= args.shard_size_cells:
            flush()
            save(final=False)
    flush()
    save(final=True)
    print(f"[OK] {len(written)} file(s), {n_cells} encoded cells")


if __name__ == "__main__":
    main()
