"""Reuse verified CZ/CRSP arrays and FinBERT cells; encode only missing tickers."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocess.step3_encode_news_finbert import main as encode_news

from .data import sha, verify_cache


def prepare(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    audit = verify_cache(args.shared_data)
    symbols = np.load(Path(args.shared_data) / "symbols.npy")
    names = pd.read_parquet(
        args.names, columns=["permno", "namedt", "nameenddt", "ticker"]
    )
    names = (
        names[names.permno.isin([int(s[1:]) for s in symbols])]
        .dropna(subset=["ticker"])
        .copy()
    )
    names["start"] = names.namedt.astype(str)
    names["end"] = names.nameenddt.astype(str)
    names = names[(names.start <= "2020-06-30") & (names.end >= "2011-06-01")]
    needed = sorted(names.ticker.unique())
    cached = torch.load(args.cached_news, map_location="cpu", weights_only=False)
    assert cached["dim"] == 768
    universe = out / "additional_news_universe.json"
    universe.write_text(
        json.dumps({"tickers": sorted(set(needed) - set(cached["data"]))}) + "\n"
    )
    shards = out / "additional_news"
    shards.mkdir(exist_ok=True)
    if not (shards / "complete.json").exists():
        for p in shards.glob("news_finbert_sparse.part*.pt"):
            p.unlink()
        csv.field_size_limit(2**31 - 1)
        encode_news(
            [
                "--news_csv",
                args.raw_news,
                "--universe",
                str(universe),
                "--start",
                "2011-06-01",
                "--end",
                "2020-06-30",
                "--out_dir",
                str(shards),
                "--model_name",
                args.finbert,
                "--device",
                args.device,
                "--batch_size",
                "8",
                "--fp16_store",
                "--shard_size_cells",
                "20000",
                "--progress_every",
                "100",
            ]
        )
        (shards / "complete.json").write_text(
            json.dumps(
                {
                    "raw_sha256": sha(args.raw_news),
                    "additional_tickers": json.loads(universe.read_text())["tickers"],
                }
            )
            + "\n"
        )
    all_news = cached["data"]
    for p in sorted(shards.glob("news_finbert_sparse.part*.pt")):
        for ticker, days in torch.load(p, map_location="cpu", weights_only=False)[
            "data"
        ].items():
            all_news.setdefault(ticker, {}).update(days)
    mapped = {}
    counts = {"mapped": 0, "ambiguous": 0, "outside_name_interval": 0}
    for ticker, group in names.groupby("ticker"):
        intervals = list(
            group[["permno", "start", "end"]].itertuples(index=False, name=None)
        )
        for day, embedding in all_news.get(ticker, {}).items():
            candidates = {int(p) for p, a, b in intervals if a <= day <= b}
            if len(candidates) != 1:
                counts["ambiguous" if candidates else "outside_name_interval"] += 1
                continue
            symbol = f"P{candidates.pop()}"
            if day in mapped.setdefault(symbol, {}):
                # Overlapping symbol histories must not silently overwrite a cell.
                raise ValueError(f"Multiple news symbols map to {symbol} on {day}")
            mapped[symbol][day] = embedding
            counts["mapped"] += 1
    torch.save({"dim": 768, "dtype": "float16", "data": mapped}, out / "news.pt")
    audit.update(
        {
            "news_mapping": counts,
            "news_sha256": sha(out / "news.pt"),
            "source_sha256": {
                k: sha(v)
                for k, v in {
                    "names": args.names,
                    "raw_news": args.raw_news,
                    "cached_news": args.cached_news,
                }.items()
            },
            "notes": [
                "Keep missing news masked; do not condition the universe on future news coverage.",
                "Use exact ticker and dated CRSP name intervals; exclude ambiguous or unmatched mappings.",
                "CZ features use pre-2011 calibration. A retained characteristic may be constant in a later training window.",
            ],
        }
    )
    (out / "data_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps({"news_mapping": counts, "features": len(audit["feature_names"])}),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    for name in ("shared_data", "names", "raw_news", "cached_news"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--out", default="data/cz_vw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--finbert", default="ProsusAI/finbert")
    prepare(p.parse_args())


if __name__ == "__main__":
    main()
