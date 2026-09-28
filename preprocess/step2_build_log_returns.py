"""Step 2: daily log returns log(p_t) - log(p_{t-1}) of the universe, saved as log_returns.npz."""

from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.common import ensure_dir, in_range, load_universe, parse_date, parse_sp500_full_multiindex_schema


def _safe_float(x) -> float:
    s = "" if x is None else str(x).strip()
    if s.lower() in {"", "nan", "none", "null"}:
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500_csv", default="data/raw/sp500_full_multiindex.csv")
    ap.add_argument("--universe", default="data/processed/universe.json")
    ap.add_argument("--start", default="2011-06-01")
    ap.add_argument("--end", default="2020-06-30")
    ap.add_argument("--out_dir", default="data/processed")
    ap.add_argument("--price_field", default="Adj Close")
    args = ap.parse_args()

    start, end = parse_date(args.start), parse_date(args.end)
    ensure_dir(args.out_dir)
    schema = parse_sp500_full_multiindex_schema(args.sp500_csv)
    tickers = [t for t in load_universe(args.universe) if (t, args.price_field) in schema.col_index]
    if not tickers:
        raise ValueError("No tickers left after filtering by price field availability.")
    col_idx = [schema.col_index[(t, args.price_field)] for t in tickers]

    dates, prices = [], []
    with open(args.sp500_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for _ in range(3):
            next(reader)
        for row in reader:
            ds = row[0].strip() if row else ""
            if not ds or ds == "Date":
                continue
            try:
                d = parse_date(ds)
            except ValueError:
                continue
            if not in_range(d, start, end):
                continue
            dates.append(ds)
            prices.append([_safe_float(row[j]) if j < len(row) else float("nan") for j in col_idx])

    if len(dates) < 10:
        raise ValueError(f"Too few rows in range {args.start}~{args.end}: got {len(dates)}")
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(np.asarray(prices, dtype=np.float64))
    log_ret = logp[1:] - logp[:-1]
    log_ret[~np.isfinite(log_ret)] = np.nan

    out_path = os.path.join(args.out_dir, "log_returns.npz")
    np.savez_compressed(out_path, dates=np.array(dates[1:], dtype="U10"), tickers=np.array(tickers, dtype="U16"),
                        log_returns=log_ret.astype(np.float32))
    print(f"[OK] wrote {out_path}: {dates[1]} .. {dates[-1]} ({len(dates) - 1} days), {len(tickers)} tickers")


if __name__ == "__main__":
    main()
