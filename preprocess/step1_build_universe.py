"""Step 1: intersect the tickers of the price file and the news file."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.common import (ensure_dir, parse_sp500_full_multiindex_schema, read_news_header_tickers,
                               read_news_long_format_tickers, save_universe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp500_csv", default="data/raw/sp500_full_multiindex.csv")
    ap.add_argument("--news_csv", default="data/raw/All_external.csv")
    ap.add_argument("--news_format", choices=["wide", "long"], default="long")
    ap.add_argument("--out_dir", default="data/processed")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    sp500_tickers = parse_sp500_full_multiindex_schema(args.sp500_csv).tickers
    if args.news_format == "wide":
        news_tickers = set(read_news_header_tickers(args.news_csv))
    else:
        news_tickers = set(read_news_long_format_tickers(args.news_csv))
    tickers = sorted(t for t in sp500_tickers if t in news_tickers)

    out_path = os.path.join(args.out_dir, "universe.json")
    save_universe(out_path, tickers, meta={
        "source": f"intersection(sp500_full_multiindex.csv tickers, {args.news_csv} tickers)",
        "news_format": args.news_format,
        "sp500_ticker_count": len(sp500_tickers),
        "news_ticker_count": len(news_tickers),
        "intersection_count": len(tickers),
    })
    print(f"[OK] wrote {out_path} (n={len(tickers)})")


if __name__ == "__main__":
    main()
