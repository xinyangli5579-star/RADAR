"""GICS sector and volatility tercile of each stock in the universe (stock_metadata.json)."""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = "data/processed"
UNIVERSE_PATH = os.path.join(DATA_DIR, "universe.json")
RETURNS_PATH = os.path.join(DATA_DIR, "log_returns.npz")
OUTPUT_PATH = os.path.join(DATA_DIR, "stock_metadata.json")
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_sectors() -> dict:
    """{ticker: GICS sector} from the Wikipedia constituents table."""
    from io import StringIO

    import requests

    resp = requests.get(WIKI_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()
    df = pd.read_html(StringIO(resp.text))[0]
    ticker_col, sector_col = "Symbol", "GICS Sector"
    if ticker_col not in df.columns or sector_col not in df.columns:
        for col in df.columns:
            if "symbol" in col.lower() or "ticker" in col.lower():
                ticker_col = col
            if "sector" in col.lower() and "sub" not in col.lower():
                sector_col = col
    return {str(r[ticker_col]).strip().replace(".", "-"): str(r[sector_col]).strip() for _, r in df.iterrows()}


def compute_volatility_groups(returns_path: str, tickers: list) -> dict:
    """{ticker: {annual_vol, vol_group}} with Low/Mid/High terciles of annualised volatility."""
    data = np.load(returns_path, allow_pickle=True)
    log_returns = data["log_returns"]
    ticker_to_idx = {t: i for i, t in enumerate(data["tickers"])}
    vols = {}
    for ticker in tickers:
        if ticker in ticker_to_idx:
            rets = log_returns[:, ticker_to_idx[ticker]]
            valid = rets[np.isfinite(rets)]
            if len(valid) >= 60:
                vols[ticker] = float(np.std(valid)) * np.sqrt(252)
    p33, p66 = np.percentile(list(vols.values()), [33.3, 66.7])
    groups = {}
    for ticker, vol in vols.items():
        group = "Low" if vol <= p33 else "Mid" if vol <= p66 else "High"
        groups[ticker] = {"annual_vol": round(vol, 4), "vol_group": group}
    return groups


def main():
    with open(UNIVERSE_PATH) as f:
        tickers = json.load(f)["tickers"]
    sectors = fetch_sp500_sectors()
    vol_groups = compute_volatility_groups(RETURNS_PATH, tickers)

    stocks, unmatched = {}, []
    for ticker in tickers:
        if ticker not in sectors:
            unmatched.append(ticker)
        vol = vol_groups.get(ticker, {"annual_vol": None, "vol_group": "Unknown"})
        stocks[ticker] = {"sector": sectors.get(ticker, "Unknown"), **vol}

    output = {"meta": {"n_tickers": len(tickers), "n_sector_matched": len(tickers) - len(unmatched),
                       "n_unmatched": len(unmatched), "unmatched_tickers": unmatched},
              "stocks": stocks}
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT_PATH} ({len(tickers) - len(unmatched)}/{len(tickers)} sectors matched)")


if __name__ == "__main__":
    main()
