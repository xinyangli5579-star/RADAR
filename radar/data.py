"""Market data loading and cross-sectional samples."""

from __future__ import annotations

import os
from bisect import bisect_left, bisect_right
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MarketData:
    log_returns: torch.Tensor  # [D, N], NaN where a stock has no price
    dates: list
    tickers: list
    news: dict  # ticker -> {date: FinBERT embedding}


def load_market_data(returns_path: str, news_path: str) -> MarketData:
    if not os.path.exists(news_path):
        raise FileNotFoundError(f"news embeddings not found: {news_path}")
    arr = np.load(returns_path, allow_pickle=True)
    news = torch.load(news_path, map_location="cpu", weights_only=False)["data"]
    return MarketData(
        log_returns=torch.tensor(arr["log_returns"], dtype=torch.float32),
        dates=[str(d) for d in arr["dates"]],
        tickers=[str(t) for t in arr["tickers"]],
        news=news,
    )


def date_range(dates, start: str, end: str):
    """Closed date interval [start, end] as a half-open index range."""
    return bisect_left(dates, start), bisect_right(dates, end)


def return_window(data: MarketData, stocks, start: int, end: int) -> torch.Tensor:
    """Log returns of `stocks` over days [start, end) as [n_stocks, L, 1]."""
    window = data.log_returns[start:end][:, stocks]
    return torch.nan_to_num(window, nan=0.0).T.unsqueeze(-1).contiguous()


def news_window(data: MarketData, stocks, start: int, end: int, news_dim: int):
    """News embeddings [n_stocks, L, news_dim] and no-news mask [n_stocks, L] over days [start, end)."""
    days = data.dates[start:end]
    seq = torch.zeros(len(stocks), len(days), news_dim)
    mask = torch.ones(len(stocks), len(days), dtype=torch.bool)
    for i, s in enumerate(stocks):
        by_date = data.news.get(data.tickers[int(s)])
        if not by_date:
            continue
        for j, d in enumerate(days):
            emb = by_date.get(d)
            if emb is not None:
                seq[i, j] = torch.as_tensor(emb, dtype=torch.float32)
                mask[i, j] = False
    return seq, mask


def stock_histories(data: MarketData, start: int, end: int, news_dim: int):
    """Per-stock return, news and mask histories over [start, end) for the context bank."""
    n_stocks = len(data.tickers)
    returns = torch.nan_to_num(data.log_returns[start:end], nan=0.0)
    news, masks = news_window(data, range(n_stocks), start, end, news_dim)
    return [returns[:, s:s + 1] for s in range(n_stocks)], list(news), list(masks)


def holding_period_returns(log_returns: torch.Tensor, mode: str = "gross") -> torch.Tensor:
    """Aggregate daily log returns into gross or net holding-period returns."""
    cumulative_log_return = log_returns.sum(dim=0)
    if mode == "gross":
        return torch.exp(cumulative_log_return)
    if mode == "net":
        return torch.expm1(cumulative_log_return)
    raise ValueError(f"return mode must be 'gross' or 'net', got {mode!r}")


class CrossSectionalDataset(Dataset):
    """One item per date t with a separate SDF target and rebalance horizon.

    Inputs use [t - seq_len, t), while the training target covers
    [t, t + target_horizon). The rebalance horizon is kept in Config for
    inference/backtesting and is intentionally independent of the SDF target.

    Target windows lie inside [start, end).
    """

    def __init__(self, data: MarketData, start: int, end: int, cfg):
        self.data, self.cfg = data, cfg
        lr = data.log_returns
        min_valid = min(cfg.n_stocks_per_sample, lr.shape[1] // 2)
        self.times = [
            t for t in range(max(start, cfg.seq_len), end - cfg.target_horizon + 1)
            if int((~torch.isnan(lr[t:t + cfg.target_horizon])).all(0).sum()) >= min_valid
        ]
        if self.times:
            print(f"[dataset] {len(self.times)} dates, {data.dates[self.times[0]]} ~ {data.dates[self.times[-1]]}")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, i):
        cfg, t = self.cfg, self.times[i]
        future = self.data.log_returns[t:t + cfg.target_horizon]
        stocks = torch.where((~torch.isnan(future)).all(0))[0]
        if len(stocks) > cfg.n_stocks_per_sample:
            stocks = stocks[torch.randperm(len(stocks))[: cfg.n_stocks_per_sample]]
        news, mask = news_window(self.data, stocks.tolist(), t - cfg.seq_len, t, cfg.news_dim)
        return {
            "return_seq": return_window(self.data, stocks, t - cfg.seq_len, t),
            "news_seq": news,
            "news_mask": mask,
            "target": holding_period_returns(future[:, stocks], cfg.target_return_mode),
            "t": t,
        }
