"""Read the existing audited CZ/CRSP cache; adapt batches, not RADAR's model."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from radar.data import news_window


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def prior_month(dates):
    p = pd.DatetimeIndex(dates).to_period("M") - 1
    return np.asarray(p.year * 100 + p.month, dtype=np.int32)


def verify_cache(folder):
    """Verify cached ranks/screens and select the same 202 using pre-2011 data only.

    The producer's seven discarded columns are constant on its 2009-2019 training
    mask. Our earlier calibration mask is verified to be a subset of that mask;
    all 202 retained columns must vary on this earlier mask independently.
    Thus no future observation is needed to decide this experiment's feature set.
    """
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    for name, digest in manifest["output_sha256"].items():
        if sha(folder / name) != digest:
            raise ValueError(f"Cache hash mismatch: {name}")
    assert len(manifest["feature_names"]) == 202
    assert len(manifest["dropped_training_constant"]) == 7
    x = np.load(folder / "monthly_features.npy", mmap_mode="r")
    dates = np.load(folder / "dates.npy")
    mi = np.load(folder / "month_index.npy")
    member = np.load(folder / "member.npy", mmap_mode="r")
    mask = np.zeros(x.shape[:2], bool)
    for t in np.flatnonzero((dates >= "2009-01-01") & (dates <= "2010-12-31")):
        mask[:, mi[t]] |= member[:, t]
    assert mask.any() and np.all(~mask | np.load(folder / "training_feature_mask.npy"))
    assert (np.ptp(x[mask], axis=0) > 1e-12).sum() == 202
    assert np.array_equal(np.load(folder / "source_months.npy")[mi], prior_month(dates))
    caps = np.load(folder / "lagged_cap.npy", mmap_mode="r")
    breaks = pd.read_csv(folder / "nyse20_breakpoints.csv").set_index("yyyymm").nyse20
    thresholds = breaks.reindex(prior_month(dates)).to_numpy()
    assert np.isfinite(thresholds).all()
    assert np.all(~member | (np.isfinite(caps) & (caps >= thresholds[None, :])))
    return {
        "manifest_sha256": sha(folder / "manifest.json"),
        "feature_names": manifest["feature_names"],
        "feature_calibration": ["2009-01-01", "2010-12-31"],
        "training_only_feature_selection_verified": True,
        "lagged_nyse20_verified": True,
    }


class Panel:
    def __init__(self, folder, news_path, start="2011-06-02", end="2020-06-30"):
        folder = Path(folder)
        dates = np.load(folder / "dates.npy")
        a, b = np.searchsorted(dates, [start, end])
        b += int(b < len(dates) and dates[b] == end)
        self.dates = dates[a:b]
        self.symbols = np.load(folder / "symbols.npy")
        self.tickers = (
            self.symbols.tolist()
        )  # original news_window contract, keyed by PERMNO
        self.monthly_features = np.load(folder / "monthly_features.npy", mmap_mode="r")
        self.month_index = np.load(folder / "month_index.npy")[a:b]
        for name in (
            "member",
            "tradable",
            "totalret",
            "terminal_events",
            "price_return",
            "lagged_cap",
        ):
            setattr(self, name, np.load(folder / f"{name}.npy", mmap_mode="r")[:, a:b])
        self.news = torch.load(news_path, map_location="cpu", weights_only=False)[
            "data"
        ]

    def ids(self, signal_day):
        # No future-return/price test is allowed here; also used at inference.
        return np.flatnonzero(self.member[:, signal_day] & self.tradable[:, signal_day])

    def features(self, ids, start, end):
        return torch.from_numpy(
            np.array(
                self.monthly_features[
                    np.asarray(ids)[:, None], self.month_index[start:end][None, :]
                ],
                copy=True,
            )
        )


class Samples(Dataset):
    """Inputs through t-1 close, execution at t close, first payoff at t+1 close."""

    def __init__(self, data, start, end, cfg):
        self.data, self.cfg = data, cfg
        assert cfg.return_dim == 202 and cfg.target_horizon == 1
        self.times = []
        for t in range(max(start, cfg.seq_len), end - 1):
            ids = data.ids(t - 1)
            if np.isfinite(data.totalret[ids, t + 1]).sum() >= 10:
                self.times.append(t)
        if not self.times:
            raise ValueError("No eligible training samples")

    def __len__(self):
        return len(self.times)

    def __getitem__(self, index):
        data, cfg, t = self.data, self.cfg, self.times[index]
        ids = data.ids(t - 1)
        # Historical target availability is used only for supervised fitting.
        ids = ids[np.isfinite(data.totalret[ids, t + 1])]
        if len(ids) > cfg.n_stocks_per_sample:
            ids = ids[torch.randperm(len(ids))[: cfg.n_stocks_per_sample].numpy()]
        news, mask = news_window(data, ids.tolist(), t - cfg.seq_len, t, cfg.news_dim)
        target = torch.tensor(np.array(data.totalret[ids, t + 1]), dtype=torch.float32)
        if cfg.target_return_mode == "gross":
            target = target + 1
        return {
            "return_seq": data.features(ids, t - cfg.seq_len, t),
            "news_seq": news,
            "news_mask": mask,
            "target": target,
            "t": t,
        }


def histories(data, start, end, news_dim):
    features = data.features(np.arange(len(data.symbols)), start, end)
    news, masks = news_window(data, range(len(data.symbols)), start, end, news_dim)
    # A future S&P entrant must not enter earlier cross-stock retrieval banks.
    member = torch.from_numpy(np.array(data.member[:, start:end], copy=True))
    masks |= ~member
    news.masked_fill_(masks.unsqueeze(-1), 0)
    features *= member.unsqueeze(-1)
    return list(features), list(news), list(masks)
