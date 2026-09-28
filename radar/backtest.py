"""Model-agnostic backtest of pre-computed daily portfolio weights."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def calculate_portfolio_metrics(daily_returns, risk_free_rate: float = 0.0,
                                trading_days_per_year: int = TRADING_DAYS) -> dict:
    """Return, risk and risk-adjusted metrics of a daily simple-return series."""
    r = np.asarray(daily_returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = len(r)
    keys = ["cumulative_return", "annualized_return", "annualized_volatility", "sharpe_ratio",
            "sortino_ratio", "max_drawdown", "calmar_ratio", "win_rate", "n_days"]
    if n == 0:
        return {k: 0.0 for k in keys}

    nav = np.cumprod(1.0 + r)
    cum_ret = float(nav[-1] - 1.0)
    ann_ret = float((1.0 + cum_ret) ** (1.0 / max(n / trading_days_per_year, 1e-8)) - 1.0)
    ann_vol = float(np.std(r, ddof=1) * np.sqrt(trading_days_per_year)) if n > 1 else 0.0

    excess = r - ((1.0 + risk_free_rate) ** (1.0 / trading_days_per_year) - 1.0)
    sharpe = float(np.mean(excess) / (np.std(excess, ddof=1) + 1e-12) * np.sqrt(trading_days_per_year))
    downside = excess[excess < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 1e-12
    sortino = float(np.mean(excess) / (downside_std + 1e-12) * np.sqrt(trading_days_per_year))

    peak = np.maximum.accumulate(nav)
    max_dd = float(np.min((nav - peak) / peak))
    calmar = float(ann_ret / (abs(max_dd) + 1e-12)) if abs(max_dd) > 1e-8 else 0.0

    return {
        "cumulative_return": cum_ret,
        "annualized_return": ann_ret,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "calmar_ratio": calmar,
        "win_rate": float(np.mean(r > 0)),
        "n_days": n,
    }


def load_local_stock_data(data_dir: str):
    data = np.load(os.path.join(data_dir, "log_returns.npz"), allow_pickle=True)
    return (data["log_returns"].astype(np.float32),
            [str(d) for d in data["dates"]],
            [str(t) for t in data["tickers"]])


def fetch_or_load_sp500_returns(start_date: str, end_date: str, cache_dir: str = "data/sp500",
                                ticker: str = "^GSPC") -> pd.DataFrame:
    """Daily S&P 500 returns as DataFrame(date, strategy_return), cached as CSV."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"sp500_{start_date}_{end_date}.csv")
    if os.path.exists(cache_file):
        df = pd.read_csv(cache_file, parse_dates=["date"])
    else:
        import yfinance as yf

        raw = yf.download(ticker, start=start_date, end=end_date, progress=False)
        close = None
        for field in ("Adj Close", "Close"):
            if isinstance(raw.columns, pd.MultiIndex) and field in raw.columns.get_level_values(0):
                sub = raw.xs(field, axis=1, level=0)
                close = sub[ticker] if ticker in sub.columns else sub.iloc[:, 0]
                break
            if not isinstance(raw.columns, pd.MultiIndex) and field in raw.columns:
                close = raw[field]
                break
        if close is None:
            raise KeyError("no 'Adj Close' or 'Close' column in the yfinance result")
        rets = close.pct_change().dropna()
        df = pd.DataFrame({"date": pd.to_datetime(rets.index).tz_localize(None), "return": rets.values})
        df.to_csv(cache_file, index=False)
    return df.rename(columns={"return": "strategy_return"})


def compute_equal_weight_returns(log_returns, dates, test_dates) -> pd.DataFrame:
    """Daily-rebalanced equal-weight portfolio of all stocks with a valid return."""
    date_to_idx = {d: i for i, d in enumerate(dates)}
    records = []
    for d in test_dates:
        lr = log_returns[date_to_idx[d]] if d in date_to_idx else np.array([])
        lr = lr[np.isfinite(lr)]
        if len(lr):
            records.append((pd.Timestamp(d), float(np.mean(np.exp(lr) - 1.0))))
    return pd.DataFrame(records, columns=["date", "strategy_return"])


def load_all_window_weights(weights_dir: str):
    """Load window_<k>.npz files ordered by k."""
    files = [f for f in os.listdir(weights_dir) if f.startswith("window_") and f.endswith(".npz")]
    if not files:
        raise FileNotFoundError(f"no window_*.npz files in {weights_dir}")
    windows = []
    for name in sorted(files, key=lambda f: int(f[len("window_"):-len(".npz")])):
        data = np.load(os.path.join(weights_dir, name), allow_pickle=True)
        windows.append({"dates": [str(d) for d in data["dates"]],
                        "tickers": [str(t) for t in data["tickers"]],
                        "weights": data["weights"]})
    return windows


def compute_strategy_returns(window_weights, log_returns, all_dates, all_tickers,
                             rebalance_freq: int = 7) -> pd.DataFrame:
    """Daily returns when rebalancing to the target weights every `rebalance_freq` days and drifting in between."""
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    ticker_to_idx = {t: i for i, t in enumerate(all_tickers)}
    n_all = log_returns.shape[1]

    targets = {}
    for win in window_weights:
        cols = np.array([ticker_to_idx.get(t, -1) for t in win["tickers"]])
        keep = cols >= 0
        for d, w in zip(win["dates"], win["weights"]):
            if d in date_to_idx:
                target = np.zeros(n_all, dtype=np.float64)
                target[cols[keep]] = w[keep]
                targets[d] = target

    records, current, since_rebalance = [], None, 0
    for d in sorted(targets):
        if current is None or since_rebalance >= rebalance_freq:
            current, since_rebalance = targets[d].copy(), 0
        simple = np.exp(log_returns[date_to_idx[d]]) - 1.0
        simple = np.where(np.isfinite(simple), simple, 0.0)
        port = float(np.dot(current, simple))
        records.append((pd.Timestamp(d), port))
        if abs(1.0 + port) > 1e-12:
            current = current * (1.0 + simple) / (1.0 + port)
        since_rebalance += 1
    return pd.DataFrame(records, columns=["date", "strategy_return"])


def metrics_table(returns: dict) -> pd.DataFrame:
    formats = [
        ("Cumulative Return", "cumulative_return", "{:.2%}"),
        ("Annualized Return", "annualized_return", "{:.2%}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
        ("Sortino Ratio", "sortino_ratio", "{:.3f}"),
        ("Max Drawdown", "max_drawdown", "{:.2%}"),
        ("Calmar Ratio", "calmar_ratio", "{:.3f}"),
        ("Volatility", "annualized_volatility", "{:.2%}"),
        ("Win Rate", "win_rate", "{:.1%}"),
        ("Trading Days", "n_days", "{:.0f}"),
    ]
    metrics = {name: calculate_portfolio_metrics(r) for name, r in returns.items()}
    return pd.DataFrame([{"Metric": label, **{name: fmt.format(m[key]) for name, m in metrics.items()}}
                         for label, key, fmt in formats])


def plot_curves(series: dict, output_dir: str):
    """NAV and drawdown plots of every return series."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.cm.tab10.colors
    fig_nav, ax_nav = plt.subplots(figsize=(14, 7), dpi=150)
    fig_dd, ax_dd = plt.subplots(figsize=(14, 5), dpi=150)
    for i, (name, df) in enumerate(series.items()):
        nav = (1.0 + df["strategy_return"]).cumprod().to_numpy()
        peak = np.maximum.accumulate(nav)
        benchmark = name in ("Equal Weight", "S&P 500")
        ax_nav.plot(df["date"], nav, label=name, color=colors[i % 10],
                    linewidth=1.6 if benchmark else 2.2, alpha=0.7 if benchmark else 1.0)
        ax_dd.fill_between(df["date"], (nav - peak) / peak, 0, alpha=0.3, color=colors[i % 10], label=name)
    for ax, title, fig, name in ((ax_nav, "NAV", fig_nav, "nav"), (ax_dd, "Drawdown", fig_dd, "drawdown")):
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"backtest_{name}.png"), bbox_inches="tight")
        plt.close(fig)


def _horizon_from_meta(weights_dir: str, default: int = 7) -> int:
    path = os.path.join(weights_dir, "run_meta.json")
    if os.path.exists(path):
        with open(path) as f:
            return int(json.load(f).get("horizon") or default)
    return default


def run_backtest(weights_dirs, data_dir: str = "data/processed", output_dir: str = "results/main",
                 sp500_cache_dir: str = "data/sp500", include_equal_weight: bool = True,
                 include_sp500: bool = True, rebalance_freq: int | None = None) -> dict:
    """Backtest each weights directory against the benchmarks and write metrics, plots and returns."""
    log_returns, dates, tickers = load_local_stock_data(data_dir)
    series = {}
    for wdir in weights_dirs:
        name = os.path.basename(os.path.normpath(wdir))
        freq = rebalance_freq or _horizon_from_meta(wdir)
        series[name] = compute_strategy_returns(load_all_window_weights(wdir), log_returns, dates, tickers, freq)
        print(f"[backtest] {name}: {len(series[name])} days, rebalance every {freq} days")

    strategy_dates = sorted({d for df in series.values() for d in df["date"].dt.strftime("%Y-%m-%d")})
    if include_equal_weight:
        series["Equal Weight"] = compute_equal_weight_returns(log_returns, dates, strategy_dates)
    if include_sp500:
        try:
            series["S&P 500"] = fetch_or_load_sp500_returns(strategy_dates[0], strategy_dates[-1], sp500_cache_dir)
        except Exception as e:
            print(f"[backtest] S&P 500 unavailable: {e}")

    os.makedirs(os.path.join(output_dir, "strategy"), exist_ok=True)
    table = metrics_table({name: df["strategy_return"].to_numpy() for name, df in series.items()})
    table.to_csv(os.path.join(output_dir, "backtest_metrics.csv"), index=False)
    print(table.to_string(index=False))
    plot_curves(series, output_dir)
    for name, df in series.items():
        df.to_csv(os.path.join(output_dir, "strategy", f"{name.replace(' ', '_').replace('/', '_')}_returns.csv"),
                  index=False)
    with open(os.path.join(output_dir, "backtest_run_meta.json"), "w") as f:
        json.dump({"created_at": datetime.now().isoformat(timespec="seconds"),
                   "weights_dirs": [os.path.normpath(d) for d in weights_dirs],
                   "date_range": [strategy_dates[0], strategy_dates[-1]],
                   "trading_days": {name: len(df) for name, df in series.items()}}, f, indent=2)
    return series


def main():
    parser = argparse.ArgumentParser(description="Backtest pre-computed portfolio weights.")
    parser.add_argument("--weights_dir", nargs="+", required=True, help="directories with window_*.npz files")
    parser.add_argument("--data_dir", default="data/processed")
    parser.add_argument("--output", default="results/main")
    parser.add_argument("--sp500_cache_dir", default="data/sp500")
    parser.add_argument("--rebalance_freq", type=int, default=None,
                        help="rebalance interval in days (default: horizon in run_meta.json, else 7)")
    parser.add_argument("--no_equal_weight", action="store_true")
    parser.add_argument("--no_sp500", action="store_true")
    args = parser.parse_args()
    run_backtest(args.weights_dir, args.data_dir, args.output, args.sp500_cache_dir,
                 not args.no_equal_weight, not args.no_sp500, args.rebalance_freq)


if __name__ == "__main__":
    main()
