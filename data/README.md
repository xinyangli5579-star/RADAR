# Data contract

The dataset is not redistributed. `preprocess/README.md` describes the raw
files and produces the following files under `data/processed/`.

`log_returns.npz`:

- `log_returns`: float array of shape `[D, N]` (NaN where a stock has no price);
- `dates`: `D` date strings;
- `tickers`: `N` ticker strings.

`news_finbert_sparse.pt`: `{"data": {ticker: {date: embedding}}}` with one
768-dimensional FinBERT embedding per ticker and news day. Tickers must match
`log_returns.npz`. The default period is 2011-06-01 to 2020-06-30.

Cached S&P 500 daily returns, when used, belong under `data/sp500/`.
