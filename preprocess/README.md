# Preprocessing

Builds the inputs of RADAR for S&P 500 constituents from 2011-06-01 to
2020-06-30: daily log returns from adjusted close prices and FinBERT
embeddings of the daily news of each stock.

## Inputs

- `data/raw/sp500_full_multiindex.csv`: two header rows (tickers, fields such as
  `Adj Close`), a `Date` placeholder row, then one row per day.
- News in one of two layouts:
  - `long` (default, FNSPID `All_external.csv`): one article per row with
    `Date`, `Stock_symbol`, `Article` / `Article_title` columns;
  - `wide`: a `date` column followed by one text column per ticker.

## Steps

```bash
# 1. Tickers present in both the price and the news file
python -m preprocess.step1_build_universe \
  --sp500_csv data/raw/sp500_full_multiindex.csv \
  --news_csv data/raw/All_external.csv --news_format long \
  --out_dir data/processed

# 2. Log-return matrix -> data/processed/log_returns.npz
python -m preprocess.step2_build_log_returns \
  --sp500_csv data/raw/sp500_full_multiindex.csv \
  --universe data/processed/universe.json \
  --start 2011-06-01 --end 2020-06-30 \
  --out_dir data/processed --price_field "Adj Close"

# 3. FinBERT [CLS] embeddings of each (ticker, day) text, written in shards
python -m preprocess.step3_encode_news_finbert \
  --news_csv data/raw/All_external.csv --news_format long \
  --universe data/processed/universe.json \
  --start 2011-06-01 --end 2020-06-30 \
  --out_dir data/processed \
  --model_name ProsusAI/finbert --device cpu --batch_size 4 \
  --fp16_store --shard_size_cells 20000

# 4. Merge the shards -> data/processed/news_finbert_sparse.pt
python -m preprocess.step4_build_dataset \
  --news_sparse_pt "data/processed/news_finbert_sparse.part*.pt" \
  --out data/processed/news_finbert_sparse.pt
```

Without `--shard_size_cells`, step 3 writes `news_finbert_sparse.pt` directly
and step 4 is not needed.

`preprocess.build_stock_metadata` writes the GICS sector and volatility tercile
of every stock to `data/processed/stock_metadata.json`.

## Look-ahead

- `log_returns[t] = log(p_t) - log(p_{t-1})`.
- A sample dated `t` uses returns and news of days `[t - seq_len, t)` as input
  and the returns of days `[t, t + target_horizon)` as the SDF target. The
  default `target_horizon=1` matches the paper's next-period notation, while
  `horizon=7` remains the default portfolio rebalancing interval.
- The default `target_return_mode=gross` converts the target log return to
  `exp(sum(log_return))`, so a value of `1` means no change in wealth as in the
  paper's SDF objective. Use `target_return_mode=net` only for legacy runs.
- News is encoded per day only; days without news are masked.
