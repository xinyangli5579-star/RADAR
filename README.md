# RADAR

Official research code for **Retrieval-Augmented Diffusion Modeling for
Stochastic Discount Factor Portfolios** (NeurIPS 2026).

The repository contains the RADAR implementation, the preprocessing pipeline,
the backtest and the ablation study. Raw/licensed data, checkpoints, logs and
baseline implementations are not redistributed.

## Structure

| Path | Purpose |
|---|---|
| `radar/config.py` | all hyper-parameters (`Config`) and ablation switches |
| `radar/model.py` | encoders, context bank and retrieval, retrieval-conditioned diffusion, portfolio head and losses |
| `radar/data.py` | market data loading and cross-sectional samples |
| `radar/train.py` | rolling-window training |
| `radar/inference.py` | checkpoint-to-portfolio-weight inference |
| `radar/backtest.py` | portfolio backtest and metrics |
| `preprocess/` | stock universe, returns, FinBERT news embeddings |
| `experiments/ablation/` | `no_news`, `no_price`, `no_retrieval` and `no_diffusion` variants |
| `data/` | data contract only; no dataset is redistributed |
| `results/` | reported values and output location |

```python
from radar import Config, RADAR

model = RADAR(Config())
```

## Environment

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Data preparation

Place the source data under `data/raw/` and follow `preprocess/README.md`. The
model expects:

```text
data/processed/log_returns.npz
data/processed/news_finbert_sparse.pt
```

## Main experiment

```bash
# Train the five rolling windows
python -m radar.train --seed 42 --device cuda --save_dir checkpoints

# Generate daily portfolio weights
python -m radar.inference \
  --results checkpoints/radar/seed_42/rolling_results.json \
  --data_dir data/processed \
  --output checkpoints/weights/radar

# Backtest (rebalancing interval is read from the weights' run_meta.json)
python -m radar.backtest --weights_dir checkpoints/weights/radar --output results/main
```

Every field of `Config` is also a command-line flag of `radar.train`, e.g.

| Flag | Default | Meaning |
|---|---|---|
| `--seq_len` / `--target_horizon` / `--horizon` | `60` / `1` / `7` | input length, SDF target horizon, and rebalancing horizon (days) |
| `--target_return_mode` | `gross` | `gross`: `exp(sum(log return))`, so `1` means no wealth change; `net`: legacy `expm1(...)` target |
| `--top_k` | `10` | retrieved neighbours per query |
| `--portfolio_loss` | `sdf` | `sdf`: `(1 - w'r)^2 + lambda ||w||^2` with gross returns by default; `return`: `-w'r` |
| `--sdf_lambda` | `1e-3` | `lambda` of the SDF loss |
| `--gamma` | `0.5` | `z = (1 - gamma) * (q_hat + c_hat) + gamma * q`; `gamma = 1` uses the price representation only |
| `--rho` | `0.002` | `L = L_pf + rho * L_diff` |
| `--fusion_act` | `relu` | activation of the bilinear fusion (`relu` or `tanh`) |
| `--timesteps` | `100` | diffusion steps (linear schedule `--beta_start`, `--beta_end`) |
| `--ablation` | `none` | `no_news`, `no_price`, `no_retrieval`, `no_diffusion` |

## Additional experiments

```bash
# Ablations
python -m experiments.ablation.run --ablation no_retrieval --device cuda
```

The reported values are listed in `results/README.md`.
