# CZ202 / value-weighted RADAR protocol

This is a separate empirical adaptation, not a claim of reproducing the paper's
377-stock sample. It reuses RADAR's architecture, diffusion, retrieval, optimizer,
validation selection and checkpoint format. Native commands keep their defaults.

The only model-input change is `return_dim=202`: lagged stock-level CZ
characteristics replace the scalar return sequence. FinBERT news remains the
second modality. Monthly characteristic values repeat over the original 60 daily
input sessions; each session uses its own immediately preceding calendar month.

## Required data and reuse

Use the existing audited cache produced by
`../prism-vq-reproduction/experiments/cz_vw/build.py`, its manifest, and the CRSP
security-name intervals. The cache is read-only and contains:

- `dates`, `symbols` (PERMNO keys), `monthly_features`, `source_months`,
  `month_index`, `member`, `tradable`, `lagged_cap`, `totalret`,
  `price_return`, `terminal_events`, and `training_feature_mask` NumPy arrays;
- `manifest.json` with hashes, 202 feature names and seven training-constant
  excluded names; `nyse20_breakpoints.csv` with the monthly NYSE20 cutoff.

The underlying sources are CZ's unsigned `Predictors.parquet` (2026.10 vintage),
CRSP CIZ daily/monthly data, and historical S&P 500 membership. The universe is
historical S&P 500 common shares, **not all US stocks**. The producer explicitly
excludes unknown prior-month size and size below the prior-month NYSE common-share
20th percentile. The adapter verifies hashes, month alignment and every retained
stock-day's size cutoff. Current-vintage data do not prove historical as-published
availability.

Feature selection is fixed using 2009–2010 calibration, entirely before the
2011–2020 experiment: all retained 202 vary there, while the seven source exclusions
are constant over a verified superset of those calibration observations. The code
checks this subset relation instead of choosing features from future test data.
One retained characteristic (`std_turn`) is constant in the first later training
window; it is not replaced with a test-selected feature or a fabricated column.

Existing FinBERT ticker/day embeddings are reused. Only additional CRSP ticker
symbols are encoded from the same `All_external.csv`. News is assigned to PERMNO
using exact ticker matches and dated name intervals; ambiguous/unmatched cells
are counted and excluded. Missing news is masked, never used to exclude a stock
based on future corpus coverage.
Cross-stock context-bank news and characteristic histories are also masked by
historical index membership, so a future entrant cannot enter an earlier bank.

## Run

Install `pip install -r experiments/cz_vw/requirements.txt`.
With environment variables pointing to the local sources:

```bash
python -m experiments.cz_vw.prepare \
  --shared_data "$CZ_CRSP_CACHE" --names "$CRSP_NAMES" \
  --raw_news "$FNSPID_CSV" --cached_news "$FINBERT_CACHE" \
  --finbert "$FINBERT_MODEL" --out data/cz_vw

python -m pytest experiments/cz_vw/test_protocol.py -q

python -m experiments.cz_vw.run \
  --shared_data "$CZ_CRSP_CACHE" --data_dir data/cz_vw \
  --factors "$FF5_UMD_PARQUET" --protocol "$PROTOCOL_FILE" \
  --output results/cz_vw --window 1
```

The default is **one prespecified rolling window**, seed 42, at most 50 epochs
with the released validation patience of 10. This bounds the protocol experiment
instead of repeating the full paper suite. Window 1 trains on 2011–2015 and tests
approximately June 2015–June 2016. Dates and configuration are saved before fitting.
`--epochs 1 --output results/cz_vw_smoke` is an integration test only. A full training
window must restart after interruption; completed checkpoints can resume evaluation.

## Portfolio and inference conventions

- Signal at close `s`, execution at close `s+1`, first total-return payoff at close
  `s+2`. Training labels follow the same delay and are purged at split boundaries.
  Inference eligibility uses only signal-time membership and tradability.
- Daily capitalization-weighted score D10−D1: +100% long and −100% short,
  200% gross. This changes the paper's seven-day native long-only evaluation.
  Ties use PERMNO; the score direction is fixed, never flipped after seeing returns.
- Gross and 10 bp per absolute traded dollar, including entry, rebalancing and
  executable final liquidation. Ten bp is the existing local protocol convention,
  not a number specified by `protocol.txt`. No borrow, financing, impact or
  collateral yield is assumed.
- Reuse the neighboring PRISM-VQ portfolio ledger: unavailable trades freeze
  holdings, executable holdings are projected to dollar neutrality, dividends
  enter total-return P&L, price-only returns drift positions, and terminal events
  settle to cash. Missing held returns are zero in primary accounting and are
  explicitly counted; final unliquidated holdings are reported.
- FF5+UMD OLS with intercept, decimal daily factors on realized payoff dates,
  Bartlett HAC20 with finite-sample correction and a t reference. Report annual
  arithmetic alpha (`252 × daily intercept`), HAC t, 95% CI and two-sided p.
  Do not subtract RF again from a zero-net spread. Gross/net p-values also receive
  a Holm correction over those two tests. One year/one seed is limited evidence.

Outputs include `alpha.csv`, cost-specific cash/P&L ledgers and weights,
`target_deciles.csv`, predictions, the data audit and locked configuration/hashes.
The gross-return SDF target is retained from the public release; the previously
identified discrepancy with the paper's net-return wording remains disclosed.

## Code reused

`portfolio.py` is the existing local PRISM-VQ CZ ledger with only its fixed end
date generalized. `fit_alpha` follows that project's estimator. Data are reused
through a verified cache, not copied into Git. Native changes are limited to
optional dataset/history/bank-batch arguments on `train_window` and optional argv
on the existing FinBERT preprocessing entry point. No model or native default is
changed.
