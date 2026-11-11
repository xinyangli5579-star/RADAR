"""Reused FF5+UMD/HAC estimator for gross and after-cost value-weighted spreads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from .portfolio import backtest


def fit_alpha(ledger, factors, lag=20):
    joined = ledger.merge(
        factors, on="date", how="left", validate="one_to_one", indicator=True
    )
    if not joined._merge.eq("both").all():
        raise ValueError("Missing factor payoff dates")
    names = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    x = np.column_stack([np.ones(len(joined)), joined[names].to_numpy(float)])
    y = joined.return_net.to_numpy(float)
    if (
        not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.linalg.matrix_rank(x) != 7
    ):
        raise ValueError("Invalid regression data or rank-deficient design")
    fit = sm.OLS(y, x, missing="raise").fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": True},
        use_t=True,
    )
    ci = fit.conf_int()[0] * 252
    return {
        "days": len(y),
        "hac_lag": lag,
        "annual_alpha": float(fit.params[0] * 252),
        "hac_t": float(fit.tvalues[0]),
        "ci_low": float(ci[0]),
        "ci_high": float(ci[1]),
        "p": float(fit.pvalues[0]),
        "condition_number": float(np.linalg.cond(x)),
        "residual_lag1": float(pd.Series(fit.resid).autocorr(1)),
    }


def evaluate(panel, predictions, factor_path, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    factors = pd.read_parquet(factor_path)
    factors["date"] = factors.date.astype(str)
    rows = []
    performance = []
    for bp in (0, 10):
        ledger, weights, deciles = backtest(panel, predictions, bp)
        ledger.to_csv(out / f"cost{bp}_ledger.csv", index=False)
        np.savez_compressed(
            out / f"cost{bp}_weights.npz",
            weights=weights,
            dates=ledger.date.to_numpy(),
            symbols=panel.symbols,
        )
        rows.append({"cost_bp": bp, **fit_alpha(ledger, factors)})
        nav = np.r_[1.0, ledger.end_nav.to_numpy()]
        performance.append(
            {
                "cost_bp": bp,
                "days": len(ledger),
                "cumulative_return": float(nav[-1] - 1),
                "max_drawdown": float((1 - nav / np.maximum.accumulate(nav)).max()),
                "missing_held_count": int(ledger.missing_held_count.sum()),
                "missing_gross_weight_days": float(ledger.missing_gross_weight.sum()),
                "final_unliquidated_dollars": float(
                    ledger.final_unliquidated_dollars.iloc[-1]
                ),
            }
        )
        if bp == 0:
            deciles.to_csv(out / "target_deciles.csv", index=False)
    alpha = pd.DataFrame(rows)
    alpha["p_holm2"] = multipletests(alpha.p, method="holm")[1]
    alpha.to_csv(out / "alpha.csv", index=False)
    (out / "performance.json").write_text(json.dumps(performance, indent=2) + "\n")
    print(alpha.to_string(index=False), flush=True)
