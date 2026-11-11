"""Value-weighted deciles and delayed CRSP accounting.

Reused from ../prism-vq-reproduction/experiments/cz_vw/portfolio.py;
only the fixed calendar end date was generalized to the input panel's end.
"""

import numpy as np
import pandas as pd


def neutral_projection(desired, lower, upper, tolerance=1e-10):
    """Minimum squared change, bounded executable holdings, zero signed dollars."""
    desired, lower, upper = [np.asarray(a, float) for a in [desired, lower, upper]]
    if not all(np.isfinite(a).all() for a in [desired, lower, upper]):
        raise ValueError("Nonfinite holdings")
    if (lower > upper).any() or lower.sum() > tolerance or upper.sum() < -tolerance:
        raise ValueError("Executable bounds cannot support dollar neutrality")
    left, right = float(np.min(desired - upper)), float(np.max(desired - lower))
    for _ in range(100):
        shift = (left + right) / 2
        value = np.clip(desired - shift, lower, upper)
        if abs(value.sum()) <= tolerance:
            return value
        if value.sum() > 0:
            left = shift
        else:
            right = shift
    raise ArithmeticError("Dollar-neutral projection failed")


def decile_targets(ids, scores, caps, n):
    ids, scores, caps = np.asarray(ids), np.asarray(scores), np.asarray(caps, float)
    assert len(ids) >= 10 and len(np.unique(ids)) == len(ids)
    assert np.isfinite(scores).all() and np.isfinite(caps).all() and (caps > 0).all()
    order = np.lexsort((ids, scores))
    bucket = np.minimum(9, np.arange(len(ids)) * 10 // len(ids))
    target = np.zeros((10, n), float)
    for d in range(10):
        chosen = order[bucket == d]
        target[d, ids[chosen]] = caps[chosen] / caps[chosen].sum()
    np.testing.assert_allclose(target.sum(1), 1)
    return target


def backtest(panel, predictions, cost_bp=0.0, adverse_missing=False):
    n = len(panel.symbols)
    previous = np.zeros(n)
    nav = 1.0
    date_map = {str(d): t for t, d in enumerate(panel.dates)}
    symbol_map = {str(s): i for i, s in enumerate(panel.symbols)}
    stop = str(panel.dates[-1])
    groups = [
        (date_map[str(d)], f)
        for d, f in predictions.groupby("date", sort=True)
        if date_map[str(d)] + 2 < len(panel.dates)
        and str(panel.dates[date_map[str(d)] + 2]) <= stop
    ]
    records, weights, deciles = [], [], []
    last = None
    for number, (t, f) in enumerate(groups):
        assert last is None or t == last + 1, (
            "Signal-session gap invalidates daily drift accounting"
        )
        last = t
        ids = np.array([symbol_map[str(s)] for s in f.symbol])
        assert (panel.member[ids, t] & panel.tradable[ids, t]).all()
        target = decile_targets(
            ids, f.prediction.to_numpy(), panel.lagged_cap[ids, t], n
        )
        desired = nav * (target[9] - target[0])
        tradable = np.asarray(panel.tradable[:, t + 1]) & ~np.asarray(
            panel.terminal_events[:, t + 1]
        )
        lower = np.where(
            tradable, np.minimum(np.minimum(previous, 0), desired), previous
        )
        upper = np.where(
            tradable, np.maximum(np.maximum(previous, 0), desired), previous
        )
        current = neutral_projection(
            desired, lower, upper, tolerance=max(1e-12, nav * 1e-10)
        )
        np.testing.assert_allclose(
            current[~tradable], previous[~tradable], rtol=0, atol=1e-12
        )
        trades = current - previous
        total = np.asarray(panel.totalret[:, t + 2], float)
        price = np.asarray(panel.price_return[:, t + 2], float)
        missing = ~np.isfinite(total)
        held = abs(current) > nav * 1e-10
        payoff = np.where(missing, 0, total)
        if adverse_missing:
            payoff = np.where(
                missing & (current > 0),
                -1.0,
                np.where(missing & (current < 0), 1.0, payoff),
            )
        pnl = float(current @ payoff)
        ending = current * (1 + np.where(np.isfinite(price), price, 0))
        ending[np.asarray(panel.terminal_events[:, t + 2])] = 0.0
        liquidation, unliquidated = 0.0, 0.0
        if number == len(groups) - 1:
            executable = np.asarray(panel.tradable[:, t + 2])
            liquidation = float(abs(ending[executable]).sum())
            unliquidated = float(abs(ending[~executable]).sum())
        traded = float(abs(trades).sum())
        fee = (traded + liquidation) * cost_bp / 1e4
        start_nav = nav
        nav += pnl - fee
        if nav <= 0:
            raise FloatingPointError(f"Portfolio insolvent on {panel.dates[t + 2]}")
        records.append(
            dict(
                signal_date=str(panel.dates[t]),
                execution_date=str(panel.dates[t + 1]),
                date=str(panel.dates[t + 2]),
                return_gross=pnl / start_nav,
                return_net=(pnl - fee) / start_nav,
                start_nav=start_nav,
                end_nav=nav,
                pnl=pnl,
                fees=fee,
                traded_dollars=traded,
                final_liquidation_dollars=liquidation,
                final_unliquidated_dollars=unliquidated,
                turnover=traded / start_nav,
                gross_exposure=float(abs(current).sum() / start_nav),
                net_exposure=float(current.sum() / start_nav),
                n_long=int((current > 1e-12).sum()),
                n_short=int((current < -1e-12).sum()),
                locked_holdings=int((held & ~tradable).sum()),
                missing_held_count=int((missing & held).sum()),
                missing_gross_weight=float(abs(current[missing]).sum() / start_nav),
                missing_price_drift_count=int((~np.isfinite(price) & held).sum()),
                target_deviation_l1=float(abs(current - desired).sum() / start_nav),
            )
        )
        weights.append(current / start_nav)
        for d in range(10):
            deciles.append(
                dict(
                    signal_date=str(panel.dates[t]),
                    date=str(panel.dates[t + 2]),
                    decile=d + 1,
                    target_return=float(target[d] @ np.where(missing, 0, total)),
                    missing_weight=float(target[d, missing].sum()),
                    stocks=int((target[d] > 0).sum()),
                )
            )
        previous = ending
    ledger = pd.DataFrame(records)
    assert len(ledger) > 0
    assert ledger.net_exposure.abs().max() < 1e-8
    np.testing.assert_allclose(
        ledger.end_nav, ledger.start_nav + ledger.pnl - ledger.fees, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.cumprod(1 + ledger.return_net), ledger.end_nav, rtol=1e-11
    )
    return ledger, np.array(weights), pd.DataFrame(deciles)
