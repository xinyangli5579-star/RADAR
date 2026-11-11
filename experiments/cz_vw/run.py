"""One protocol run, using the released RADAR model and training loop."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from radar.config import Config
from radar.data import news_window
from radar.model import RADAR
from radar.train import rolling_windows, set_seed, train_window

from .data import Panel, Samples, histories, sha, verify_cache
from .evaluate import evaluate


@torch.no_grad()
def predict(model, data, start, end, cfg):
    rows = []
    # t is execution day; signal t-1 uses only already observed features/news.
    for t in range(max(start, cfg.seq_len), end - 1):
        ids = data.ids(t - 1)
        if len(ids) < 10:
            raise ValueError(
                f"Insufficient ex-ante eligible stocks on {data.dates[t - 1]}"
            )
        news, mask = news_window(data, ids.tolist(), t - cfg.seq_len, t, cfg.news_dim)
        torch.manual_seed(cfg.seed + t)
        z = model(
            data.features(ids, t - cfg.seq_len, t).to(cfg.device),
            news.to(cfg.device),
            mask.to(cfg.device),
            current_t=t,
        )
        _, scores = model.portfolio_head(z)
        rows.extend(
            {
                "date": str(data.dates[t - 1]),
                "symbol": str(data.symbols[i]),
                "prediction": float(score),
            }
            for i, score in zip(ids, scores.cpu().numpy())
        )
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shared_data", required=True)
    p.add_argument("--data_dir", default="data/cz_vw")
    p.add_argument("--factors", required=True)
    p.add_argument(
        "--protocol", help="Optional user protocol file to hash into the run lock"
    )
    p.add_argument("--output", default="results/cz_vw")
    p.add_argument("--device", default="cuda")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument(
        "--window",
        type=int,
        default=1,
        help="One prespecified window, not test-selected",
    )
    args = p.parse_args()
    torch.set_num_threads(2)
    cfg = Config(
        return_dim=202, horizon=1, num_workers=0, device=args.device, epochs=args.epochs
    )
    data_dir = Path(args.data_dir)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    audit = json.loads((data_dir / "data_audit.json").read_text())
    assert sha(data_dir / "news.pt") == audit["news_sha256"]
    assert verify_cache(args.shared_data)["manifest_sha256"] == audit["manifest_sha256"]
    data = Panel(args.shared_data, data_dir / "news.pt")
    windows = rolling_windows(len(data.dates), cfg)
    window = windows[args.window - 1]
    code = list(Path(__file__).parent.glob("*.py")) + list(Path("radar").glob("*.py"))
    lock = {
        "config": asdict(cfg),
        "window": window,
        "window_number": args.window,
        "data_audit_sha256": sha(data_dir / "data_audit.json"),
        "factor_sha256": sha(args.factors),
        "user_protocol_sha256": sha(args.protocol) if args.protocol else None,
        "code_sha256": {str(f): sha(f) for f in code},
        "scope": "Single prespecified annual out-of-sample window; not a full paper replication.",
    }
    lock_path = out / "locked.json"
    if lock_path.exists():
        assert json.loads(lock_path.read_text()) == lock, (
            "Inputs/settings changed under existing run"
        )
    else:
        lock_path.write_text(json.dumps(lock, indent=2) + "\n")
    result_path = out / "training.json"
    if not result_path.exists():
        entry = train_window(
            args.window,
            window,
            data,
            cfg,
            str(out),
            dataset_cls=Samples,
            history_fn=histories,
            bank_max_tokens=8192,
        )
        if not np.isfinite(entry["best_val_loss"]) or entry["best_val_loss"] <= 0:
            raise ValueError(
                "No finite positive validation loss; refuse an invalid checkpoint"
            )
        result_path.write_text(json.dumps(entry, indent=2) + "\n")
    entry = json.loads(result_path.read_text())
    checkpoint = torch.load(
        out / entry["checkpoint"], map_location=cfg.device, weights_only=True
    )
    model = RADAR(cfg).to(cfg.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    set_seed(cfg.seed)
    model.build_context_bank(
        *histories(data, window[0], window[1], cfg.news_dim),
        start_idx=window[0],
        max_tokens=8192,
    )
    predictions = predict(model, data, window[2], window[3], cfg)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    evaluate(data, predictions, args.factors, out)
    (out / "complete.json").write_text(
        json.dumps(
            {
                "completed": True,
                "scope": lock["scope"],
                "prediction_days": int(predictions.date.nunique()),
                "training": entry,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
