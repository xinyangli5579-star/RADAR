# Ablations

Each variant switches off exactly one component of the full model; everything
else (rolling windows, seed, objective, optimisation, inference and backtest)
is identical to the main run.

| Variant | Change relative to RADAR |
|---|---|
| `no_news` | news inputs are zeroed and fully masked (training, context bank and inference) |
| `no_price` | return inputs are zeroed (training, context bank and inference) |
| `no_retrieval` | diffusion noise is standard Gaussian `N(0, I)` instead of the retrieved statistics |
| `no_diffusion` | no diffusion: `q_hat = q`, `c_hat = c`, trained with the portfolio loss only |

Context-bank segments are always delimited by news days, so `no_news` and
`no_price` keep the same bank structure and change only the encoded inputs.

```bash
python -m experiments.ablation.run --ablation no_news --device cuda
python -m experiments.ablation.run --ablation no_price --device cuda
python -m experiments.ablation.run --ablation no_retrieval --device cuda
python -m experiments.ablation.run --ablation no_diffusion --device cuda

python -m radar.backtest \
  --weights_dir checkpoints/weights/radar checkpoints/weights/no_news \
                checkpoints/weights/no_price checkpoints/weights/no_retrieval \
                checkpoints/weights/no_diffusion \
  --output results/ablation
```

The same variants can be trained directly with `python -m radar.train --ablation <variant>`.
