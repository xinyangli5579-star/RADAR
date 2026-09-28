"""Experiment configuration."""

from __future__ import annotations

from dataclasses import dataclass

# Components switched off by each ablation.
ABLATIONS = {
    "none": {},
    "no_news": {"use_news": False},
    "no_price": {"use_price": False},
    "no_retrieval": {"use_retrieval": False},
    "no_diffusion": {"use_diffusion": False},
}

CHOICES = {
    "fusion_act": ("relu", "tanh"),
    "portfolio_loss": ("sdf", "return"),
    "target_return_mode": ("gross", "net"),
    "ablation": tuple(ABLATIONS),
}


@dataclass
class Config:
    # Data and output
    returns_path: str = "data/processed/log_returns.npz"
    news_path: str = "data/processed/news_finbert_sparse.pt"
    save_dir: str = "checkpoints"
    run_name: str = ""

    # Rolling windows
    train_years: int = 4
    test_years: int = 1
    step_years: int = 1
    trading_days_per_year: int = 252
    val_ratio: float = 0.1

    # Samples
    seq_len: int = 60
    # Training target period; the paper's SDF notation uses the next period.
    target_horizon: int = 1
    # Gross returns make 1.0 the no-change baseline in the SDF objective.
    target_return_mode: str = "gross"
    # Portfolio rebalance / holding interval used at inference and backtest.
    horizon: int = 7
    n_stocks_per_sample: int = 256

    # Encoders
    hidden_units: int = 64
    return_dim: int = 1
    news_dim: int = 768
    fusion_act: str = "relu"

    # Diffusion
    timesteps: int = 100
    beta_start: float = 1e-4
    beta_end: float = 0.02

    # Context bank and retrieval
    top_k: int = 10
    embedding_bank_size: int = 400_000
    bank_refresh_freq: int = 5

    # Objective: L = L_pf + rho * L_diff, z = (1 - gamma) * (q_hat + c_hat) + gamma * q
    portfolio_loss: str = "sdf"
    sdf_lambda: float = 1e-3
    gamma: float = 0.5
    rho: float = 0.002

    # Optimisation
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip: float = 1.0
    early_stop_patience: int = 10
    num_workers: int = 4
    seed: int = 42
    device: str = "cuda"

    # Ablation
    ablation: str = "none"
    use_price: bool = True
    use_news: bool = True
    use_retrieval: bool = True
    use_diffusion: bool = True

    def __post_init__(self):
        for name, allowed in CHOICES.items():
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        for flag, value in ABLATIONS[self.ablation].items():
            setattr(self, flag, value)

    @property
    def default_run_name(self) -> str:
        return self.run_name or ("radar" if self.ablation == "none" else self.ablation)
