"""RADAR: retrieval-augmented diffusion representations with a portfolio head."""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    """Single-layer GRU returning all hidden states and the last one."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        full, last = self.gru(x)
        return full, last[-1]


class AttnPooling(nn.Module):
    """Additive attention pooling over time, queried by the last hidden state."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, full, last):
        score = self.V(torch.tanh(self.W1(last).unsqueeze(1) + self.W2(full)))
        return (torch.softmax(score, dim=1) * full).sum(dim=1)


class PriceEncoder(nn.Module):
    """GRU and attention pooling over the return sequence."""

    def __init__(self, hidden_units: int, return_dim: int = 1):
        super().__init__()
        self.gru = GRUEncoder(return_dim, hidden_units)
        self.attn = AttnPooling(hidden_units)

    def forward(self, return_seq):
        return self.attn(*self.gru(return_seq))


class TextEncoder(nn.Module):
    """Per-day projection of news embeddings, then GRU and attention pooling over days."""

    def __init__(self, hidden_units: int, news_dim: int = 768):
        super().__init__()
        self.day_gru = GRUEncoder(news_dim, hidden_units)
        self.day_attn = AttnPooling(hidden_units)
        self.gru = GRUEncoder(hidden_units, hidden_units)
        self.attn = AttnPooling(hidden_units)

    def forward(self, news_seq, news_mask=None):
        B, L, D = news_seq.shape
        days = self.day_attn(*self.day_gru(news_seq.reshape(B * L, 1, D))).reshape(B, L, -1)
        if news_mask is not None:
            days = days * (~news_mask).unsqueeze(-1).to(days.dtype)
        return self.attn(*self.gru(days))


class Denoiser(nn.Module):
    """One self-attention layer over [noisy price, noisy text, market state, step] tokens."""

    def __init__(self, hidden_units: int):
        super().__init__()
        self.hidden_units = hidden_units
        self.ln = nn.LayerNorm(hidden_units, elementwise_affine=False)
        self.w_q = nn.Linear(hidden_units, hidden_units, bias=False)
        self.w_k = nn.Linear(hidden_units, hidden_units)
        self.w_v = nn.Linear(hidden_units, hidden_units)
        for layer in (self.w_q, self.w_k, self.w_v):
            nn.init.xavier_normal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

    def step_embedding(self, s):
        half = self.hidden_units // 2
        freqs = torch.exp(-math.log(10000) / (half - 1) * torch.arange(half, device=s.device))
        angles = s.float()[:, None] * freqs[None, :]
        return torch.cat([angles.sin(), angles.cos()], dim=-1)

    def forward(self, xq, xc, m, s):
        h = self.ln(torch.stack([xq, xc, m, self.step_embedding(s)], dim=1))
        attn = torch.softmax(self.w_q(h) @ self.w_k(h).transpose(-1, -2) * self.hidden_units ** -0.5, dim=-1)
        out = attn @ self.w_v(h)
        return out[:, 0], out[:, 1]


class PortfolioHead(nn.Module):
    """MLP scores followed by a softmax over the stocks of one date."""

    def __init__(self, hidden_dim: int = 64, temperature: float = 1.0):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.temperature = temperature

    def forward(self, embeddings):
        scores = self.score_net(embeddings).squeeze(-1)
        return F.softmax(scores / self.temperature, dim=0), scores


class RADAR(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        H = cfg.hidden_units
        self.price_encoder = PriceEncoder(H, cfg.return_dim)
        self.text_encoder = TextEncoder(H, cfg.news_dim)
        self.fusion = nn.Bilinear(H, H, H)
        self.denoiser = Denoiser(H)
        self.portfolio_head = PortfolioHead(H)
        self._register_schedule(cfg.timesteps, cfg.beta_start, cfg.beta_end)
        self.set_bank(None, None)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def uses_bank(self) -> bool:
        return self.cfg.use_retrieval and self.cfg.use_diffusion

    # ------------------------------------------------------------------ schedule

    def _register_schedule(self, T, beta_start, beta_end):
        """Linear beta schedule; entry s-1 holds the value for diffusion step s = 1..T."""
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        alphas = 1.0 - betas
        abar = torch.cumprod(alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1, dtype=torch.float64), abar[:-1]])
        schedule = {
            "sqrt_abar": abar.sqrt(),
            "r": (1.0 - abar).sqrt(),
            "r_prev": (1.0 - abar_prev).sqrt(),
            "coef_a": betas * abar_prev.sqrt() / (1.0 - abar),
            "coef_b": alphas.sqrt() * (1.0 - abar_prev) / (1.0 - abar),
            "beta_tilde": betas * (1.0 - abar_prev) / (1.0 - abar),
        }
        for name, value in schedule.items():
            self.register_buffer(name, value.float(), persistent=False)

    @staticmethod
    def _at(buffer, s):
        if isinstance(s, int):
            return buffer[s - 1]
        return buffer[s - 1].unsqueeze(-1)

    # ------------------------------------------------------------------ encoders

    def encode(self, return_seq, news_seq, news_mask=None):
        """Price, text and fused market representations; ablated modalities are zeroed."""
        if not self.cfg.use_price:
            return_seq = torch.zeros_like(return_seq)
        if not self.cfg.use_news:
            news_seq = torch.zeros_like(news_seq)
            news_mask = torch.ones(news_seq.shape[:2], dtype=torch.bool, device=news_seq.device)
        q = self.price_encoder(return_seq)
        c = self.text_encoder(news_seq, news_mask)
        m = self.fusion(q, c)
        m = F.relu(m) if self.cfg.fusion_act == "relu" else torch.tanh(m)
        return q, c, m

    # ------------------------------------------------------------------ context bank

    def set_bank(self, embeddings, times):
        self.bank = embeddings
        self.bank_times = times
        self.bank_normed = None if embeddings is None else F.normalize(embeddings, dim=-1)

    @torch.no_grad()
    def build_context_bank(self, returns, news, news_masks, start_idx: int = 0, max_tokens: int = 65536):
        """Encode every stock's segments [news day, next news day) into the context bank.

        returns, news, news_masks: per-stock tensors of shape [D, 1], [D, news_dim] and [D].
        Each segment is time-stamped with the global index of its last day.
        """
        was_training = self.training
        self.eval()
        segments = defaultdict(list)
        for i, mask in enumerate(news_masks):
            days = torch.where(~mask)[0].tolist()
            for a, b in zip(days[:-1], days[1:]):
                segments[b - a].append((i, a, b))

        embeddings, times = [], []
        for length, segs in sorted(segments.items()):
            size = max(1, max_tokens // length)
            for j in range(0, len(segs), size):
                part = segs[j:j + size]
                ret = torch.stack([returns[i][a:b] for i, a, b in part]).to(self.device)
                nws = torch.stack([news[i][a:b] for i, a, b in part]).to(self.device)
                msk = torch.stack([news_masks[i][a:b] for i, a, b in part]).to(self.device)
                embeddings.append(self.encode(ret, nws, msk)[2])
                times.extend(start_idx + b - 1 for _, _, b in part)
        self.train(was_training)

        if not embeddings:
            self.set_bank(None, None)
            print("[bank] no segments; falling back to standard Gaussian noise")
            return
        bank = torch.cat(embeddings)
        times = torch.tensor(times, dtype=torch.long, device=self.device)
        if bank.shape[0] > self.cfg.embedding_bank_size:
            keep = torch.randperm(bank.shape[0])[: self.cfg.embedding_bank_size].to(self.device)
            bank, times = bank[keep], times[keep]
        self.set_bank(bank, times)
        print(f"[bank] {bank.shape[0]} segments")

    @torch.no_grad()
    def select_neighbors(self, z, current_t=None):
        """Indices and selection mask of the cosine-similar bank segments ending before `current_t`."""
        sim = F.normalize(z, dim=-1) @ self.bank_normed.T
        n_valid = sim.shape[1]
        if current_t is not None:
            valid = self.bank_times < int(current_t)
            n_valid = int(valid.sum())
            sim = sim.masked_fill(~valid, float("-inf"))

        k = min(self.cfg.top_k, n_valid)
        if k < 2:
            return None, None
        idx = sim.topk(k, dim=-1).indices
        return idx, torch.ones_like(idx, dtype=torch.bool)

    @torch.no_grad()
    def noise_stats(self, z, current_t=None):
        """Mean and standard deviation of the retrieved segments (standard normal without retrieval)."""
        mu, sigma = torch.zeros_like(z), torch.ones_like(z)
        if not self.cfg.use_retrieval or self.bank is None:
            return mu, sigma
        idx, selected = self.select_neighbors(z, current_t)
        if idx is None:
            return mu, sigma
        neighbors = self.bank[idx]
        w = selected.unsqueeze(-1).to(neighbors.dtype)
        count = w.sum(dim=1)
        mu = (neighbors * w).sum(dim=1) / count
        var = ((neighbors - mu.unsqueeze(1)).square() * w).sum(dim=1) / count
        return mu, (var + 1e-8).sqrt()

    # ------------------------------------------------------------------ diffusion

    def q_sample(self, x0, s, mu, sigma, eps=None):
        """Forward marginal at step s with retrieved noise N(mu, sigma^2)."""
        eps = torch.randn_like(x0) if eps is None else eps
        return self._at(self.sqrt_abar, s) * x0 + self._at(self.r, s) * (mu + sigma * eps)

    def _posterior_sample(self, x0_hat, xs, s: int, mu, sigma):
        """One reverse step from s to s-1 under the retrieval-conditioned posterior."""
        mean = (self.coef_a[s - 1] * x0_hat
                + self.coef_b[s - 1] * (xs - self.r[s - 1] * mu)
                + self.r_prev[s - 1] * mu)
        return mean + self.beta_tilde[s - 1].sqrt() * sigma * torch.randn_like(xs)

    def reverse(self, q, c, m, stats_q, stats_c):
        """Initialise at step T with the retrieved statistics and denoise back to step 0."""
        T = self.cfg.timesteps
        xq = self.q_sample(q, T, *stats_q)
        xc = self.q_sample(c, T, *stats_c)
        for s in range(T, 0, -1):
            steps = torch.full((q.shape[0],), s, dtype=torch.long, device=q.device)
            q_hat, c_hat = self.denoiser(xq, xc, m, steps)
            if s == 1:
                return q_hat, c_hat
            xq = self._posterior_sample(q_hat, xq, s, *stats_q)
            xc = self._posterior_sample(c_hat, xc, s, *stats_c)

    def diffusion_loss(self, q, c, m, stats_q, stats_c):
        """x0-prediction error of both modalities at a uniformly sampled step."""
        s = torch.randint(1, self.cfg.timesteps + 1, (q.shape[0],), device=q.device)
        q_hat, c_hat = self.denoiser(self.q_sample(q, s, *stats_q), self.q_sample(c, s, *stats_c), m, s)
        err = (q.detach() - q_hat).square().sum(-1) + (c.detach() - c_hat).square().sum(-1)
        return err.mean()

    # ------------------------------------------------------------------ representation and losses

    def _represent(self, return_seq, news_seq, news_mask, current_t, with_diffusion_loss):
        q, c, m = self.encode(return_seq, news_seq, news_mask)
        diff_loss = q.new_zeros(())
        if self.cfg.use_diffusion:
            stats_q = self.noise_stats(q, current_t)
            stats_c = self.noise_stats(c, current_t)
            q_hat, c_hat = self.reverse(q, c, m, stats_q, stats_c)
            if with_diffusion_loss:
                diff_loss = self.diffusion_loss(q, c, m, stats_q, stats_c)
        else:
            q_hat, c_hat = q, c
        z = (1.0 - self.cfg.gamma) * (q_hat + c_hat) + self.cfg.gamma * q
        return z, diff_loss

    def forward(self, return_seq, news_seq, news_mask=None, current_t=None):
        """Market representation z of each stock."""
        return self._represent(return_seq, news_seq, news_mask, current_t, with_diffusion_loss=False)[0]

    def portfolio_loss(self, weights, returns):
        """SDF pricing error on gross returns, or the negative portfolio return."""
        portfolio_return = (weights * returns).sum()
        if self.cfg.portfolio_loss == "sdf":
            return (1.0 - portfolio_return).square() + self.cfg.sdf_lambda * weights.square().sum()
        return -portfolio_return

    def compute_loss(self, return_seq, news_seq, news_mask, returns, current_t=None):
        """Total loss L_pf + rho * L_diff for the stocks of one date."""
        z, diff_loss = self._represent(return_seq, news_seq, news_mask, current_t, with_diffusion_loss=True)
        weights, _ = self.portfolio_head(z)
        pf_loss = self.portfolio_loss(weights, returns)
        return pf_loss + self.cfg.rho * diff_loss, pf_loss, diff_loss, weights
