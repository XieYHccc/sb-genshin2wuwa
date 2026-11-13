"""EDM 重参数化系数：离线实测标定与查表（doc/sbflow-genshin-wuwa-plan.md §2）。

D(x, τ, d) = c_skip(τ)·x + c_out(τ)·F(c_in(τ)·x, τ, d) 预测速度；
c_in 让网络输入单位方差，c_skip 取最小二乘最优，c_out 让回归目标单位方差。
正反向各标定一张表（d=1 正向，d=0 反向），在真实的配对数据上实测，不套高斯假设。
"""
from __future__ import annotations

import math

import torch

TAU_MIN, TAU_MAX = 1e-3, 1 - 1e-3   # 端点奇点：训练和推理都 clamp（§2.5）


@torch.no_grad()
def calibrate_coeffs(z0: torch.Tensor, z1: torch.Tensor, eps: float, n_tau: int = 64, seed: int = 0):
    """z0, z1：配对好的样本（与训练完全相同的现裁 + 配对流程攒出来的）。返回 (taus, {1: (n,3), 0: (n,3)})。"""
    g = torch.Generator(device=z0.device).manual_seed(seed)
    taus = torch.linspace(TAU_MIN, TAU_MAX, n_tau, dtype=torch.float64)
    tables = {}
    for d in (1, 0):
        rows = []
        for tau in taus.tolist():
            t = tau if d == 1 else 1 - tau            # 网络时间 τ → 物理时间 t
            z = torch.randn(z0.shape, generator=g, device=z0.device)
            xt = (1 - t) * z0 + t * z1 + math.sqrt(eps * t * (1 - t)) * z
            v = (z1 - xt) / (1 - t) if d == 1 else (z0 - xt) / t
            xt, v = xt.double(), v.double()
            var_x = xt.var().item()
            var_v = v.var().item()
            cov = ((xt - xt.mean()) * (v - v.mean())).mean().item()
            c_skip = cov / var_x
            c_out = max(var_v - cov ** 2 / var_x, 1e-8) ** 0.5
            c_in = var_x ** -0.5
            rows.append((c_in, c_skip, c_out))
        tables[d] = torch.tensor(rows, dtype=torch.float32)
    return taus.float(), tables


def analytic_coeffs(taus: torch.Tensor, eps: float, sigma_d: float = 1.0, rho: float = 0.0):
    """§2.2 的解析式（高斯假设）。只用于自检。"""
    t = taus.double()
    s2 = sigma_d ** 2 * ((1 - t) ** 2 + t ** 2 + 2 * rho * t * (1 - t)) + eps * t * (1 - t)
    su2 = 2 * sigma_d ** 2 * (1 - rho)
    cov = -(su2 / 2 * (1 - 2 * t) + eps * t)
    c_skip = cov / s2
    c_out = (su2 + eps * t / (1 - t) - cov ** 2 / s2).clamp_min(1e-16).sqrt()
    return torch.stack([s2.rsqrt(), c_skip, c_out], -1).float()


def self_check(device="cuda", n=4096, shape=(32, 16, 16)):
    """合成单位方差高斯、随机配对：ε=0 时 τ=0/1 应得到 §2.3 的值；ε>0 时应与解析式一致。"""
    g = torch.Generator(device=device).manual_seed(123)
    z0 = torch.randn((n, *shape), generator=g, device=device)
    z1 = torch.randn((n, *shape), generator=g, device=device)
    out = {}
    for eps in (0.0, 0.05, 0.2):
        taus, tab = calibrate_coeffs(z0, z1, eps, n_tau=33)
        ana = analytic_coeffs(taus, eps)
        err = max((tab[d] - ana).abs().max().item() for d in (0, 1))
        out[eps] = {"max_abs_err_vs_analytic": err,
                    "tau0": [round(v, 4) for v in tab[1][0].tolist()],
                    "tau1": [round(v, 4) for v in tab[1][-1].tolist()]}
    return out


def save_coeff(path, taus, tables, **meta):
    torch.save({"taus": taus, "tables": tables, **meta}, path)


def load_coeff(path, device="cuda"):
    c = torch.load(path, map_location="cpu", weights_only=False)
    return c["taus"].to(device), {d: t.to(device) for d, t in c["tables"].items()}, c


def lookup(coeff, tau: torch.Tensor, d: torch.Tensor):
    """coeff = (taus, tables)；tau、d 形状 (B,)。按 d 选表、按 τ 线性插值。返回 c_in, c_skip, c_out，各 (B,1,1,1)。"""
    taus, tables = coeff[0], coeff[1]
    tau = tau.float().clamp(taus[0].item(), taus[-1].item())
    i = torch.searchsorted(taus, tau).clamp(1, len(taus) - 1)
    w = ((tau - taus[i - 1]) / (taus[i] - taus[i - 1])).unsqueeze(-1)
    fwd = tables[1][i - 1] * (1 - w) + tables[1][i] * w
    bwd = tables[0][i - 1] * (1 - w) + tables[0][i] * w
    c = torch.where(d.bool().unsqueeze(-1), fwd, bwd)
    return [c[:, k].view(-1, 1, 1, 1) for k in range(3)]
