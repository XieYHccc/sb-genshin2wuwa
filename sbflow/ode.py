"""推理：概率流 ODE dx = (v→ − v←)/2 · dt（doc/sbflow-genshin-wuwa-plan.md §6）。

约定：域 0 = 原神，域 1 = 鸣潮；物理时间 t 从 0（原神端）到 1（鸣潮端）。
网络 F(x, τ, d)：d=1 时 τ=t 预测 v→；d=0 时 τ=1−t 预测 v←（网络看到的 τ=0 永远是“出发端”）。
整帧单次前向，不切 tile；正反两个场拼成一次前向。永远用 ODE，不用 SDE 处理视频。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from sbflow.coeffs import TAU_MAX, TAU_MIN, load_coeff, lookup
from sbflow.frames import load_overlay_mask
from sbflow.model import DiT, DiTConfig


def denoise(net, x, tau, d, coeff):
    """预测的速度 v = c_skip·x + c_out·F(c_in·x)。"""
    tau = tau.clamp(TAU_MIN, TAU_MAX)
    c_in, c_skip, c_out = lookup(coeff, tau, d)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        f = net(c_in * x, tau, d)
    return c_skip * x + c_out * f.float()


def _pf_velocity(net, x, t: float, coeff):
    """物理时间 t 处的概率流速度 (v→ − v←)/2，两个场一次前向。"""
    B = x.shape[0]
    tt = torch.full((B,), t, device=x.device)
    tau = torch.cat([tt, 1 - tt])                       # 正向 τ=t，反向 τ=1−t
    d = torch.cat([torch.ones(B, device=x.device), torch.zeros(B, device=x.device)])
    v = denoise(net, torch.cat([x, x]), tau, d, coeff)
    return 0.5 * (v[:B] - v[B:])


@torch.no_grad()
def translate(net, z_src, coeff, n_steps=32, direction="0to1", solver="euler"):
    """z_src: 整帧（或任意网格）scaled latent (B,32,H,W)。direction: '0to1' 原神→鸣潮，'1to0' 反过来。"""
    x = z_src.float().clone()
    sgn = 1.0 if direction == "0to1" else -1.0
    ts = np.linspace(0, 1, n_steps + 1)
    if sgn < 0:
        ts = ts[::-1]
    for i in range(n_steps):
        t0, t1 = float(ts[i]), float(ts[i + 1])
        dt = t1 - t0                                     # 带符号
        v0 = _pf_velocity(net, x, t0, coeff)
        if solver == "heun" and i < n_steps - 1:         # 最后一步不做校正（端点奇点）
            x1 = x + v0 * dt
            v1 = _pf_velocity(net, x1, t1, coeff)
            x = x + 0.5 * (v0 + v1) * dt
        else:
            x = x + v0 * dt
    return x


# ---------------------------------------------------------------------------
# 读取训练好的 run
# ---------------------------------------------------------------------------
def load_run(run_dir, ckpt: str = "latest", use_ema=True, device="cuda"):
    run = Path(run_dir)
    cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
    net = DiT(DiTConfig(**cfg["model"])).to(device).eval()
    path = run / "ckpt" / (f"{ckpt}.pt" if not str(ckpt).endswith(".pt") else ckpt)
    sd = torch.load(path, map_location=device, weights_only=False)
    net.load_state_dict(sd["ema" if use_ema else "net"])
    net.requires_grad_(False)
    taus, tables, _ = load_coeff(run / "coeff.pt", device)
    mask = load_overlay_mask(run / "overlay_mask.png")
    return {"net": net, "coeff": (taus, tables), "config": cfg, "step": sd.get("step"),
            "overlay_mask": mask, "nfe": cfg.get("nfe", 32)}


@torch.no_grad()
def translate_latent(bundle, z, direction="0to1", nfe=None, solver="euler"):
    return translate(bundle["net"], z, bundle["coeff"], nfe or bundle["nfe"], direction, solver)
