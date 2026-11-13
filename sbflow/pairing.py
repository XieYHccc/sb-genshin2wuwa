"""打包数据的加载与 minibatch OT 配对（doc/sbflow-genshin-wuwa-plan.md §5.1）。

在与 latent 网格对齐的 pixel 特征（1/16 缩略图或 DINO patch token）上配对，在 latent 上训练：
    1. 两域各随机取 B 帧整帧 latent
    2. 本 batch 共用一个裁窗，裁出同一尺寸的窗口
    3. 在裁出来的区域上算逐位置对齐的代价
    4. 精确 minibatch OT = 线性指派（Birkhoff–von Neumann），scipy 的 linear_sum_assignment
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from sbflow.frames import LAT_H, LAT_W

GAMES = ("genshin", "wuwa")   # 域 0 = 原神，域 1 = 鸣潮


@dataclass
class Domain:
    lat: torch.Tensor            # (M,32,18,32) fp16，GPU
    thumb: torch.Tensor          # (M,3,36,64) uint8，GPU
    dino: torch.Tensor | None    # (M,64,18,32) fp16，GPU，可选
    rows: list[dict]             # meta 行（idx, src_file, session, flip, split）

    def __len__(self):
        return len(self.lat)


def load_domain(packed: str | Path, game: str, split: str, device="cuda", dino=False) -> Domain:
    packed = Path(packed)
    with open(packed / f"{game}_meta.csv", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    idx = np.array([int(r["idx"]) for r in meta if r["split"] == split])
    rows = [meta[i] for i in idx]
    lat = torch.from_numpy(np.load(packed / f"{game}_lat.npy", mmap_mode="r")[idx]).to(device)
    thumb = torch.from_numpy(np.load(packed / f"{game}_thumb.npy", mmap_mode="r")[idx]).to(device)
    d = None
    if dino:
        d = torch.from_numpy(np.load(packed / f"{game}_dino.npy", mmap_mode="r")[idx]).to(device)
    return Domain(lat, thumb, d, rows)


def _win(win) -> tuple[int, int]:
    if win is None or win == "full":
        return LAT_H, LAT_W
    if isinstance(win, int):
        return win, win
    return tuple(win)


def pair_cost(D0: Domain, D1: Domain, i0, i1, oy, ox, wh, ww, cost: str) -> torch.Tensor:
    """裁窗区域上的逐位置对齐代价矩阵 (len(i0), len(i1))。"""
    if cost == "rgb":   # 缩略图每个 latent 格 = 2×2 像素；逐像素平方误差（[0,1] 尺度，按像素平均）
        ys, xs = slice(2 * oy, 2 * (oy + wh)), slice(2 * ox, 2 * (ox + ww))
        f0 = D0.thumb[i0, :, ys, xs].flatten(1).float() / 255
        f1 = D1.thumb[i1, :, ys, xs].flatten(1).float() / 255
        return torch.cdist(f0, f1).pow(2) / f0.shape[1]
    if cost == "dino":  # 逐位置的 patch token 余弦相似度，不池化（池化只比“有什么”，不比构图）
        d0 = F.normalize(D0.dino[i0, :, oy:oy + wh, ox:ox + ww].flatten(2).float(), dim=1)
        d1 = F.normalize(D1.dino[i1, :, oy:oy + wh, ox:ox + ww].flatten(2).float(), dim=1)
        return 1 - torch.einsum("adp,bdp->ab", d0, d1) / (wh * ww)
    raise ValueError(cost)


def sample_pairs(D0: Domain, D1: Domain, B: int, win=16, ot_ratio=0.8, cost="rgb", pool_mult=1,
                 gen: torch.Generator | None = None, return_info=False):
    """返回 z0, z1：(B,32,h,w) float32。前 k = B·ot_ratio 对走 OT，其余保持随机配对。

    pool_mult > 1 时在 k·pool_mult 个候选上解 OT，再随机取 k 对（配对池大于训练 batch，§5.1）。
    gen：CPU 上的 torch.Generator，给定时整个采样可复现（固定 val 集用）。
    """
    dev = D0.lat.device
    wh, ww = _win(win)
    r = lambda n, hi: torch.randint(hi, (n,), generator=gen).to(dev)
    oy = int(torch.randint(LAT_H - wh + 1, (1,), generator=gen))   # 本 batch 共用一个裁窗
    ox = int(torch.randint(LAT_W - ww + 1, (1,), generator=gen))
    k = int(round(B * ot_ratio))
    P = k * pool_mult
    i0 = torch.cat([r(P, len(D0)), r(B - k, len(D0))])
    i1 = torch.cat([r(P, len(D1)), r(B - k, len(D1))])
    info = {}
    if k > 0:
        C = pair_cost(D0, D1, i0[:P], i1[:P], oy, ox, wh, ww, cost)
        _, j = linear_sum_assignment(C.double().cpu().numpy())
        j = torch.as_tensor(j, device=dev)
        if return_info:
            info["cost_ot"] = C[torch.arange(P, device=dev), j].mean().item()
            info["cost_rand"] = C.diagonal().mean().item()        # i0、i1 独立随机 → 对角线就是随机配对
        sel = torch.randperm(P, generator=gen)[:k].to(dev) if pool_mult > 1 else torch.arange(k, device=dev)
        i0 = torch.cat([i0[:P][sel], i0[P:]])
        i1 = torch.cat([i1[:P][j][sel], i1[P:]])
    z0 = D0.lat[i0, :, oy:oy + wh, ox:ox + ww].float()
    z1 = D1.lat[i1, :, oy:oy + wh, ox:ox + ww].float()
    if return_info:
        info.update(oy=oy, ox=ox)
        return z0, z1, info
    return z0, z1
