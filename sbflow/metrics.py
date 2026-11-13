"""评估指标（doc/sbflow-genshin-wuwa-plan.md §7.3、§7.4）。

所有比较都在“AE 重建空间”里做：输入用 decode(z_src)，参照的真实目标域帧也用 decode(z_tgt)，
这样 DC-AE 自身的重建损失不会混进转换模型的指标里。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from sbflow.dino import DINO_EVAL, Dino


class Evaluator:
    def __init__(self, device="cuda"):
        import lpips
        self.dino = Dino(DINO_EVAL, device)
        self.lpips = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        self.lpips.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def feats(self, x: torch.Tensor, bs=32):
        """x: (N,3,H,W) [-1,1]（CPU 或 GPU）→ cls (N,768) 和 patch token (N,576,768)，都在 GPU、float32。"""
        cls, pat = [], []
        for i in range(0, len(x), bs):
            c, p = self.dino(x[i:i + bs].to(self.device))
            cls.append(c); pat.append(p)
        return torch.cat(cls), torch.cat(pat)

    @torch.no_grad()
    def lpips_dist(self, a: torch.Tensor, b: torch.Tensor, bs=16) -> torch.Tensor:
        out = []
        for i in range(0, len(a), bs):
            out.append(self.lpips(a[i:i + bs].to(self.device).float(), b[i:i + bs].to(self.device).float()).flatten())
        return torch.cat(out)


@torch.no_grad()
def struct_dist(pa: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
    """DINO self-similarity 距离：‖S(a) − S(b)‖_F² / N²，S 是归一化 patch token 的两两余弦相似度（Splice）。"""
    out = []
    for a, b in zip(pa, pb):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        out.append(((a @ a.T - b @ b.T) ** 2).mean())
    return torch.stack(out)


@torch.no_grad()
def kid(fx: torch.Tensor, fy: torch.Tensor, n_subsets=100, subset_size=1000, seed=0):
    """KID：多项式核 k(a,b) = (aᵀb/d + 1)³ 的无偏 MMD²，随机子集取平均。返回 (mean, std)。"""
    fx, fy = fx.double(), fy.double()
    d = fx.shape[1]
    m = min(len(fx), len(fy), subset_size)
    g = torch.Generator().manual_seed(seed)
    vals = []
    for _ in range(n_subsets):
        x = fx[torch.randperm(len(fx), generator=g)[:m].to(fx.device)]
        y = fy[torch.randperm(len(fy), generator=g)[:m].to(fy.device)]
        kxx = (x @ x.T / d + 1) ** 3
        kyy = (y @ y.T / d + 1) ** 3
        kxy = (x @ y.T / d + 1) ** 3
        mmd = ((kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
               + (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))
               - 2 * kxy.mean())
        vals.append(mmd.item())
        if m == len(fx) == len(fy):   # 全集只有一个子集
            break
    return float(np.mean(vals)), float(np.std(vals))


@torch.no_grad()
def nn_similarity(f_query: torch.Tensor, f_bank: torch.Tensor) -> torch.Tensor:
    """每个 query 到 bank 的最大余弦相似度（DINO 全局特征）。查背图用（§7.1）。"""
    q, b = F.normalize(f_query.float(), dim=-1), F.normalize(f_bank.float(), dim=-1)
    return (q @ b.T).max(1).values


# ---------------------------------------------------------------------------
# Reinhard 全局调色（CIE Lab 空间均值 / 方差匹配）。Lab 转换在 GPU 上用 torch 做（sRGB、D65）
# ---------------------------------------------------------------------------
_M = torch.tensor([[0.4124564, 0.3575761, 0.1804375],
                   [0.2126729, 0.7151522, 0.0721750],
                   [0.0193339, 0.1191920, 0.9503041]], dtype=torch.float64)
_WHITE = torch.tensor([0.95047, 1.0, 1.08883], dtype=torch.float64)
_D = 6 / 29


def rgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) [-1,1] sRGB → (B,3,H,W) CIE Lab（L∈[0,100]），float32。"""
    c = ((x.float().clamp(-1, 1) + 1) / 2)
    lin = torch.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    xyz = torch.einsum("ij,bjhw->bihw", _M.to(x.device, torch.float32), lin)
    t = xyz / _WHITE.to(x.device, torch.float32).view(1, 3, 1, 1)
    f = torch.where(t > _D ** 3, t.clamp_min(1e-12) ** (1 / 3), t / (3 * _D ** 2) + 4 / 29)
    return torch.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], 1)


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    """CIE Lab → sRGB [-1,1]（超出色域的部分 clamp）。"""
    fy = (lab[:, 0] + 16) / 116
    f = torch.stack([fy + lab[:, 1] / 500, fy, fy - lab[:, 2] / 200], 1)
    t = torch.where(f > _D, f ** 3, 3 * _D ** 2 * (f - 4 / 29))
    xyz = t * _WHITE.to(lab.device, torch.float32).view(1, 3, 1, 1)
    lin = torch.einsum("ij,bjhw->bihw", torch.linalg.inv(_M).to(lab.device, torch.float32), xyz).clamp(0, 1)
    c = torch.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin.clamp_min(1e-12) ** (1 / 2.4) - 0.055)
    return c.clamp(0, 1) * 2 - 1


def lab_stats(x: torch.Tensor, device="cuda", bs=32):
    """每张图 Lab 三通道的均值和标准差，各 (B,3) numpy。"""
    ms, ss = [], []
    for i in range(0, len(x), bs):
        lab = rgb_to_lab(x[i:i + bs].to(device)).flatten(2)
        ms.append(lab.mean(-1).cpu()); ss.append(lab.std(-1).cpu())
    return torch.cat(ms).numpy(), torch.cat(ss).numpy()


@torch.no_grad()
def reinhard(src: torch.Tensor, ref_mean, ref_std, device="cuda", bs=32) -> torch.Tensor:
    """把 src 每张图的 Lab 均值 / 方差匹配到 ref_mean / ref_std（形状 (3,) 或 (B,3)）。返回 CPU [-1,1]。"""
    rm = torch.as_tensor(np.broadcast_to(ref_mean, (len(src), 3)).copy(), dtype=torch.float32)
    rs = torch.as_tensor(np.broadcast_to(ref_std, (len(src), 3)).copy(), dtype=torch.float32)
    out = []
    for i in range(0, len(src), bs):
        lab = rgb_to_lab(src[i:i + bs].to(device))
        m = lab.mean((2, 3), keepdim=True)
        s = lab.std((2, 3), keepdim=True) + 1e-6
        r_m = rm[i:i + bs].to(device).view(-1, 3, 1, 1)
        r_s = rs[i:i + bs].to(device).view(-1, 3, 1, 1)
        out.append(lab_to_rgb((lab - m) / s * r_s + r_m).cpu())
    return torch.cat(out)


@torch.no_grad()
def tone_collapse_ratio(ev: Evaluator, x_in: torch.Tensor, y_out: torch.Tensor) -> torch.Tensor:
    """ρ = LPIPS(调色后的输入, 输出) / LPIPS(输入, 输出)。<0.15 基本只调了色；0.25–0.6 正常；>0.8 结构被大改（§7.4）。"""
    m, s = lab_stats(y_out)
    x_rec = reinhard(x_in, m, s)
    return ev.lpips_dist(x_rec, y_out) / ev.lpips_dist(x_in, y_out).clamp_min(1e-6)
