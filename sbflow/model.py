"""DiT + AdaLN-Zero + 2D 轴向 RoPE（doc/sbflow-genshin-wuwa-plan.md §5.2）。

参照 facebookresearch/DiT 的 models.py，改动：
    - 类别标签那一路换成方向 d 的 nn.Embedding(2, hidden)，和 τ 的 embedding 相加后送进 AdaLN-Zero
    - 去掉绝对位置编码，在每层 attention 的 q/k 上加 2D 轴向 RoPE（只依赖相对位移）
    - patch_size=1，token 网格可变：窗口 16×16 = 256 token，整帧 32×18 = 576 token
    - q/k 加 RMSNorm（QK-norm），在 RoPE 之前；稳定 attention logit，对序列变长也更稳
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DiTConfig:
    in_channels: int = 32
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_ratio: float = 4.0
    rope_theta: float = 10000.0
    qk_norm: bool = True

    def to_dict(self):
        return asdict(self)


PRESETS = {
    "S": DiTConfig(hidden_size=384, depth=12, num_heads=6),    # pilot，约 33M
    "B": DiTConfig(hidden_size=768, depth=12, num_heads=12),   # main，约 130M
}


# ---------------------------------------------------------------------------
# 2D 轴向 RoPE：head_dim 前一半编码 y、后一半编码 x，每轴 head_dim/4 个频率
# ---------------------------------------------------------------------------
def _rope_1d(pos: torch.Tensor, dim: int, theta: float):
    freqs = theta ** (-torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim)
    ang = pos.double()[:, None] * freqs[None]
    return ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)


def rope_2d(h: int, w: int, head_dim: int, theta: float, device):
    ys, xs = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij")
    cy, sy = _rope_1d(ys.flatten(), head_dim // 2, theta)
    cx, sx = _rope_1d(xs.flatten(), head_dim // 2, theta)
    return torch.cat([cy, cx], -1).float(), torch.cat([sy, sx], -1).float()   # (N, head_dim)


def apply_rope(q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q: (B, heads, N, D)。相邻两维 (a, b) 一组旋转。"""
    q2 = torch.stack([-q[..., 1::2], q[..., 0::2]], -1).flatten(-2)
    return q * cos.to(q.dtype) + q2 * sin.to(q.dtype)


# ---------------------------------------------------------------------------
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimestepEmbedder(nn.Module):
    """τ ∈ [0,1] 先乘 1000，再走 DiT 的正弦编码 + MLP。"""

    def __init__(self, hidden_size, freq_dim=256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(nn.Linear(freq_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, tau):
        t = tau.float() * 1000
        half = self.freq_dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], -1)
        return self.mlp(emb)


class RMSNorm(nn.Module):
    """权重随输入 dtype 走，autocast 下也能用融合 kernel。"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight.to(x.dtype), self.eps)


class Attention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True):
        super().__init__()
        self.h = num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        hd = dim // num_heads
        self.q_norm = RMSNorm(hd) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(hd) if qk_norm else nn.Identity()

    def forward(self, x, cos, sin):
        B, N, C = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v)
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class DiTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qk_norm=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, num_heads, qk_norm)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c, cos, sin):
        sh1, sc1, g1, sh2, sc2, g2 = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + g1.unsqueeze(1) * self.attn(modulate(self.norm1(x), sh1, sc1), cos, sin)
        x = x + g2.unsqueeze(1) * self.mlp(modulate(self.norm2(x), sh2, sc2))
        return x


class FinalLayer(nn.Module):
    def __init__(self, dim, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(dim, out_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


class DiT(nn.Module):
    def __init__(self, cfg: DiTConfig):
        super().__init__()
        self.cfg = cfg
        D = cfg.hidden_size
        self.head_dim = D // cfg.num_heads
        assert self.head_dim % 4 == 0, "2D RoPE 要求 head_dim 能被 4 整除"
        self.x_embedder = nn.Linear(cfg.in_channels, D)            # patch_size = 1
        self.t_embedder = TimestepEmbedder(D)
        self.d_embedder = nn.Embedding(2, D)                        # 方向：1 = 正向（0→1），0 = 反向
        self.blocks = nn.ModuleList([DiTBlock(D, cfg.num_heads, cfg.mlp_ratio, cfg.qk_norm) for _ in range(cfg.depth)])
        self.final_layer = FinalLayer(D, cfg.in_channels)
        self._rope_cache: dict = {}
        self.initialize_weights()

    def initialize_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.d_embedder.weight, std=0.02)
        for b in self.blocks:                                      # AdaLN-Zero：每个块初始为恒等
            nn.init.zeros_(b.adaLN_modulation[-1].weight)
            nn.init.zeros_(b.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)             # 输出初始为 0 → D = c_skip·x
        nn.init.zeros_(self.final_layer.linear.bias)

    def rope(self, h, w, device):
        key = (h, w, device)
        if key not in self._rope_cache:
            self._rope_cache[key] = rope_2d(h, w, self.head_dim, self.cfg.rope_theta, device)
        return self._rope_cache[key]

    def forward(self, x, tau, d):
        """x: (B,C,H,W)（已乘 c_in）；tau: (B,) 网络时间；d: (B,) 方向。返回 F，形状同 x。"""
        B, C, H, W = x.shape
        h = self.x_embedder(x.flatten(2).transpose(1, 2))
        c = self.t_embedder(tau) + self.d_embedder(d.long())
        cos, sin = self.rope(H, W, x.device)
        for blk in self.blocks:
            h = blk(h, c, cos, sin)
        out = self.final_layer(h, c)
        return out.transpose(1, 2).reshape(B, C, H, W)


def build_model(size: str = "B", **overrides) -> DiT:
    cfg = DiTConfig(**{**PRESETS[size].to_dict(), **overrides})
    return DiT(cfg)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
