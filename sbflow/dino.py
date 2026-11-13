"""DINOv2 特征（doc/sbflow-genshin-wuwa-plan.md §4.4 配对特征、§7.3 评估指标）。

整帧以 448×252 输入，patch 14 → 32×18 的 patch 网格，和 DC-AE latent 网格逐格对齐。
用带 registers 的版本：patch token 更干净（没有高范数伪影 token）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

import sbflow  # noqa: F401

DINO_PAIR = "facebook/dinov2-with-registers-small"   # 配对代价：384 维，PCA 到 64 维存盘
DINO_EVAL = "facebook/dinov2-with-registers-base"    # 评估指标：768 维（比配对用的大一号，避免“自己考自己”）
DINO_W, DINO_H = 448, 252
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


class Dino:
    def __init__(self, name: str = DINO_PAIR, device="cuda", dtype=torch.float16):
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained(name, dtype=dtype).to(device).eval()
        self.model.requires_grad_(False)
        self.n_reg = getattr(self.model.config, "num_register_tokens", 0)
        self.dim = self.model.config.hidden_size
        self.device, self.dtype = device, dtype
        self.mean = torch.tensor(_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(_STD, device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor):
        """x: (B,3,H,W)，范围 [-1,1]，16:9。返回 cls (B,C) 和 patch (B, 18*32, C)，float32。"""
        x = (x.to(self.device).float() + 1) / 2
        if x.shape[-2:] != (DINO_H, DINO_W):
            x = F.interpolate(x, size=(DINO_H, DINO_W), mode="bicubic", antialias=True, align_corners=False)
        x = ((x.clamp(0, 1) - self.mean) / self.std).to(self.dtype)
        h = self.model(pixel_values=x).last_hidden_state.float()
        return h[:, 0], h[:, 1 + self.n_reg:]
