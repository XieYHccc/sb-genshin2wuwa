"""DC-AE 加载与编解码（doc/sbflow-genshin-wuwa-plan.md §4.2、§4.3）。

latent 一律以“乘过 scaling_factor”的形式在代码里流转（约单位方差），解码前再除回去。
"""
from __future__ import annotations

import torch

import sbflow  # noqa: F401  (设置 HF 环境变量)

DCAE = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"


def load_ae(device="cuda", dtype=torch.float16, model=DCAE):
    from diffusers import AutoencoderDC
    ae = AutoencoderDC.from_pretrained(model, torch_dtype=dtype).to(device).eval()
    ae.requires_grad_(False)
    return ae


@torch.no_grad()
def encode(ae, x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) [-1,1] → scaled latent (B,32,H/32,W/32)，dtype 与 AE 相同。"""
    sf = ae.config.scaling_factor           # 从配置读，不要硬编码（sana-1.1 是 0.41407）
    return ae.encode(x.to(ae.device, ae.dtype)).latent * sf


@torch.no_grad()
def decode(ae, z: torch.Tensor) -> torch.Tensor:
    """scaled latent → 图像 (B,3,H,W)，float32，clamp 到 [-1,1]。"""
    sf = ae.config.scaling_factor
    return ae.decode(z.to(ae.device, ae.dtype) / sf).sample.float().clamp(-1, 1)
