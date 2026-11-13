"""规范分辨率下的帧读取与预处理（doc/sbflow-genshin-wuwa-plan.md §4.1、§3.4）。

训练（打包）、评估、推理必须走同一个函数，保证像素尺度和预处理完全一致：
    1920×1080 原始帧 → Lanczos 缩放到 1024×576 → 用两域并集掩码修补固定叠加层
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

CANON_W, CANON_H = 1024, 576                 # 规范分辨率（§4.1）
LAT_W, LAT_H = CANON_W // 32, CANON_H // 32  # DC-AE f32 的 latent 网格：32×18
THUMB_W, THUMB_H = CANON_W // 16, CANON_H // 16  # 配对用 RGB 缩略图：64×36（每个 latent 格 2×2 像素）

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp"}


def to_canonical(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")
    if img.size != (CANON_W, CANON_H):
        img = img.resize((CANON_W, CANON_H), Image.LANCZOS)
    return img


def load_overlay_mask(path: str | Path | None) -> np.ndarray | None:
    """固定叠加层掩码（规范分辨率，uint8，255 = 需要修补）。文件不存在时返回 None。"""
    if path is None or not Path(path).exists():
        return None
    m = np.asarray(Image.open(path).convert("L"))
    assert m.shape == (CANON_H, CANON_W), f"overlay mask 尺寸应为 {CANON_W}x{CANON_H}，实际 {m.shape[::-1]}"
    return np.where(m > 127, 255, 0).astype(np.uint8)


def fill_overlay(arr: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """用 Telea inpainting 抹掉固定叠加层（UID 文字、隐藏 UI 的图标等）。两域用同一张并集掩码。"""
    if mask is None or not mask.any():
        return arr
    import cv2
    return cv2.inpaint(np.ascontiguousarray(arr), mask, 5, cv2.INPAINT_TELEA)


def load_canonical(path: str | Path, mask: np.ndarray | None = None) -> np.ndarray:
    """读一帧 → (576, 1024, 3) uint8 RGB。mask 给定时修补叠加层。"""
    with Image.open(path) as im:
        arr = np.asarray(to_canonical(im))
    return fill_overlay(arr, mask)


def to_tensor(arr: np.ndarray) -> torch.Tensor:
    """(H,W,3) uint8 → (3,H,W) float32，范围 [-1, 1]。"""
    return torch.from_numpy(np.array(arr, dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(127.5).sub_(1)


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """(3,H,W) 或 (B,3,H,W)，范围 [-1,1] → uint8 HWC / BHWC（CPU numpy）。"""
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte()
    return x.movedim(-3, -1).cpu().numpy()


def thumb_from_canonical(arr: np.ndarray) -> np.ndarray:
    """规范分辨率 RGB → 1/16 缩略图 (3, 36, 64) uint8：16×16 块平均，和 latent 网格逐格对齐（§4.4）。"""
    h, w, _ = arr.shape
    t = arr.reshape(h // 16, 16, w // 16, 16, 3).astype(np.float32).mean(axis=(1, 3))
    return np.round(t).astype(np.uint8).transpose(2, 0, 1)


def list_images(src: str | Path) -> list[Path]:
    src = Path(src)
    if src.is_file():
        return [src]
    return sorted(p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT and ".tmp" not in p.suffixes)
