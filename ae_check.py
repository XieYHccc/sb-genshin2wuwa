"""
DC-AE 重建质量闸门（doc/sbflow-genshin-wuwa-plan.md §4.2，go/no-go）。

把一批游戏截图过 encode → decode，输出 [原图 | 重建 | 误差×4] 对照图和 PSNR，
人眼检查描边、远景植被、水面高光、草地纹理有没有糊掉。

用法：
    python ae_check.py --images data/10_frames/genshin/s00               # 512×512 中心裁
    python ae_check.py --images data/10_frames/genshin/s01 --full        # 规范分辨率 1024×576 整帧
输出：data/ae_check/<游戏>_<会话>[_full]/*.png + summary.txt（文件名以 PSNR 开头，最差的排最前）
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sbflow.ae import DCAE, decode, encode, load_ae
from sbflow.frames import list_images, to_canonical

RES = 512
ROOT = Path(__file__).resolve().parent


def load_center_crop(path: Path, res=RES) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((res, res), Image.LANCZOS)


def out_name(src: Path, full: bool) -> str:
    name = f"{src.parent.name}_{src.name}" if src.parent.name in {"genshin", "wuwa"} else src.name
    return name + ("_full" if full else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default=DCAE)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--full", action="store_true", help="规范分辨率 1024×576 整帧（默认 512×512 中心裁）")
    ap.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")

    src = Path(args.images)
    files = list_images(src)
    if not files:
        raise SystemExit(f"{src} 下没有图片")
    random.seed(args.seed)
    files = random.sample(files, min(args.n, len(files)))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    ae = load_ae(dev, dtype, args.model)
    load = (lambda p: to_canonical(Image.open(p))) if args.full else load_center_crop
    print(f"device={dev}  dtype={args.dtype}  scaling_factor={ae.config.scaling_factor}  "
          f"images={len(files)}  mode={'full 1024x576' if args.full else 'center 512'}")

    out = ROOT / "data" / "ae_check" / out_name(src, args.full)
    out.mkdir(parents=True, exist_ok=True)
    psnrs, lat_stats = [], []
    for i in range(0, len(files), args.bs):
        batch = files[i:i + args.bs]
        imgs = [np.asarray(load(p)) for p in batch]
        x = torch.stack([torch.from_numpy(im).permute(2, 0, 1) for im in imgs]).float()
        x = (x / 127.5 - 1).to(dev)
        z = encode(ae, x)
        lat_stats.append(z.float().cpu())
        y = decode(ae, z)
        for p, a, yi in zip(batch, imgs, y):
            b = ((yi + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
            psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-8))
            psnrs.append((psnr, p.name))
            err = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            Image.fromarray(np.concatenate([a, b, err], axis=1)).save(out / f"{psnr:05.2f}dB_{p.stem}.png")
        print(f"{i + len(batch)}/{len(files)}")

    z = torch.cat(lat_stats)
    psnrs.sort()
    vals = np.array([p for p, _ in psnrs])
    lines = [
        f"model: {args.model}  dtype: {args.dtype}  mode: {'full 1024x576' if args.full else 'center 512'}",
        f"PSNR mean {vals.mean():.2f} dB, median {np.median(vals):.2f}, min {vals[0]:.2f} ({psnrs[0][1]}), max {vals[-1]:.2f}",
        f"scaled latent: shape {tuple(z.shape[1:])}, mean {z.mean():.4f}, std {z.std():.4f}  (std≈1 说明 scaling_factor 合适)",
        "最差 5 张（重点人眼看这些）: " + ", ".join(n for _, n in psnrs[:5]),
    ]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"对照图: {out}  （左原图 | 中重建 | 右误差×4）")


if __name__ == "__main__":
    main()
