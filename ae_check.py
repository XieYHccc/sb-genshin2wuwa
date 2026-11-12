"""
DC-AE 重建质量闸门（plan §3.2 / §8.1 第 1 步，go/no-go）。

把一批游戏截图过 encode → decode，输出 [原图 | 重建 | 放大误差] 对照图和 PSNR，
人眼检查描边、远景植被、水面高光有没有糊掉。

用法：
    python ae_check.py --images data/probe                 # 任意截图目录
    python ae_check.py --images data/10_frames/genshin --n 50
输出：data/ae_check/<目录名>/*.png + summary.txt
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "mit-han-lab/dc-ae-f32c32-sana-1.1-diffusers"
RES = 512


def load_center_crop(path: Path, res=RES) -> Image.Image:
    img = Image.open(path).convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    return img.resize((res, res), Image.LANCZOS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--bs", type=int, default=4)
    args = ap.parse_args()

    import sys; sys.stdout.reconfigure(errors="replace")
    from diffusers import AutoencoderDC

    src = Path(args.images)
    files = sorted(p for p in src.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
    if not files:
        raise SystemExit(f"{src} 下没有图片")
    random.seed(0)
    files = random.sample(files, min(args.n, len(files)))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ae = AutoencoderDC.from_pretrained(args.model, torch_dtype=torch.float32).to(dev).eval()
    sf = ae.config.scaling_factor
    print(f"device={dev}  scaling_factor={sf}  images={len(files)}")

    out = Path(__file__).resolve().parent / "data" / "ae_check" / src.name
    out.mkdir(parents=True, exist_ok=True)
    psnrs, lat_stats = [], []
    for i in range(0, len(files), args.bs):
        batch = files[i:i + args.bs]
        imgs = [load_center_crop(p) for p in batch]
        x = torch.stack([torch.from_numpy(np.array(im)).permute(2, 0, 1) for im in imgs]).float()
        x = (x / 127.5 - 1).to(dev)
        with torch.no_grad():
            z = ae.encode(x).latent
            lat_stats.append((z * sf).float().cpu())
            y = ae.decode(z).sample.clamp(-1, 1)
        for p, xi, yi in zip(batch, x, y):
            a = ((xi + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            b = ((yi + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
            mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
            psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-8))
            psnrs.append((psnr, p.name))
            err = np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
            Image.fromarray(np.concatenate([a, b, err], axis=1)).save(out / f"{psnr:05.2f}dB_{p.stem}.png")
        print(f"{i + len(batch)}/{len(files)}")

    z = torch.cat(lat_stats)
    psnrs.sort()
    lines = [
        f"model: {args.model}",
        f"PSNR mean {np.mean([p for p, _ in psnrs]):.2f} dB, min {psnrs[0][0]:.2f} ({psnrs[0][1]}), max {psnrs[-1][0]:.2f}",
        f"scaled latent: shape {tuple(z.shape[1:])}, mean {z.mean():.4f}, std {z.std():.4f}  (std≈1 说明 scaling_factor 合适)",
        "最差 5 张（重点人眼看这些）: " + ", ".join(n for _, n in psnrs[:5]),
    ]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"对照图: {out}  （左原图 | 中重建 | 右误差×4）")


if __name__ == "__main__":
    main()
