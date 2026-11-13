"""
平移等变测试（doc/sbflow-genshin-wuwa-plan.md §8.2）：不需要视频，估计“相位变化”带来的闪烁。

在规范分辨率的整帧上取一个比整帧窄 32 像素的窗口（水平 960×576、垂直 1024×544），
把窗口平移 k = 1…32 像素，分别过 f，再在公共区域（去掉边界 margin）里和 k=0 的结果比较：
    f = AE 重建（encode → decode）           → 这条曲线是闪烁的下限
    f = 完整转换（encode → ODE → decode）   → 加 --run 时测；和 AE 曲线之差是转换模型引入的
k=32 正好是一个 latent token，严格平移等变的网络在 k=32 处应接近 k=0（只剩边界上下文的影响）。

用法：
    python ae_shift_test.py --images data/10_frames/genshin/s00 --n 8
    python ae_shift_test.py --images data/10_frames/genshin/s00 --n 8 --run runs/<run>   # 加测完整转换
输出：data/shift_test/<名字>/curve.csv、curve.png、summary.txt
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sbflow.ae import decode, encode, load_ae
from sbflow.frames import CANON_H, CANON_W, list_images, load_canonical, to_tensor

ROOT = Path(__file__).resolve().parent
MAX_K = 32
MARGIN = 32  # 公共区域再去掉离窗口边界 32 像素以内的部分（AE / 网络的边界效应）


def crops(x: torch.Tensor, axis: str):
    """x: (3,576,1024) → 33 个平移窗口 (33,3,h,w)。"""
    if axis == "x":
        return torch.stack([x[:, :, k:k + CANON_W - MAX_K] for k in range(MAX_K + 1)])
    return torch.stack([x[:, k:k + CANON_H - MAX_K, :] for k in range(MAX_K + 1)])


def shift_mse(y: torch.Tensor, axis: str) -> np.ndarray:
    """y: (33,3,h,w) 各窗口的输出。返回每个 k 与 k=0 在公共区域内的 MSE（像素值 0–255 尺度）。"""
    y = (y.float() + 1) * 127.5
    ref = y[0]
    n = (ref.shape[-1] if axis == "x" else ref.shape[-2])
    out = []
    for k in range(MAX_K + 1):
        lo, hi = k + MARGIN, n - MARGIN          # 规范坐标下的公共列（行）区间
        if axis == "x":
            a, b = ref[:, :, lo:hi], y[k][:, :, lo - k:hi - k]
        else:
            a, b = ref[:, lo:hi, :], y[k][:, lo - k:hi - k, :]
        out.append(((a - b) ** 2).mean().item())
    return np.array(out)


def psnr(mse):
    return 10 * np.log10(255 ** 2 / np.maximum(mse, 1e-10))


def plot(curves: dict, path: Path, title: str):
    """简单折线图：横轴 k=1..32，纵轴 PSNR(dB)。"""
    W, H, L, R, T, B = 900, 420, 60, 20, 40, 50
    ys = np.concatenate([v[1:] for v in curves.values()])
    lo, hi = np.floor(ys.min() - 1), np.ceil(min(ys.max(), 80) + 1)
    im = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(im)
    X = lambda k: L + (k - 1) / (MAX_K - 1) * (W - L - R)
    Y = lambda v: T + (hi - min(v, hi)) / (hi - lo) * (H - T - B)
    for v in np.arange(lo, hi + 1e-6, max(1, round((hi - lo) / 8))):
        d.line([(L, Y(v)), (W - R, Y(v))], fill=(225, 225, 225))
        d.text((8, Y(v) - 6), f"{v:.0f}dB", fill="black")
    for k in (1, 8, 16, 24, 32):
        d.line([(X(k), T), (X(k), H - B)], fill=(235, 235, 235))
        d.text((X(k) - 6, H - B + 8), str(k), fill="black")
    colors = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40)]
    for i, (name, v) in enumerate(curves.items()):
        pts = [(X(k), Y(v[k])) for k in range(1, MAX_K + 1)]
        d.line(pts, fill=colors[i % 4], width=2)
        d.text((L + 10 + 230 * i, 12), f"— {name}", fill=colors[i % 4])
    d.text((W // 2 - 120, H - 22), f"shift k (px)   {title}", fill="black")
    im.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--run", default=None, help="训练好的 run 目录；给了就加测完整转换")
    ap.add_argument("--direction", default=None, help="0to1（原神→鸣潮）或 1to0；默认按路径里的游戏名推断")
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--bs", type=int, default=11)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    src = Path(args.images)
    files = list_images(src)
    random.seed(args.seed)
    files = sorted(random.sample(files, min(args.n, len(files))))
    ae = load_ae("cuda", torch.float16)

    model = None
    if args.run:
        from sbflow.ode import load_run, translate_latent
        model = load_run(args.run)
        direction = args.direction or ("0to1" if "genshin" in str(src) else "1to0")
        mask = model["overlay_mask"]
    else:
        mask = None

    curves = {}
    for axis in ("x", "y"):
        mse_ae, mse_tr = [], []
        for p in files:
            x = to_tensor(load_canonical(p, mask))
            c = crops(x, axis)
            ys_ae, ys_tr = [], []
            for i in range(0, len(c), args.bs):
                z = encode(ae, c[i:i + args.bs].cuda())
                ys_ae.append(decode(ae, z).cpu())
                if model is not None:
                    zt = translate_latent(model, z.float(), direction=direction, nfe=args.nfe)
                    ys_tr.append(decode(ae, zt).cpu())
            mse_ae.append(shift_mse(torch.cat(ys_ae), axis))
            if model is not None:
                mse_tr.append(shift_mse(torch.cat(ys_tr), axis))
            print(f"axis {axis}: {p.name}")
        curves[f"AE {axis}"] = psnr(np.mean(mse_ae, 0))
        if model is not None:
            curves[f"translate {axis}"] = psnr(np.mean(mse_tr, 0))

    name = args.name or (f"{src.parent.name}_{src.name}" if src.parent.name in {"genshin", "wuwa"} else src.name)
    if args.run:
        name += f"_{Path(args.run).name}"
    out = ROOT / "data" / "shift_test" / name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k"] + [f"psnr_{k.replace(' ', '_')}" for k in curves])
        for k in range(1, MAX_K + 1):
            w.writerow([k] + [f"{v[k]:.3f}" for v in curves.values()])
    plot(curves, out / "curve.png", name)

    lines = [f"images: {src} ×{len(files)}   margin {MARGIN}px   {'run ' + args.run if args.run else 'AE only'}"]
    for k_name, v in curves.items():
        mid = v[1:MAX_K]                      # k = 1..31（不含整 token 平移）
        lines.append(f"{k_name:>12}: mean(k=1..31) {mid.mean():.2f} dB   min {mid.min():.2f} dB (k={mid.argmin() + 1})"
                     f"   k=16 {v[16]:.2f}   k=32 {v[32]:.2f}")
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"输出: {out}")


if __name__ == "__main__":
    main()
