"""
推理命令行（doc/sbflow-genshin-wuwa-plan.md §6）：单张图、目录或连续片段。

完整流程：帧 → 缩放到 1024×576 → 修补固定叠加层 → DC-AE 编码 ×scaling_factor → 概率流 ODE
         → ÷scaling_factor → DC-AE 解码。整帧单次前向，不切 tile；确定性，同一输入永远同一输出。

用法：
    python translate.py --run runs/p_e0.05_rgb --input data/10_frames/genshin/s00 --direction 0to1
    python translate.py --run runs/main --input some.png --nfe 16 --side-by-side
direction：0to1 = 原神→鸣潮，1to0 = 鸣潮→原神；不给时按输入路径里的游戏名推断。
输出：默认 runs/<run>/translate/<输入名>_<方向>_nfe<N>/
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sbflow.ae import decode, encode, load_ae
from sbflow.frames import list_images, load_canonical, to_tensor, to_uint8
from sbflow.ode import load_run, translate_latent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--direction", choices=["0to1", "1to0"], default=None)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--solver", choices=["euler", "heun"], default="euler")
    ap.add_argument("--ckpt", default="latest", help="latest 或 step_0020000 之类")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--side-by-side", action="store_true", help="输出 [输入 | 转换] 拼图")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    torch.backends.cuda.matmul.allow_tf32 = True

    src = Path(args.input)
    direction = args.direction or ("1to0" if "wuwa" in str(src).lower() else "0to1")
    files = list_images(src)
    if not files:
        raise SystemExit(f"{src} 下没有图片")
    bundle = load_run(args.run, args.ckpt)
    ae = load_ae("cuda", torch.float16)
    out = Path(args.out) if args.out else Path(args.run) / "translate" / f"{src.stem}_{direction}_nfe{args.nfe}"
    out.mkdir(parents=True, exist_ok=True)
    print(f"run {args.run}（step {bundle['step']}，EMA）  {direction}  NFE {args.nfe}  {len(files)} 张 → {out}")

    t_all = time.time()
    for i in range(0, len(files), args.bs):
        batch = files[i:i + args.bs]
        arrs = [load_canonical(p, bundle["overlay_mask"]) for p in batch]
        x = torch.stack([to_tensor(a) for a in arrs]).cuda()
        z = encode(ae, x).float()
        zt = translate_latent(bundle, z, direction, args.nfe, args.solver)
        y = to_uint8(decode(ae, zt))
        for p, a, yi in zip(batch, arrs, y):
            img = np.concatenate([a, yi], axis=1) if args.side_by_side else yi
            Image.fromarray(img).save(out / f"{p.stem}.png")
        print(f"{i + len(batch)}/{len(files)}")
    print(f"完成，{(time.time() - t_all) / len(files) * 1000:.0f} ms/张（含读写盘）")


if __name__ == "__main__":
    main()
