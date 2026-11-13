"""
打包（doc/sbflow-genshin-wuwa-plan.md §4.3–4.5）：整帧 latent（含水平翻转）、配对用缩略图和 DINO 特征。

读 data/20_filtered/{game}.csv 里 keep=1 的帧 → 规范分辨率 1024×576（Lanczos）→ 用 overlay_mask.png 修补
固定叠加层 → 原图和水平翻转各一份：
    latent   DC-AE f32c32 编码（fp16）后乘 scaling_factor，(32,18,32) fp16。翻转是先翻图再重新编码，
             因为 AE 不是严格翻转等变的
    thumb    1/16 缩略图 (3,36,64) uint8，16×16 块平均，和 latent 网格逐格对齐
    dino     DINOv2-small(registers) 在 448×252 上的 patch token (18×32)，两域 train 帧上拟合一套 PCA
             （居中、不白化）降到 64 维，(64,18,32) fp16
行号 idx = 2*i + flip（i 是清单里第 i 个保留帧）。

用法：
    python pack.py
输出：data/30_packed/{game}_lat.npy、{game}_thumb.npy、{game}_dino.npy、{game}_meta.csv、
      dino_pca.pt、overlay_mask.png、pack_info.json、check/（叠加层修补前后对照）
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sbflow.ae import DCAE, encode, load_ae
from sbflow.dino import DINO_PAIR, Dino
from sbflow.frames import (LAT_H, LAT_W, THUMB_H, THUMB_W, load_canonical, load_overlay_mask,
                           thumb_from_canonical, to_tensor)

ROOT = Path(__file__).resolve().parent
FILTERED = ROOT / "data" / "20_filtered"
OUT = ROOT / "data" / "30_packed"
GAMES = ("genshin", "wuwa")


def read_list(game: str):
    with open(FILTERED / f"{game}.csv", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["keep"] == "1"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--bs", type=int, default=8, help="每批帧数（翻转后翻倍）")
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--pca-tokens", type=int, default=400_000, help="拟合 PCA 用的 token 数上限")
    ap.add_argument("--no-dino", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "check").mkdir(exist_ok=True)

    mask_path = FILTERED / "overlay_mask.png"
    mask = load_overlay_mask(mask_path)
    if mask is not None:
        shutil.copy(mask_path, OUT / "overlay_mask.png")
    ae = load_ae("cuda", torch.float16)
    dino = None if args.no_dino else Dino(DINO_PAIR)
    info = {"created": datetime.now().isoformat(timespec="seconds"), "dcae": DCAE,
            "scaling_factor": ae.config.scaling_factor, "dino": None if args.no_dino else DINO_PAIR,
            "overlay_mask": mask is not None, "games": {}}
    raw_dino = {}

    for game in args.games.split(","):
        rows = read_list(game)
        n = len(rows)
        lat = np.zeros((2 * n, 32, LAT_H, LAT_W), np.float16)
        thumb = np.zeros((2 * n, 3, THUMB_H, THUMB_W), np.uint8)
        dfeat = None if dino is None else np.zeros((2 * n, LAT_H * LAT_W, dino.dim), np.float16)
        t0 = time.time()
        load = lambda r: load_canonical(ROOT / r["file"], mask)
        with ThreadPoolExecutor(8) as ex:
            it = ex.map(load, rows)
            for b0 in range(0, n, args.bs):
                arrs = [next(it) for _ in range(min(args.bs, n - b0))]
                if b0 == 0 and mask is not None:      # 叠加层修补前后对照（左：原图，右：修补后）
                    raw = load_canonical(ROOT / rows[0]["file"], None)
                    ys, xs = np.nonzero(mask)
                    y0, y1 = max(ys.min() - 24, 0), min(ys.max() + 24, raw.shape[0])
                    for (cx0, cx1) in ((0, 160), (860, 1024)):
                        a, b = raw[:, cx0:cx1], arrs[0][:, cx0:cx1]
                        crop = np.concatenate([a, b], axis=1)
                        crop = crop[:80] if cx0 == 0 else crop[-60:]
                        Image.fromarray(crop).resize((crop.shape[1] * 3, crop.shape[0] * 3), Image.NEAREST).save(
                            OUT / "check" / f"{game}_overlay_{cx0}.png")
                x = torch.stack([to_tensor(a) for a in arrs]).cuda()
                x = torch.stack([x, x.flip(-1)], 1).flatten(0, 1)       # 交错：原图、翻转、原图、翻转…
                i0, i1 = 2 * b0, 2 * b0 + len(x)
                lat[i0:i1] = encode(ae, x).cpu().numpy().astype(np.float16)
                for j, a in enumerate(arrs):
                    t = thumb_from_canonical(a)
                    thumb[i0 + 2 * j] = t
                    thumb[i0 + 2 * j + 1] = t[:, :, ::-1]
                if dino is not None:
                    _, p = dino(x)
                    dfeat[i0:i1] = p.cpu().numpy().astype(np.float16)
                if (b0 // args.bs) % 20 == 0:
                    print(f"  {game}: {b0 + len(arrs)}/{n}  {time.time() - t0:.0f}s", flush=True)

        np.save(OUT / f"{game}_lat.npy", lat)
        np.save(OUT / f"{game}_thumb.npy", thumb)
        with open(OUT / f"{game}_meta.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "src_file", "session", "flip", "split"])
            for i, r in enumerate(rows):
                for flip in (0, 1):
                    w.writerow([2 * i + flip, r["file"], r["session"], flip, r["split"]])
        split = np.repeat(np.array([r["split"] for r in rows]), 2)
        lt = lat[split == "train"].astype(np.float32)
        info["games"][game] = {
            "frames": n, "rows": 2 * n,
            "train_rows": int((split == "train").sum()), "val_rows": int((split == "val").sum()),
            "latent_mean": float(lt.mean()), "latent_std": float(lt.std()),
            "latent_channel_mean": [round(float(v), 4) for v in lt.mean((0, 2, 3))],
            "latent_channel_std": [round(float(v), 4) for v in lt.std((0, 2, 3))],
            "seconds": round(time.time() - t0, 1),
        }
        print(f"{game}: {n} 帧 → {2 * n} 行，train latent mean {lt.mean():.4f} std {lt.std():.4f}，"
              f"{time.time() - t0:.0f}s")
        if dfeat is not None:
            raw_dino[game] = (dfeat, split)

    if raw_dino:
        # 两域 train 行上拟合一套 PCA（居中、不白化）
        rng = np.random.default_rng(0)
        pool = np.concatenate([d[s == "train"].reshape(-1, d.shape[-1]) for d, s in raw_dino.values()])
        sub = pool[rng.choice(len(pool), min(args.pca_tokens, len(pool)), replace=False)]
        X = torch.from_numpy(sub).cuda().float()
        mean = X.mean(0)
        Xc = X - mean
        cov = Xc.T @ Xc / len(Xc)
        evals, evecs = torch.linalg.eigh(cov)
        evals, evecs = evals.flip(0), evecs.flip(1)
        comp = evecs[:, :args.pca_dim]                     # (C, k)
        ratio = float(evals[:args.pca_dim].sum() / evals.sum())
        torch.save({"mean": mean.cpu(), "components": comp.cpu(), "explained": evals[:args.pca_dim].cpu(),
                    "explained_ratio": ratio, "model": DINO_PAIR}, OUT / "dino_pca.pt")
        print(f"DINO PCA {dino.dim}→{args.pca_dim}：解释方差 {ratio:.1%}")
        info["dino_pca_explained_ratio"] = ratio
        for game, (d, _) in raw_dino.items():
            out = np.zeros((len(d), args.pca_dim, LAT_H, LAT_W), np.float16)
            for i0 in range(0, len(d), 256):
                t = torch.from_numpy(d[i0:i0 + 256]).cuda().float()
                p = (t - mean) @ comp                       # (b, 576, k)
                out[i0:i0 + 256] = p.transpose(1, 2).reshape(-1, args.pca_dim, LAT_H, LAT_W).cpu().numpy()
            np.save(OUT / f"{game}_dino.npy", out)

    (OUT / "pack_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"输出: {OUT}")


if __name__ == "__main__":
    main()
