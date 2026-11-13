"""评估面板（doc/sbflow-genshin-wuwa-plan.md §7.2）。

固定一批 val 帧（固定种子选一次，存成 runs/<run>/panel_frames.json），整帧 1024×576，用 EMA 模型、
固定 NFE 跑到 ODE 终点，输出 `源图 | 转换后` 对照网格，两个方向各一张，并算指标：
    快速面板：每方向 panel_n 张；LPIPS、结构距离、tone-collapse ρ、KID
    完整面板：全部 val 帧；再加 NFE–质量曲线（4/8/16/32）、背图检查（最近邻）、窗口 vs 整帧一致性（§5.4）
两个 baseline 和模型用同一批输入：不转换（identity），以及 Reinhard 全局调色到目标域 train 的 Lab 统计。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sbflow.ae import decode, load_ae
from sbflow.dino import DINO_EVAL
from sbflow.metrics import (Evaluator, kid, lab_stats, nn_similarity, reinhard, struct_dist,
                            tone_collapse_ratio)
from sbflow.ode import translate
from sbflow.pairing import GAMES

DIRS = (("0to1", 0, 1), ("1to0", 1, 0))   # (名字, 源域, 目标域)
CACHE_VERSION = 2                          # eval_cache.pt 的格式 / 口径变了就加一（2：Lab 改用 torch 实现）


def _unflipped(D):
    return np.array([i for i, r in enumerate(D.rows) if r["flip"] == "0"])


def _to_u8(x: torch.Tensor) -> torch.Tensor:
    return ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def _from_u8(x: torch.Tensor) -> torch.Tensor:
    return x.float() / 127.5 - 1


def grid(src: torch.Tensor, out: torch.Tensor, cols=4, w=384, h=216) -> Image.Image:
    """src/out: (N,3,H,W) uint8 → 每行 cols 对 [源 | 转换]。"""
    n = len(src)
    rows = (n + cols - 1) // cols
    im = Image.new("RGB", (cols * 2 * w + (cols - 1) * 8, rows * h), "white")
    for i in range(n):
        x0, y0 = (i % cols) * (2 * w + 8), (i // cols) * h
        for j, t in enumerate((src[i], out[i])):
            tile = Image.fromarray(t.permute(1, 2, 0).numpy()).resize((w, h), Image.BILINEAR)
            im.paste(tile, (x0 + j * w, y0))
    return im


class Panel:
    def __init__(self, run_dir, packed, D_val, D_train, cfg, device="cuda"):
        self.run, self.packed, self.cfg, self.dev = Path(run_dir), Path(packed), cfg, device
        self.Dv, self.Dt = D_val, D_train
        self.val_idx = [_unflipped(D) for D in D_val]
        self.train_idx = [_unflipped(D) for D in D_train]
        pf = self.run / "panel_frames.json"
        if pf.exists():
            sel = json.loads(pf.read_text(encoding="utf-8"))
            self.quick_idx = [np.array(sel[g]["quick"]) for g in GAMES]
        else:
            rng = np.random.default_rng(cfg["seed"])
            self.quick_idx = [np.sort(rng.choice(v, min(cfg["panel_n"], len(v)), replace=False)) for v in self.val_idx]
            pf.write_text(json.dumps({g: {"quick": q.tolist(), "files": [D.rows[i]["src_file"] for i in q]}
                                      for g, q, D in zip(GAMES, self.quick_idx, D_val)},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
        # 完整面板：全部 val 帧，超过 256 张时用固定种子抽 256 张（§7.2）
        rng = np.random.default_rng(cfg["seed"] + 1)
        self.full_idx = [v if len(v) <= 256 else np.sort(rng.choice(v, 256, replace=False)) for v in self.val_idx]
        self.ready = False

    # ------------------------------------------------------------------
    def _decode_rows(self, D, idx, bs=8):
        out = []
        for i in range(0, len(idx), bs):
            out.append(_to_u8(decode(self.ae, D.lat[idx[i:i + bs]].float())))
        return torch.cat(out)

    def setup(self):
        if self.ready:
            return
        self.ae = load_ae(self.dev, torch.float16)
        self.ev = Evaluator(self.dev)
        # val 帧的 AE 重建（输入与参照都在重建空间里比较）
        self.val_img = [self._decode_rows(D, v) for D, v in zip(self.Dv, self.val_idx)]
        self.val_cls = [self.ev.feats(_from_u8(x))[0] for x in self.val_img]
        # train 帧：DINO 全局特征（背图检查的 bank）和 Lab 统计（Reinhard baseline），按打包数据缓存
        cache_p = self.packed / "eval_cache.pt"
        files = [[D.rows[i]["src_file"] for i in t] for D, t in zip(self.Dt, self.train_idx)]
        cache = torch.load(cache_p, weights_only=False) if cache_p.exists() else None
        if (cache is None or cache.get("model") != DINO_EVAL or cache.get("files") != files
                or cache.get("version") != CACHE_VERSION):
            cache = {"model": DINO_EVAL, "files": files, "version": CACHE_VERSION,
                     "train_cls": [], "lab_mean": [], "lab_std": []}
            for D, t in zip(self.Dt, self.train_idx):
                cls, ms, ss = [], [], []
                for i in range(0, len(t), 32):
                    x = _from_u8(self._decode_rows(D, t[i:i + 32]))
                    cls.append(self.ev.feats(x)[0].cpu())
                    m, s = lab_stats(x)
                    ms.append(m); ss.append(s)
                cache["train_cls"].append(torch.cat(cls))
                cache["lab_mean"].append(np.concatenate(ms).mean(0))
                cache["lab_std"].append(np.concatenate(ss).mean(0))
            torch.save(cache, cache_p)
        self.train_cls = [c.to(self.dev) for c in cache["train_cls"]]
        self.lab = list(zip(cache["lab_mean"], cache["lab_std"]))
        # 背图检查的参照：held-out 真实目标域帧到 train 的最近邻相似度
        self.nn_ref = [nn_similarity(self.val_cls[b], self.train_cls[b]).mean().item() for b in (0, 1)]
        self.ready = True

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _translate(self, net, z, direction, nfe, bs=16):
        return torch.cat([translate(net, z[i:i + bs], self.coeff, nfe, direction) for i in range(0, len(z), bs)])

    def _decode(self, z, bs=8):
        return torch.cat([_to_u8(decode(self.ae, z[i:i + bs])) for i in range(0, len(z), bs)])

    def _metrics(self, x, fx, y_u8, b):
        """x：输入（[-1,1] float）；fx：输入的 DINO 特征 (cls, patch)，同一批输入只算一次。"""
        y = _from_u8(y_u8)
        cy, py = self.ev.feats(y)
        return {
            "kid": kid(cy, self.val_cls[b])[0] * 1e3,                 # ×1e3 便于阅读
            "lpips": self.ev.lpips_dist(x, y).mean().item(),
            "struct": struct_dist(fx[1], py).mean().item() * 1e3,
            "tone_rho": tone_collapse_ratio(self.ev, x, y).median().item(),
            "nn_sim": nn_similarity(cy, self.train_cls[b]).mean().item(),
        }

    def _baselines(self, x, fx, b):
        cr = self.ev.feats(reinhard(x, *self.lab[b]))[0]
        return {"kid_identity": kid(fx[0], self.val_cls[b])[0] * 1e3, "kid_reinhard": kid(cr, self.val_cls[b])[0] * 1e3}

    def _consistency(self, net, z, direction, nfe):
        """§5.4：同一帧整帧推理 vs 切成 16×16 窗口逐块推理，比较窗口内部区域（去掉 2 格边）。"""
        full = self._translate(net, z, direction, nfe)
        full_img = self._decode(full)
        rel, tone, psnr = [], [], []
        for oy in (0, z.shape[2] - 16):
            for ox in (0, z.shape[3] - 16):
                zw = z[:, :, oy:oy + 16, ox:ox + 16]
                ow = self._translate(net, zw, direction, nfe)
                a = ow[:, :, 2:14, 2:14]
                f = full[:, :, oy + 2:oy + 14, ox + 2:ox + 14]
                s = zw[:, :, 2:14, 2:14]
                rel.append(((a - f) ** 2).mean((1, 2, 3)) / ((f - s) ** 2).mean((1, 2, 3)).clamp_min(1e-8))
                wi = self._decode(ow)[:, :, 64:448, 64:448].float()
                fi = full_img[:, :, oy * 32 + 64:oy * 32 + 448, ox * 32 + 64:ox * 32 + 448].float()
                tone.append((wi.mean((2, 3)) - fi.mean((2, 3))).abs().mean(1))
                mse = ((wi - fi) ** 2).mean((1, 2, 3)).clamp_min(1e-6)
                psnr.append(10 * torch.log10(255 ** 2 / mse))
        return {"win_rel_diff": torch.cat(rel).mean().item(), "win_tone_shift": torch.cat(tone).mean().item(),
                "win_psnr": torch.cat(psnr).mean().item()}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, net, coeff, step, full=False, writer=None, nfe=32):
        self.setup()
        self.coeff = coeff
        out_dir = self.run / "samples"
        out_dir.mkdir(exist_ok=True)
        tag = "full" if full else "panel"
        res = {}
        for name, a, b in DIRS:
            idx = self.full_idx[a] if full else self.quick_idx[a]
            pos = np.searchsorted(self.val_idx[a], idx)              # 在 val_img 里的位置
            x_u8 = self.val_img[a][pos]
            x = _from_u8(x_u8)
            fx = self.ev.feats(x)
            z = self.Dv[a].lat[idx].float()
            zt = self._translate(net, z, name, nfe)
            y_u8 = self._decode(zt)
            m = self._metrics(x, fx, y_u8, b)
            m.update(self._baselines(x, fx, b))
            m["nn_sim_ref"] = self.nn_ref[b]
            n_show = min(32, len(x_u8))
            grid(x_u8[:n_show], y_u8[:n_show]).save(out_dir / f"{tag}_{step:07d}_{name}.jpg", quality=90)
            if writer is not None:
                g = grid(x_u8[:8], y_u8[:8], cols=2, w=256, h=144)
                writer.add_image(f"{tag}/{name}", np.asarray(g), step, dataformats="HWC")
            if full:
                # NFE–质量曲线（§7.3、§9.4）：前 32 张，只算 KID 和结构；同一批帧的 NFE=nfe 结果作参照
                n_c = min(32, len(z))
                m[f"nfe{nfe}_kid_sub"] = kid(self.ev.feats(_from_u8(y_u8[:n_c]))[0], self.val_cls[b])[0] * 1e3
                for k in (4, 8, 16):
                    yk = self._decode(self._translate(net, z[:n_c], name, k))
                    cyk, pyk = self.ev.feats(_from_u8(yk))
                    mse = ((yk.float() - y_u8[:n_c].float()) ** 2).mean((1, 2, 3)).clamp_min(1e-6)
                    m[f"nfe{k}_kid_sub"] = kid(cyk, self.val_cls[b])[0] * 1e3
                    m[f"nfe{k}_struct"] = struct_dist(fx[1][:n_c], pyk).mean().item() * 1e3
                    m[f"nfe{k}_psnr_vs_nfe{nfe}"] = (10 * torch.log10(255 ** 2 / mse)).mean().item()
                m.update(self._consistency(net, z[:16], name, nfe))
            res[name] = m
            if writer is not None:
                for k, v in m.items():
                    writer.add_scalar(f"{tag}_{name}/{k}", v, step)
        with open(self.run / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, "kind": tag, "nfe": nfe, **res}, ensure_ascii=False) + "\n")
        return res
