"""
训练薛定谔桥流 teacher（doc/sbflow-genshin-wuwa-plan.md §5），带评估面板（§7）和断点续训（§5.7）。

用法（仓库根目录，先激活 .venv）：
    python train.py --run smoke --preset smoke                          # 冒烟测试，约 10 分钟
    python train.py --run p_e0.05_rgb --preset pilot --eps 0.05         # pilot：DiT-S，16×16 窗口，2.5 万步
    python train.py --run main --preset main                            # main：DiT-B，15 万步
    python train.py --run p_e0.05_rgb --eval-only                       # 用最新 checkpoint 跑一次完整面板
    tensorboard --logdir runs                                            # 看曲线和对照图

同名 run 再次启动会无条件从 runs/<run>/ckpt/latest.pt 续训。决定训练目标的超参（模型、ε、配对、
batch、lr……）以 runs/<run>/config.json 为准，续训时命令行改不了；步数和日志 / 面板频率可以改。
Ctrl+C 会先存 latest.pt 再退出。
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sbflow  # noqa: F401
from sbflow.coeffs import TAU_MAX, TAU_MIN, calibrate_coeffs, load_coeff, lookup, save_coeff, self_check
from sbflow.model import DiT, DiTConfig, PRESETS as MODEL_PRESETS, count_params
from sbflow.pairing import GAMES, load_domain, sample_pairs

ROOT = Path(__file__).resolve().parent

# 决定训练目标的超参：建 run 时写进 config.json，续训时不可改
FROZEN = dict(model="S", eps=0.05, cost="rgb", ot_ratio=0.8, pool_mult=1, win=16, bs=128, lr=1e-4, warmup=2000,
              wd=0.0, ema=0.9999, full_every=0, full_bs=64, seed=0, rope_theta=10000.0, qk_norm=True,
              n_cal_batches=32, packed="data/30_packed")
# 运行期参数：续训时命令行可以改
OPERATIONAL = dict(steps=25000, log_every=50, val_every=1000, panel_every=2500, full_panel_every=12500,
                   ckpt_every=1000, archive_every=5000, panel_n=32, nfe=32, max_minutes=0.0)
PRESETS = {
    "smoke": dict(model="S", steps=1500, warmup=200, log_every=25, val_every=250, panel_every=500,
                  full_panel_every=1500, ckpt_every=250, archive_every=1000, panel_n=16),
    "pilot": dict(model="S", steps=25000, warmup=2000, log_every=50, val_every=1000, panel_every=2500,
                  full_panel_every=12500, ckpt_every=1000, archive_every=5000, panel_n=32),
    "main": dict(model="B", steps=150000, warmup=2000, log_every=50, val_every=1000, panel_every=5000,
                 full_panel_every=20000, ckpt_every=1000, archive_every=20000, panel_n=32),
}


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="runs/<run> 目录名")
    ap.add_argument("--preset", choices=list(PRESETS), default=None)
    ap.add_argument("--eval-only", action="store_true", help="加载最新 checkpoint，跑一次完整面板后退出")
    ap.add_argument("--bench", type=int, default=0, help="只跑 N 步测速（不存 checkpoint、不写日志）")
    for k, v in {**FROZEN, **OPERATIONAL}.items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            ap.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"), default=None)
        else:
            ap.add_argument(flag, type=type(v), default=None)
    return ap.parse_args()


def resolve_config(args, run: Path) -> dict:
    cli = {k: getattr(args, k) for k in {**FROZEN, **OPERATIONAL} if getattr(args, k) is not None}
    preset = PRESETS.get(args.preset or "", {})
    path = run / "config.json"
    if path.exists():
        cfg = json.loads(path.read_text(encoding="utf-8"))
        for k, v in cli.items():
            old = cfg.get("model_cfg_name") if k == "model" else cfg.get(k)
            if k in FROZEN and old != v:
                print(f"! 续训：{k} 以 config.json 为准（{old}），忽略命令行的 {v}")
            elif k in OPERATIONAL:
                cfg[k] = v
        for k, v in OPERATIONAL.items():
            cfg.setdefault(k, v)
    else:
        cfg = {**FROZEN, **OPERATIONAL, **preset, **cli}
        cfg["model_cfg_name"] = cfg["model"]
        cfg["model"] = {**MODEL_PRESETS[cfg["model"]].to_dict(), "rope_theta": cfg["rope_theta"],
                        "qk_norm": cfg["qk_norm"]}
        cfg["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    return cfg


def write_config(cfg, run: Path):
    tmp = run / "config.json.tmp"
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, run / "config.json")


# ---------------------------------------------------------------------------
# 训练目标（§5.3）
# ---------------------------------------------------------------------------
def bridge_loss(net, z0, z1, coeff, eps, tau=None, d=None, noise=None):
    """布朗桥插值 + 两个速度场 + EDM 重参数化。返回逐样本 loss、回归目标、τ。"""
    B, dev = z0.shape[0], z0.device
    if tau is None:
        tau = torch.rand(B, device=dev)
    tau = tau.clamp(TAU_MIN, TAU_MAX)
    if d is None:
        d = (torch.rand(B, device=dev) < 0.5).float()       # 每个样本随机一个方向
    t = torch.where(d.bool(), tau, 1 - tau).view(-1, 1, 1, 1)   # 物理时间：d=1 用 τ，d=0 用 1−τ
    z = torch.randn_like(z0) if noise is None else noise
    xt = (1 - t) * z0 + t * z1 + (eps * t * (1 - t)).sqrt() * z
    v = torch.where(d.view(-1, 1, 1, 1).bool(), (z1 - xt) / (1 - t), (z0 - xt) / t)
    c_in, c_skip, c_out = lookup(coeff, tau, d)
    target = (v - c_skip * xt) / c_out
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = net(c_in * xt, tau, d)
    per = (pred.float() - target).pow(2).mean((1, 2, 3))
    return per, target, tau


def tau_bucket(tau, n=8):
    return (tau * n).long().clamp(0, n - 1)


@torch.no_grad()
def ema_update(ema, net, decay):
    ep = [p for p in ema.parameters()]
    np_ = [p.detach() for p in net.parameters()]
    torch._foreach_mul_(ep, decay)
    torch._foreach_add_(ep, np_, alpha=1 - decay)


def build_fixed(D0, D1, cfg, seed, n_batches=8):
    """固定 τ 网格、固定噪声、固定配对的 loss 集（§7.1），只用来看 train / val 差距。"""
    g = torch.Generator().manual_seed(seed)
    B = cfg["bs"]
    out = []
    for _ in range(n_batches):
        z0, z1 = sample_pairs(D0, D1, B, cfg["win"], cfg["ot_ratio"], cfg["cost"], cfg["pool_mult"], gen=g)
        tau = ((torch.arange(B) + 0.5) / B).to(z0.device)
        d = (torch.arange(B) % 2).float().to(z0.device)
        noise = torch.randn(z0.shape, generator=g).to(z0.device)
        out.append((z0, z1, tau, d, noise))
    return out


@torch.no_grad()
def fixed_loss(net, batches, coeff, eps, nb=8):
    tot, cnt = torch.zeros(nb, device="cuda"), torch.zeros(nb, device="cuda")
    for z0, z1, tau, d, noise in batches:
        per, _, tau_c = bridge_loss(net, z0, z1, coeff, eps, tau, d, noise)
        b = tau_bucket(tau_c, nb)
        tot.index_add_(0, b, per); cnt.index_add_(0, b, torch.ones_like(per))
    per_bucket = tot / cnt.clamp_min(1)
    return (tot.sum() / cnt.sum()).item(), per_bucket.tolist()


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------
def save_ckpt(path: Path, net, ema, opt, step, full=True):
    obj = {"net": net.state_dict(), "ema": ema.state_dict(), "step": step}
    if full:
        obj.update(opt=opt.state_dict(), rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   py_rng=random.getstate(), np_rng=np.random.get_state())
    tmp = path.with_suffix(".pt.tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)                     # 原子替换，断电也不会写坏


def load_ckpt(path: Path, net, ema, opt):
    sd = torch.load(path, map_location="cpu", weights_only=False)   # RNG 状态必须留在 CPU
    net.load_state_dict(sd["net"]); ema.load_state_dict(sd["ema"])
    if opt is not None and "opt" in sd:
        opt.load_state_dict(sd["opt"])
        torch.set_rng_state(sd["rng"]); torch.cuda.set_rng_state_all(sd["cuda_rng"])
        random.setstate(sd["py_rng"]); np.random.set_state(sd["np_rng"])
    return sd["step"]


# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    sys.stdout.reconfigure(errors="replace", line_buffering=True)
    run = ROOT / "runs" / args.run
    (run / "ckpt").mkdir(parents=True, exist_ok=True)
    cfg = resolve_config(args, run)
    if not args.bench:
        write_config(cfg, run)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])

    packed = ROOT / cfg["packed"]
    use_dino = cfg["cost"] == "dino"
    Dtr = [load_domain(packed, g, "train", dino=use_dino) for g in GAMES]
    Dva = [load_domain(packed, g, "val", dino=use_dino) for g in GAMES]
    if (packed / "overlay_mask.png").exists() and not (run / "overlay_mask.png").exists():
        shutil.copy(packed / "overlay_mask.png", run / "overlay_mask.png")
    print(f"run {run.name}: train {len(Dtr[0])}/{len(Dtr[1])} 行，val {len(Dva[0])}/{len(Dva[1])} 行"
          f"（原神/鸣潮，含翻转）")

    # EDM 系数：用与训练完全相同的现裁 + 配对流程标定（§2.5），随 run 保存
    cpath = run / "coeff.pt"
    if not cpath.exists():
        g = torch.Generator().manual_seed(cfg["seed"] + 1)
        pairs = [sample_pairs(Dtr[0], Dtr[1], cfg["bs"], cfg["win"], cfg["ot_ratio"], cfg["cost"],
                              cfg["pool_mult"], gen=g) for _ in range(cfg["n_cal_batches"])]
        z0 = torch.cat([p[0] for p in pairs]); z1 = torch.cat([p[1] for p in pairs])
        rho = F.cosine_similarity((z0 - z0.mean()).flatten(1), (z1 - z1.mean()).flatten(1)).mean().item()
        taus, tables = calibrate_coeffs(z0, z1, cfg["eps"])
        chk = self_check()
        save_coeff(cpath, taus, tables, eps=cfg["eps"], cost=cfg["cost"], ot_ratio=cfg["ot_ratio"],
                   win=cfg["win"], pool_mult=cfg["pool_mult"], n_pairs=len(z0), self_check=chk,
                   stats={"var_z0": z0.var().item(), "var_z1": z1.var().item(), "mean_z0": z0.mean().item(),
                          "mean_z1": z1.mean().item(), "pair_corr": rho})
        print(f"标定 EDM 系数：{len(z0)} 对，var z0 {z0.var():.3f} / z1 {z1.var():.3f}，配对相关 ρ≈{rho:.3f}；"
              f"自检 ε=0 τ=0 {chk[0.0]['tau0']} τ=1 {chk[0.0]['tau1']}")
        del pairs, z0, z1
    taus, tables, cmeta = load_coeff(cpath)
    coeff = (taus, tables)

    net = DiT(DiTConfig(**cfg["model"])).cuda()
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=cfg["wd"], fused=True)
    print(f"DiT-{cfg.get('model_cfg_name', '?')}: {count_params(net) / 1e6:.1f}M 参数；ε={cfg['eps']} cost={cfg['cost']} "
          f"ot_ratio={cfg['ot_ratio']} win={cfg['win']} bs={cfg['bs']}")

    step = 0
    latest = run / "ckpt" / "latest.pt"
    if latest.exists() and not args.bench:
        step = load_ckpt(latest, net, ema, opt)
        print(f"从 {latest} 续训，step {step}")

    from sbflow.panel import Panel
    panel = Panel(run, packed, Dva, Dtr, cfg)
    if args.eval_only:
        from torch.utils.tensorboard import SummaryWriter
        res = panel.evaluate(ema, coeff, step, full=True, writer=SummaryWriter(run / "tb"), nfe=cfg["nfe"])
        print(json.dumps(res, ensure_ascii=False, indent=1))
        return

    writer = None
    if not args.bench:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(run / "tb", purge_step=step if step > 0 else None)
        writer.add_text("config", "```\n" + json.dumps(cfg, ensure_ascii=False, indent=1) + "\n```", step)
    fixed_tr = build_fixed(Dtr[0], Dtr[1], cfg, seed=1234)
    fixed_va = build_fixed(Dva[0], Dva[1], cfg, seed=4321)

    nb = 8
    acc = {"loss": 0.0, "n": 0, "bl": torch.zeros(nb, device="cuda"), "bf2": torch.zeros(nb, device="cuda"),
           "bn": torch.zeros(nb, device="cuda"), "gn": 0.0, "ot": [], "rand": []}
    t_start, t_log = time.time(), time.time()
    total = args.bench or cfg["steps"]
    print(f"开始训练：step {step} → {total}")
    try:
        while step < total:
            lr = cfg["lr"] * min(1.0, (step + 1) / max(1, cfg["warmup"]))
            for gr in opt.param_groups:
                gr["lr"] = lr
            full_bucket = cfg["full_every"] > 0 and step % cfg["full_every"] == cfg["full_every"] - 1
            want_info = step % cfg["log_every"] == 0
            z0, z1, info = sample_pairs(Dtr[0], Dtr[1], cfg["full_bs"] if full_bucket else cfg["bs"],
                                        "full" if full_bucket else cfg["win"], cfg["ot_ratio"], cfg["cost"],
                                        cfg["pool_mult"], return_info=True)
            per, target, tau = bridge_loss(net, z0, z1, coeff, cfg["eps"])
            loss = per.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ema_update(ema, net, min(cfg["ema"], (1 + step) / (10 + step)))
            step += 1

            # 累计日志量（不同步 GPU）
            b = tau_bucket(tau, nb)
            acc["bl"].index_add_(0, b, per.detach())
            acc["bf2"].index_add_(0, b, target.pow(2).mean((1, 2, 3)))
            acc["bn"].index_add_(0, b, torch.ones_like(per))
            acc["loss"] += loss.detach(); acc["gn"] += gn.detach(); acc["n"] += 1
            if want_info and "cost_ot" in info:
                acc["ot"].append(info["cost_ot"]); acc["rand"].append(info["cost_rand"])

            if args.bench:
                if step == 10:
                    torch.cuda.synchronize(); t_b = time.time()
                continue
            if step % cfg["log_every"] == 0:
                dt = (time.time() - t_log) / acc["n"]
                l, gnm = (acc["loss"] / acc["n"]).item(), (acc["gn"] / acc["n"]).item()
                writer.add_scalar("train/loss", l, step)
                writer.add_scalar("train/grad_norm", gnm, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/sec_per_step", dt, step)
                bl = (acc["bl"] / acc["bn"].clamp_min(1)).tolist()
                bf2 = (acc["bf2"] / acc["bn"].clamp_min(1)).tolist()
                for i in range(nb):
                    writer.add_scalar(f"loss_by_tau/b{i}", bl[i], step)
                    writer.add_scalar(f"calib_Ftarget_m2/b{i}", bf2[i], step)   # 应接近 1（§2.5 自检）
                if acc["ot"]:
                    writer.add_scalar("pair/cost_ot", np.mean(acc["ot"]), step)
                    writer.add_scalar("pair/cost_random", np.mean(acc["rand"]), step)
                eta = (total - step) * dt / 3600
                print(f"step {step:>7}  loss {l:.4f}  gn {gnm:.3f}  lr {lr:.2e}  {dt * 1000:.0f} ms/step  "
                      f"F²[{min(bf2):.2f},{max(bf2):.2f}]  ETA {eta:.1f} h")
                acc = {"loss": 0.0, "n": 0, "bl": torch.zeros(nb, device="cuda"), "bf2": torch.zeros(nb, device="cuda"),
                       "bn": torch.zeros(nb, device="cuda"), "gn": 0.0, "ot": [], "rand": []}
                t_log = time.time()
            if step % cfg["val_every"] == 0 or step == total:
                ltr, _ = fixed_loss(ema, fixed_tr, coeff, cfg["eps"])
                lva, lvb = fixed_loss(ema, fixed_va, coeff, cfg["eps"])
                writer.add_scalar("fixed/train_loss", ltr, step)
                writer.add_scalar("fixed/val_loss", lva, step)
                writer.add_scalar("fixed/val_minus_train", lva - ltr, step)
                for i, v in enumerate(lvb):
                    writer.add_scalar(f"fixed_val_by_tau/b{i}", v, step)
                print(f"  [fixed] train {ltr:.4f}  val {lva:.4f}  gap {lva - ltr:+.4f}")
            if step % cfg["ckpt_every"] == 0 or step == total:
                save_ckpt(latest, net, ema, opt, step)
            if cfg["archive_every"] and step % cfg["archive_every"] == 0:
                save_ckpt(run / "ckpt" / f"step_{step:07d}.pt", net, ema, opt, step, full=False)
            is_full = step % cfg["full_panel_every"] == 0 or step == total
            if step % cfg["panel_every"] == 0 or is_full:
                t0 = time.time()
                res = panel.evaluate(ema, coeff, step, full=is_full, writer=writer, nfe=cfg["nfe"])
                for name, m in res.items():
                    print(f"  [{'full' if is_full else 'panel'} {name}] KID×1e3 {m['kid']:.2f}（identity "
                          f"{m['kid_identity']:.2f} / reinhard {m['kid_reinhard']:.2f}）  struct×1e3 {m['struct']:.2f}  "
                          f"LPIPS {m['lpips']:.3f}  ρ {m['tone_rho']:.2f}  NN {m['nn_sim']:.3f}（参照 {m['nn_sim_ref']:.3f}）")
                print(f"  面板耗时 {time.time() - t0:.0f}s → {run / 'samples'}；"
                      f"至今峰值显存 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
                t_log += time.time() - t0
            if cfg["max_minutes"] and time.time() - t_start > cfg["max_minutes"] * 60:
                print("达到 max_minutes，保存并退出")
                save_ckpt(latest, net, ema, opt, step)
                break
    except KeyboardInterrupt:
        print(f"\n中断：保存 step {step} 到 {latest}")
        save_ckpt(latest, net, ema, opt, step)
    if args.bench:
        torch.cuda.synchronize()
        n = total - 10
        print(f"bench: {(time.time() - t_b) / n * 1000:.1f} ms/step（{n} 步平均），"
              f"峰值显存 {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
