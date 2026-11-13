"""
离线过滤与配平（doc/sbflow-genshin-wuwa-plan.md §3.4、§3.5）。

在线过滤只看得到局部，这里补全局视野的步骤（按顺序，先命中的原因记账）：
    1. dup      全量 pHash 去重（跨会话、跨滑动窗口），按时间顺序保留先出现的
    2. night    时段筛查：游戏内时间是采集时间的连续函数，所以按“整段时间”判，不逐帧判。
                会话目录下有 exclude.txt（人工判读的时段）就用它；否则用 CLIP 的
                P(夜/黄昏 | 时段) 沿时间轴取滑动中位数，超过阈值的整段排除，并提示人工复核
    3. content  CLIP zero-shot 逐帧筛查：对话框 / 战斗特效 / 室内 / 角色特写
    4. blur     清晰度复核：在剩下的帧里，两域各按自己分布的 q 分位（默认 3%）定阈值
    5. guard    验证集保护带（见下）
按会话切分 train / val（§3.5）：--val-sessions 里的会话作 val。某个域的 val 不足 --min-val 张时，
从该域最后一个 train 会话的时间末尾切一段作临时 val，并丢掉切点前 --guard-s 秒的帧，避免相邻帧泄漏。

另外从全部帧的逐像素时域方差里找出固定叠加层（UID 文字、隐藏 UI 的图标），两域取并集、膨胀后
存成 overlay_mask.png（规范分辨率）。打包和推理都用它修补叠加层，免得模型去学两边 UI 的差异。

逐帧特征增量缓存在 data/20_filtered/cache_<game>.npz：改阈值重跑只要几秒，新增会话只解码新帧。

用法：
    python filter_frames.py --clip
输出：data/20_filtered/{game}.csv（全部帧，带 keep / reason / split）、overlay_mask.png、report.txt、
      sheets/（随机抽检、各类被丢帧、每个会话的时间线）
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import imagehash
import numpy as np
import torch
from PIL import Image, ImageDraw

import sbflow  # noqa: F401
from sbflow.frames import CANON_H, CANON_W, to_canonical

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "10_frames"
OUT = ROOT / "data" / "20_filtered"
GAMES = ("genshin", "wuwa")

# 与 capture.py 相同的统计口径
CORNER_FRAC = 0.12
PHASH_MIN_DIST = 6

CLIP_MODEL = "openai/clip-vit-base-patch16"
PROMPTS = {   # 每类若干条提示词，取平均文本特征
    "day": ["a video game screenshot of an outdoor nature landscape in bright daylight",
            "a sunny daytime video game scene with grass, trees and a blue sky"],
    "night": ["a video game screenshot of a landscape at night",
              "a dark night scene in a video game with moonlight and stars"],
    "dusk": ["a video game screenshot at sunset with an orange and pink sky",
             "a dim dusk scene in a video game"],
    "ui": ["a video game screenshot with a dialog box, menu or large text on screen",
           "a video game screenshot with a popup window and subtitles"],
    "combat": ["a video game combat scene with a monster and glowing special effects",
               "a video game screenshot with explosions, beams and magic effects"],
    "indoor": ["an indoor scene inside a building in a video game",
               "a dark cave in a video game"],
    "closeup": ["a close-up of a video game character filling the screen"],
}
CONTENT = ("ui", "combat", "indoor", "closeup")
REASONS = ("dup", "night", "content", "blur", "guard")


# ---------------------------------------------------------------------------
# 逐帧特征
# ---------------------------------------------------------------------------
def frame_feats(path: Path):
    with Image.open(path) as im:
        im = im.convert("RGB")
        rgb = np.asarray(im)
        ph = imagehash.phash(im.resize((256, 144)))
        canon = np.asarray(to_canonical(im))
        clip_in = im.resize((398, 224), Image.BICUBIC)        # 16:9 → 短边 224，处理器再中心裁
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = float(gray.mean())
    small = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    h, w = small.shape
    c = int(min(h, w) * CORNER_FRAC)
    edges = cv2.Canny(small, 100, 200)
    corner = max(float((e > 0).mean()) for e in
                 (edges[:c, :c], edges[:c, -c:], edges[-c:, :c], edges[-c:, -c:]))
    return dict(luma=luma, sharp=sharp, corner=corner, phash=int(str(ph), 16)), canon, clip_in


def build_cache(game: str, files: list[Path], bs: int = 64):
    """解码全部帧：统计量、pHash、CLIP 图像特征，并累计规范分辨率下的逐像素均值 / 方差。"""
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    n = len(files)
    stats = {k: np.zeros(n, np.float64) for k in ("luma", "sharp", "corner")}
    phash = np.zeros(n, np.uint64)
    emb = np.zeros((n, model.config.projection_dim), np.float16)
    s1 = np.zeros((CANON_H, CANON_W, 3), np.float64)
    s2 = np.zeros_like(s1)
    pending, pend_idx = [], []

    def flush():
        if not pending:
            return
        with torch.no_grad():
            px = proc(images=pending, return_tensors="pt")["pixel_values"].cuda().half()
            f = model.get_image_features(pixel_values=px)
            f = f if torch.is_tensor(f) else f.pooler_output   # transformers 5 返回 ModelOutput，pooler_output 是投影后的特征
            f = torch.nn.functional.normalize(f.float(), dim=-1)
        emb[pend_idx] = f.cpu().numpy().astype(np.float16)
        pending.clear(); pend_idx.clear()

    with ThreadPoolExecutor(8) as ex:
        for i, (st, canon, clip_in) in enumerate(ex.map(frame_feats, files)):
            for k in stats:
                stats[k][i] = st[k]
            phash[i] = np.uint64(st["phash"])
            c = canon.astype(np.float64)
            s1 += c; s2 += c * c
            pending.append(clip_in); pend_idx.append(i)
            if len(pending) >= bs:
                flush()
            if (i + 1) % 500 == 0 or i + 1 == n:
                print(f"  {game}: {i + 1}/{n}", flush=True)
    flush()
    return dict(files=np.array([p.relative_to(ROOT).as_posix() for p in files]), phash=phash,
                emb=emb, s1=s1, s2=s2, **stats)


def update_cache(game: str, files: list[Path], cache_path: Path, rebuild=False):
    """增量缓存：已缓存的帧直接复用，只解码新增的帧；有帧被删掉时全量重建。"""
    rel = [p.relative_to(ROOT).as_posix() for p in files]
    old = None
    if cache_path.exists() and not rebuild:
        old = dict(np.load(cache_path, allow_pickle=False))
        if "s1" not in old or not set(map(str, old["files"])) <= set(rel):
            old = None
    have = set(map(str, old["files"])) if old is not None else set()
    new_files = [p for p, r in zip(files, rel) if r not in have]
    if old is not None and not new_files:
        c = old
    else:
        print(f"解码 {game}: 新增 {len(new_files)} 帧（已缓存 {len(have)}）")
        add = build_cache(game, new_files)
        if old is None:
            c = add
        else:
            c = {k: np.concatenate([old[k], add[k]]) for k in ("files", "phash", "emb", "luma", "sharp", "corner")}
            c["s1"], c["s2"] = old["s1"] + add["s1"], old["s2"] + add["s2"]
        np.savez(cache_path, **c)
    pos = {str(f): i for i, f in enumerate(c["files"])}
    order = np.array([pos[r] for r in rel])            # 按当前文件顺序排列
    out = {k: c[k][order] for k in ("files", "phash", "emb", "luma", "sharp", "corner")}
    n = len(rel)
    mean = c["s1"] / n
    out["std"] = np.sqrt(np.maximum(c["s2"] / n - mean ** 2, 0)).astype(np.float32)
    return out


_TEXT = None


def clip_probs(emb: np.ndarray) -> dict[str, np.ndarray]:
    """CLIP zero-shot：对全部类别做 softmax。"""
    global _TEXT
    if _TEXT is None:
        from transformers import CLIPModel, CLIPTokenizer
        model = CLIPModel.from_pretrained(CLIP_MODEL, dtype=torch.float32).eval()
        tok = CLIPTokenizer.from_pretrained(CLIP_MODEL)
        feats = []
        with torch.no_grad():
            for ps in PROMPTS.values():
                t = model.get_text_features(**tok(ps, padding=True, return_tensors="pt"))
                t = t if torch.is_tensor(t) else t.pooler_output
                feats.append(torch.nn.functional.normalize(torch.nn.functional.normalize(t, dim=-1).mean(0), dim=0))
        _TEXT = torch.stack(feats)
    p = (100.0 * torch.from_numpy(emb.astype(np.float32)) @ _TEXT.T).softmax(-1).numpy()
    return {k: p[:, i] for i, k in enumerate(PROMPTS)}


# ---------------------------------------------------------------------------
# 过滤步骤
# ---------------------------------------------------------------------------
def frame_time(f: str) -> tuple[str, int]:
    """文件名 {game}_sNN_YYYYMMDD_HHMMSS_mmm.png → (排序键, 当天秒数)。"""
    parts = Path(f).stem.split("_")
    date, hms, ms = parts[2], parts[3], parts[4]
    return f"{date}{hms}{ms}", int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:6])


def hms(s: int) -> str:
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def popcount64(x: np.ndarray) -> np.ndarray:
    b = x.astype(np.uint64).view(np.uint8).reshape(*x.shape, 8)
    return np.unpackbits(b, axis=-1).sum(-1)


def global_dedup(phash: np.ndarray, order: np.ndarray, min_dist: int):
    """按时间顺序贪心：和已保留帧的最小汉明距离 < min_dist 就丢。"""
    keep = np.zeros(len(phash), bool)
    dup_of = np.full(len(phash), -1)
    kept: list[int] = []
    for i in order:
        if kept:
            d = popcount64(phash[kept] ^ phash[i])
            j = int(d.argmin())
            if d[j] < min_dist:
                dup_of[i] = kept[j]
                continue
        keep[i] = True
        kept.append(i)
    return keep, dup_of


def read_exclude(path: Path) -> list[tuple[int, int, str]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        span, *why = line.split(maxsplit=1)
        a, b = span.split("-")
        to_s = lambda t: sum(int(x) * m for x, m in zip(t.split(":"), (3600, 60, 1)))
        out.append((to_s(a), to_s(b), why[0] if why else ""))
    return out


def rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    return np.array([np.median(x[max(0, i - w):i + w + 1]) for i in range(len(x))])


def overlay_mask(std_maps: list[np.ndarray], thr: float, dilate: int, min_area: int):
    """时域标准差很小的像素 = 每帧都一样的叠加层。两域取并集，去掉碎点，闭运算连成块，再膨胀。"""
    m = np.zeros((CANON_H, CANON_W), np.uint8)
    for s in std_maps:
        m |= (s.max(-1) < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for i in range(1, n):
        if st[i][4] >= min_area:
            out[lab == i] = 1
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    out = cv2.dilate(out, np.ones((2 * dilate + 1, 2 * dilate + 1), np.uint8))
    n, _, st, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    boxes = [tuple(int(v) for v in st[i][:4]) for i in range(1, n)]
    return (out * 255).astype(np.uint8), boxes


# ---------------------------------------------------------------------------
# 拼图
# ---------------------------------------------------------------------------
def contact_sheet(paths: list[str], captions: list[str], out: Path, cols=10, tw=192, th=108):
    if not paths:
        return
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    d = ImageDraw.Draw(sheet)

    def load(p):
        with Image.open(ROOT / p) as im:
            return im.convert("RGB").resize((tw, th), Image.BILINEAR)
    with ThreadPoolExecutor(8) as ex:
        thumbs = list(ex.map(load, paths))
    for i, (t, cap) in enumerate(zip(thumbs, captions)):
        x, y = (i % cols) * tw, (i // cols) * th
        sheet.paste(t, (x, y))
        if cap:
            d.rectangle([x, y, x + 6 * len(cap) + 4, y + 11], fill="black")
            d.text((x + 2, y), cap, fill="yellow")
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, quality=88)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", default=",".join(GAMES))
    ap.add_argument("--val-sessions", default="00,05", help="作验证集的会话号（§3.5）")
    ap.add_argument("--min-val", type=int, default=150, help="某域 val 少于此数时，从最后一个 train 会话末尾补")
    ap.add_argument("--val-tail", type=float, default=0.15, help="补 val 时至少切走该会话保留帧的这个比例")
    ap.add_argument("--guard-s", type=float, default=45, help="临时 val 切点前丢掉的秒数")
    ap.add_argument("--sharp-q", type=float, default=0.03,
                    help="清晰度按各域分布的这个分位定阈值。被动采集已在线去掉运动模糊，分位再高丢的就是天空多的开阔景")
    ap.add_argument("--night-thr", type=float, default=0.35, help="没有 exclude.txt 时：P(夜|时段) 滑动中位数阈值")
    ap.add_argument("--smooth", type=int, default=60, help="滑动中位数半窗（帧）")
    ap.add_argument("--content-max", type=float, default=0.5, help="P(ui/combat/indoor/closeup) 之和超过它就丢")
    ap.add_argument("--mask-std", type=float, default=25.0, help="叠加层检测：时域 std 低于它的像素")
    ap.add_argument("--mask-dilate", type=int, default=3)
    ap.add_argument("--mask-min-area", type=int, default=20)
    ap.add_argument("--rebuild", action="store_true", help="忽略缓存重新解码")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(errors="replace")
    OUT.mkdir(parents=True, exist_ok=True)
    val_sessions = {s.strip() for s in args.val_sessions.split(",") if s.strip()}

    report, std_maps = [], []
    for g in args.games.split(","):
        files_p = sorted(p for p in (SRC / g).glob("s*/*.png") if ".tmp" not in p.suffixes)
        c = update_cache(g, files_p, OUT / f"cache_{g}.npz", args.rebuild)
        std_maps.append(c["std"])

        files = [str(f) for f in c["files"]]
        n = len(files)
        sess = np.array([Path(f).parent.name[1:] for f in files])
        key_t = [frame_time(f) for f in files]
        tsec = np.array([t for _, t in key_t])
        order = np.argsort([k for k, _ in key_t], kind="stable")
        split = np.where(np.isin(sess, list(val_sessions)), "val", "train").astype(object)
        reason = np.array([""] * n, dtype=object)
        p = clip_probs(c["emb"])
        p_night = (p["night"] + p["dusk"]) / (p["day"] + p["night"] + p["dusk"])
        p_content = sum(p[k] for k in CONTENT)
        tag = np.array(list(PROMPTS))[np.stack([p[k] for k in PROMPTS]).argmax(0)]
        night_s = np.zeros(n)
        report.append(f"== {g}: {n} 帧")

        # 1. 全量 pHash 去重
        keep_d, dup_of = global_dedup(c["phash"], order, PHASH_MIN_DIST)
        reason[~keep_d] = "dup"

        # 2. 时段：按会话、按时间顺序
        for s in sorted(set(sess)):
            idx = order[sess[order] == s]
            night_s[idx] = rolling_median(p_night[idx], args.smooth)
            exc = SRC / g / f"s{s}" / "exclude.txt"
            if exc.exists():
                bad = np.zeros(len(idx), bool)
                for a, b, why in read_exclude(exc):
                    bad |= (tsec[idx] >= a) & (tsec[idx] <= b)
                how = f"人工 exclude.txt（{len(read_exclude(exc))} 段）"
            else:
                bad = night_s[idx] > args.night_thr
                how = f"自动：P(夜) 滑动中位数 > {args.night_thr}，请看 timeline 拼图复核"
            hit = idx[bad & (reason[idx] == "")]
            reason[hit] = "night"
            report.append(f"   s{s} 时段筛查 [{how}]：排除 {bad.sum()} 帧（{hms(tsec[idx].min())}–{hms(tsec[idx].max())}）")

        # 3. 逐帧内容
        reason[(reason == "") & (p_content > args.content_max)] = "content"

        # 4. 清晰度：剩下的帧里按本域 q 分位
        alive = reason == ""
        thr_sharp = float(np.quantile(c["sharp"][alive], args.sharp_q))
        reason[alive & (c["sharp"] < thr_sharp)] = "blur"

        # 5. 验证集：不足时从最后一个 train 会话末尾补，切点前留保护带
        keep = reason == ""
        n_val = int((keep & (split == "val")).sum())
        if n_val < args.min_val:
            train_sess = sorted(set(sess[split == "train"]))
            s = train_sess[-1]
            idx = order[(sess[order] == s) & keep[order]]        # 该会话保留帧，按时间
            n_tail = max(int(np.ceil(args.val_tail * len(idx))), args.min_val - n_val)
            tail = idx[-n_tail:]
            split[tail] = "val"
            t_cut = tsec[tail[0]]
            guard = idx[(tsec[idx] >= t_cut - args.guard_s) & (tsec[idx] < t_cut)]
            reason[guard] = "guard"
            report.append(f"   val 不足（{n_val} < {args.min_val}）：把 s{s} 从 {hms(t_cut)} 起的 {n_tail} 帧划为临时 val，"
                          f"切点前 {args.guard_s:.0f}s 的 {len(guard)} 帧作保护带丢弃")
        keep = reason == ""

        with open(OUT / f"{g}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "session", "split", "keep", "reason", "time", "luma", "sharp", "corner",
                        "phash", "dup_of", "p_night", "p_night_smooth", "p_content", "clip_top"])
            for i in order:
                w.writerow([files[i], sess[i], split[i], int(keep[i]), reason[i], hms(tsec[i]),
                            f"{c['luma'][i]:.1f}", f"{c['sharp'][i]:.1f}", f"{c['corner'][i]:.4f}",
                            f"{int(c['phash'][i]):016x}", files[dup_of[i]] if dup_of[i] >= 0 else "",
                            f"{p_night[i]:.3f}", f"{night_s[i]:.3f}", f"{p_content[i]:.3f}", tag[i]])

        report.append(f"   保留 {keep.sum()}（{keep.mean():.1%}）；清晰度阈值 {thr_sharp:.1f}（{args.sharp_q:.0%} 分位）")
        report.append("   丢弃：" + "，".join(f"{r} {int((reason == r).sum())}" for r in REASONS))
        for s in sorted(set(sess)):
            m = sess == s
            report.append(f"   s{s}: 原始 {m.sum():>5}  保留 train {int((keep & m & (split == 'train')).sum()):>5}"
                          f"  val {int((keep & m & (split == 'val')).sum()):>4}")
        report.append(f"   合计：train {int((keep & (split == 'train')).sum())}，val {int((keep & (split == 'val')).sum())}")

        # 拼图
        rng = random.Random(args.seed)
        kept_idx = list(np.flatnonzero(keep))
        pick = sorted(rng.sample(kept_idx, min(100, len(kept_idx))), key=lambda i: key_t[i][0])
        contact_sheet([files[i] for i in pick], [f"{split[i]} {tag[i]} c{p_content[i]:.2f}" for i in pick],
                      OUT / "sheets" / f"{g}_keep_random100.jpg")
        for r in REASONS:
            idx = np.flatnonzero(reason == r)
            if not len(idx):
                continue
            if r == "content":
                idx = idx[np.argsort(p_content[idx])]
            elif r == "blur":
                idx = idx[np.argsort(-c["sharp"][idx])]
            else:
                idx = np.array(sorted(rng.sample(list(idx), min(100, len(idx))), key=lambda i: key_t[i][0]))
            pick = idx[:100]
            contact_sheet([files[i] for i in pick],
                          [f"{hms(tsec[i])} {tag[i]} s{c['sharp'][i]:.0f}" for i in pick],
                          OUT / "sheets" / f"{g}_drop_{r}.jpg")
        for s in sorted(set(sess)):    # 每个会话的时间线：每 20 秒一帧
            idx = order[sess[order] == s]
            pick, last = [], -10 ** 9
            for i in idx:
                if tsec[i] - last >= 20:
                    pick.append(i); last = tsec[i]
            caps = [f"{hms(tsec[i])} n{night_s[i]:.2f} {'KEEP' if keep[i] else reason[i]}" for i in pick]
            contact_sheet([files[i] for i in pick], caps, OUT / "sheets" / f"timeline_{g}_s{s}.jpg")

    mask, boxes = overlay_mask(std_maps, args.mask_std, args.mask_dilate, args.mask_min_area)
    Image.fromarray(mask).save(OUT / "overlay_mask.png")
    report.append(f"== overlay mask（两域并集）：{int((mask > 0).sum())} 像素（{(mask > 0).mean():.3%}），"
                  f"外接框 (x,y,w,h)：{boxes}")
    vis = np.concatenate([np.clip(s.max(-1) * 4, 0, 255).astype(np.uint8) for s in std_maps], axis=0)
    Image.fromarray(vis).save(OUT / "sheets" / "temporal_std_x4.png")
    (OUT / "report.txt").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
