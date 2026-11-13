"""
被动截屏采集脚本（doc/sbflow-genshin-wuwa-plan.md §3.2）。

安全边界：只做被动屏幕捕获（与 OBS/Discord 相同的 GDI BitBlt 截屏），
不注入进程、不读写游戏内存、不发送任何键鼠输入。热键只“读取”按键状态。

用法：
    python capture.py --list-windows                      # 列出可见窗口，确认游戏窗口标题
    python capture.py --game genshin --probe              # 抓 1 帧检查（黑屏/分辨率/亮度）
    python capture.py --game genshin --session 01         # 正式采集
    python capture.py --game wuwa    --session 01 --target 1200

控制：
    游戏窗口失焦 / 最小化 → 自动暂停（切出去看菜单、聊天都不会污染数据）
    F9  手动暂停/继续（进菜单、过场、战斗、传送时用）
    F10 结束本次会话（也可在控制台 Ctrl+C）
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import queue
import re
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path

import cv2
import imagehash
import mss
import numpy as np
import win32gui
import win32process
import psutil
from PIL import Image

MSS = getattr(mss, "MSS", None) or mss.mss

# ============================================================================
# 可配置阈值（两域必须用同一套；离线阶段 plan §3.4 会按分布再复核）
# ============================================================================
TARGET_FPS = 1.5               # 采样频率，1–2 帧/秒
LUMA_MIN, LUMA_MAX = 60, 230   # 平均亮度（0–255）范围，之外丢弃（黑场/过曝/夜景/洞穴）
SHARP_MIN = 60.0               # Laplacian 方差下限（在 960x540 灰度图上算），低于视为模糊
CORNER_EDGE_MAX = 0.20         # 四角任一区域 Canny 边缘像素占比上限（UI 残留检测）
CORNER_FRAC = 0.12             # 角区域边长 = 画面短边 * CORNER_FRAC
PHASH_WINDOW = 300             # 滑动窗口去重：只和最近 N 张已保存帧比较
PHASH_MIN_DIST = 6             # 汉明距离 < 此值视为近重复
EXPECT_SIZE = (1920, 1080)     # 期望客户区分辨率（不一致时只警告）
PNG_COMPRESS = 1               # PNG 压缩等级（1=快，无损）

GAMES = {
    # 窗口标题正则 + 进程名（小写），二者满足其一即视为匹配
    "genshin": {
        "title": r"^(原神|Genshin Impact)$",
        "exe": {"yuanshen.exe", "genshinimpact.exe"},
    },
    "wuwa": {
        "title": r"^(鸣潮|鳴潮|Wuthering Waves)",
        "exe": {"client-win64-shipping.exe", "wuthering waves.exe"},
    },
}

VK_F9, VK_F10 = 0x78, 0x79
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "10_frames"


# ============================================================================
# 窗口相关
# ============================================================================
def set_dpi_aware():
    """让坐标是物理像素，否则 125%/150% 缩放下截取区域会错位。"""
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def window_exe(hwnd) -> str:
    try:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        return psutil.Process(pid).name().lower()
    except Exception:
        return "?"


def list_windows():
    out = []

    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.strip():
                out.append((hwnd, title, window_exe(hwnd)))
    win32gui.EnumWindows(cb, None)
    return out


def find_game_window(game: str, title_override: str | None):
    spec = GAMES[game]
    pat = re.compile(title_override or spec["title"])
    cands = []
    for hwnd, title, exe in list_windows():
        if pat.search(title) or exe in spec["exe"]:
            cands.append((hwnd, title, exe))
    if not cands:
        return None
    # 优先“标题+进程都匹配”的，其次客户区最大的（排除启动器小窗）
    def score(c):
        hwnd, title, exe = c
        _, _, w, h = win32gui.GetClientRect(hwnd)
        return (bool(pat.search(title)) and exe in spec["exe"], w * h)
    cands.sort(key=score, reverse=True)
    return cands[0]


def client_rect_on_screen(hwnd):
    """返回客户区（不含标题栏/边框）在屏幕上的 (left, top, width, height)。"""
    _, _, w, h = win32gui.GetClientRect(hwnd)
    left, top = win32gui.ClientToScreen(hwnd, (0, 0))
    return left, top, w, h


def key_pressed(vk) -> bool:
    """只读查询按键状态（GetAsyncKeyState），不安装钩子、不发送输入。"""
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x0001)


# ============================================================================
# 过滤
# ============================================================================
def frame_stats(rgb: np.ndarray):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    luma = float(gray.mean())
    small = cv2.resize(gray, (960, 540), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(small, cv2.CV_64F).var())
    # 四角边缘密度：UI（小地图、任务栏、按钮）是高对比硬边
    h, w = small.shape
    c = int(min(h, w) * CORNER_FRAC)
    edges = cv2.Canny(small, 100, 200)
    corners = [edges[:c, :c], edges[:c, -c:], edges[-c:, :c], edges[-c:, -c:]]
    corner = max(float((e > 0).mean()) for e in corners)
    return luma, sharp, corner


class Deduper:
    def __init__(self, window=PHASH_WINDOW, min_dist=PHASH_MIN_DIST):
        self.recent = deque(maxlen=window)
        self.min_dist = min_dist

    def check(self, img: Image.Image):
        h = imagehash.phash(img.resize((256, 144)))
        dmin = min((h - r for r in self.recent), default=64)
        return dmin >= self.min_dist, h, dmin

    def add(self, h):
        self.recent.append(h)


# ============================================================================
# 异步写盘（每帧独立文件，崩溃不丢已写帧）
# ============================================================================
class Writer(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q: queue.Queue = queue.Queue(maxsize=16)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                break
            path, img = item
            tmp = path.with_suffix(".tmp.png")
            img.save(tmp, compress_level=PNG_COMPRESS)
            tmp.replace(path)

    def put(self, path, img):
        self.q.put((path, img))

    def close(self):
        self.q.put(None)
        self.join()


# ============================================================================
# 主流程
# ============================================================================
def grab(sct, hwnd) -> np.ndarray:
    left, top, w, h = client_rect_on_screen(hwnd)
    shot = sct.grab({"left": left, "top": top, "width": w, "height": h})
    return np.asarray(shot)[:, :, 2::-1].copy()   # BGRA -> RGB


def probe(args, hwnd, title):
    with MSS() as sct:
        rgb = grab(sct, hwnd)
    luma, sharp, corner = frame_stats(rgb)
    out = ROOT / "data" / "probe"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.game}_probe_{datetime.now():%H%M%S}.png"
    Image.fromarray(rgb).save(path)
    h, w = rgb.shape[:2]
    print(f"窗口: {title!r}  客户区: {w}x{h}")
    print(f"亮度={luma:.1f}  清晰度={sharp:.1f}  角落边缘={corner:.3f}")
    print(f"已保存: {path}")
    ok = True
    if luma < 5 and rgb.std() < 3:
        print("✗ 抓到的是黑屏 → 游戏很可能是独占全屏，请改为【无边框窗口】后重试。")
        ok = False
    if (w, h) != EXPECT_SIZE:
        print(f"! 客户区不是 {EXPECT_SIZE[0]}x{EXPECT_SIZE[1]}，两边游戏请保持一致。")
    if not (LUMA_MIN <= luma <= LUMA_MAX):
        print("! 亮度不在采集范围内（这一帧会被丢弃），检查是否夜间/室内/菜单。")
    if sharp < SHARP_MIN:
        print("! 清晰度低于阈值（站定不动时也低的话，说明 SHARP_MIN 对这款游戏偏高）。")
    if corner > CORNER_EDGE_MAX:
        print("! 角落边缘密度偏高，可能 UI 没隐藏干净（或该处恰好是密集植被）。")
    print("✓ 截屏正常" if ok else "")


def run(args, hwnd, title):
    sess_dir = DATA_DIR / args.game / f"s{args.session}"
    sess_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(sess_dir.glob("*.png")))
    if existing:
        print(f"! {sess_dir} 已有 {existing} 张，本次会追加。")

    log_f = open(sess_dir / "frames.csv", "a", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    if log_f.tell() == 0:
        log.writerow(["time", "result", "luma", "sharp", "corner", "phash_dmin", "file"])

    dedup = Deduper()
    # 用本会话已有帧预热去重窗口（断点续采时不重复）
    for p in sorted(sess_dir.glob("*.png"))[-PHASH_WINDOW:]:
        try:
            dedup.add(imagehash.phash(Image.open(p).convert("RGB").resize((256, 144))))
        except Exception:
            pass

    writer = Writer()
    writer.start()
    cnt = Counter()
    saved = existing
    paused_manual = False
    t_start = time.time()
    period = 1.0 / args.fps
    key_pressed(VK_F9); key_pressed(VK_F10)   # 清掉历史按键位

    print(f"采集 {args.game} 会话 s{args.session} → {sess_dir}")
    print("切到游戏开始走路。失焦自动暂停；F9 暂停/继续；F10 或 Ctrl+C 结束。\n")
    last_status = ""
    try:
        with MSS() as sct:
            while True:
                t0 = time.time()
                if key_pressed(VK_F10):
                    break
                if key_pressed(VK_F9):
                    paused_manual = not paused_manual
                if not win32gui.IsWindow(hwnd):
                    print("\n游戏窗口已关闭。")
                    break

                focused = win32gui.GetForegroundWindow() == hwnd
                minimized = win32gui.IsIconic(hwnd)
                if paused_manual or not focused or minimized:
                    state = "手动暂停" if paused_manual else "失焦暂停"
                else:
                    state = "采集中"
                    rgb = grab(sct, hwnd)
                    cnt["grabbed"] += 1
                    luma, sharp, corner = frame_stats(rgb)
                    result, dmin, fname = "", "", ""
                    if not (LUMA_MIN <= luma <= LUMA_MAX):
                        result = "luma"
                    elif sharp < SHARP_MIN:
                        result = "blur"
                    elif corner > CORNER_EDGE_MAX:
                        result = "ui"
                    else:
                        img = Image.fromarray(rgb)
                        ok, h, dmin = dedup.check(img)
                        if not ok:
                            result = "dup"
                        else:
                            result = "saved"
                            dedup.add(h)
                            fname = (f"{args.game}_s{args.session}_"
                                     f"{datetime.now():%Y%m%d_%H%M%S_%f}"[:-3] + ".png")
                            writer.put(sess_dir / fname, img)
                            saved += 1
                    cnt[result] += 1
                    log.writerow([f"{t0:.3f}", result, f"{luma:.1f}", f"{sharp:.1f}",
                                  f"{corner:.4f}", dmin, fname])

                el = int(time.time() - t_start)
                status = (f"\r[{state}] {el // 60:02d}:{el % 60:02d}  有效 {saved}"
                          f"{'/' + str(args.target) if args.target else ''}  | 抓 {cnt['grabbed']}"
                          f"  丢: 亮度 {cnt['luma']} 模糊 {cnt['blur']} UI {cnt['ui']} 重复 {cnt['dup']}   ")
                if status != last_status:
                    sys.stdout.write(status); sys.stdout.flush()
                    last_status = status

                if args.target and saved >= args.target:
                    print("\n已达到目标数量。")
                    break
                if args.minutes and el >= args.minutes * 60:
                    print("\n已达到时长上限。")
                    break
                time.sleep(max(0.0, period - (time.time() - t0)) if state == "采集中" else 0.1)
    except KeyboardInterrupt:
        pass
    finally:
        writer.close()
        log_f.close()
        stats = {
            "game": args.game, "session": args.session, "window": title,
            "started": datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
            "duration_s": round(time.time() - t_start, 1),
            "saved_total_in_session": saved, "counts": dict(cnt),
            "thresholds": {
                "fps": args.fps, "luma": [LUMA_MIN, LUMA_MAX], "sharp_min": SHARP_MIN,
                "corner_edge_max": CORNER_EDGE_MAX, "phash_window": PHASH_WINDOW,
                "phash_min_dist": PHASH_MIN_DIST,
            },
        }
        stats_path = sess_dir / f"session_stats_{datetime.now():%Y%m%d_%H%M%S}.json"
        stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结束。本会话有效帧 {saved}，统计已写入 {stats_path.name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--game", choices=list(GAMES))
    ap.add_argument("--session", default="01", help="会话编号，用于按 session 切分 train/val")
    ap.add_argument("--window", help="覆盖窗口标题正则")
    ap.add_argument("--fps", type=float, default=TARGET_FPS)
    ap.add_argument("--target", type=int, default=0, help="本会话有效帧达到此数自动结束")
    ap.add_argument("--minutes", type=float, default=0, help="时长上限（分钟）")
    ap.add_argument("--list-windows", action="store_true")
    ap.add_argument("--probe", action="store_true", help="抓一帧做检查后退出")
    args = ap.parse_args()

    set_dpi_aware()
    sys.stdout.reconfigure(errors="replace")
    if args.list_windows:
        for hwnd, title, exe in list_windows():
            _, _, w, h = win32gui.GetClientRect(hwnd)
            print(f"{hwnd:>10}  {w:>5}x{h:<5} {exe:<32} {title}")
        return
    if not args.game:
        ap.error("需要 --game")

    found = find_game_window(args.game, args.window)
    if not found:
        sys.exit(f"找不到 {args.game} 窗口。先启动游戏，或用 --list-windows 查标题后传 --window。")
    hwnd, title, exe = found
    print(f"找到窗口: {title!r} ({exe})")
    if args.probe:
        # 给 3 秒切到游戏，保证游戏画面在最前面
        for i in range(3, 0, -1):
            print(f"{i} 秒后截图，请切到游戏窗口…"); time.sleep(1)
        probe(args, hwnd, title)
    else:
        run(args, hwnd, title)


if __name__ == "__main__":
    main()
