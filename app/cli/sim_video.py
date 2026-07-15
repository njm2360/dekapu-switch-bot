"""シミュレーションのフレームCSVを「一人称3D + 上面2D地図」の並置動画にする。

左は占有グリッドへのDDAレイキャスティング(擬似3D)、右は歩行可能グリッド上の
軌跡・現在位置・目標点。rawvideo を ffmpeg に直接パイプして mp4 を書く。

例:
    sim-video --csv frames.csv --map maps/XXXX/room.npz --out out.mp4
    sim-video --csv frames.csv --out traj_only.mp4 --speed 2
"""

import argparse
import csv
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from app.mapping.mapper import RoomMapper
from app.spatial.navigation import NavGrid

FOV_DEG = 90.0
MAX_DIST = 25.0

CEIL = np.array([46, 52, 64], np.uint8)
FLOOR = np.array([76, 70, 60], np.uint8)
WALL = np.array([170, 150, 120], np.float64)
TARGET_COLOR = (255, 80, 40)


# ---- CSV ---------------------------------------------------------------
def load_frames(path: Path) -> dict[str, np.ndarray]:
    """フレームCSVを列名→配列の辞書で読む(余分な列は無視)。"""
    need = ["t", "x", "z", "yaw", "pitch", "tx", "tz", "yaw_err", "turn", "fwd"]
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"empty csv: {path}")
    out = {k: np.array([float(r[k]) for r in rows]) for k in need if k in rows[0]}
    missing = [k for k in need if k not in out]
    if missing:
        raise SystemExit(f"csv missing columns: {missing}")
    return out


# ---- レイキャスティング --------------------------------------------------
def raycast(
    solid: np.ndarray,
    cell: float,
    xmin: float,
    zmin: float,
    px: float,
    pz: float,
    dirs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """DDA法。dirs(W,2)=(dx,dz) の各レイの (距離[m], side 0=X面/1=Z面) を返す。"""
    rows, cols = solid.shape
    gx = (px - xmin) / cell
    gz = (pz - zmin) / cell
    w = len(dirs)
    dx, dz = dirs[:, 0], dirs[:, 1]
    mapx = np.full(w, int(gx), np.int64)
    mapz = np.full(w, int(gz), np.int64)
    with np.errstate(divide="ignore"):
        ddx = np.abs(1.0 / dx)
        ddz = np.abs(1.0 / dz)
    stepx = np.where(dx < 0, -1, 1)
    stepz = np.where(dz < 0, -1, 1)
    sdx = np.where(dx < 0, gx - mapx, mapx + 1.0 - gx) * ddx
    sdz = np.where(dz < 0, gz - mapz, mapz + 1.0 - gz) * ddz
    dist = np.zeros(w)
    side = np.zeros(w, np.int8)
    active = np.ones(w, bool)
    max_steps = rows + cols + 2
    for _ in range(max_steps):
        if not active.any():
            break
        take_x = active & (sdx < sdz)
        take_z = active & ~take_x
        mapx[take_x] += stepx[take_x]
        dist[take_x] = sdx[take_x]
        sdx[take_x] += ddx[take_x]
        side[take_x] = 0
        mapz[take_z] += stepz[take_z]
        dist[take_z] = sdz[take_z]
        sdz[take_z] += ddz[take_z]
        side[take_z] = 1
        inb = (mapx >= 0) & (mapx < cols) & (mapz >= 0) & (mapz < rows)
        hit = np.zeros(w, bool)
        hit[inb] = solid[mapz[inb], mapx[inb]]
        active &= inb & ~hit & (dist * cell < MAX_DIST)
    return np.clip(dist * cell, 1e-3, MAX_DIST), side


def render_3d(
    solid: np.ndarray,
    grid: NavGrid,
    x: float,
    z: float,
    yaw: float,
    pitch: float,
    tx: float,
    tz: float,
    w: int,
    h: int,
) -> np.ndarray:
    """一人称ビュー(h, w, 3)を描く。yaw規約: +Z基準で+右回り、dir=(sin,cos)。"""
    half = math.radians(FOV_DEG / 2)
    rel = np.arctan(np.linspace(-math.tan(half), math.tan(half), w))  # 左→右
    ang = math.radians(yaw) + rel
    dirs = np.stack([np.sin(ang), np.cos(ang)], axis=1)
    b = grid.bounds
    dist, side = raycast(solid, grid.cell, b.xmin, b.zmin, x, z, dirs)
    perp = dist * np.cos(rel)  # 魚眼補正

    vhalf = math.atan(math.tan(half) * h / w)  # 垂直半FOV
    horizon = h / 2 + (h / 2) * math.tan(math.radians(pitch)) / math.tan(vhalf)
    wall_h = (h * 0.9) / np.maximum(perp, 0.05)
    top = horizon - wall_h / 2
    bot = horizon + wall_h / 2

    shade = np.clip(1.0 - dist / MAX_DIST, 0.15, 1.0)
    shade = np.where(side == 1, shade, shade * 0.7)  # X面は暗く
    wall_rgb = (WALL[None, :] * shade[:, None]).astype(np.uint8)  # (w,3)

    rows = np.arange(h)[:, None]
    img = np.where(rows[:, :, None] < horizon, CEIL, FLOOR)
    img = np.broadcast_to(img, (h, w, 3)).copy()
    mask = (rows >= top[None, :]) & (rows < bot[None, :])
    img[mask] = np.broadcast_to(wall_rgb[None, :, :], (h, w, 3))[mask]

    # 目標点ビルボード(壁より手前なら描く)
    dx, dz = tx - x, tz - z
    tdist = math.hypot(dx, dz)
    trel = math.atan2(dx, dz) - math.radians(yaw)
    trel = (trel + math.pi) % (2 * math.pi) - math.pi
    if tdist > 0.05 and abs(trel) < half:
        col = int((math.tan(trel) / math.tan(half) + 1) / 2 * (w - 1))
        if tdist < dist[col] + 0.3:
            tperp = tdist * math.cos(trel)
            size = int(np.clip((h * 0.25) / max(tperp, 0.2), 3, h // 3))
            cy = int(horizon + (h * 0.45) / max(tperp, 0.2))  # 床レベル付近
            y0, y1 = max(0, cy - size), min(h, cy)
            x0, x1 = max(0, col - size // 3), min(w, col + size // 3 + 1)
            if y0 < y1 and x0 < x1:
                img[y0:y1, x0:x1] = TARGET_COLOR
    return img


# ---- 2D地図 ---------------------------------------------------------------
class MapPane:
    """上面図ペイン。背景をキャッシュし、通過軌跡は差分描画する。"""

    def __init__(self, grid: NavGrid | None, data: dict, w: int, h: int):
        self.w, self.h = w, h
        if grid is not None:
            b = grid.bounds
            self.xmin, self.zmin, self.zmax = b.xmin, b.zmin, b.zmax
            gw, gd = b.width, b.depth
        else:
            xs, zs = data["x"], data["z"]
            pad = 1.0
            self.xmin, self.zmin = xs.min() - pad, zs.min() - pad
            self.zmax = zs.max() + pad
            gw = xs.max() + pad - self.xmin
            gd = self.zmax - self.zmin
        self.s = min(w / gw, h / gd)
        bg = np.full((h, w, 3), 24, np.uint8)
        if grid is not None:
            # ペイン各画素→グリッドセルの最近傍サンプル(上下反転で+Z上向き)
            xs = (np.arange(w) + 0.5) / self.s + self.xmin
            zs = self.zmax - (np.arange(h) + 0.5) / self.s
            ci = ((xs - grid.bounds.xmin) / grid.cell).astype(int)
            ri = ((zs - grid.bounds.zmin) / grid.cell).astype(int)
            ok = (
                (ci >= 0)[None, :]
                & (ci < grid.shape[1])[None, :]
                & (ri >= 0)[:, None]
                & (ri < grid.shape[0])[:, None]
            )
            free = np.zeros((h, w), bool)
            rr = np.clip(ri, 0, grid.shape[0] - 1)
            cc = np.clip(ci, 0, grid.shape[1] - 1)
            free = grid.free[rr[:, None], cc[None, :]] & ok
            bg[free] = (60, 66, 78)
            bg[~free & ok] = (36, 38, 44)
        # 経路全体(予定線)を薄く
        for px, py in zip(*self.to_px(data["x"], data["z"])):
            bg[max(py, 0) : py + 1, max(px, 0) : px + 1] = (90, 90, 100)
        self.bg = bg
        self.trail = np.zeros((h, w), bool)
        self._drawn = 0
        self.data = data

    def to_px(self, x, z):
        px = np.clip(((np.asarray(x) - self.xmin) * self.s).astype(int), 0, self.w - 1)
        py = np.clip(((self.zmax - np.asarray(z)) * self.s).astype(int), 0, self.h - 1)
        return px, py

    def _disc(self, img, px, py, r, color):
        y0, y1 = max(0, py - r), min(self.h, py + r + 1)
        x0, x1 = max(0, px - r), min(self.w, px + r + 1)
        img[y0:y1, x0:x1] = color

    def render(self, idx: int) -> np.ndarray:
        d = self.data
        if idx + 1 > self._drawn:  # 通過済み軌跡を差分で焼き込む
            px, py = self.to_px(d["x"][self._drawn : idx + 1], d["z"][self._drawn : idx + 1])
            self.trail[py, px] = True
            self._drawn = idx + 1
        img = self.bg.copy()
        img[self.trail] = (80, 200, 120)
        tx, ty = self.to_px(d["tx"][idx], d["tz"][idx])
        self._disc(img, int(tx), int(ty), 3, TARGET_COLOR)
        px, py = self.to_px(d["x"][idx], d["z"][idx])
        self._disc(img, int(px), int(py), 3, (80, 160, 255))
        # 向き矢印(yaw: +Z基準+右回り → 画面は+Z上向きなので dy=-cos)
        yaw = math.radians(d["yaw"][idx])
        for i in range(2, 14):
            ax = int(px + math.sin(yaw) * i)
            ay = int(py - math.cos(yaw) * i)
            if 0 <= ax < self.w and 0 <= ay < self.h:
                img[ay, ax] = (255, 255, 80)
        return img


# ---- HUD ---------------------------------------------------------------
def draw_hud(frame: np.ndarray, d: dict, idx: int, hud_h: int) -> np.ndarray:
    """下部にバー(turn/fwd)とテキストを描く。"""
    h, w, _ = frame.shape
    y0 = h - hud_h
    frame[y0:, :] = (18, 18, 22)
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img)
    cx = w // 4 + 20
    bw = w // 4 - 50

    def bar(y, val, lo, hi, color, label):
        dr.rectangle([cx - bw, y, cx + bw, y + 10], outline=(90, 90, 90))
        n = max(-1.0, min(1.0, (2 * (val - lo) / (hi - lo)) - 1.0))
        dr.rectangle(sorted_box(cx, cx + int(n * bw), y, y + 10), fill=color)
        dr.text((cx - bw - 8, y), label, fill=(200, 200, 200), anchor="rm")

    bar(y0 + 8, d["turn"][idx], -1, 1, (255, 180, 60), "turn")
    bar(y0 + 26, d["fwd"][idx], -1, 1, (80, 200, 120), "fwd")
    txt = (
        f"t={d['t'][idx]:6.2f}s  yaw={d['yaw'][idx]:7.1f}  "
        f"yaw_err={d['yaw_err'][idx]:6.1f}  turn={d['turn'][idx]:+.2f}  fwd={d['fwd'][idx]:+.2f}"
    )
    dr.text((w // 2 + 10, y0 + 14), txt, fill=(220, 220, 220))
    return np.asarray(img)


def sorted_box(x0, x1, y0, y1):
    return [min(x0, x1), y0, max(x0, x1), y1]


# ---- main ---------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="シムのフレームCSVを3D+2D並置動画に出力する")
    p.add_argument("--csv", required=True, help="フレームCSV")
    p.add_argument("--map", default=None, help="room.npz(省略時は2Dは軌跡のみ)")
    p.add_argument("--out", required=True, help="出力 mp4")
    p.add_argument("--fps", type=int, default=60, help="動画fps(既定60)")
    p.add_argument("--size", default="960x480", help="動画サイズ WxH")
    p.add_argument("--speed", type=float, default=1.0, help="再生倍率")
    p.add_argument("--png-every", type=int, default=0, help="Nフレーム毎にPNGも保存(0=無効)")
    p.add_argument("--png-dir", default=None, help="PNG出力先(既定: outと同じ場所)")
    args = p.parse_args()

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found in PATH")

    d = load_frames(Path(args.csv))
    w, h = (int(v) for v in args.size.lower().split("x"))
    w -= w % 2
    h -= h % 2
    hud_h = 44
    view_h = h - hud_h
    half_w = w // 2

    grid = None
    if args.map:
        grid = NavGrid.from_mapper(RoomMapper.load(args.map))
    solid = ~grid.free if grid is not None else None
    pane = MapPane(grid, d, half_w, view_h)

    t = d["t"]
    n_frames = max(1, int((t[-1] - t[0]) / args.speed * args.fps) + 1)
    frame_times = t[0] + np.arange(n_frames) / args.fps * args.speed
    idxs = np.searchsorted(t, frame_times, side="right") - 1
    idxs = np.clip(idxs, 0, len(t) - 1)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    png_dir = Path(args.png_dir) if args.png_dir else out.parent
    cmd = [
        ffmpeg, "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}", "-r", str(args.fps), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert proc.stdin is not None
    try:
        for k, i in enumerate(idxs):
            frame = np.empty((h, w, 3), np.uint8)
            if solid is not None:
                frame[:view_h, :half_w] = render_3d(
                    solid, grid, d["x"][i], d["z"][i], d["yaw"][i], d["pitch"][i],
                    d["tx"][i], d["tz"][i], half_w, view_h,
                )
            else:
                frame[:view_h, :half_w] = 30
            frame[:view_h, half_w:] = pane.render(int(i))
            frame = draw_hud(frame, d, int(i), hud_h)
            proc.stdin.write(frame.tobytes())
            if args.png_every and k % args.png_every == 0:
                png_dir.mkdir(parents=True, exist_ok=True)
                Image.fromarray(frame).save(png_dir / f"frame_{k:05d}.png")
    finally:
        proc.stdin.close()
        proc.wait()
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed with code {proc.returncode}")
    print(f"wrote {out} ({n_frames} frames, {n_frames / args.fps:.1f}s)")


if __name__ == "__main__":
    main()
