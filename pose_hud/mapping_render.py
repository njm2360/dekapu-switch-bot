from pathlib import Path

import numpy as np

from .mapping import RoomMapper


def render_map(
    mapper: RoomMapper,
    out_path: str | Path,
    cell: float = 0.1,
    show_occupancy: bool = True,
    title: str | None = None,
) -> Path:
    """matplotlib で床平面の間取り図(トップダウン)を PNG 保存する。

    matplotlib 未導入なら ImportError。CLI 側で occupancy PNG にフォールバックする。
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # ヘッドレス
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(
            "matplotlib is required for render_map (uv sync --extra map)"
        ) from exc

    if len(mapper) == 0:
        raise ValueError("no points to render")

    out_path = Path(out_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pts = mapper.points
    w, d = mapper.dimensions()

    fig, ax = plt.subplots(figsize=(8, 8))

    if show_occupancy and len(mapper) >= 2:
        occ = mapper.occupancy_grid(cell=cell)
        ax.imshow(
            occ.grid,
            origin="lower",
            extent=occ.bounds.as_extent(),
            cmap="Greys",
            alpha=0.35,
            interpolation="nearest",
            aspect="equal",
        )

    # 歩行軌跡(=壁の輪郭)。セグメント分割(ペンアップ)をまたいでは繋がない。
    for i, seg in enumerate(mapper.segment_points()):
        ax.plot(seg[:, 0], seg[:, 1], "-", color="#1f77b4", lw=1.2,
                label="walked path" if i == 0 else None)
    ax.plot(pts[0, 0], pts[0, 1], "o", color="#2ca02c", ms=9, label="start")
    ax.plot(pts[-1, 0], pts[-1, 1], "s", color="#d62728", ms=8, label="end")

    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, ls=":", alpha=0.5)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(title or f"Room map  {w:.2f} x {d:.2f} m  ({len(mapper)} pts)")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def save_occupancy_png(
    mapper: RoomMapper,
    out_path: str | Path,
    cell: float = 0.1,
    upscale: int = 4,
) -> Path:
    """Pillow で占有グリッドを白黒PNG保存する(matplotlib 無し環境向け)。

    Pillow も無ければ ImportError。
    """
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - 環境依存
        raise ImportError(
            "Pillow required for save_occupancy_png (uv sync --extra debug)"
        ) from exc

    occ = mapper.occupancy_grid(cell=cell)
    # origin=lower に合わせて上下反転(画像は上が row0)
    img = np.where(occ.grid[::-1], 0, 255).astype(np.uint8)  # 通過=黒
    if upscale > 1:
        img = np.repeat(np.repeat(img, upscale, axis=0), upscale, axis=1)
    out_path = Path(out_path).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img, mode="L").save(out_path)
    return out_path


def save_map(mapper: RoomMapper, out_prefix: str | Path, cell: float = 0.1) -> Path:
    """地図を1枚のPNGに保存する。matplotlib→Pillow→(不可なら例外)の順にフォールバック。"""
    out_prefix = Path(out_prefix)
    try:
        return render_map(mapper, out_prefix, cell=cell)
    except ImportError:
        return save_occupancy_png(mapper, out_prefix, cell=cell)
