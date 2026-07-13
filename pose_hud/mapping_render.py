from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # ヘッドレス
import matplotlib.pyplot as plt

from .mapping import RoomMapper


def render_map(
    mapper: RoomMapper,
    out_path: str | Path,
    cell: float = 0.1,
    show_occupancy: bool = True,
    title: str | None = None,
) -> Path:
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

    # 歩行軌跡(=壁の輪郭)。セグメント分割をまたいでは繋がない。
    for i, seg in enumerate(mapper.segment_points()):
        ax.plot(
            seg[:, 0],
            seg[:, 1],
            "-",
            color="#1f77b4",
            lw=1.2,
            label="walked path" if i == 0 else None,
        )
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
