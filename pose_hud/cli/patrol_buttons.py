"""サンプル: 保存した部屋マップを読み込み、指定座標のボタンを壁を避けて巡回する。

歩行軌跡マップ(map_room.py の .npz)から歩行可能グリッドを作り、各ボタン座標へ
A* で壁を迂回する経路を計画する。ライブ実行では PoseReader の位置フィードバックを見ながら
OSC でアバターを移動・旋回させ、到着したらボタンを向く(実クリックはまだ行わない)。

    # まず計画だけ確認(VRChat不要。--dry-run でマップ隣に plan.png を自動保存)
    uv run patrol-buttons --map maps/room.npz --target 3.0,1.2,5.0 --target -1.0,1.0,2.0 --dry-run

    # 実際に OSC で巡回(VRChat 起動 + HUD_Enable が必要)
    uv run patrol-buttons --map maps/room.npz --target 3.0,1.2,5.0 --target -1.0,1.0,2.0

"""

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path

from pose_hud import RoomMapper
from pose_hud.cli._ctl_log import make_log
from pose_hud.control import PID
from pose_hud.navigation import (
    NavGrid,
    forward_factor,
    heading_error,
    pitch_error,
    plan_path,
)

Target = tuple[str, tuple[float, float], float]


def _parse_targets(args) -> list[Target]:
    targets: list[Target] = []
    for i, spec in enumerate(args.target or []):
        parts = [float(v) for v in spec.split(",")]
        if len(parts) != 3:  # x,y,z(高さ必須)
            raise SystemExit(f"--target は 'x,y,z' 形式で(高さ必須): {spec!r}")
        targets.append((f"t{i + 1}", (parts[0], parts[2]), parts[1]))
    if args.buttons:
        data = json.loads(Path(args.buttons).read_text(encoding="utf-8"))
        for rec in data if isinstance(data, list) else [data]:
            p = rec.get("result", {}).get("point") or rec.get("point")
            if not p:
                continue
            if p.get("y") is None:
                raise SystemExit(
                    f"buttons JSON にy(高さ)がありません: {rec.get('name')}"
                )
            targets.append((rec.get("name", "button"), (p["x"], p["z"]), p["y"]))
    return targets


def _target_xyz(tgt_xz, tgt_y):
    """ボタンの3D座標(x, y, z)。"""
    return (tgt_xz[0], tgt_y, tgt_xz[1])


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _plan_tour(
    grid: NavGrid,
    start: tuple[float, float],
    targets: list[tuple[str, tuple[float, float]]],
):
    """start から順に各ターゲットへ計画。(name, target, Path|None, next_start) を列挙。"""
    cur = start
    legs = []
    for name, tgt, _y in targets:
        path = plan_path(grid, cur, tgt)
        legs.append((name, tgt, path))
        if path is not None:
            cur = path.reached_goal_cell
    return legs


def _render_plan(grid: NavGrid, start, legs, out: Path) -> Path | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plan図スキップ] matplotlib 未導入 (uv sync --extra map)")
        return None
    b = grid.bounds
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(
        grid.free,
        origin="lower",
        extent=b.as_extent(),
        cmap="Greys_r",
        alpha=0.5,
        interpolation="nearest",
        aspect="equal",
    )
    ax.plot(start[0], start[1], "o", color="#2ca02c", ms=10, label="start")
    for name, tgt, path in legs:
        ax.plot(tgt[0], tgt[1], "X", color="#d62728", ms=11)
        ax.annotate(name, tgt, textcoords="offset points", xytext=(6, 6), fontsize=8)
        if path is not None:
            wx = [p[0] for p in path.waypoints]
            wz = [p[1] for p in path.waypoints]
            ax.plot(wx, wz, "-", lw=1.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title("Patrol plan (white=walkable)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = out.with_suffix(".png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


# ---- ライブ制御(PID + OSC) ----------------------------------------------
def _nav_pids(g):
    """移動(nav)用の (yaw, 前進) PID。yaw は穏やかなゲイン・デッドゾーン補償なし
    (連続追従では膝以下が自然なデッドバンドとして働き、暴れを防ぐ)。"""
    yaw = PID(
        kp=g.nav_turn_kp, ki=g.nav_turn_ki, kd=g.nav_turn_kd,
        out_min=-1.0, out_max=1.0, i_limit=0.5,
    )
    fwd = PID(
        kp=g.fwd_kp, ki=0.0, kd=g.fwd_kd, out_min=0.0, out_max=g.speed, i_limit=0.0
    )
    return yaw, fwd


def _face_pids(g):
    """正対(face)用の (yaw, pitch) PID。yaw は視点軸のデッドゾーンを飛び越えるため
    out_knee(膝)補償つき。最後の数度を漏れ速度で這わず、11deg/s 以上で詰める。"""
    yaw = PID(
        kp=g.turn_kp, ki=g.turn_ki, kd=g.turn_kd,
        out_min=-1.0, out_max=1.0, i_limit=g.turn_ilim, out_knee=g.turn_knee,
    )
    pitch = PID(
        kp=g.pitch_kp, ki=g.pitch_ki, kd=g.pitch_kd,
        out_min=-1.0, out_max=1.0, i_limit=g.pitch_ilim, out_knee=g.pitch_knee,
    )
    return yaw, pitch


def _next_frame(reader, last_t, last_time, wait_cap=2.0):
    """新しいフレーム(time_ms が変化)が来るまで待って (pose, dt, now) を返す。

    HUD が wait_cap 秒来なければ (None, 0, now)。dt は前フレームからの実経過秒。
    """
    deadline = time.monotonic() + wait_cap
    while time.monotonic() < deadline:
        pose = reader.get_latest()
        if pose is not None and pose.time_ms != last_t:
            now = time.monotonic()
            return pose, min(now - last_time, 0.2), now   # dt は 0.2s で頭打ち
        time.sleep(0.002)
    return None, 0.0, time.monotonic()


def _navigate_to(reader, osc, grid, tgt_xz, tgt_y, name, g, log):
    pose = reader.get_latest()
    if pose is None:
        print(f"  [{name}] 現在位置が取れません(HUD?)")
        return False
    start = (pose.position[0], pose.position[2])
    path = plan_path(grid, start, tgt_xz)
    if path is None:
        print(f"  [{name}] 経路なし(到達不能)")
        return False
    print(
        f"  [{name}] 経路 {len(path.waypoints)}点 / {path.length:.1f}m"
        + ("(壁面ボタン→最寄り床へ)" if path.goal_blocked else "")
    )

    wps = path.waypoints
    yaw_pid, fwd_pid = _nav_pids(g)
    idx = 1 if len(wps) > 1 else 0
    last_t = None
    last_time = t0 = time.monotonic()
    while idx < len(wps):
        pose, dt, now = _next_frame(reader, last_t, last_time)
        if pose is None:
            print(f"  [{name}] HUD lost, abort nav")
            break
        last_t, last_time = pose.time_ms, now
        cur = (pose.position[0], pose.position[2])

        # 追い越したウェイポイントはスキップ(連続追従・停止しない)
        prev_idx = idx
        while idx < len(wps) - 1 and _dist(cur, wps[idx]) < g.arrive:
            idx += 1
        if idx != prev_idx:
            yaw_pid.reset_derivative()   # 目標急変での微分キック(turnスパイク)を防ぐ
        target = wps[idx]
        final = idx == len(wps) - 1
        err, dist = heading_error(cur, pose.yaw_deg, target)
        if final and dist < g.arrive:
            break

        turn = yaw_pid.update(err, dt)
        ff = forward_factor(err)
        # 最終WPは距離PIDで減速、途中は巡航速度。どちらも向きズレで滑らかに減速。
        speed = (fwd_pid.update(dist, dt) if final else g.speed) * ff
        osc.look(turn)
        osc.move(forward=speed)

        log.row(t=now - t0, phase="nav", target=name, wp=idx, dt=dt,
                x=pose.position[0], y=pose.position[1], z=pose.position[2],
                yaw=pose.yaw_deg, pitch=pose.pitch_deg,
                tx=target[0], tz=target[1], dist=dist, yaw_err=err,
                turn_p=yaw_pid.last_p, turn_i=yaw_pid.last_i,
                turn_d=yaw_pid.last_d, turn=turn, fwd=speed, fwd_factor=ff)
        if now - t0 > g.nav_timeout:
            print(f"  [{name}] nav timeout")
            break
    osc.stop()

    yaw_err, pitch_err = _face(reader, osc, grid, tgt_xz, tgt_y, name, g, log)
    print(f"  [{name}] arrived. aim yaw_err={yaw_err:+.2f}° pitch_err={pitch_err:+.2f}°")
    return True


def _face(reader, osc, grid, tgt_xz, tgt_y, name, g, log):
    """ボタンを向くまで PID 旋回。yaw/pitch が tol 未満を settle 回連続で達成したら収束。

    戻り値: (最終 yaw誤差[deg], 最終 pitch誤差[deg])。
    """
    yaw_pid, pitch_pid = _face_pids(g)
    target_xyz = _target_xyz(tgt_xz, tgt_y)
    last_t = None
    last_time = t0 = time.monotonic()
    settle = 0
    yaw_err = pitch_err = 0.0
    while time.monotonic() - t0 < g.face_timeout:
        pose, dt, now = _next_frame(reader, last_t, last_time)
        if pose is None:
            break
        last_t, last_time = pose.time_ms, now
        cur = (pose.position[0], pose.position[2])
        yaw_err, _ = heading_error(cur, pose.yaw_deg, tgt_xz)
        pitch_err = pitch_error(pose.position, pose.forward, target_xyz)

        if abs(yaw_err) < g.face_tol and abs(pitch_err) < g.face_tol:
            settle += 1
            if settle >= g.settle:
                break
        else:
            settle = 0

        # tol 内は各軸の出力を0にゲート(膝補償の最小旋回で行き過ぎるのを防ぐ)
        turn = 0.0 if abs(yaw_err) < g.face_tol else yaw_pid.update(yaw_err, dt)
        pitch_cmd = 0.0 if abs(pitch_err) < g.face_tol else pitch_pid.update(pitch_err, dt)
        osc.look(turn, pitch=pitch_cmd)

        log.row(t=now - t0, phase="face", target=name, dt=dt,
                x=pose.position[0], y=pose.position[1], z=pose.position[2],
                yaw=pose.yaw_deg, pitch=pose.pitch_deg,
                tx=target_xyz[0], ty=target_xyz[1], tz=target_xyz[2],
                yaw_err=yaw_err, pitch_err=pitch_err,
                turn_p=yaw_pid.last_p, turn_i=yaw_pid.last_i,
                turn_d=yaw_pid.last_d, turn=turn,
                pitch_p=pitch_pid.last_p, pitch_i=pitch_pid.last_i,
                pitch_d=pitch_pid.last_d, pitch_cmd=pitch_cmd)
    osc.stop()
    return yaw_err, pitch_err


def _run_live(grid, targets, g):
    from pose_hud import PoseReader
    from pose_hud.capture import WindowsVRChatCapture
    from pose_hud.osc import VRChatOSC

    log_path = None
    if not g.no_log:
        log_path = g.log or str(
            Path("logs") / f"patrol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log = make_log(log_path)
    if log_path:
        print(f"control log: {log_path}")

    reader = PoseReader(source=WindowsVRChatCapture(window_title=g.window))
    reader.start()
    osc = VRChatOSC()
    osc.hud_enable(True)
    print("waiting for HUD...")
    for _ in range(100):
        if reader.get_latest() is not None:
            break
        time.sleep(0.1)
    try:
        for name, tgt_xz, tgt_y in targets:
            print(f"-> {name} {tgt_xz} y={tgt_y}")
            _navigate_to(reader, osc, grid, tgt_xz, tgt_y, name, g, log)
        print("patrol done.")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        osc.close()
        reader.stop()
        log.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patrol buttons on a saved room map, avoiding walls"
    )
    parser.add_argument(
        "--map", required=True, help="部屋マップ .npz(map_room.py 出力)"
    )
    parser.add_argument(
        "--target", action="append", metavar="X,Y,Z", help="ボタン座標(複数可)"
    )
    parser.add_argument("--buttons", metavar="JSON", help="ボタン座標をまとめたJSON")
    parser.add_argument("--start", metavar="X,Z", help="開始位置(既定: 軌跡の先頭)")
    parser.add_argument("--cell", type=float, default=0.1, help="グリッド解像度[m]")
    parser.add_argument(
        "--radius", type=float, default=0.25, help="アバター半径=壁クリアランス[m]"
    )
    parser.add_argument(
        "--gap-close", type=float, default=0.3, help="軌跡の隙間を塞ぐ距離[m]"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="計画のみ(OSC送信しない)"
    )
    parser.add_argument(
        "--save-plan", metavar="PNG",
        help="計画図の保存先(未指定でも --dry-run 時はマップ隣に plan.png を自動保存)",
    )
    parser.add_argument("--window", default="VRChat", help="キャプチャ対象ウィンドウ")
    parser.add_argument("--speed", type=float, default=0.7, help="巡航前進速度の上限(0..1)")
    parser.add_argument("--arrive", type=float, default=0.35, help="ウェイポイント到達半径[m]")
    parser.add_argument("--face-tol", type=float, default=1.0, help="正対とみなす角度[deg]")
    parser.add_argument("--settle", type=int, default=3, help="収束に必要な連続tol達成フレーム数")
    parser.add_argument("--nav-timeout", type=float, default=60.0, help="移動の打切り秒")
    parser.add_argument("--face-timeout", type=float, default=12.0, help="正対の打切り秒")
    # ログ(既定ON。PID チューニング用の詳細CSV)
    parser.add_argument("--log", metavar="CSV", help="制御ログの保存先(既定 logs/patrol_<日時>.csv)")
    parser.add_argument("--no-log", action="store_true", help="制御ログを書かない")
    # PID ゲイン(yaw/pitch は度→軸[-1,1]、fwd は m→速度。実機はログを見て要調整)
    # face(正対)yaw: 視点軸の膝(≈0.55)以下が無反応なので out_knee で飛び越える。
    parser.add_argument("--turn-kp", type=float, default=0.08)
    parser.add_argument("--turn-ki", type=float, default=0.01)
    parser.add_argument("--turn-kd", type=float, default=0.006)
    parser.add_argument("--turn-ilim", type=float, default=0.5, help="yaw積分項の絶対上限")
    parser.add_argument("--turn-knee", type=float, default=0.55,
                        help="face yaw のデッドゾーン補償(視点軸の膝。0で無効)")
    # nav(移動中)yaw: kp0.08 は移動には強すぎ暴れるので穏やかに。膝補償は入れない。
    parser.add_argument("--nav-turn-kp", type=float, default=0.035)
    parser.add_argument("--nav-turn-ki", type=float, default=0.015)
    parser.add_argument("--nav-turn-kd", type=float, default=0.004)
    parser.add_argument("--pitch-kp", type=float, default=0.035)
    parser.add_argument("--pitch-ki", type=float, default=0.015)
    parser.add_argument("--pitch-kd", type=float, default=0.004)
    parser.add_argument("--pitch-ilim", type=float, default=0.5, help="pitch積分項の絶対上限")
    parser.add_argument("--pitch-knee", type=float, default=0.0,
                        help="face pitch のデッドゾーン補償(既定0=無効。上下が粘るなら 0.5 程度)")
    parser.add_argument("--fwd-kp", type=float, default=1.5)
    parser.add_argument("--fwd-kd", type=float, default=0.1)
    args = parser.parse_args()

    mapper = RoomMapper.load(args.map)
    grid = NavGrid.from_mapper(
        mapper, cell=args.cell, avatar_radius=args.radius, gap_close=args.gap_close
    )
    free_ratio = grid.free.mean()
    print(
        f"map: {len(mapper)}pts  grid {grid.shape[1]}x{grid.shape[0]}  "
        f"walkable {free_ratio:.0%}  dims {tuple(round(v,2) for v in mapper.dimensions())}m"
    )

    targets = _parse_targets(args)
    if not targets:
        parser.error("--target x,y,z を1つ以上、または --buttons を指定してください")

    if args.start:
        sx, sz = (float(v) for v in args.start.split(","))
        start = (sx, sz)
    else:
        p0 = mapper.points[0]
        start = (float(p0[0]), float(p0[1]))

    legs = _plan_tour(grid, start, targets)
    print(f"\nplan from {tuple(round(v,2) for v in start)}:")
    for name, tgt, path in legs:
        if path is None:
            print(f"  {name} {tgt}: 到達不能")
        else:
            note = " (壁→最寄り床)" if path.goal_blocked else ""
            print(f"  {name} {tgt}: {len(path.waypoints)}wp / {path.length:.2f}m{note}")

    # --save-plan 指定時はそこへ。dry-run なら未指定でもマップ隣に plan.png を自動保存。
    plan_out = args.save_plan or (
        str(Path(args.map).with_name("plan.png")) if args.dry_run else None
    )
    if plan_out:
        png = _render_plan(grid, start, legs, Path(plan_out))
        if png:
            print(f"plan figure: {png}")

    if args.dry_run:
        return
    _run_live(grid, targets, args)


if __name__ == "__main__":
    main()
