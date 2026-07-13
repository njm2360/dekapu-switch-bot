"""保存した部屋マップを読み込み、指定座標のボタンを壁を避けて巡回する。

歩行軌跡マップ(map_room.py の .npz)から歩行可能グリッドを作り、各ボタン座標へ A* で
壁を迂回する経路を計画する。ライブ実行では PoseReader の位置フィードバックを見ながら
アバターを移動・旋回させ、到着したらボタンを向く(実クリックはまだ行わない)。

視点(look)と移動(move)のアクチュエータ、および PID 制御器は注入式。既定は移動・視点とも
OSC。`--look mouse` で視点だけ DirectInput(相対マウス)に差し替えできる(要 pydirectinput)。

    # まず計画だけ確認(VRChat不要。--dry-run でマップ隣に plan.png を自動保存)
    uv run patrol-buttons --map maps/room.npz --target 3.0,1.2,5.0 --target -1.0,1.0,2.0 --dry-run

    # 実際に巡回(VRChat 起動 + HUD_Enable が必要)
    uv run patrol-buttons --map maps/room.npz --target 3.0,1.2,5.0 --target -1.0,1.0,2.0
"""

import argparse
import dataclasses
import json
import math
import time
from datetime import datetime
from pathlib import Path

from pose_hud.cli._ctl_log import ControlLog
from pose_hud.controller import (
    FaceControllers,
    NavControllers,
    PatrolGains,
    face_controllers,
    nav_controllers,
)
from pose_hud.mapping import RoomMapper
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
    """start から順に各ターゲットへ計画。(name, target, Path|None) を列挙。"""
    cur = start
    legs = []
    for name, tgt, _y in targets:
        path = plan_path(grid, cur, tgt)
        legs.append((name, tgt, path))
        if path is not None:
            cur = path.reached_goal_cell
    return legs


def _render_plan(grid: NavGrid, start, legs, out: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


# ---- ライブ制御(アクチュエータ + 制御器を注入) --------------------------
def _next_frame(reader, last_t, last_time, wait_cap=2.0):
    """新しいフレーム(time_ms が変化)が来るまで待って (pose, dt, now) を返す。

    HUD が wait_cap 秒来なければ (None, 0, now)。dt は前フレームからの実経過秒。
    """
    deadline = time.monotonic() + wait_cap
    while time.monotonic() < deadline:
        pose = reader.get_latest()
        if pose is not None and pose.time_ms != last_t:
            now = time.monotonic()
            return pose, min(now - last_time, 0.2), now  # dt は 0.2s で頭打ち
        time.sleep(0.002)
    return None, 0.0, time.monotonic()


def _navigate_to(
    reader, look, move, grid, tgt_xz, tgt_y, name, g: PatrolGains,
    nav: NavControllers, face: FaceControllers, log,
):
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
    nav.yaw.reset()
    nav.forward.reset()
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
            nav.yaw.reset_derivative()  # 目標が急に変わったとき turn が跳ねるのを防ぐ
        target = wps[idx]
        final = idx == len(wps) - 1
        err, dist = heading_error(cur, pose.yaw_deg, target)
        if final and dist < g.arrive:
            break

        turn = nav.yaw.update(err, dt)
        ff = forward_factor(err)
        # 最終WPは距離制御器で減速、途中は巡航速度。どちらも向きズレで滑らかに減速。
        speed = (nav.forward.update(dist, dt) if final else g.speed) * ff
        look.look(turn)
        move.move(forward=speed)

        log.row(t=now - t0, phase="nav", target=name, wp=idx, dt=dt,
                x=pose.position[0], y=pose.position[1], z=pose.position[2],
                yaw=pose.yaw_deg, pitch=pose.pitch_deg,
                tx=target[0], tz=target[1], dist=dist, yaw_err=err,
                turn_p=nav.yaw.last_p, turn_i=nav.yaw.last_i,
                turn_d=nav.yaw.last_d, turn=turn, fwd=speed, fwd_factor=ff)
        if now - t0 > g.nav_timeout:
            print(f"  [{name}] nav timeout")
            break
    look.stop()
    move.stop()

    yaw_err, pitch_err = _face(reader, look, tgt_xz, tgt_y, name, g, face, log)
    print(f"  [{name}] arrived. aim yaw_err={yaw_err:+.2f}° pitch_err={pitch_err:+.2f}°")
    return True


def _face(reader, look, tgt_xz, tgt_y, name, g: PatrolGains, face: FaceControllers, log):
    """ボタンを向くまで PID 旋回。yaw/pitch が tol 未満を settle 回連続で達成したら収束。

    戻り値: (最終 yaw誤差[deg], 最終 pitch誤差[deg])。
    """
    face.yaw.reset()
    face.pitch.reset()
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

        # tol 未満は制御器側が指令を0にする(不感帯補償による最小の旋回で行き過ぎるのを防ぐ)。
        turn = face.yaw.update(yaw_err, dt)
        pitch_cmd = face.pitch.update(pitch_err, dt)
        look.look(turn, pitch_cmd)

        log.row(t=now - t0, phase="face", target=name, dt=dt,
                x=pose.position[0], y=pose.position[1], z=pose.position[2],
                yaw=pose.yaw_deg, pitch=pose.pitch_deg,
                tx=target_xyz[0], ty=target_xyz[1], tz=target_xyz[2],
                yaw_err=yaw_err, pitch_err=pitch_err,
                turn_p=face.yaw.last_p, turn_i=face.yaw.last_i,
                turn_d=face.yaw.last_d, turn=turn,
                pitch_p=face.pitch.last_p, pitch_i=face.pitch.last_i,
                pitch_d=face.pitch.last_d, pitch_cmd=pitch_cmd)
    look.stop()
    return yaw_err, pitch_err


def _build_look(args, osc):
    """--look に応じて LookActuator を返す(mouse 以外は OSC を流用)。"""
    if args.look == "mouse":
        from pose_hud.actuator import MouseLookActuator

        return MouseLookActuator(
            yaw_gain=args.mouse_yaw_gain, pitch_gain=args.mouse_pitch_gain
        )
    return osc


def _run_live(grid, targets, args, gains: PatrolGains):
    from pose_hud.capture import WindowsVRChatCapture
    from pose_hud.osc import VRChatOSC
    from pose_hud.reader import PoseReader

    log_path = Path("logs") / f"patrol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = ControlLog(log_path)
    print(f"control log: {log_path}")

    reader = PoseReader(source=WindowsVRChatCapture())
    reader.start()
    osc = VRChatOSC()
    move = osc  # 移動は OSC(pydirect版は使わない想定)
    look = _build_look(args, osc)  # 視点は OSC or マウス
    nav = nav_controllers(gains)
    face = face_controllers(gains)

    osc.hud_enable(True)
    print(f"look={args.look}  waiting for HUD...")
    for _ in range(100):
        if reader.get_latest() is not None:
            break
        time.sleep(0.1)
    try:
        for name, tgt_xz, tgt_y in targets:
            print(f"-> {name} {tgt_xz} y={tgt_y}")
            _navigate_to(reader, look, move, grid, tgt_xz, tgt_y, name, gains, nav, face, log)
        print("patrol done.")
    except KeyboardInterrupt:
        print("\ninterrupted.")
    finally:
        look.stop()
        osc.close()
        reader.stop()
        log.close()


def _add_gain_args(parser) -> None:
    """PID ゲイン等のチューニングフラグを追加(既定値は PatrolGains が単一の出所)。"""
    d = PatrolGains()
    parser.add_argument("--speed", type=float, default=d.speed, help="巡航前進速度の上限(0..1)")
    parser.add_argument("--arrive", type=float, default=d.arrive, help="ウェイポイント到達半径[m]")
    parser.add_argument("--face-tol", type=float, default=d.face_tol, help="正対とみなす角度[deg]")
    parser.add_argument("--settle", type=int, default=d.settle, help="収束に必要な、連続で正対を保ったフレーム数")
    parser.add_argument("--nav-timeout", type=float, default=d.nav_timeout, help="移動の打切り秒")
    parser.add_argument("--face-timeout", type=float, default=d.face_timeout, help="正対の打切り秒")
    # 正対(face)の yaw: 視点軸は約0.55以下がほとんど反応しないので out_deadzone で飛び越える(OSC用)。
    parser.add_argument("--turn-kp", type=float, default=d.turn_kp)
    parser.add_argument("--turn-ki", type=float, default=d.turn_ki)
    parser.add_argument("--turn-kd", type=float, default=d.turn_kd)
    parser.add_argument("--turn-ilim", type=float, default=d.turn_ilim, help="yaw積分項の絶対上限")
    parser.add_argument("--turn-deadzone", type=float, default=d.turn_deadzone,
                        help="正対 yaw の不感帯補償(視点軸が反応しない範囲。0で無効。マウス時は0推奨)")
    # 移動中(nav)の yaw: 移動には強すぎて暴れるので穏やかに。不感帯補償は入れない。
    parser.add_argument("--nav-turn-kp", type=float, default=d.nav_turn_kp)
    parser.add_argument("--nav-turn-ki", type=float, default=d.nav_turn_ki)
    parser.add_argument("--nav-turn-kd", type=float, default=d.nav_turn_kd)
    parser.add_argument("--pitch-kp", type=float, default=d.pitch_kp)
    parser.add_argument("--pitch-ki", type=float, default=d.pitch_ki)
    parser.add_argument("--pitch-kd", type=float, default=d.pitch_kd)
    parser.add_argument("--pitch-ilim", type=float, default=d.pitch_ilim, help="pitch積分項の絶対上限")
    parser.add_argument("--pitch-deadzone", type=float, default=d.pitch_deadzone,
                        help="正対 pitch の不感帯補償(既定0=無効。上下がなかなか合わないなら 0.5 程度)")
    parser.add_argument("--fwd-kp", type=float, default=d.fwd_kp)
    parser.add_argument("--fwd-kd", type=float, default=d.fwd_kd)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patrol buttons on a saved room map, avoiding walls"
    )
    parser.add_argument("--map", required=True, help="部屋マップ .npz(map_room.py 出力)")
    parser.add_argument("--target", action="append", metavar="X,Y,Z", help="ボタン座標(複数可)")
    parser.add_argument("--buttons", metavar="JSON", help="ボタン座標をまとめたJSON")
    parser.add_argument("--cell", type=float, default=0.1, help="グリッド解像度[m]")
    parser.add_argument("--radius", type=float, default=0.25, help="アバター半径=壁クリアランス[m]")
    parser.add_argument("--gap-close", type=float, default=0.3, help="軌跡の隙間を塞ぐ距離[m]")
    parser.add_argument("--dry-run", action="store_true",
                        help="計画のみ(操作しない)。マップ隣に plan.png を自動保存")
    # アクチュエータ選択(視点のみ差し替え可。移動は OSC 固定)
    parser.add_argument("--look", choices=("osc", "mouse"), default="osc",
                        help="視点アクチュエータ(mouse=DirectInput相対マウス。要 pydirectinput)")
    parser.add_argument("--mouse-yaw-gain", type=float, default=40.0,
                        help="マウス視点の水平ゲイン[px/指令]")
    parser.add_argument("--mouse-pitch-gain", type=float, default=40.0,
                        help="マウス視点の上下ゲイン[px/指令]")
    _add_gain_args(parser)
    args = parser.parse_args()

    # チューニング定数を1オブジェクトに集約(フラグは PatrolGains の既定を上書き)。
    gains = PatrolGains(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(PatrolGains)}
    )

    mapper = RoomMapper.load(args.map)
    grid = NavGrid.from_mapper(
        mapper, cell=args.cell, avatar_radius=args.radius, gap_close=args.gap_close
    )
    free_ratio = grid.free.mean()
    print(
        f"map: {len(mapper)}pts  grid {grid.shape[1]}x{grid.shape[0]}  "
        f"walkable {free_ratio:.0%}  dims {tuple(round(v, 2) for v in mapper.dimensions())}m"
    )

    targets = _parse_targets(args)
    if not targets:
        parser.error("--target x,y,z を1つ以上、または --buttons を指定してください")

    p0 = mapper.points[0]
    start = (float(p0[0]), float(p0[1]))

    legs = _plan_tour(grid, start, targets)
    print(f"\nplan from {tuple(round(v, 2) for v in start)}:")
    for name, tgt, path in legs:
        if path is None:
            print(f"  {name} {tgt}: 到達不能")
        else:
            note = " (壁→最寄り床)" if path.goal_blocked else ""
            print(f"  {name} {tgt}: {len(path.waypoints)}wp / {path.length:.2f}m{note}")

    if args.dry_run:
        png = _render_plan(grid, start, legs, Path(args.map).with_name("plan.png"))
        print(f"plan figure: {png}")
        return

    _run_live(grid, targets, args, gains)


if __name__ == "__main__":
    main()
