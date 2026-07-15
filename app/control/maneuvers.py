"""誘導を構成する制御ループ部品(1フェーズ=1関数)。

実機 I/O には依存せず、PoseSource / LookActuator / MoveActuator の抽象だけで
動く(ヘッドレスでテスト可能)。follow_path / follow_path_hold_view は体の移動追従
(後者は視点を回さない)、aim_at / turn_to は視点合わせ、strafe_align は横移動での
最終照準。経路計画やフェーズの連結は pilot.Pilot が担う。
"""

import logging
import math
import time
from dataclasses import dataclass
from typing import Callable, Protocol

from .actuator import LookActuator, MoveActuator
from .controller import (
    AxisController,
    FaceControllers,
    NavControllers,
    PatrolGains,
    TranslateControllers,
)
from .guidance import forward_factor, heading_error, pitch_error, wrap180
from ..spatial.navigation import Path
from ..core.pose import Pose
from .telemetry import AxisAccumulator, AxisMetrics, NullRecorder, Recorder

logger = logging.getLogger(__name__)


class PoseSource(Protocol):
    def get_latest(self) -> Pose | None: ...


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> None: ...


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _cum_arclen(wps: list[tuple[float, float]]) -> list[float]:
    """ポリラインの累積弧長(cum[0]=0、cum[-1]=全長)。"""
    cum = [0.0]
    for i in range(len(wps) - 1):
        cum.append(cum[-1] + _dist(wps[i], wps[i + 1]))
    return cum


def _project_arclen(
    cur: tuple[float, float],
    wps: list[tuple[float, float]],
    cum: list[float],
    hint: int,
) -> tuple[int, float]:
    """cur を経路ポリラインへ射影し (セグメント番号, 始点からの弧長) を返す。

    hint(前回のセグメント)以降だけを見て、経路が自分に近接して戻ってくる形でも
    射影が後退しないようにする。
    """
    best_d: float | None = None
    best_i, best_s = max(0, min(hint, len(wps) - 2)), cum[-1]
    for i in range(max(0, hint), len(wps) - 1):
        ax, az = wps[i]
        bx, bz = wps[i + 1]
        dx, dz = bx - ax, bz - az
        l2 = dx * dx + dz * dz
        if l2 <= 1e-12:
            continue
        t = ((cur[0] - ax) * dx + (cur[1] - az) * dz) / l2
        t = max(0.0, min(1.0, t))
        d = math.hypot(cur[0] - (ax + t * dx), cur[1] - (az + t * dz))
        if best_d is None or d < best_d:
            best_d, best_i, best_s = d, i, cum[i] + t * math.sqrt(l2)
    return best_i, best_s


def _point_at_arclen(
    wps: list[tuple[float, float]], cum: list[float], s: float
) -> tuple[float, float]:
    """弧長 s の位置の点(範囲外は端点でクランプ)。"""
    if s <= 0.0:
        return wps[0]
    if s >= cum[-1]:
        return wps[-1]
    for i in range(len(wps) - 1):
        if cum[i + 1] >= s:
            seg = cum[i + 1] - cum[i]
            if seg <= 1e-12:
                return wps[i + 1]
            t = (s - cum[i]) / seg
            return (
                wps[i][0] + t * (wps[i + 1][0] - wps[i][0]),
                wps[i][1] + t * (wps[i + 1][1] - wps[i][1]),
            )
    return wps[-1]


def _next_frame(
    reader: PoseSource,
    last_t: int | None,
    last_time: float,
    *,
    clock: Clock = time,
    wait_cap: float = 2.0,
    dt_cap: float = 0.2,
    poll: float = 0.002,
) -> tuple[Pose | None, float, float]:
    deadline = clock.monotonic() + wait_cap
    while clock.monotonic() < deadline:
        pose = reader.get_latest()
        if pose is not None and pose.time_ms != last_t:
            now = clock.monotonic()
            return pose, min(now - last_time, dt_cap), now
        clock.sleep(poll)
    return None, 0.0, clock.monotonic()


@dataclass
class NavResult:
    reached: bool  # 経路が見つかった(到達可能)
    arrived: bool  # 最終ウェイポイント付近まで到達した
    reason: str  # "arrived" | "unreachable" | "no_pose" | "hud_lost" | "timeout"
    path: Path | None  # 計画した経路(follow() 直接指定時は None)
    elapsed: float  # 追従に要した秒
    frames: int  # 処理フレーム数
    yaw: AxisMetrics | None = (
        None  # 進行方向 yaw の応答指標(制御フレームが無ければ None)
    )


@dataclass
class AimResult:
    converged: bool  # yaw/pitch とも許容内を settle 回連続で達成した
    yaw_err: float  # 最終 yaw 誤差[deg]
    pitch_err: float  # 最終 pitch 誤差[deg]
    elapsed: float
    frames: int
    yaw: AxisMetrics | None = None  # yaw 軸の応答指標
    pitch: AxisMetrics | None = None  # pitch 軸の応答指標(pitch 未制御時は None)
    reason: str = ""  # "converged" | "timeout" | "hud_lost" | "stuck"(align のみ)


def follow_path(
    reader: PoseSource,
    look: LookActuator,
    move: MoveActuator,
    waypoints: list[tuple[float, float]],
    gains: PatrolGains,
    nav: NavControllers,
    *,
    clock: Clock = time,
    recorder: Recorder | None = None,
    name: str = "",
) -> NavResult:
    rec = recorder or NullRecorder()
    track = not isinstance(rec, NullRecorder)
    wps = list(waypoints)
    if not wps:
        return NavResult(True, True, "arrived", None, 0.0, 0)

    nav.yaw.reset()
    nav.forward.reset()
    cum = _cum_arclen(wps)
    total = cum[-1]
    seg = 0
    last_t: int | None = None
    last_time = t0 = clock.monotonic()
    frames = 0
    reason = "arrived"
    yaw_acc = AxisAccumulator() if track else None
    try:
        while True:
            pose, dt, now = _next_frame(reader, last_t, last_time, clock=clock)
            if pose is None:
                reason = "hud_lost"
                logger.warning("[%s] HUD lost, abort nav", name)
                break
            last_t, last_time = pose.time_ms, now
            frames += 1
            cur = (pose.position[0], pose.position[2])

            seg, s_proj = _project_arclen(cur, wps, cum, seg)
            end_dist = _dist(cur, wps[-1])
            if total - s_proj < gains.arrive and end_dist < gains.arrive:
                break
            carrot_s = s_proj + gains.nav_lookahead
            final = carrot_s >= total  # carrot が末端にクランプ=減速フェーズ
            target = _point_at_arclen(wps, cum, carrot_s)
            err, _ = heading_error(cur, pose.yaw_deg, target)

            turn = nav.yaw.update(err, dt)
            ff = forward_factor(err)
            speed = (nav.forward.update(end_dist, dt) if final else gains.speed) * ff
            look.look(turn)
            move.move(forward=speed)

            if track:
                rec.row(
                    t=now - t0,
                    phase="nav",
                    target=name,
                    wp=seg + 1,
                    dt=dt,
                    x=pose.position[0],
                    y=pose.position[1],
                    z=pose.position[2],
                    yaw=pose.yaw_deg,
                    pitch=pose.pitch_deg,
                    tx=target[0],
                    tz=target[1],
                    dist=end_dist,
                    yaw_err=err,
                    turn_p=nav.yaw.last_p,
                    turn_i=nav.yaw.last_i,
                    turn_d=nav.yaw.last_d,
                    turn=turn,
                    fwd=speed,
                    fwd_factor=ff,
                )
                yaw_acc.update(err, turn, now - t0, dt, gains.face_tol)
            if now - t0 > gains.nav_timeout:
                reason = "timeout"
                logger.warning("[%s] nav timeout", name)
                break
    finally:
        # 例外(Ctrl+C・OSC/マウスエラー等)で抜けてもアバターを確実に止める
        look.stop()
        move.stop()
    elapsed = clock.monotonic() - t0
    logger.debug(
        "[%s] nav end: %s wp=%d/%d frames=%d %.2fs",
        name,
        reason,
        seg + 1,
        len(wps),
        frames,
        elapsed,
    )
    return NavResult(
        reached=True,
        arrived=(reason == "arrived"),
        reason=reason,
        path=None,
        elapsed=elapsed,
        frames=frames,
        yaw=yaw_acc.snapshot() if (track and frames) else None,
    )


def follow_path_hold_view(
    reader: PoseSource,
    look: LookActuator,
    move: MoveActuator,
    waypoints: list[tuple[float, float]],
    gains: PatrolGains,
    ctl: TranslateControllers,
    *,
    clock: Clock = time,
    recorder: Recorder | None = None,
    name: str = "",
) -> NavResult:
    rec = recorder or NullRecorder()
    track = not isinstance(rec, NullRecorder)
    wps = list(waypoints)
    if not wps:
        return NavResult(True, True, "arrived", None, 0.0, 0)

    ctl.forward.reset()
    ctl.strafe.reset()
    idx = 1 if len(wps) > 1 else 0
    last_t: int | None = None
    last_time = t0 = clock.monotonic()
    frames = 0
    reason = "arrived"
    try:
        while idx < len(wps):
            pose, dt, now = _next_frame(reader, last_t, last_time, clock=clock)
            if pose is None:
                reason = "hud_lost"
                logger.warning("[%s] HUD lost, abort move", name)
                break
            last_t, last_time = pose.time_ms, now
            frames += 1
            cur = (pose.position[0], pose.position[2])

            prev_idx = idx
            while idx < len(wps) - 1 and _dist(cur, wps[idx]) < gains.arrive:
                idx += 1
            if idx != prev_idx:
                # 目標が急に変わったとき前後・左右の誤差が跳ねて D 項がスパイクするのを防ぐ
                ctl.forward.reset_derivative()
                ctl.strafe.reset_derivative()
            target = wps[idx]
            final = idx == len(wps) - 1
            dist = _dist(cur, target)
            if final and dist < gains.arrive:
                break

            # 目標への世界誤差を、現在の体の向きで前方向/右方向へ射影する(視点は回さない)
            ex, ez = target[0] - cur[0], target[1] - cur[1]
            yr = math.radians(pose.yaw_deg)
            fwd_err = ex * math.sin(yr) + ez * math.cos(yr)
            right_err = ex * math.cos(yr) - ez * math.sin(yr)

            fwd = ctl.forward.update(fwd_err, dt)
            strafe = ctl.strafe.update(right_err, dt)
            move.move(forward=fwd, strafe=strafe)
            look.look(0.0, 0.0)  # 視点はゼロ指令で保持(前フェーズの残留指令も打ち消す)

            if track:
                rec.row(
                    t=now - t0,
                    phase="move",
                    target=name,
                    wp=idx,
                    dt=dt,
                    x=pose.position[0],
                    y=pose.position[1],
                    z=pose.position[2],
                    yaw=pose.yaw_deg,
                    pitch=pose.pitch_deg,
                    tx=target[0],
                    tz=target[1],
                    dist=dist,
                    fwd_err=fwd_err,
                    right_err=right_err,
                    fwd=fwd,
                    strafe=strafe,
                )
            if now - t0 > gains.nav_timeout:
                reason = "timeout"
                logger.warning("[%s] move timeout", name)
                break
    finally:
        # 例外で抜けても移動・視点を確実に止める
        move.stop()
        look.stop()
    elapsed = clock.monotonic() - t0
    logger.debug(
        "[%s] move end: %s wp=%d/%d frames=%d %.2fs",
        name,
        reason,
        idx,
        len(wps),
        frames,
        elapsed,
    )
    return NavResult(
        reached=True,
        arrived=(reason == "arrived"),
        reason=reason,
        path=None,
        elapsed=elapsed,
        frames=frames,
    )


def _face_loop(
    reader: PoseSource,
    look: LookActuator,
    gains: PatrolGains,
    face: FaceControllers,
    *,
    errors: Callable[[Pose], tuple[float, float]],
    control_pitch: bool,
    phase: str,
    extra: dict[str, float],
    clock: Clock,
    recorder: Recorder | None,
    name: str,
) -> AimResult:
    """正対系ループの共通コア。errors(pose) が (yaw誤差, pitch誤差)[deg] を返す。

    control_pitch=False のときは pitch を制御せず(指令0)、収束判定も yaw のみ。
    extra は記録行に足す列(aim_at のターゲット座標など)。
    """
    rec = recorder or NullRecorder()
    track = not isinstance(
        rec, NullRecorder
    )  # 記録先が無ければ行組立も指標計算もしない
    face.yaw.reset()
    face.pitch.reset()
    last_t: int | None = None
    last_time = t0 = clock.monotonic()
    frames = 0
    settle = 0
    converged = False
    reason = "timeout"
    yaw_err = pitch_err = 0.0
    yaw_acc = AxisAccumulator() if track else None
    pitch_acc = AxisAccumulator() if (track and control_pitch) else None
    try:
        while clock.monotonic() - t0 < gains.face_timeout:
            pose, dt, now = _next_frame(reader, last_t, last_time, clock=clock)
            if pose is None:
                reason = "hud_lost"
                break
            last_t, last_time = pose.time_ms, now
            frames += 1
            yaw_err, pitch_err = errors(pose)

            pitch_ok = not control_pitch or abs(pitch_err) < gains.face_tol
            if abs(yaw_err) < gains.face_tol and pitch_ok:
                settle += 1
                if settle >= gains.settle:
                    converged = True
                    reason = "converged"
                    break
            else:
                settle = 0

            turn = face.yaw.update(yaw_err, dt)
            pitch_cmd = face.pitch.update(pitch_err, dt) if control_pitch else 0.0
            look.look(turn, pitch_cmd)

            if track:
                rec.row(
                    t=now - t0,
                    phase=phase,
                    target=name,
                    dt=dt,
                    x=pose.position[0],
                    y=pose.position[1],
                    z=pose.position[2],
                    yaw=pose.yaw_deg,
                    pitch=pose.pitch_deg,
                    **extra,
                    yaw_err=yaw_err,
                    pitch_err=pitch_err,
                    turn_p=face.yaw.last_p,
                    turn_i=face.yaw.last_i,
                    turn_d=face.yaw.last_d,
                    turn=turn,
                    pitch_p=face.pitch.last_p,
                    pitch_i=face.pitch.last_i,
                    pitch_d=face.pitch.last_d,
                    pitch_cmd=pitch_cmd,
                )
                yaw_acc.update(yaw_err, turn, now - t0, dt, gains.face_tol)
                if pitch_acc is not None:
                    pitch_acc.update(pitch_err, pitch_cmd, now - t0, dt, gains.face_tol)
    finally:
        # 例外で抜けても視点の回転を確実に止める
        look.stop()
    elapsed = clock.monotonic() - t0
    logger.debug(
        "[%s] %s end: %s yaw_err=%+.2f pitch_err=%+.2f frames=%d %.2fs",
        name,
        phase,
        reason,
        yaw_err,
        pitch_err,
        frames,
        elapsed,
    )
    return AimResult(
        converged=converged,
        yaw_err=yaw_err,
        pitch_err=pitch_err,
        elapsed=elapsed,
        frames=frames,
        yaw=yaw_acc.snapshot() if (track and frames) else None,
        pitch=pitch_acc.snapshot() if (pitch_acc is not None and frames) else None,
        reason=reason,
    )


def strafe_align(
    reader: PoseSource,
    look: LookActuator,
    move: MoveActuator,
    target_xyz: tuple[float, float, float],
    gains: PatrolGains,
    face: FaceControllers,
    strafe: AxisController,
    *,
    clock: Clock = time,
    recorder: Recorder | None = None,
    name: str = "",
) -> AimResult:
    """最終照準: 視点(yaw)は回さず、体の横移動で誤差を潰す。

    視点軸は指令0.50以下が効かず、不感帯補償するとリミットサイクルになる
    (gain-tuning.md 参照)。移動軸は連続的に効くため、粗い正対(aim_at)後の
    残り yaw 誤差を横ずれ e = dist·sin(yaw_err)[m] に換算して横移動で吸収
    すれば発振が原理的に出ない。pitch は従来どおり視点で合わせる。

    収束: |e| < align_tol かつ |pitch_err| < face_tol を settle 回連続。
    角のボタン等で移動方向が壁に塞がれた場合は、指令を出しても
    align_stuck_time 秒間 align_stuck_eps[m] 以上動けないことを検出して
    打ち切る(reason="stuck"。デッドロック防止)。
    """
    rec = recorder or NullRecorder()
    track = not isinstance(rec, NullRecorder)
    face.pitch.reset()
    strafe.reset()
    tgt_xz = (target_xyz[0], target_xyz[2])
    last_t: int | None = None
    last_time = t0 = clock.monotonic()
    frames = 0
    settle = 0
    converged = False
    reason = "timeout"
    yaw_err = pitch_err = 0.0
    lat_acc = AxisAccumulator() if track else None
    pitch_acc = AxisAccumulator() if track else None
    # スタック検出: 窓の開始時刻・窓内の移動経路長Σ|Δpos|・指令を出したかを追跡する
    # (経路長で見ると、その場往復や微速クロールを「動けていない」と誤判定しない)
    win_t = t0
    win_prev: tuple[float, float] | None = None
    win_path = 0.0
    win_commanded = False
    try:
        while clock.monotonic() - t0 < gains.align_timeout:
            pose, dt, now = _next_frame(reader, last_t, last_time, clock=clock)
            if pose is None:
                reason = "hud_lost"
                break
            last_t, last_time = pose.time_ms, now
            frames += 1
            cur = (pose.position[0], pose.position[2])
            yaw_err, dist = heading_error(cur, pose.yaw_deg, tgt_xz)
            lat_err = dist * math.sin(math.radians(yaw_err))  # +なら目標が右
            pitch_err = pitch_error(pose.position, pose.forward, target_xyz)

            if abs(lat_err) < gains.align_tol and abs(pitch_err) < gains.face_tol:
                settle += 1
                if settle >= gains.settle:
                    converged = True
                    reason = "converged"
                    break
            else:
                settle = 0

            # 目標が右(+)なら右へ動くと視線上に乗る
            strafe_cmd = strafe.update(lat_err, dt)
            pitch_cmd = face.pitch.update(pitch_err, dt)
            move.move(strafe=strafe_cmd)
            look.look(0.0, pitch_cmd)

            # スタック検出(壁に押し付けて動けない)
            if win_prev is None:
                win_t, win_prev = now, cur
            else:
                win_path += _dist(cur, win_prev)
                win_prev = cur
            win_commanded = win_commanded or abs(strafe_cmd) > 1e-3
            if now - win_t >= gains.align_stuck_time:
                if win_commanded and win_path < gains.align_stuck_eps:
                    reason = "stuck"
                    break
                win_t, win_prev, win_path, win_commanded = now, cur, 0.0, False

            if track:
                rec.row(
                    t=now - t0,
                    phase="align",
                    target=name,
                    dt=dt,
                    x=pose.position[0],
                    y=pose.position[1],
                    z=pose.position[2],
                    yaw=pose.yaw_deg,
                    pitch=pose.pitch_deg,
                    tx=target_xyz[0],
                    ty=target_xyz[1],
                    tz=target_xyz[2],
                    dist=dist,
                    yaw_err=yaw_err,
                    pitch_err=pitch_err,
                    lat_err=lat_err,
                    strafe_p=strafe.last_p,
                    strafe_i=strafe.last_i,
                    strafe_d=strafe.last_d,
                    strafe=strafe_cmd,
                    pitch_p=face.pitch.last_p,
                    pitch_i=face.pitch.last_i,
                    pitch_d=face.pitch.last_d,
                    pitch_cmd=pitch_cmd,
                )
                lat_acc.update(lat_err, strafe_cmd, now - t0, dt, gains.align_tol)
                pitch_acc.update(pitch_err, pitch_cmd, now - t0, dt, gains.face_tol)
    finally:
        # 例外で抜けても移動・視点を確実に止める
        move.stop()
        look.stop()
    elapsed = clock.monotonic() - t0
    logger.debug(
        "[%s] align end: %s yaw_err=%+.2f pitch_err=%+.2f frames=%d %.2fs",
        name,
        reason,
        yaw_err,
        pitch_err,
        frames,
        elapsed,
    )
    return AimResult(
        converged=converged,
        yaw_err=yaw_err,
        pitch_err=pitch_err,
        elapsed=elapsed,
        frames=frames,
        yaw=lat_acc.snapshot() if (track and frames) else None,  # 誤差=横ずれ[m]
        pitch=pitch_acc.snapshot() if (track and frames) else None,
        reason=reason,
    )


def aim_at(
    reader: PoseSource,
    look: LookActuator,
    target_xyz: tuple[float, float, float],
    gains: PatrolGains,
    face: FaceControllers,
    *,
    clock: Clock = time,
    recorder: Recorder | None = None,
    name: str = "",
) -> AimResult:
    tgt_xz = (target_xyz[0], target_xyz[2])

    def errors(pose: Pose) -> tuple[float, float]:
        cur = (pose.position[0], pose.position[2])
        yaw_err, _ = heading_error(cur, pose.yaw_deg, tgt_xz)
        return yaw_err, pitch_error(pose.position, pose.forward, target_xyz)

    return _face_loop(
        reader,
        look,
        gains,
        face,
        errors=errors,
        control_pitch=True,
        phase="face",
        extra={"tx": target_xyz[0], "ty": target_xyz[1], "tz": target_xyz[2]},
        clock=clock,
        recorder=recorder,
        name=name,
    )


def turn_to(
    reader: PoseSource,
    look: LookActuator,
    yaw_deg: float,
    gains: PatrolGains,
    face: FaceControllers,
    *,
    pitch_deg: float | None = None,
    clock: Clock = time,
    recorder: Recorder | None = None,
    name: str = "",
) -> AimResult:
    def errors(pose: Pose) -> tuple[float, float]:
        yaw_err = wrap180(yaw_deg - pose.yaw_deg)
        return yaw_err, 0.0 if pitch_deg is None else (pitch_deg - pose.pitch_deg)

    return _face_loop(
        reader,
        look,
        gains,
        face,
        errors=errors,
        control_pitch=pitch_deg is not None,
        phase="turn",
        extra={},
        clock=clock,
        recorder=recorder,
        name=name,
    )
