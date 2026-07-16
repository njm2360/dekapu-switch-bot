"""同定済みプラント(plant.json)上で正対ループを回すオフラインゲイン検証 CLI。

本番と同じ turn_to を回し、初期誤差ごとの収束時間・振動・オーバーシュートを表にする。
PID ゲインは patrol-buttons と同じフラグで上書きできる(実機前の当たり付け用)。
"""

import argparse
import dataclasses
from pathlib import Path

from app.cli._logging import setup_logging
from app.control.telemetry import ControlLog, ListRecorder
from app.cli.patrol_buttons import _add_gain_args
from app.control.controller import PatrolGains, face_controllers
from app.control.maneuvers import turn_to
from app.sysid.simplant import SimClock, SimulatedVRChat
from app.sysid.identify import PlantModel


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(
        description="plant.json 上で正対(turn_to)ループを回しゲインを検証する"
    )
    parser.add_argument(
        "--model", required=True, help="probe-axes が出力した plant.json"
    )
    parser.add_argument(
        "--yaw-err",
        default="30,10,5,2",
        help="試す初期 yaw 誤差[deg]のCSV",
    )
    parser.add_argument(
        "--pitch-err",
        type=float,
        default=None,
        help="初期 pitch 誤差[deg](省略時は yaw のみ制御)",
    )
    parser.add_argument(
        "--log", default=None, help="全試行のフレーム記録 CSV の出力先(任意)"
    )
    _add_gain_args(parser)
    args = parser.parse_args()

    gains = PatrolGains(
        **{f.name: getattr(args, f.name) for f in dataclasses.fields(PatrolGains)}
    )
    plant = PlantModel.load(args.model)
    errs = [float(v) for v in args.yaw_err.split(",") if v.strip()]

    recorder = ControlLog(Path(args.log)) if args.log else ListRecorder()
    print(f"model: {args.model}  (dt {plant.dt_mean * 1000:.0f} ms)")
    print(
        f"gains: kp={gains.turn_kp} ki={gains.turn_ki} kd={gains.turn_kd} "
        f"deadzone={gains.turn_deadzone} tol={gains.face_tol}°"
    )
    print(
        f"{'err':>8}  {'result':6}  {'time':>6}  {'final':>7}  "
        f"{'osc':>3}  {'overshoot':>9}  {'settle':>6}"
    )
    ok = 0
    for err in errs:
        sim = SimulatedVRChat(plant)
        res = turn_to(
            sim,
            sim,
            err,  # sim は yaw=0 で始まるので目標=初期誤差
            gains,
            face_controllers(gains),
            pitch_deg=args.pitch_err,  # 同上(pitch も 0 始まり)
            clock=SimClock(sim),
            recorder=recorder,
            name=f"err{err:g}",
        )
        ok += res.converged
        m = res.yaw
        print(
            f"{err:+7.1f}°  {'OK' if res.converged else 'NG':6}  "
            f"{res.elapsed:5.2f}s  {res.yaw_err:+6.2f}°  "
            f"{m.osc if m else '-':>3}  "
            f"{f'{m.overshoot:.2f}°' if m else '-':>9}  "
            f"{f'{m.settle_time:.2f}s' if m and m.settle_time is not None else '-':>6}"
        )
    print(f"\nconverged {ok}/{len(errs)}")
    if args.log:
        recorder.close()
        print(f"frame log: {args.log}")


if __name__ == "__main__":
    main()
