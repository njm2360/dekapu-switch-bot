"""VRChat PoseTelemetryHUD デコーダの CLI デモ。

pose_hud パッケージを使い、VRChat ウィンドウから 6DoF(位置・向き)を読み取って
標準出力に表示する。実運用ではライブラリとして ``from pose_hud import PoseReader``
を import して使うこと(この main.py は動作確認・キャリブレーション用のフロントエンド)。

前提:
    - Windows で VRChat をネイティブ解像度・ウィンドウ表示で起動していること。
    - アバターの HUD を OSC で ON にすること
      (`/avatar/parameters/HUD_Enable = true`)。OFF だと MAGIC 不一致で読めない。
    - ゲーム内メニューを開くと HUD が隠れて読めない(異常ではない)。

使い方:
    uv run main.py                       # VRChat ウィンドウを読み続けて表示
    uv run main.py --stats               # 1秒ごとに統計(fps/成功率など)も表示
    uv run main.py --dump out/dbg        # 連続失敗時にデバッグダンプを保存
    uv run main.py --window "VRChat"     # キャプチャ対象ウィンドウタイトルを指定
    uv run main.py --offset-x 16 --block 4   # グリッド定数を注入(キャリブレーション)
    # 停止は Ctrl+C。

オプション:
    --window TITLE      キャプチャ対象ウィンドウタイトル(既定: "VRChat")
    --stats             1秒ごとに ReaderStats(実効fps/キャプチャfps/成功率/重複数/
                        連続失敗数)を表示する
    --dump PREFIX       連続失敗が --warn-after を超えたら、直近フレームを
                        <PREFIX>.npy / .txt(あれば .png)に1回だけ保存する。
                        グリッドが見つからないときの目視確認・キャリブレーション用
    --warn-after N      連続失敗をこの回数超えたら警告(+ダンプ)。既定: 120
    --offset-x N        グリッド左上Xオフセット(px)。既定: 8(確定仕様値)
    --offset-y N        グリッド左上Yオフセット(px)。既定: 8(確定仕様値)
    --block N           1ビットの一辺(px)。既定: 2(確定仕様値)
                        ※ offset/block はシェーダー側と一致必須。既定値の変更は要相談。

出力の見かた:
    pos=( +1.500,  -2.250, +42.000)  yaw= +12.34  pitch= -5.67  t=123456
      pos … カメラのワールド座標 [m](Unity: Y-up, 左手系)
      yaw … +Z 基準の水平角 [deg]、pitch … 上向き正の仰角 [deg]
      t   … _VRChatTimeNetworkMs(フレーム識別用。ラップあり)

終了時に総グラブ数・新規フレーム数・成功率のサマリを表示する。
"""

import argparse
import logging
import time

from pose_hud import PoseReader, WindowNotFoundError


def main() -> None:
    from pose_hud.spec import DEFAULT_SPEC, GridSpec

    parser = argparse.ArgumentParser(description="VRChat 6DoF HUD decoder demo")
    parser.add_argument("--window", default="VRChat", help="キャプチャ対象ウィンドウタイトル")
    parser.add_argument("--stats", action="store_true", help="統計を定期表示")
    parser.add_argument("--dump", metavar="PREFIX", help="連続失敗時のデバッグダンプ保存先プレフィックス")
    parser.add_argument("--warn-after", type=int, default=120, help="連続失敗警告のしきい値")
    # グリッド定数の注入(既定は CLAUDE.md の確定仕様値。キャリブレーション用に上書き可)
    parser.add_argument("--offset-x", type=int, default=DEFAULT_SPEC.offset_x, help="グリッドXオフセット(px)")
    parser.add_argument("--offset-y", type=int, default=DEFAULT_SPEC.offset_y, help="グリッドYオフセット(px)")
    parser.add_argument("--block", type=int, default=DEFAULT_SPEC.block, help="1ビットの一辺(px)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    spec = GridSpec(offset_x=args.offset_x, offset_y=args.offset_y, block=args.block)
    if (spec.offset_x, spec.offset_y, spec.block) != (
        DEFAULT_SPEC.offset_x, DEFAULT_SPEC.offset_y, DEFAULT_SPEC.block
    ):
        print(f"[spec] injected offset=({spec.offset_x},{spec.offset_y}) block={spec.block} "
              f"capture={spec.capture_w}x{spec.capture_h}")

    from pose_hud.capture import WindowsVRChatCapture

    try:
        source = WindowsVRChatCapture(spec, window_title=args.window)
    except RuntimeError as exc:
        parser.error(str(exc))

    dumped = False

    def on_warning(stats) -> None:
        nonlocal dumped
        if args.dump and not dumped:
            path = reader.dump_debug(args.dump)
            print(f"[debug] dumped capture to {path.with_suffix('.npy')} (+.txt/.png)")
            dumped = True

    reader = PoseReader(source=source, spec=spec, on_frame=None, on_warning=on_warning,
                        warn_after=args.warn_after)
    reader.start()
    print("reading VRChat HUD... (Ctrl+C to stop)")

    last_stats = time.monotonic()
    try:
        for pose in reader.poses():
            dumped = False  # 有効フレームが来たらダンプ抑止を解除
            print(
                f"pos=({pose.position[0]:+8.3f}, {pose.position[1]:+8.3f}, "
                f"{pose.position[2]:+8.3f})  yaw={pose.yaw_deg:+7.2f}  "
                f"pitch={pose.pitch_deg:+6.2f}  t={pose.time_ms}"
            )
            if args.stats and time.monotonic() - last_stats >= 1.0:
                s = reader.get_stats()
                print(
                    f"  [stats] frame_fps={s.frame_fps:5.1f} capture_fps={s.capture_fps:5.1f} "
                    f"ok={s.success_rate:5.1%} dup={s.duplicate_skipped} "
                    f"consec_fail={s.consecutive_fail}"
                )
                last_stats = time.monotonic()
    except KeyboardInterrupt:
        pass
    except WindowNotFoundError as exc:
        print(f"error: {exc}")
    finally:
        reader.stop()
        s = reader.get_stats()
        print(f"\nstopped. grabbed={s.frames_grabbed} new_frames={s.new_frames} "
              f"ok_rate={s.success_rate:.1%}")


if __name__ == "__main__":
    main()
