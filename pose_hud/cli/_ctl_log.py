"""制御ループの詳細ログ(CSV)。PID チューニング・ぎこちなさのデバッグ用。

毎tick(新フレームごと)の姿勢・誤差・PID内訳・OSC指令を1行ずつ書き出す。
`ControlLog(path)` で有効化、`ControlLog(None)`(= NullLog)で無効。
"""

import csv


FIELDS = [
    "t",          # 開始からの経過秒
    "phase",      # nav / face
    "target",     # ターゲット名
    "wp",         # 追従中のウェイポイント番号(navのみ)
    "dt",         # 前フレームからの実経過秒
    "x", "y", "z",
    "yaw", "pitch",
    "tx", "ty", "tz",
    "dist",       # ターゲットまでの水平距離[m]
    "yaw_err", "pitch_err",
    "turn_p", "turn_i", "turn_d", "turn",   # yaw(LookHorizontal)PID内訳と出力
    "pitch_p", "pitch_i", "pitch_d", "pitch_cmd",
    "fwd",        # Vertical(前進)出力
    "fwd_factor",  # 向きズレによる前進減衰係数
]


class ControlLog:
    """CSV に制御状態を書き出すロガー。"""

    def __init__(self, path):
        self.path = path
        self._f = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(self._f, fieldnames=FIELDS, extrasaction="ignore")
        self._w.writeheader()

    def row(self, **kw) -> None:
        self._w.writerow({k: kw.get(k, "") for k in FIELDS})
        self._f.flush()   # クラッシュしても直近まで残す

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


class NullLog:
    """何もしないロガー(--no-log 時)。"""

    path = None

    def row(self, **kw) -> None:
        pass

    def close(self) -> None:
        pass


def make_log(path):
    """path が None なら NullLog、そうでなければ CSV ロガー。"""
    return NullLog() if path is None else ControlLog(path)
