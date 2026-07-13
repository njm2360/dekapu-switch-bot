import math
from dataclasses import dataclass, field


def wrap180(deg: float) -> float:
    """角度を (-180, 180] に正規化(最短回りの誤差に使う)。"""
    return (deg + 180.0) % 360.0 - 180.0


@dataclass
class PID:
    """離散 PID。

    update(error, dt) を毎周期呼ぶ。出力は [out_min, out_max] にクランプ。
    積分項は i_limit で絶対値制限し、出力飽和中は積分を止める(anti-windup)。
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    out_min: float = -1.0
    out_max: float = 1.0
    i_limit: float = 1.0  # 積分項(ki*∫e)の絶対値上限
    # 出力デッドゾーン補償。>0 なら非ゼロ出力を最低でも out_knee まで押し上げる
    # (VRChat の視点軸のように、閾値以下がほぼ無反応な非線形を打ち消す)。
    out_knee: float = 0.0
    _i: float = field(default=0.0, init=False, repr=False)
    _prev: float | None = field(default=None, init=False, repr=False)
    # 直近 update() の内訳(ログ/デバッグ用)
    last_p: float = field(default=0.0, init=False, repr=False)
    last_i: float = field(default=0.0, init=False, repr=False)
    last_d: float = field(default=0.0, init=False, repr=False)
    last_out: float = field(default=0.0, init=False, repr=False)

    def reset(self) -> None:
        self._i = 0.0
        self._prev = None
        self.last_p = self.last_i = self.last_d = self.last_out = 0.0

    def reset_derivative(self) -> None:
        """微分履歴だけをリセット(積分は保持)。目標が急変する時の微分キック抑制用。"""
        self._prev = None

    def update(self, error: float, dt: float) -> float:
        p = self.kp * error

        # 微分(計測誤差ノイズに素直。必要なら呼び出し側で平滑化)
        d = 0.0
        if self._prev is not None and dt > 0.0:
            d = self.kd * (error - self._prev) / dt
        self._prev = error

        # まず P+D と現在の積分で仮出力を作り、飽和していなければ積分を進める
        unsat = p + self.ki * self._i + d
        if dt > 0.0 and self.ki != 0.0:
            if self.out_min < unsat < self.out_max or (error * self._i) < 0.0:
                self._i += error * dt
                i_term = self.ki * self._i
                if i_term > self.i_limit:
                    self._i = self.i_limit / self.ki
                elif i_term < -self.i_limit:
                    self._i = -self.i_limit / self.ki

        i = self.ki * self._i
        out = max(self.out_min, min(self.out_max, p + i + d))
        # デッドゾーン補償: 非ゼロ出力を最低 out_knee まで押し上げ、無反応域を飛ばす。
        # 生出力 |out|→1 で 1、微小出力で out_knee ちょうど(符号は保持)。
        if self.out_knee > 0.0 and abs(out) > 1e-3:
            out = math.copysign(self.out_knee + (1.0 - self.out_knee) * min(abs(out), 1.0), out)
        self.last_p, self.last_i, self.last_d, self.last_out = p, i, d, out
        return out
