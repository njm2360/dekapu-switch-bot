"""操作アクチュエータ: 視点(look)・移動(move)・押下(interact)を独立に差し替えるための IF。

制御ループは LookActuator / MoveActuator へ指令値 [-1,1] を出すだけで、実際の注入方法
(OSC / DirectInput)は実装側が吸収する。look と move は別プロトコルなので、視点は
マウス・移動は OSC、のように片方だけ差し替えられる。

InteractActuator は連続軸ではなく単発の押下(press/release/click)なので、
PoseSource や gains に依存しない。呼び出し側(Pilot.click 等)が都度どちらの実装
(OSC の /input/UseRight か、マウスクリックか)を渡すかを選べる。

VRChat の HUD 表示切替(`/avatar/parameters/HUD_Enable`)はアクチュエータではなく OSC
固有の操作なので、ここには含めない(`osc.VRChatOSC.hud_enable` を使う)。

`osc.VRChatOSC` は look / move / interact / stop を備えるため、全プロトコルをそのまま
満たす(OSC 経由ならアダプタ不要)。
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class LookActuator(Protocol):
    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class MoveActuator(Protocol):
    def move(self, forward: float = 0.0, strafe: float = 0.0) -> None: ...

    def stop(self) -> None: ...


@runtime_checkable
class InteractActuator(Protocol):
    def press(self) -> None: ...

    def release(self) -> None: ...

    def click(self) -> None: ...


class MouseLookActuator:
    """DirectInput(相対マウス移動)で視点を操作する LookActuator。

    制御指令 [-1,1] を、1フレームあたりのマウス移動量[px]へ線形に変換する
    (実質的には速度指令として働く)。
    VRChat デスクトップのマウス視点は加速なしが前提。操作するにはウィンドウに
    フォーカスが必要。マウスには OSC の視点軸のような不感帯が無いので、PID の
    out_deadzone は 0 でよい(ゲイン[px/指令]は OSC 版とは別物。実機で要校正)。

    ``move_rel`` を差し替えるとテストできる(既定は ``pydirectinput.moveRel``)。
    """

    def __init__(
        self,
        yaw_gain: float = 40.0,
        pitch_gain: float = 40.0,
        invert_pitch: bool = True,  # 画面Yは下が正。pitch+(上)は dy<0
        move_rel=None,
    ):
        if move_rel is None:
            import pydirectinput

            pydirectinput.PAUSE = 0.0
            move_rel = pydirectinput.moveRel
        self.yaw_gain = yaw_gain
        self.pitch_gain = pitch_gain
        self.invert_pitch = invert_pitch
        self._move_rel = move_rel

    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None:
        dx = int(round(turn * self.yaw_gain))
        dy = int(round(pitch * self.pitch_gain))
        if self.invert_pitch:
            dy = -dy
        if dx or dy:
            self._move_rel(dx, dy)

    def stop(self) -> None:
        pass


class MouseClickActuator:
    """pydirectinput の左クリックで interact する InteractActuator。"""

    def __init__(self):
        import pydirectinput

        pydirectinput.PAUSE = 0.0
        self._pdi = pydirectinput

    def press(self) -> None:
        self._pdi.mouseDown()

    def release(self) -> None:
        self._pdi.mouseUp()

    def click(self) -> None:
        self._pdi.click()
