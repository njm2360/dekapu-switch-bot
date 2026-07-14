from pythonosc.udp_client import SimpleUDPClient


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


class VRChatOSC:
    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self.client = SimpleUDPClient(host, port)

    # ---- 連続軸(-1..1) ------------------------------------------------
    def axis(self, name: str, value: float) -> None:
        self.client.send_message(f"/input/{name}", _clamp(float(value)))

    def move(self, forward: float = 0.0, strafe: float = 0.0) -> None:
        """前後(forward)と左右ストレイフ(strafe)を同時指定。"""
        self.axis("Vertical", forward)
        self.axis("Horizontal", strafe)

    def look(self, turn: float = 0.0, pitch: float = 0.0) -> None:
        """水平旋回(+で右)。pitch を与えると上下視点も動かす。"""
        self.axis("LookHorizontal", turn)
        if pitch:
            self.look_vertical(pitch)

    def look_vertical(self, pitch: float = 0.0) -> None:
        """上下視点。+で上。"""
        self.axis("LookVertical", pitch)

    def stop(self) -> None:
        """移動・旋回を全停止(軸を0に戻す)。"""
        self.move(0.0, 0.0)
        self.axis("LookHorizontal", 0.0)
        self.axis("LookVertical", 0.0)

    # ---- ボタン(0/1) --------------------------------------------------
    def button(self, name: str, pressed: bool) -> None:
        self.client.send_message(f"/input/{name}", 1 if pressed else 0)

    def jump(self) -> None:
        self.button("Jump", True)
        self.button("Jump", False)

    # ---- アバターパラメータ --------------------------------------------
    def avatar_param(self, name: str, value) -> None:
        self.client.send_message(f"/avatar/parameters/{name}", value)

    def hud_enable(self, on: bool = True) -> None:
        self.avatar_param("HUD_Enable", bool(on))

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "VRChatOSC":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
