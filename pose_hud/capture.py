import sys
from typing import Protocol

import numpy as np

from .spec import DEFAULT_SPEC, GridSpec


class FrameSource(Protocol):
    """グリッドを内包する領域(クライアント左上原点)を返すもの。"""

    def grab(self) -> np.ndarray:
        """HxWx3(以上) の uint8 画像を返す。frame[0,0] がクライアント左上。"""
        ...

    def close(self) -> None: ...


class ArrayFrameSource:
    """固定 numpy 配列を返すテスト/リプレイ用ソース。差し替え可能。"""

    def __init__(self, frame: np.ndarray):
        self._frame = frame

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame = frame

    def grab(self) -> np.ndarray:
        return self._frame

    def close(self) -> None:
        pass


def _enable_dpi_awareness() -> None:
    """プロセスを Per-Monitor DPI Aware にし、物理ピクセルで矩形を得られるようにする。

    VRChat はネイティブ解像度で描くため、論理ピクセルへ丸められるとブロック境界が壊れる。
    """
    import ctypes

    # Per-Monitor v2 (-4) -> Per-Monitor (2) -> System と順にフォールバック
    try:
        ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def find_window_rect(title: str = "VRChat") -> tuple[int, int, int, int] | None:
    """タイトルからウィンドウを探し、クライアント領域の (left, top, width, height) を
    スクリーン座標(物理px)で返す。見つからなければ None。"""
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32

    # まず完全一致で FindWindow、ダメなら可視ウィンドウを走査して部分一致
    hwnd = user32.FindWindowW(None, title)
    if not hwnd:
        hwnd = _find_window_substring(title)
    if not hwnd:
        return None

    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    width, height = rect.right - rect.left, rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return None  # 最小化中など

    pt = wintypes.POINT(0, 0)
    if not user32.ClientToScreen(hwnd, ctypes.byref(pt)):
        return None
    return pt.x, pt.y, width, height


def _find_window_substring(needle: str):
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    needle_low = needle.lower()
    found = []

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def _cb(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        if needle_low in buf.value.lower():
            found.append(hwnd)
            return False
        return True

    user32.EnumWindows(WNDENUMPROC(_cb), 0)
    return found[0] if found else None


class WindowsVRChatCapture:
    """VRChat のクライアント左上付近の小領域を mss で高速キャプチャする FrameSource。

    グリッド(64x24px)を内包する capture_w x capture_h 領域のみを掴むので 60fps を狙える。
    ウィンドウ矩形はキャッシュし、掴めなくなったら再解決する。
    """

    def __init__(self, spec: GridSpec = DEFAULT_SPEC, window_title: str = "VRChat"):
        if sys.platform != "win32":
            raise RuntimeError(
                "WindowsVRChatCapture is Windows-only; inject a FrameSource"
            )
        import mss  # 遅延 import(テスト環境に mss/win 依存を持ち込まない)

        _enable_dpi_awareness()
        self.spec = spec
        self.window_title = window_title
        self._sct = mss.mss()
        self._rect: tuple[int, int, int, int] | None = None

    def _resolve_rect(self) -> tuple[int, int, int, int] | None:
        rect = find_window_rect(self.window_title)
        self._rect = rect
        return rect

    def grab(self) -> np.ndarray:
        rect = self._rect or self._resolve_rect()
        if rect is None:
            raise WindowNotFoundError(f'window "{self.window_title}" not found')
        left, top, cw, ch = rect
        region = {
            "left": left,
            "top": top,
            "width": min(self.spec.capture_w, cw),
            "height": min(self.spec.capture_h, ch),
        }
        try:
            shot = self._sct.grab(region)
        except Exception:
            # ウィンドウが移動/クローズした可能性。次回再解決させる。
            self._rect = None
            raise
        # mss は BGRA。RGB和で二値化するのでチャンネル順は不問。先頭3chのみ使う。
        return np.asarray(shot)[:, :, :3]

    def refresh_window(self) -> None:
        """ウィンドウ矩形を強制再解決する(解像度/位置変更後に呼ぶ)。"""
        self._resolve_rect()

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


class WindowNotFoundError(RuntimeError):
    """VRChat ウィンドウが見つからない。"""
