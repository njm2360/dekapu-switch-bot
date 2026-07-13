import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterator

import numpy as np

from .capture import FrameSource
from .decode import DecodeResult, DecodeStatus, Pose, decode_pose
from .spec import DEFAULT_SPEC, GridSpec

logger = logging.getLogger("pose_hud")


def _fps_from(times: deque[float]) -> float:
    """monotonic タイムスタンプ列から実効fpsを推定する。"""
    if len(times) < 2:
        return 0.0
    span = times[-1] - times[0]
    return (len(times) - 1) / span if span > 0 else 0.0


@dataclass
class ReaderStats:
    """読み取り統計のスナップショット。"""

    frames_grabbed: int = 0  # キャプチャ回数(重複含む総グラブ数)
    decode_ok: int = 0  # 検証OK(重複含む)
    decode_fail: int = 0  # MAGIC/チェックサム不一致
    new_frames: int = 0  # 新規(time_ms 更新)ポーズ数
    duplicate_skipped: int = 0  # 同一 time_ms の二重読み
    consecutive_fail: int = 0  # 連続デコード失敗数(成功でリセット)
    last_status: DecodeStatus | None = None
    capture_fps: float = 0.0  # グラブのスループット
    frame_fps: float = 0.0  # 新規ポーズの実効fps
    magic_mismatch: int = 0
    checksum_mismatch: int = 0

    @property
    def success_rate(self) -> float:
        """検証OK率(全グラブに対する割合)。"""
        total = self.decode_ok + self.decode_fail
        return self.decode_ok / total if total else 0.0


class PoseReader:
    """VRChat HUD からポーズを読み続けるリーダ。

    使い方::

        reader = PoseReader()              # 既定で WindowsVRChatCapture を使用
        reader.start()
        pose = reader.get_latest()         # 最新ポーズ(なければ None)
        for pose in reader.poses():        # 新フレームをブロッキング取得
            ...
        reader.stop()

    テストや非Windows環境では ``source=ArrayFrameSource(frame)`` を注入する。
    """

    def __init__(
        self,
        source: FrameSource | None = None,
        spec: GridSpec = DEFAULT_SPEC,
        on_frame: Callable[[Pose], None] | None = None,
        on_warning: Callable[[ReaderStats], None] | None = None,
        warn_after: int = 120,
        poll_interval: float = 0.0,
        stats_window: int = 120,
    ):
        """source 未指定なら WindowsVRChatCapture を遅延生成する。

        warn_after: 連続失敗がこの数を超えたら on_warning を1回発火し警告ログを出す
                    (メニュー開きっぱなし・HUD_Enable=false の検出)。
        poll_interval: グラブ間の待機秒。0 なら全力(mssのグラブ律速)。
        """
        if source is None:
            from .capture import WindowsVRChatCapture

            source = WindowsVRChatCapture(spec)
        self.source = source
        self.spec = spec
        self.on_frame = on_frame
        self.on_warning = on_warning
        self.warn_after = warn_after
        self.poll_interval = poll_interval

        self.stats = ReaderStats()
        self._latest: Pose | None = None
        self._latest_result: DecodeResult | None = None
        self._last_frame: np.ndarray | None = None
        self._last_time_ms: int | None = None

        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._warned = False
        self._queue: queue.Queue[Pose] = queue.Queue()

        self._capture_times: deque[float] = deque(maxlen=stats_window)
        self._frame_times: deque[float] = deque(maxlen=stats_window)

    # ---- ライフサイクル -------------------------------------------------
    def start(self) -> "PoseReader":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="PoseReader", daemon=True
        )
        self._thread.start()
        return self

    def stop(self, join: bool = True, timeout: float = 2.0) -> None:
        self._stop.set()
        if join and self._thread:
            self._thread.join(timeout)
        self.source.close()

    def __enter__(self) -> "PoseReader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # ---- 取得API -------------------------------------------------------
    def get_latest(self) -> Pose | None:
        """直近の有効ポーズ。まだ無ければ None。スレッドセーフ。"""
        with self._lock:
            return self._latest

    def get_stats(self) -> ReaderStats:
        """統計のコピーを返す。"""
        with self._lock:
            return replace(self.stats)

    def poses(self, timeout: float | None = None) -> Iterator[Pose]:
        """新規ポーズをブロッキングで yield し続けるジェネレータ。

        timeout 秒新フレームが来なければ終了(None なら stop() まで無限)。
        """
        while not self._stop.is_set():
            try:
                yield self._queue.get(timeout=timeout if timeout is not None else 0.5)
            except queue.Empty:
                if timeout is not None:
                    return
                continue

    # ---- 内部ループ ----------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> DecodeResult:
        """1フレームをデコードして統計・状態を更新する(単体テスト可能)。"""
        now = time.monotonic()
        result = decode_pose(frame, self.spec)

        with self._lock:
            self.stats.frames_grabbed += 1
            self.stats.last_status = result.status
            self._last_frame = frame
            self._capture_times.append(now)
            self._update_fps()

            if result.ok:
                self.stats.decode_ok += 1
                self.stats.consecutive_fail = 0
                self._warned = False
                pose = result.pose
                assert pose is not None
                if pose.time_ms == self._last_time_ms:
                    self.stats.duplicate_skipped += 1
                else:
                    self._last_time_ms = pose.time_ms
                    self._latest = pose
                    self._latest_result = result
                    self.stats.new_frames += 1
                    self._frame_times.append(now)
                    self._emit(pose)
            else:
                self.stats.decode_fail += 1
                self.stats.consecutive_fail += 1
                if result.status is DecodeStatus.MAGIC_MISMATCH:
                    self.stats.magic_mismatch += 1
                else:
                    self.stats.checksum_mismatch += 1
                self._maybe_warn()

        return result

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.source.grab()
            except Exception as exc:  # noqa: BLE001 - ウィンドウ消失等は回復対象
                logger.debug("grab failed: %s", exc)
                self._stop.wait(0.2)
                continue
            self.process_frame(frame)
            if self.poll_interval:
                self._stop.wait(self.poll_interval)

    # ---- ヘルパ(ロック保持前提) ---------------------------------------
    def _emit(self, pose: Pose) -> None:
        self._queue.put(pose)
        if self.on_frame is not None:
            try:
                self.on_frame(pose)
            except Exception:  # noqa: BLE001 - コールバックの失敗でループを止めない
                logger.exception("on_frame callback raised")

    def _maybe_warn(self) -> None:
        if not self._warned and self.stats.consecutive_fail >= self.warn_after:
            self._warned = True
            logger.warning(
                "no valid HUD for %d consecutive frames (last=%s). "
                "menu open? HUD_Enable=false? wrong window?",
                self.stats.consecutive_fail,
                self.stats.last_status,
            )
            if self.on_warning is not None:
                try:
                    self.on_warning(replace(self.stats))
                except Exception:  # noqa: BLE001
                    logger.exception("on_warning callback raised")

    def _update_fps(self) -> None:
        self.stats.capture_fps = _fps_from(self._capture_times)
        self.stats.frame_fps = _fps_from(self._frame_times)

    # ---- デバッグ / キャリブレーション ----------------------------------
    def dump_debug(self, path: str | Path) -> Path:
        """直近フレームと復元ワードを保存し、グリッド検出不良を目視確認する。

        <path>.npy(生配列)を必ず保存。PIL があれば <path>.png も。<path>.txt に
        ワード16進とステータスを書く。保存先パスを返す。
        """
        with self._lock:
            frame = None if self._last_frame is None else self._last_frame.copy()
            result = self._latest_result
            status = self.stats.last_status
        if frame is None:
            raise RuntimeError("no frame captured yet")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path.with_suffix(".npy"), frame)

        lines = [f"status: {status}"]
        # 直近の生ワードを常に再デコードして確認できるようにする
        from .decode import decode_words

        words = decode_words(frame, self.spec)
        lines += [f"word[{i:2d}] = 0x{int(w):08X}" for i, w in enumerate(words)]
        if result is not None and result.pose is not None:
            p = result.pose
            lines.append(
                f"pose: pos={p.position} yaw={p.yaw_deg:.2f} pitch={p.pitch_deg:.2f}"
            )
        path.with_suffix(".txt").write_text("\n".join(lines), encoding="utf-8")

        try:
            from PIL import Image  # optional

            Image.fromarray(frame[:, :, :3][:, :, ::-1]).save(path.with_suffix(".png"))
        except Exception:  # noqa: BLE001 - PIL 無しでも .npy があれば足りる
            logger.debug("PIL not available; skipped PNG dump")
        return path
