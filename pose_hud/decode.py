import enum
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .spec import (
    DEFAULT_SPEC,
    IDX_CHECKSUM,
    IDX_FWD,
    IDX_MAGIC,
    IDX_POS,
    IDX_TIME,
    IDX_UP,
    GridSpec,
)


class DecodeStatus(enum.Enum):
    OK = "ok"
    MAGIC_MISMATCH = "magic_mismatch"  # HUD非表示 / メニューで隠れている等
    CHECKSUM_MISMATCH = "checksum_mismatch"  # 読み取りノイズ / 途中フレーム


@dataclass(frozen=True)
class Pose:
    """復元された 6DoF ポーズ(Unity座標系: Y-up, 左手系, 単位メートル)。"""

    time_ms: int  # _VRChatTimeNetworkMs (ラップあり)
    position: tuple[float, float, float]
    forward: tuple[float, float, float]
    up: tuple[float, float, float]

    @property
    def yaw_deg(self) -> float:
        """+Z 基準の yaw。atan2(fwd.x, fwd.z)。"""
        return math.degrees(math.atan2(self.forward[0], self.forward[2]))

    @property
    def pitch_deg(self) -> float:
        """上向きが正の pitch。asin(fwd.y)。"""
        return math.degrees(math.asin(max(-1.0, min(1.0, self.forward[1]))))

    @property
    def roll_deg(self) -> float:
        """up ベクトルから求めた roll。デスクトップでは常に≒0(VR対応用)。"""
        fwd = np.asarray(self.forward, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        # forward まわりで world-up を投影した右手系の傾き
        right = np.cross(up, fwd)
        world_up_proj = np.cross(fwd, right)
        return math.degrees(
            math.atan2(
                np.dot(right, [0.0, 1.0, 0.0]), np.dot(world_up_proj, [0.0, 1.0, 0.0])
            )
        )


@dataclass(frozen=True)
class DecodeResult:
    status: DecodeStatus
    words: np.ndarray  # uint32[12] 生ワード(常に埋まる)
    pose: Pose | None = None

    @property
    def ok(self) -> bool:
        return self.status is DecodeStatus.OK


@lru_cache(maxsize=8)
def _sampling_geometry(spec: GridSpec):
    """ブロック中心座標と MSB-左のパック重みをキャッシュして返す。"""
    cols = np.arange(spec.cols)
    rows = np.arange(spec.rows)
    half = spec.block // 2
    cx = spec.offset_x + cols * spec.block + half  # (cols,)
    cy = spec.offset_y + rows * spec.block + half  # (rows,)
    # MSB が左端 => 列0が最上位ビット
    weights = np.uint64(1) << np.arange(spec.cols - 1, -1, -1, dtype=np.uint64)
    return cy, cx, weights


def sample_bits(frame: np.ndarray, spec: GridSpec = DEFAULT_SPEC) -> np.ndarray:
    """フレームからブロック中心を一括サンプルし (rows, cols) の bool 配列を返す。

    frame: HxWx3 以上 (BGR/RGB どちらでも / アルファ付きも可)。
    グリッド原点はクライアント左上 = frame[0,0] を前提。
    """
    if frame.ndim != 3:
        raise ValueError(f"frame must be HxWxC, got shape {frame.shape}")
    cy, cx, _ = _sampling_geometry(spec)
    need_h = int(cy[-1]) + 1
    need_w = int(cx[-1]) + 1
    if frame.shape[0] < need_h or frame.shape[1] < need_w:
        raise ValueError(
            f"frame {frame.shape[:2]} too small for grid (need >= {need_h}x{need_w})"
        )
    # (rows, cols, C) を一括ギャザー。C はRGBの先頭3chのみ使用。
    samples = frame[np.ix_(cy, cx)][:, :, :3].astype(np.uint16)
    rgb_sum = samples.sum(axis=2)  # (rows, cols)
    return rgb_sum > spec.threshold


def pack_words(bits: np.ndarray, spec: GridSpec = DEFAULT_SPEC) -> np.ndarray:
    """(rows, cols) bool 配列を uint32[rows] ワードにパックする(MSBが左端)。"""
    _, _, weights = _sampling_geometry(spec)
    words = bits.astype(np.uint64) @ weights  # (rows,)
    return words.astype(np.uint32)


def decode_words(frame: np.ndarray, spec: GridSpec = DEFAULT_SPEC) -> np.ndarray:
    """フレームから 12 ワードを復元する(検証なし)。uint32[12]。"""
    return pack_words(sample_bits(frame, spec), spec)


def validate_words(words: np.ndarray, spec: GridSpec = DEFAULT_SPEC) -> DecodeStatus:
    """MAGIC と XOR チェックサムを検証する。"""
    if int(words[IDX_MAGIC]) != spec.magic:
        return DecodeStatus.MAGIC_MISMATCH
    xor = np.bitwise_xor.reduce(words[:IDX_CHECKSUM].astype(np.uint32))
    if np.uint32(xor) != words[IDX_CHECKSUM]:
        return DecodeStatus.CHECKSUM_MISMATCH
    return DecodeStatus.OK


def words_to_pose(words: np.ndarray) -> Pose:
    """検証済みワードから Pose を構築する(float32ビットパターンを解釈)。"""
    floats = words[2:11].astype(np.uint32).view(np.float32)
    return Pose(
        time_ms=int(words[IDX_TIME]),
        position=tuple(float(v) for v in floats[0:3]),
        forward=tuple(float(v) for v in floats[3:6]),
        up=tuple(float(v) for v in floats[6:9]),
    )


def decode_pose(frame: np.ndarray, spec: GridSpec = DEFAULT_SPEC) -> DecodeResult:
    """フレームをデコードし検証まで行う。

    CLAUDE.md のデコード検証手順 3〜4 を実施(手順5の重複フレーム判定は PoseReader 側)。
    """
    words = decode_words(frame, spec)
    status = validate_words(words, spec)
    pose = words_to_pose(words) if status is DecodeStatus.OK else None
    return DecodeResult(status=status, words=words, pose=pose)
