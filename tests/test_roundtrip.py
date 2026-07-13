"""合成画像でのエンコード→デコード ラウンドトリップ検証(実VRChat不要 / CI可能)。"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pose_hud import (
    ArrayFrameSource,
    DecodeStatus,
    GridSpec,
    PoseReader,
    decode_pose,
    decode_words,
    pack_pose_words,
    render_grid,
    render_pose,
)
from pose_hud.spec import MAGIC

SPEC = GridSpec()

SAMPLE_POSES = [
    (0, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    (123456, (1.5, -2.25, 42.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    (0xFFFFFFFF, (-1000.125, 7.0, 3.5), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    (999, (12.34, 56.78, -90.12), (0.6, 0.0, 0.8), (0.0, 1.0, 0.0)),
]


@pytest.mark.parametrize("time_ms,pos,fwd,up", SAMPLE_POSES)
def test_roundtrip_pose(time_ms, pos, fwd, up):
    frame = render_pose(time_ms, pos, fwd, up, SPEC)
    result = decode_pose(frame, SPEC)

    assert result.status is DecodeStatus.OK
    p = result.pose
    assert p.time_ms == (time_ms & 0xFFFFFFFF)
    # float32 で往復するので厳密一致(丸めなし)
    assert p.position == tuple(np.float32(v) for v in pos)
    assert p.forward == tuple(np.float32(v) for v in fwd)
    assert p.up == tuple(np.float32(v) for v in up)


def test_yaw_pitch_values():
    # +X を向く => yaw = atan2(1, 0) = 90度
    frame = render_pose(0, (0, 0, 0), (1.0, 0.0, 0.0), (0, 1, 0), SPEC)
    p = decode_pose(frame, SPEC).pose
    assert math.isclose(p.yaw_deg, 90.0, abs_tol=1e-4)
    assert math.isclose(p.pitch_deg, 0.0, abs_tol=1e-4)

    # 45度上向き
    inv = 1.0 / math.sqrt(2)
    frame = render_pose(0, (0, 0, 0), (0.0, inv, inv), (0, 1, 0), SPEC)
    p = decode_pose(frame, SPEC).pose
    assert math.isclose(p.pitch_deg, 45.0, abs_tol=1e-3)


def test_words_roundtrip():
    words = pack_pose_words(42, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    assert int(words[0]) == MAGIC
    decoded = decode_words(render_grid(words, SPEC), SPEC)
    np.testing.assert_array_equal(words, decoded)


def test_magic_mismatch_on_blank():
    frame = np.zeros((SPEC.capture_h, SPEC.capture_w, 3), dtype=np.uint8)
    result = decode_pose(frame, SPEC)
    assert result.status is DecodeStatus.MAGIC_MISMATCH
    assert result.pose is None


def test_checksum_mismatch_detected():
    words = pack_pose_words(1, (1.0, 0, 0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    words[3] ^= np.uint32(1)  # 位置ビットを1つ壊す(チェックサムはそのまま)
    frame = render_grid(words, SPEC)
    result = decode_pose(frame, SPEC)
    assert result.status is DecodeStatus.CHECKSUM_MISMATCH


def test_grid_offset_within_larger_canvas():
    # クライアント左上原点に描いたグリッドを、より大きなキャンバスでも読める
    words = pack_pose_words(7, (3.0, 2.0, 1.0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    frame = render_grid(words, SPEC, canvas_shape=(80, 160))
    assert decode_pose(frame, SPEC).status is DecodeStatus.OK


def test_alpha_channel_frame():
    # mss は BGRA を返す。4ch でも先頭3chで読めること。
    frame3 = render_pose(5, (1.0, 1.0, 1.0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    alpha = np.full((*frame3.shape[:2], 1), 255, dtype=np.uint8)
    frame4 = np.concatenate([frame3, alpha], axis=2)
    assert decode_pose(frame4, SPEC).status is DecodeStatus.OK


def test_no_python_pixel_loop_is_fast():
    # ベクトル化のスモークテスト: 1000回デコードしても十分速い
    frame = render_pose(0, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    import time

    t0 = time.perf_counter()
    for _ in range(1000):
        decode_pose(frame, SPEC)
    dt = time.perf_counter() - t0
    assert dt < 2.0, f"1000 decodes took {dt:.2f}s (too slow?)"


def test_custom_block_size():
    spec = GridSpec(block=3, offset_x=4, offset_y=6, capture_w=120, capture_h=60)
    frame = render_pose(11, (9.0, 8.0, 7.0), (0, 0, 1.0), (0, 1.0, 0), spec)
    result = decode_pose(frame, spec)
    assert result.status is DecodeStatus.OK
    assert result.pose.position == (np.float32(9.0), np.float32(8.0), np.float32(7.0))


def test_capture_auto_derived_from_geometry():
    # capture_w/h 未指定 => オフセット+グリッド+マージンから自動導出
    spec = GridSpec()
    assert spec.capture_w == spec.offset_x + spec.grid_w + spec.capture_margin  # 8+64+8=80
    assert spec.capture_h == spec.offset_y + spec.grid_h + spec.capture_margin  # 8+24+8=40


@pytest.mark.parametrize("offx,offy,block", [(0, 0, 1), (16, 12, 2), (4, 4, 4), (30, 20, 5)])
def test_injected_offset_and_block_roundtrip(offx, offy, block):
    # capture を明示せずオフセット/ブロックだけ注入しても破綻せず往復できる
    spec = GridSpec(offset_x=offx, offset_y=offy, block=block)
    frame = render_pose(77, (1.0, -2.0, 3.0), (0.6, 0.0, 0.8), (0, 1.0, 0), spec)
    result = decode_pose(frame, spec)
    assert result.status is DecodeStatus.OK
    assert result.pose.position == (np.float32(1.0), np.float32(-2.0), np.float32(3.0))


def test_capture_too_small_for_offset_rejected():
    with pytest.raises(ValueError):
        GridSpec(offset_x=8, block=2, capture_w=60, capture_h=40)  # 8+64=72 > 60


def test_invalid_geometry_rejected():
    with pytest.raises(ValueError):
        GridSpec(block=0)
    with pytest.raises(ValueError):
        GridSpec(offset_x=-1)


# ---- PoseReader 統合(合成ソース) --------------------------------------
def test_pose_reader_with_array_source():
    frame = render_pose(100, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    reader = PoseReader(source=ArrayFrameSource(frame), spec=SPEC)

    # スレッドを使わず単体で1フレーム処理
    result = reader.process_frame(frame)
    assert result.ok
    assert reader.get_latest().position == (np.float32(1.0), np.float32(2.0), np.float32(3.0))
    assert reader.get_stats().new_frames == 1

    # 同一 time_ms は重複としてスキップ
    reader.process_frame(frame)
    stats = reader.get_stats()
    assert stats.duplicate_skipped == 1
    assert stats.new_frames == 1


def test_pose_reader_consecutive_fail_and_warning():
    blank = np.zeros((SPEC.capture_h, SPEC.capture_w, 3), dtype=np.uint8)
    warnings = []
    reader = PoseReader(
        source=ArrayFrameSource(blank), spec=SPEC,
        warn_after=3, on_warning=lambda s: warnings.append(s.consecutive_fail),
    )
    for _ in range(5):
        reader.process_frame(blank)
    stats = reader.get_stats()
    assert stats.consecutive_fail == 5
    assert stats.magic_mismatch == 5
    assert len(warnings) == 1  # 1回だけ発火
    assert warnings[0] >= 3


def test_pose_reader_new_frame_resets_fail():
    good = render_pose(1, (0, 0, 0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    blank = np.zeros((SPEC.capture_h, SPEC.capture_w, 3), dtype=np.uint8)
    reader = PoseReader(source=ArrayFrameSource(blank), spec=SPEC)
    reader.process_frame(blank)
    reader.process_frame(blank)
    assert reader.get_stats().consecutive_fail == 2
    reader.process_frame(good)
    assert reader.get_stats().consecutive_fail == 0


def test_pose_reader_callback_and_generator():
    reader = PoseReader(source=ArrayFrameSource(None), spec=SPEC)
    seen = []
    reader.on_frame = seen.append
    frame = render_pose(50, (5.0, 0, 0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    reader.process_frame(frame)
    assert len(seen) == 1
    # ジェネレータ側にも積まれている
    gen = reader.poses(timeout=0.1)
    assert next(gen).time_ms == 50


def test_dump_debug(tmp_path):
    frame = render_pose(9, (1.0, 2.0, 3.0), (0, 0, 1.0), (0, 1.0, 0), SPEC)
    reader = PoseReader(source=ArrayFrameSource(frame), spec=SPEC)
    reader.process_frame(frame)
    out = reader.dump_debug(tmp_path / "dbg")
    assert out.with_suffix(".npy").exists()
    assert out.with_suffix(".txt").exists()
    loaded = np.load(out.with_suffix(".npy"))
    np.testing.assert_array_equal(loaded, frame)
