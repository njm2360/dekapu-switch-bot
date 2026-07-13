"""PID コントローラと照準誤差ヘルパのテスト。"""

from __future__ import annotations

import math

import pytest

from pose_hud import PID, aim_angle, heading_error, pitch_error, wrap180
from pose_hud.decode import Pose


def test_wrap180():
    assert wrap180(0) == 0
    assert wrap180(190) == pytest.approx(-170)
    assert wrap180(-190) == pytest.approx(170)
    assert abs(wrap180(180)) == pytest.approx(180)   # 境界は ±180 どちらでも可


def test_pid_proportional_only():
    pid = PID(kp=0.5)
    assert pid.update(2.0, 0.1) == pytest.approx(1.0)   # 0.5*2=1.0
    assert pid.update(-10.0, 0.1) == pytest.approx(-1.0)  # クランプ


def test_pid_output_clamped():
    pid = PID(kp=1.0, out_min=-0.5, out_max=0.5)
    assert pid.update(10.0, 0.1) == pytest.approx(0.5)
    assert pid.update(-10.0, 0.1) == pytest.approx(-0.5)


def test_pid_integral_accumulates_and_removes_steady_state():
    # P だけだと届かない小さな一定誤差でも、I があれば出力が育つ
    pid = PID(kp=0.01, ki=0.5, kd=0.0, i_limit=1.0)
    out1 = pid.update(1.0, 0.1)
    out2 = pid.update(1.0, 0.1)
    out3 = pid.update(1.0, 0.1)
    assert out1 < out2 < out3            # 積分で増加
    assert out3 > out1


def test_pid_integral_clamped_by_i_limit():
    pid = PID(kp=0.0, ki=1.0, kd=0.0, out_min=-10, out_max=10, i_limit=0.3)
    for _ in range(100):
        out = pid.update(1.0, 0.1)
    assert out == pytest.approx(0.3, abs=1e-9)   # i_limit で頭打ち


def test_pid_derivative_reacts_to_change():
    pid = PID(kp=0.0, ki=0.0, kd=1.0, out_min=-100, out_max=100)
    pid.update(0.0, 0.1)                  # prev=0
    out = pid.update(1.0, 0.1)           # d = (1-0)/0.1 = 10
    assert out == pytest.approx(10.0)


def test_pid_exposes_term_breakdown():
    pid = PID(kp=0.1, ki=0.5, kd=0.2, out_min=-10, out_max=10)
    pid.update(2.0, 0.1)              # prev=None → d=0
    out = pid.update(3.0, 0.1)
    # 内訳の合計が出力に一致(飽和していない前提)
    assert pid.last_p == pytest.approx(0.1 * 3.0)
    assert pid.last_d == pytest.approx(0.2 * (3.0 - 2.0) / 0.1)
    assert pid.last_out == pytest.approx(out)
    assert pid.last_p + pid.last_i + pid.last_d == pytest.approx(out)


def test_pid_deadzone_compensation():
    # out_knee>0 なら微小な非ゼロ出力でも最低 knee まで押し上げる(符号保持)
    pid = PID(kp=0.01, out_knee=0.55)
    out = pid.update(1.0, 0.1)        # 生出力 0.01 → knee にブースト
    assert out == pytest.approx(0.55 + (1 - 0.55) * 0.01, abs=1e-6)
    neg = pid.update(-1.0, 0.1)
    assert neg < 0 and abs(neg) == pytest.approx(0.55 + (1 - 0.55) * 0.01, abs=1e-6)
    # ちょうど0(誤差0)は0のまま(不感帯)
    assert PID(kp=0.1, out_knee=0.55).update(0.0, 0.1) == 0.0
    # 生出力が飽和(1.0)なら 1.0 のまま
    big = PID(kp=1.0, out_knee=0.55).update(5.0, 0.1)
    assert big == pytest.approx(1.0)


def test_pid_reset_derivative_keeps_integral():
    pid = PID(kp=0.0, ki=1.0, kd=1.0, out_min=-10, out_max=10)
    pid.update(1.0, 0.1)
    pid.update(1.0, 0.1)
    i_before = pid.last_i
    pid.reset_derivative()            # 積分は保持、微分履歴のみクリア
    out = pid.update(1.0, 0.1)
    assert pid.last_d == 0.0          # prev=None → d=0(微分キックなし)
    assert pid.last_i >= i_before     # 積分は保持されて増えている


def test_pid_reset():
    pid = PID(kp=0.0, ki=1.0)
    pid.update(1.0, 0.1); pid.update(1.0, 0.1)
    pid.reset()
    # reset 後は積分ゼロから
    assert pid.update(1.0, 0.1) == pytest.approx(1.0 * 0.1)


def test_pid_converges_in_sim():
    """1次系(角度)を PID で 0 に収束できることをシミュレーションで確認。"""
    pid = PID(kp=0.05, ki=0.02, kd=0.01, out_min=-1, out_max=1, i_limit=0.5)
    angle = 60.0                          # 初期誤差60度
    dt = 0.05
    for _ in range(800):
        cmd = pid.update(angle, dt)       # +で右回転が必要
        angle -= cmd * 90.0 * dt          # コマンドに比例して角度が減る(90deg/s @cmd=1)
    assert abs(angle) < 5.0               # 収束(振動を経て±5度以内)


# ---- 照準誤差 -----------------------------------------------------------
def test_heading_error_right_is_positive():
    # +Z を向いて原点。ターゲットが右(+X)→ +90度、距離5
    err, dist = heading_error((0.0, 0.0), 0.0, (5.0, 0.0))
    assert err == pytest.approx(90.0)
    assert dist == pytest.approx(5.0)


def test_heading_error_shortest_way():
    # 現在 yaw=170、目標方向が -170(=190)→ 最短は +20度側ではなく…
    err, _ = heading_error((0.0, 0.0), 170.0, (math.sin(math.radians(-170)), math.cos(math.radians(-170))))
    assert abs(err) == pytest.approx(20.0, abs=1e-6)


def test_pitch_error_up_positive():
    # 視点(0,1,0)、水平を向いている。ボタンが上(y=3, 前方z=5)→ もっと上を向く必要=+
    eye = (0.0, 1.0, 0.0)
    forward = (0.0, 0.0, 1.0)             # pitch 0
    target = (0.0, 3.0, 5.0)
    err = pitch_error(eye, forward, target)
    assert err > 0
    assert err == pytest.approx(math.degrees(math.atan2(2.0, 5.0)), abs=1e-6)


def test_pitch_error_zero_when_aligned():
    eye = (0.0, 1.0, 0.0)
    target = (0.0, 1.0, 5.0)              # 同じ高さ真正面
    forward = (0.0, 0.0, 1.0)
    assert pitch_error(eye, forward, target) == pytest.approx(0.0, abs=1e-9)


def test_aim_angle_zero_when_forward_hits_target():
    eye = (0.0, 1.0, 0.0)
    target = (3.0, 2.0, 4.0)
    d = (3.0, 1.0, 4.0)
    n = math.sqrt(9 + 1 + 16)
    forward = (d[0] / n, d[1] / n, d[2] / n)
    assert aim_angle(eye, forward, target) == pytest.approx(0.0, abs=1e-3)


def test_aim_angle_from_pose():
    # Pose.forward が真上90度ずれ
    eye = (0.0, 1.0, 0.0)
    target = (0.0, 1.0, 5.0)             # 真正面(+Z)
    forward = (1.0, 0.0, 0.0)            # +X を向いている → 90度ずれ
    assert aim_angle(eye, forward, target) == pytest.approx(90.0, abs=1e-6)
