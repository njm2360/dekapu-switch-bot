"""フレーム単位の照準幾何(現在ポーズ → 誤差・係数)。

navigation(オフラインの経路計画)とは役割が別で、制御ループが毎フレーム呼ぶ
純粋関数群。角度の正規化 wrap180 の定義もここに集約する。
"""

import math

from ..core.vec import Vec2, Vec3


def wrap180(deg: float) -> float:
    """角度を [-180, 180) に正規化する(最短回りの誤差に使う。wrap180(180)==-180)。"""
    return (deg + 180.0) % 360.0 - 180.0


def heading_error(
    cur_xz: Vec2, cur_yaw_deg: float, target_xz: Vec2
) -> tuple[float, float]:
    """target への (yaw誤差[deg], 水平距離[m]) を返す。yaw誤差は最短回り、+で右。"""
    d = target_xz - cur_xz
    desired_yaw = math.degrees(math.atan2(d.x, d.z))
    return wrap180(desired_yaw - cur_yaw_deg), d.norm()


def pitch_error(
    eye_xyz: Vec3,
    cur_forward: Vec3,
    target_xyz: Vec3,
    *,
    min_horiz: float = 0.0,
) -> float:
    """視線の pitch 誤差[deg]。+ は「もっと上を向く必要」。

    現在 pitch は forward.y から、目標 pitch は視点→ボタンの仰角から求める。

    min_horiz>0 なら水平距離をその値で下限クランプする(移動中の事前整合用)。真下付近でも
    目標 pitch が ±90° へ発散せず、standoff 相当で頭打ちになって到着地点の仰角に一致する。
    最終照準は真値が要るので既定 0。
    """
    d = target_xyz - eye_xyz
    horiz = d.xz.norm()
    if min_horiz > 0.0:
        horiz = max(horiz, min_horiz)
    desired_pitch = math.degrees(math.atan2(d.y, horiz))
    fy = max(-1.0, min(1.0, cur_forward[1]))
    current_pitch = math.degrees(math.asin(fy))
    return desired_pitch - current_pitch


def aim_angle(
    eye_xyz: Vec3,
    cur_forward: Vec3,
    target_xyz: Vec3,
) -> float:
    """視線 forward と「視点→ボタン」方向との実際のなす角[deg](総合ずれの指標)。"""
    d = target_xyz - eye_xyz
    if d.norm() < 1e-9:
        return 0.0
    f = cur_forward.normalized()
    cos = max(-1.0, min(1.0, d.normalized().dot(f)))
    return math.degrees(math.acos(cos))


def standoff_point(xyz: Vec3, face_yaw_deg: float, dist: float) -> Vec2:
    """目標の正面 dist [m] に立つ位置の XZ。face_yaw_deg は目標面の法線方向(+Z基準)。

    dist<=0 なら目標の直下 XZ。現在地には依存しない(壁裏への回り込みを防ぐ)。
    """
    xz = xyz.xz
    if dist <= 0.0:
        return xz
    y = math.radians(face_yaw_deg)
    return xz + Vec2(math.sin(y), math.cos(y)) * dist


def forward_factor(yaw_err_deg: float, cutoff_deg: float = 90.0) -> float:
    """前進速度の減衰係数 [0,1]。正対で1、|yaw_err| >= cutoff で0(cos ベース)。

    その場停止→旋回のガクつきを避け、向きのズレに応じて滑らかに減速する。
    """
    a = abs(yaw_err_deg)
    if a >= cutoff_deg:
        return 0.0
    return max(0.0, math.cos(math.radians(a)))
