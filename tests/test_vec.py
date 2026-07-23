from __future__ import annotations

import math

import numpy as np
import pytest

from vrc_autopilot.core.vec import Vec2, Vec3


class TestVec2:
    def test_add_sub_is_elementwise_not_concat(self):
        a = Vec2(1.0, 2.0)
        b = Vec2(0.5, -1.0)
        assert a + b == Vec2(1.5, 1.0)
        assert a - b == Vec2(0.5, 3.0)
        assert isinstance(a + b, Vec2)

    def test_ops_accept_plain_tuple(self):
        assert Vec2(1.0, 2.0) - (1.0, 1.0) == Vec2(0.0, 1.0)

    def test_plain_tuple_on_left_is_not_concat(self):
        # tuple.__add__ の連結ではなく、サブクラスの反射演算(要素和)になること
        assert (1.0, 2.0) + Vec2(3.0, 4.0) == Vec2(4.0, 6.0)
        assert (1.0, 2.0) - Vec2(3.0, 4.0) == Vec2(-2.0, -2.0)

    def test_scalar_mul(self):
        assert Vec2(1.0, -2.0) * 2.0 == Vec2(2.0, -4.0)
        assert 2.0 * Vec2(1.0, -2.0) == Vec2(2.0, -4.0)

    def test_norm_dist(self):
        assert Vec2(3.0, 4.0).norm() == pytest.approx(5.0)
        assert Vec2(1.0, 1.0).dist((4.0, 5.0)) == pytest.approx(5.0)

    def test_tuple_compat(self):
        v = Vec2(1.0, 2.0)
        assert v == (1.0, 2.0)
        x, z = v
        assert (x, z) == (1.0, 2.0)
        assert v[0] == 1.0 and v[1] == 2.0


class TestVec3:
    def test_add_sub(self):
        a = Vec3(1.0, 2.0, 3.0)
        assert a + (1.0, 1.0, 1.0) == Vec3(2.0, 3.0, 4.0)
        assert a - (1.0, 1.0, 1.0) == Vec3(0.0, 1.0, 2.0)

    def test_plain_tuple_on_left_is_not_concat(self):
        assert (1.0, 1.0, 1.0) + Vec3(1.0, 2.0, 3.0) == Vec3(2.0, 3.0, 4.0)
        assert (1.0, 1.0, 1.0) - Vec3(1.0, 2.0, 3.0) == Vec3(0.0, -1.0, -2.0)

    def test_scalar_mul(self):
        assert Vec3(1.0, 2.0, 3.0) * -1.0 == Vec3(-1.0, -2.0, -3.0)

    def test_xz_projection(self):
        assert Vec3(1.0, 2.0, 3.0).xz == Vec2(1.0, 3.0)
        assert isinstance(Vec3(1.0, 2.0, 3.0).xz, Vec2)

    def test_norm_dot(self):
        assert Vec3(2.0, 3.0, 6.0).norm() == pytest.approx(7.0)
        assert Vec3(1.0, 2.0, 3.0).dot((4.0, 5.0, 6.0)) == pytest.approx(32.0)

    def test_normalized(self):
        n = Vec3(0.0, 0.0, 5.0).normalized()
        assert n.norm() == pytest.approx(1.0)
        # ゼロベクトルでも発散しない
        z = Vec3(0.0, 0.0, 0.0).normalized()
        assert math.isfinite(z.norm())

    def test_tuple_compat_and_numpy(self):
        v = Vec3(1.0, 2.0, 3.0)
        assert v == (1.0, 2.0, 3.0)
        arr = np.asarray(v, dtype=np.float64)
        assert arr.shape == (3,)
        assert float(np.linalg.norm(arr)) == pytest.approx(v.norm())
