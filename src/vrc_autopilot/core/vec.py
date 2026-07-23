from __future__ import annotations

import math
from typing import NamedTuple


class Vec2(NamedTuple):
    x: float
    z: float

    def __add__(self, o) -> Vec2:
        return Vec2(self.x + o[0], self.z + o[1])

    def __radd__(self, o) -> Vec2:
        return Vec2(o[0] + self.x, o[1] + self.z)

    def __sub__(self, o) -> Vec2:
        return Vec2(self.x - o[0], self.z - o[1])

    def __rsub__(self, o) -> Vec2:
        return Vec2(o[0] - self.x, o[1] - self.z)

    def __mul__(self, k: float) -> Vec2:
        return Vec2(self.x * k, self.z * k)

    __rmul__ = __mul__

    def norm(self) -> float:
        return math.hypot(self.x, self.z)

    def dot(self, o) -> float:
        return self.x * o[0] + self.z * o[1]

    def dist(self, o) -> float:
        return math.hypot(self.x - o[0], self.z - o[1])


class Vec3(NamedTuple):
    x: float
    y: float
    z: float

    def __add__(self, o) -> Vec3:
        return Vec3(self.x + o[0], self.y + o[1], self.z + o[2])

    def __radd__(self, o) -> Vec3:
        return Vec3(o[0] + self.x, o[1] + self.y, o[2] + self.z)

    def __sub__(self, o) -> Vec3:
        return Vec3(self.x - o[0], self.y - o[1], self.z - o[2])

    def __rsub__(self, o) -> Vec3:
        return Vec3(o[0] - self.x, o[1] - self.y, o[2] - self.z)

    def __mul__(self, k: float) -> Vec3:
        return Vec3(self.x * k, self.y * k, self.z * k)

    __rmul__ = __mul__

    @property
    def xz(self) -> Vec2:
        return Vec2(self.x, self.z)

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def dot(self, o) -> float:
        return self.x * o[0] + self.y * o[1] + self.z * o[2]

    def normalized(self, eps: float = 1e-12) -> Vec3:
        return self * (1.0 / (self.norm() + eps))
