"""Pilot(高レベル誘導API)でボタンを巡回する。patrol-buttons CLI の中身と同じ構成。

visit はボタン正面への移動 → 正対 → 横移動での最終照準をひとまとめにしたもの。
ゲインの既定値と根拠は docs/gain-tuning.md。
"""

import logging
from pathlib import Path

from app.control.controller import PatrolGains
from app.control.pilot import Pilot
from app.control.recording import ControlLog
from app.mapping.mapper import RoomMapper
from app.spatial.navigation import NavGrid

logging.basicConfig(level=logging.INFO)

grid = NavGrid.from_mapper(RoomMapper.load("maps/<日時>/room.npz"), avatar_radius=0.25)

# (名前, (x, y, z), face_yaw)。座標は find-button の出力、face_yaw は壁の外向き法線[deg]
TARGETS = [
    ("switch1", (3.0, 1.2, 5.0), 180.0),
    ("switch2", (-1.0, 1.0, 2.0), 90.0),
]

# recorder は省略可。渡すと制御ログが残り、log-video で再生できる
log = ControlLog(Path("logs/example_patrol.csv"))

# 実機 I/O(HUD キャプチャ + OSC)を接続。視点をマウスにするなら look=MouseLookActuator(...)
with Pilot.connect(grid, gains=PatrolGains(speed=0.7), recorder=log) as pilot:
    pilot.wait_for_hud()
    for name, nav, aim in pilot.patrol(TARGETS):
        print(name, nav.reason, aim.reason if aim else "-")

    # フェーズ単位でも呼べる
    pilot.goto((0.0, 0.0))  # 壁を避けて移動
    pilot.translate_to((1.0, 1.0))  # 視点を固定したまま並進
    pilot.aim((3.0, 1.2, 5.0))  # その場で正対(yaw+pitch)
    pilot.align((3.0, 1.2, 5.0))  # 横移動で視線上に載せる
    pilot.turn_to(90.0)  # 指定方位を向く

log.close()
