"""VRChatOSC の送信テスト(VRChat不要、ループバックUDPで受信して python-osc で解析)。"""

from __future__ import annotations

import socket

import pytest
from pythonosc.osc_message import OscMessage

from app.control.osc import VRChatOSC


def _receiver():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(1.0)
    return sock, sock.getsockname()[1]


def _recv(sock):
    data, _ = sock.recvfrom(2048)
    msg = OscMessage(data)
    return msg.address, list(msg.params)


def test_move_sends_clamped_axes():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.move(forward=2.0, strafe=-3.0)   # クランプされる
        a1, p1 = _recv(sock)
        a2, p2 = _recv(sock)
        assert a1 == "/input/Vertical" and p1[0] == pytest.approx(1.0)
        assert a2 == "/input/Horizontal" and p2[0] == pytest.approx(-1.0)
    finally:
        sock.close()


def test_look_turn_and_pitch():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.look(0.3, pitch=0.5)
        a1, p1 = _recv(sock)
        a2, p2 = _recv(sock)
        assert a1 == "/input/LookHorizontal" and p1[0] == pytest.approx(0.3)
        assert a2 == "/input/LookVertical" and p2[0] == pytest.approx(0.5)
    finally:
        sock.close()


def test_look_without_pitch_sends_only_horizontal():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.look(0.2)                        # pitch=0 -> LookVertical は送らない
        a1, p1 = _recv(sock)
        assert a1 == "/input/LookHorizontal" and p1[0] == pytest.approx(0.2)
        with pytest.raises(socket.timeout):
            _recv(sock)
    finally:
        sock.close()


def test_look_vertical_direct():
    sock, port = _receiver()
    try:
        VRChatOSC("127.0.0.1", port).look_vertical(-0.4)
        a, p = _recv(sock)
        assert a == "/input/LookVertical" and p[0] == pytest.approx(-0.4)
    finally:
        sock.close()


def test_button_is_int():
    sock, port = _receiver()
    try:
        VRChatOSC("127.0.0.1", port).button("Jump", True)
        a, p = _recv(sock)
        assert a == "/input/Jump"
        assert p == [1] and isinstance(p[0], int)
    finally:
        sock.close()


def test_hud_enable_avatar_param():
    sock, port = _receiver()
    try:
        VRChatOSC("127.0.0.1", port).hud_enable(True)
        a, p = _recv(sock)
        assert a == "/avatar/parameters/HUD_Enable"
        assert p == [True]
    finally:
        sock.close()


def test_stop_zeroes_axes():
    sock, port = _receiver()
    try:
        osc = VRChatOSC("127.0.0.1", port)
        osc.stop()
        seen = {}
        for _ in range(4):
            a, p = _recv(sock)
            seen[a] = p[0]
        assert seen["/input/Vertical"] == pytest.approx(0.0)
        assert seen["/input/Horizontal"] == pytest.approx(0.0)
        assert seen["/input/LookHorizontal"] == pytest.approx(0.0)
        assert seen["/input/LookVertical"] == pytest.approx(0.0)
    finally:
        sock.close()
