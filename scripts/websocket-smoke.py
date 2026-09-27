#!/usr/bin/env python3
"""Exercise one complete same-origin controller turn through the proxy."""

from __future__ import annotations

import base64
import json
import os
import socket
import struct


HOST = "127.0.0.1"
PORT = 8080
ORIGIN = f"http://{HOST}:{PORT}"
CONNECT_HOST = os.environ.get("FLY_SMOKE_HOST", HOST)


def _read_exact(connection: socket.socket, length: int) -> bytes:
    result = bytearray()
    while len(result) < length:
        chunk = connection.recv(length - len(result))
        if not chunk:
            raise AssertionError("WebSocket closed before the smoke turn completed")
        result.extend(chunk)
    return bytes(result)


def _send_json(connection: socket.socket, message: dict[str, object]) -> None:
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    mask = os.urandom(4)
    length = len(payload)
    if length < 126:
        header = bytes((0x81, 0x80 | length))
    elif length <= 0xFFFF:
        header = bytes((0x81, 0xFE)) + struct.pack("!H", length)
    else:
        header = bytes((0x81, 0xFF)) + struct.pack("!Q", length)
    masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
    connection.sendall(header + mask + masked)


def _receive_json(connection: socket.socket) -> dict[str, object]:
    first, second = _read_exact(connection, 2)
    opcode = first & 0x0F
    length = second & 0x7F
    if length == 126:
        length = struct.unpack("!H", _read_exact(connection, 2))[0]
    elif length == 127:
        length = struct.unpack("!Q", _read_exact(connection, 8))[0]
    if second & 0x80:
        raise AssertionError("server WebSocket frames must not be masked")
    payload = _read_exact(connection, length)
    if opcode == 0x8:
        raise AssertionError("server closed the WebSocket during the smoke turn")
    if opcode != 0x1:
        raise AssertionError(f"expected a text WebSocket frame, received opcode {opcode}")
    message = json.loads(payload)
    if not isinstance(message, dict):
        raise AssertionError("expected a JSON object from the simulation server")
    return message


def main() -> None:
    websocket_key = base64.b64encode(os.urandom(16)).decode("ascii")
    request = (
        "GET /api/simulation HTTP/1.1\r\n"
        f"Host: {HOST}:{PORT}\r\n"
        f"Origin: {ORIGIN}\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {websocket_key}\r\n"
        "Sec-WebSocket-Version: 13\r\n"
        "\r\n"
    ).encode("ascii")

    with socket.create_connection((CONNECT_HOST, PORT), timeout=20) as connection:
        connection.sendall(request)
        response = bytearray()
        while b"\r\n\r\n" not in response:
            chunk = connection.recv(4096)
            if not chunk:
                break
            response.extend(chunk)

        status_line = bytes(response).partition(b"\r\n")[0].decode("ascii", "replace")
        if status_line != "HTTP/1.1 101 Switching Protocols":
            raise AssertionError(f"expected WebSocket HTTP 101, received {status_line!r}")

        envelope = {
            "version": 2,
            "sessionId": "s-smoketest1",
            "episodeId": "e-smoketest1",
            "simulationTime": 0.0,
        }
        _send_json(connection, {
            "type": "hello",
            **envelope,
            "sequence": 0,
            "supportedVersions": [2],
            "uiBuild": "docker-smoke",
        })
        _send_json(connection, {
            "type": "configure",
            **envelope,
            "sequence": 1,
            "population": 80,
            "backend": "cpu",
            "seed": 7,
            "speed": 1.0,
        })
        ready = _receive_json(connection)
        if ready.get("type") != "ready":
            raise AssertionError(f"expected ready, received {ready.get('type')!r}")

        _send_json(connection, {
            "type": "observation",
            **envelope,
            "sequence": 2,
            "simulationTime": 1.0,
            "gameStep": 0,
            "observation": [0.0] * 517,
            "reward": 0.0,
        })
        received = [_receive_json(connection) for _ in range(3)]
        types = [message.get("type") for message in received]
        if types != ["neural_keyframe", "intention", "action_result"]:
            raise AssertionError(f"unexpected controller turn: {types!r}")
        if len(received[0].get("updates", [])) != 80:
            raise AssertionError("expected activity for all 80 controller neurons")

    print("ready -> neural_keyframe(80) -> intention -> action_result")


if __name__ == "__main__":
    main()
