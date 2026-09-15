#!/usr/bin/env python3
"""Require a same-origin WebSocket upgrade through the production proxy."""

from __future__ import annotations

import base64
import os
import socket


HOST = "127.0.0.1"
PORT = 8080
ORIGIN = f"http://{HOST}:{PORT}"


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

    with socket.create_connection((HOST, PORT), timeout=5) as connection:
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
    print(status_line)


if __name__ == "__main__":
    main()
