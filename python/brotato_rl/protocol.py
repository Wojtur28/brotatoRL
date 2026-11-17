"""Message framing shared with bridge.gd: 4-byte little-endian length + UTF-8 JSON."""
import json
import socket
import struct


class BridgeError(RuntimeError):
    """The game instance stopped responding or sent something unexpected."""


class Connection:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

    def send(self, msg: dict) -> None:
        body = json.dumps(msg, separators=(",", ":")).encode("utf-8")
        try:
            self.sock.sendall(struct.pack("<I", len(body)) + body)
        except OSError as exc:
            raise BridgeError(f"send failed: {exc}") from exc

    def recv(self, timeout: float) -> dict:
        self.sock.settimeout(timeout)
        (size,) = struct.unpack("<I", self._read_exact(4))
        body = self._read_exact(size)
        try:
            return json.loads(body)
        except ValueError as exc:
            raise BridgeError(f"bad JSON from game: {body[:200]!r}") from exc

    def _read_exact(self, n: int) -> bytes:
        chunks, left = [], n
        while left:
            try:
                chunk = self.sock.recv(left)
            except socket.timeout as exc:
                raise BridgeError("timed out waiting for the game") from exc
            except OSError as exc:
                raise BridgeError(f"connection error: {exc}") from exc
            if not chunk:
                raise BridgeError("game closed the connection")
            chunks.append(chunk)
            left -= len(chunk)
        return b"".join(chunks)

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass
