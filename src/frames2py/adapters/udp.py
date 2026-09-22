"""UDP event stream adapter.

Receives and sends event batches over UDP.  The wire format is a
simple binary protocol:

    Header (12 bytes):
        - magic:      uint32 LE  (0xF2P10001)
        - n_events:   uint32 LE
        - reserved:   uint32 LE

    Payload (n_events * 13 bytes):
        - Contiguous EVENT_DTYPE structs

This is intentionally minimal -- no framing, no acknowledgement, no
retransmission.  UDP is lossy by design, which aligns with the
frames2py philosophy of best-effort visualization.
"""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterator

import numpy as np

from frames2py.core.types import EVENT_DTYPE, BatchMeta, EventBatch

_MAGIC = 0xF2E10001
_HEADER_SIZE = 12
_EVENT_SIZE = EVENT_DTYPE.itemsize  # 13 bytes
_MAX_UDP_PAYLOAD = 65507
_MAX_EVENTS_PER_PACKET = (_MAX_UDP_PAYLOAD - _HEADER_SIZE) // _EVENT_SIZE


def from_udp(
    host: str = "0.0.0.0",
    port: int = 5000,
    timeout: float = 5.0,
) -> Iterator[tuple[EventBatch, BatchMeta]]:
    """Yield event batches received over UDP.

    Binds a UDP socket and yields batches as they arrive.  The iterator
    terminates after *timeout* seconds of inactivity.

    Parameters:
        host: Bind address.
        port: Bind port.
        timeout: Socket timeout in seconds.  The iterator exits after
            this duration of silence.

    Yields:
        ``(EventBatch, BatchMeta)`` tuples.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.settimeout(timeout)

    meta = BatchMeta(source="udp")

    try:
        while True:
            try:
                data, _addr = sock.recvfrom(_MAX_UDP_PAYLOAD)
            except socket.timeout:
                return

            if len(data) < _HEADER_SIZE:
                continue

            magic, n_events, _reserved = struct.unpack_from("<III", data)
            if magic != _MAGIC:
                continue

            expected = _HEADER_SIZE + n_events * _EVENT_SIZE
            if len(data) < expected:
                continue

            payload = data[_HEADER_SIZE:expected]
            batch = np.frombuffer(payload, dtype=EVENT_DTYPE).copy()
            yield batch, meta
    finally:
        sock.close()


def to_udp(
    events: EventBatch,
    host: str = "127.0.0.1",
    port: int = 5000,
) -> int:
    """Send an event batch over UDP.

    Splits large batches into multiple packets to stay within the UDP
    MTU.

    Parameters:
        events: Event array with dtype :data:`EVENT_DTYPE`.
        host: Destination address.
        port: Destination port.

    Returns:
        Number of UDP packets sent.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packets_sent = 0

    try:
        offset = 0
        n = len(events)
        while offset < n:
            end = min(offset + _MAX_EVENTS_PER_PACKET, n)
            segment = events[offset:end]
            seg_len = len(segment)

            header = struct.pack("<III", _MAGIC, seg_len, 0)
            payload = np.ascontiguousarray(segment).tobytes()
            sock.sendto(header + payload, (host, port))
            packets_sent += 1
            offset = end
    finally:
        sock.close()

    return packets_sent


def pack_events_udp(events: EventBatch) -> bytes:
    """Serialize events into the UDP wire format (header + payload).

    Useful for custom transport implementations.

    Parameters:
        events: Event array with dtype :data:`EVENT_DTYPE`.

    Returns:
        Wire-format bytes ready to send.
    """
    header = struct.pack("<III", _MAGIC, len(events), 0)
    payload = np.ascontiguousarray(events).tobytes()
    return header + payload


def unpack_events_udp(data: bytes) -> EventBatch | None:
    """Deserialize events from the UDP wire format.

    Parameters:
        data: Raw bytes received from a UDP socket.

    Returns:
        Event array, or ``None`` if the data is malformed.
    """
    if len(data) < _HEADER_SIZE:
        return None

    magic, n_events, _reserved = struct.unpack_from("<III", data)
    if magic != _MAGIC:
        return None

    expected = _HEADER_SIZE + n_events * _EVENT_SIZE
    if len(data) < expected:
        return None

    return np.frombuffer(
        data[_HEADER_SIZE:expected], dtype=EVENT_DTYPE
    ).copy()
