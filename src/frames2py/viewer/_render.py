"""``render()``: a snapshot as an RGB image, in NumPy alone."""

from __future__ import annotations

import math
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from frames2py.publish import Snapshot

AUTO_SCALE_MIN_VALUES: Final = 100
"""With fewer nonzero values than this, automatic scaling uses their maximum."""
AUTO_SCALE_PERCENTILE: Final = 99.0

RGB = NDArray[np.uint8]


def render(snapshot: Snapshot, *, scale: float | None = None, window_us: float = 50_000) -> RGB:
    """The snapshot as a new ``(height, width, 3)`` uint8 RGB image.

    Reads ``snapshot.frame`` and ``snapshot.meta`` and never writes either. The mapping
    follows the frame's dtype and shape, which are fixed per kernel:

    - ``(H, W)`` uint32 (``event_count``) and ``(H, W)`` float32 (``exp_decay``,
      ``timestamp_decay``): grey, ``floor(255 * min(v, s) / s)``.
    - ``(H, W, 2)`` uint32 (``polarity``; channel 0 OFF, channel 1 ON): OFF in blue, ON in
      red and green (yellow), both white, each ``floor(255 * min(v, s) / s)``.
    - ``(H, W)`` uint64 (``time_surface``): ``floor(255 * max(0, 1 - (T - v) / window_us))``
      with ``T`` the snapshot's watermark; ``v == 0``, no event yet, is black.

    ``s`` is *scale* when given. With ``scale=None`` it comes from the frame's nonzero values
    (both channels together for ``polarity``): their maximum if there are fewer than 100,
    otherwise their 99th percentile (NumPy's default linear interpolation, in float64).
    Values above ``s`` show at full brightness. A frame with no nonzero value is black.

    Args:
        snapshot: A published snapshot, such as ``engine.snapshot()`` returns.
        scale: The value shown at full brightness, or ``None`` for automatic scaling.
        window_us: For ``time_surface``: how far behind the watermark, in µs, a pixel's
            last event fades to black.

    Raises:
        TypeError: *snapshot* is not a ``Snapshot``, or its frame's dtype and shape are not
            one of the above.
        ValueError: *scale* or *window_us* is not a finite positive number.
    """
    if not isinstance(snapshot, Snapshot):
        raise TypeError(f"render() needs a frames2py.publish.Snapshot, got {type(snapshot).__name__}")
    scale = None if scale is None else _positive(scale, "scale")
    window_us = _positive(window_us, "window_us")
    frame = snapshot.frame
    kind = _kind(frame)
    if kind == "time":
        return _grey(_time_levels(frame, snapshot.meta.watermark, window_us))
    if kind == "grey":
        return _grey(_levels(frame, _scale(frame, scale)))
    s = _scale(frame, scale)
    out = np.empty(frame.shape[:2] + (3,), dtype=np.uint8)
    on = _levels(frame[:, :, 1], s)
    out[:, :, 0] = on
    out[:, :, 1] = on
    out[:, :, 2] = _levels(frame[:, :, 0], s)
    return out


def check_options(scale: float | None, window_us: float) -> None:
    """``render()``'s argument checks, for callers that validate before rendering."""
    if scale is not None:
        _positive(scale, "scale")
    _positive(window_us, "window_us")


def _kind(frame: Any) -> str:
    if not isinstance(frame, np.ndarray):
        raise TypeError(f"the snapshot's frame is not an array: {type(frame).__name__}")
    if frame.ndim == 2 and frame.dtype == np.uint64:
        return "time"
    if frame.ndim == 2 and frame.dtype in (np.uint32, np.float32):
        return "grey"
    if frame.ndim == 3 and frame.shape[2] == 2 and frame.dtype == np.uint32:
        return "polarity"
    raise TypeError(
        f"no rendering for a frame of dtype {frame.dtype} and shape {frame.shape}; supported: (H, W) uint32, "
        "float32 or uint64, and (H, W, 2) uint32"
    )


def _positive(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    number = float(value)
    if not (math.isfinite(number) and number > 0):
        raise ValueError(f"{name} must be a finite positive number, got {value!r}")
    return number


def _scale(frame: NDArray[Any], scale: float | None) -> float | None:
    """The value shown at full brightness; ``None`` when the frame has no nonzero value."""
    if scale is not None:
        return scale
    values = frame[frame != 0]
    if values.size == 0:
        return None
    if values.size < AUTO_SCALE_MIN_VALUES:
        return float(values.max())
    return float(np.percentile(values.astype(np.float64), AUTO_SCALE_PERCENTILE))


def _levels(values: NDArray[Any], s: float | None) -> RGB:
    """``floor(255 * min(v, s) / s)`` as uint8; all zero when *s* is ``None``."""
    if s is None:
        return np.zeros(values.shape, dtype=np.uint8)
    wide = values.astype(np.float64)
    levels = np.floor(np.minimum(wide, s) * 255.0 / s)
    np.minimum(levels, 255.0, out=levels)
    levels[wide >= s] = 255.0
    return levels.astype(np.uint8)


def _time_levels(frame: NDArray[np.uint64], watermark: int | None, window_us: float) -> RGB:
    """``floor(255 * max(0, 1 - (T - v) / window_us))`` for ``v != 0``, else 0."""
    if watermark is None:
        return np.zeros(frame.shape, dtype=np.uint8)
    t = np.uint64(watermark)
    age = np.where(frame <= t, t - np.minimum(frame, t), np.uint64(0)).astype(np.float64)
    remaining = np.maximum(window_us - age, 0.0)
    levels = np.floor(remaining * 255.0 / window_us)
    np.minimum(levels, 255.0, out=levels)
    levels[frame == 0] = 0.0
    return levels.astype(np.uint8)


def _grey(levels: RGB) -> RGB:
    return np.repeat(levels[:, :, np.newaxis], 3, axis=2)
