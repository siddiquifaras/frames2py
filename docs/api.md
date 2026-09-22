# frames2py Python API Reference

**Version:** 0.1.0  
**Status:** Complete API documentation matching the implementation.

---

## 1. Top-Level Package

```python
import frames2py
```

### 1.1 Public Exports

| Symbol | Description |
|-------|-------------|
| `Engine` | Non-blocking event-to-frame accumulation engine |
| `Viewer` | Asynchronous snapshot viewer (daemon thread) |
| `Recorder` | Snapshot recorder to MP4 |
| `Telemetry` | Periodic engine health monitor |
| `TelemetrySample` | Single telemetry data point |
| `EVENT_DTYPE` | Canonical NumPy dtype for events |
| `BatchMeta` | Optional metadata for event batches |
| `EngineStats` | Live engine statistics |
| `EventBatch` | Type alias for event array |
| `FrameView` | Type alias for frame array |
| `KernelState` | Type alias for kernel accumulator state |
| `OverflowPolicy` | Ring buffer overflow strategy enum |
| `SnapshotMeta` | Metadata published with each snapshot |
| `Kernel` | Protocol for accumulation kernels |
| `__version__` | Package version string |

---

## 2. Core Types

### 2.1 EVENT_DTYPE

```python
frames2py.EVENT_DTYPE  # np.dtype
```

Structured dtype for events: `[("t", "<u8"), ("x", "<u2"), ("y", "<u2"), ("p", "u1")]`.

- `t`: uint64, timestamp (microseconds)
- `x`: uint16, column
- `y`: uint16, row
- `p`: uint8, polarity (0=OFF, 1=ON)

### 2.2 OverflowPolicy

```python
class OverflowPolicy(enum.Enum):
    DROP_OLDEST = "drop_oldest"   # Evict oldest chunk when full
    DROP_NEWEST = "drop_newest"   # Discard incoming chunk when full
```

### 2.3 BatchMeta

```python
@dataclass(frozen=True)
class BatchMeta:
    monotonic: bool = False      # Timestamps non-decreasing?
    reordered: bool = False      # Adapter sorted events?
    source: str = ""              # e.g. "h5", "aedat4"
    sensor_size: tuple[int, int] | None = None
```

### 2.4 SnapshotMeta

```python
@dataclass(frozen=True)
class SnapshotMeta:
    timestamp: int = 0           # Latest event timestamp (µs)
    seq: int = 0                 # Publication sequence number
    events_accumulated: int = 0   # Total events since reset
    events_dropped: int = 0       # Events dropped since previous snapshot
    wall_time_ns: int = 0         # time.monotonic_ns() when published
```

### 2.5 EngineStats

```python
@dataclass
class EngineStats:
    events_ingested: int = 0
    events_dropped: int = 0
    chunks_dropped: int = 0
    buffer_fill_ratio: float = 0.0   # [0.0, 1.0]
    snapshots_published: int = 0
    uptime_ns: int = 0
```

---

## 3. Engine

### 3.1 Constructor

```python
Engine(
    sensor_size: tuple[int, int],
    kernel: str | Kernel = "event_count",
    buffer_capacity: int = 64,
    chunk_size: int = 65_536,
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    snapshot_interval_ms: float = 0.0,
    frame_dtype: np.dtype = np.dtype(np.float32),
) -> Engine
```

| Parameter | Description |
|-----------|-------------|
| `sensor_size` | `(width, height)` of the event sensor |
| `kernel` | Kernel name or `Kernel` instance |
| `buffer_capacity` | Number of chunk slots in ring buffer |
| `chunk_size` | Max events per chunk slot |
| `overflow_policy` | `DROP_OLDEST` or `DROP_NEWEST` |
| `snapshot_interval_ms` | Min ms between snapshots; `0` = every ingest |
| `frame_dtype` | NumPy dtype for output frame |

### 3.2 Methods

#### ingest

```python
def ingest(
    self,
    events: EventBatch,
    meta: BatchMeta | None = None,
) -> None
```

Ingest an event batch. Non-blocking. Validates `events` (1-D structured array, C-contiguous, fields `t`, `x`, `y`, `p`).

#### latest_snapshot

```python
def latest_snapshot(self) -> tuple[FrameView, SnapshotMeta] | None
```

Return the most recently published snapshot, or `None` if none available or write in progress. Thread-safe.

#### reset

```python
def reset(self) -> None
```

Clear ring buffer, kernel state, seqlock, and counters. Engine remains running.

#### stop

```python
def stop(self) -> None
```

Set `running = False`. Subsequent `ingest()` calls are no-ops.

#### start

```python
def start(self) -> None
```

Set `running = True`. Re-enable after `stop()`.

### 3.3 Properties

| Property | Type | Description |
|----------|------|-------------|
| `stats` | `EngineStats` | Live engine statistics |
| `latency_tracker` | `LatencyTracker` | Per-call accumulate latency |
| `running` | `bool` | Whether engine accepts events |
| `sensor_size` | `tuple[int, int]` | `(width, height)` |
| `kernel` | `Kernel` | Active accumulation kernel |

---

## 4. LatencyTracker

```python
class LatencyTracker:
    def __init__(self, window_size: int = 10_000) -> None
    def record(self, duration_ns: int) -> None
    def percentile(self, q: float) -> float   # Returns ms
    def reset(self) -> None
    @property
    def count(self) -> int
```

Rolling-window tracker for accumulate-call durations. Used by Telemetry.

---

## 5. Viewer

```python
Viewer(
    engine: Engine,
    backend: str = "opencv",
    fps: float = 30.0,
    window_name: str = "frames2py",
    colormap: str | None = None,
) -> Viewer
```

| Parameter | Description |
|-----------|-------------|
| `backend` | `"opencv"` or `"headless"` |
| `fps` | Target polling rate |
| `colormap` | OpenCV colormap name (e.g. `"viridis"`, `"hot"`) or `None` for grayscale |

**Methods:**
- `start() -> Viewer`  -  Start daemon thread
- `stop() -> None`  -  Stop and join thread
- `add_overlay(overlay: Overlay) -> None`

**Properties:**
- `frames_shown: int`
- `frames_dropped: int`

---

## 6. Recorder

```python
Recorder(
    engine: Engine,
    output_path: str,
    fps: float = 30.0,
    codec: str = "mp4v",
) -> Recorder
```

**Methods:**
- `start() -> Recorder`
- `stop() -> None`

**Properties:**
- `frames_written: int`

---

## 7. Telemetry

```python
Telemetry(
    engine: Engine,
    poll_interval_ms: float = 1000.0,
    viewer: Viewer | None = None,
) -> Telemetry
```

**Methods:**
- `start() -> Telemetry`
- `stop() -> None`

**Properties:**
- `history: list[TelemetrySample]`
- `latest: TelemetrySample | None`

### 7.1 TelemetrySample

```python
@dataclass(frozen=True)
class TelemetrySample:
    wall_time_ns: int = 0
    events_ingested: int = 0
    events_dropped: int = 0
    chunks_dropped: int = 0
    buffer_fill_ratio: float = 0.0
    snapshots_published: int = 0
    viewer_frames_shown: int | None = None
    viewer_frames_dropped: int | None = None
    accumulate_ms_p50: float = 0.0
    accumulate_ms_p95: float = 0.0
    accumulate_ms_p99: float = 0.0
```

---

## 8. Kernel Protocol

```python
class Kernel(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def channels(self) -> int: ...

    def init_state(
        self,
        sensor_size: tuple[int, int],
        frame_dtype: np.dtype = np.dtype(np.float32),
    ) -> KernelState: ...

    def accumulate(self, events: EventBatch, state: KernelState) -> None: ...

    def snapshot(self, state: KernelState, out: FrameView) -> SnapshotMeta: ...

    def reset(self, state: KernelState) -> None: ...
```

**Built-in kernels:** `event_count`, `polarity`, `time_surface`, `exp_decay`.

---

## 9. Display Overlays

```python
from frames2py.display.overlays import (
    FPSOverlay,
    TimestampOverlay,
    StatsOverlay,
    BBoxOverlay,
    BBox,
)
```

### 9.1 BBox

```python
@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = ""
    color: tuple[int, int, int] = (0, 255, 0)
    confidence: float = 1.0
```

### 9.2 Overlay Usage

```python
viewer.add_overlay(FPSOverlay())
viewer.add_overlay(StatsOverlay(engine))
bbox = BBoxOverlay()
viewer.add_overlay(bbox)
bbox.update([BBox(100, 50, 300, 200, label="car")])
```

---

## 10. Adapters

Adapters are optional and use lazy imports. Install with extras: `frames2py[adapter-h5]`, etc.

### 10.1 HDF5

```python
from frames2py.adapters.h5 import from_h5, to_h5

# Read
for batch, meta in from_h5(path, chunk_size=50_000, dataset="events"):
    engine.ingest(batch)

# Write
to_h5(path, events)
```

### 10.2 AEDAT4

```python
from frames2py.adapters.aedat4 import from_aedat4, to_aedat4

for batch, meta in from_aedat4(path, chunk_size=50_000):
    engine.ingest(batch)

to_aedat4(path, events, sensor_size=(width, height))
```

### 10.3 UDP

```python
from frames2py.adapters.udp import from_udp, to_udp, pack_events_udp, unpack_events_udp

for batch, meta in from_udp(host="127.0.0.1", port=9000, timeout=1.0):
    engine.ingest(batch)

to_udp(events, host="127.0.0.1", port=9000)
```

### 10.4 Format Converter

```python
from frames2py.adapters.convert import convert, read_events, write_events

n = convert("input.h5", "output.aedat4")
events = read_events("data.npy")
write_events("out.csv", events)
```

---

## 11. Validation

```python
from frames2py.core.types import validate_event_batch

events = validate_event_batch(arr)  # Raises TypeError/ValueError if invalid
```

Checks: 1-D structured array, C-contiguous, fields `t`, `x`, `y`, `p`.
