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

## 5. Viewer and recorder

The prototype `Viewer` and `Recorder` are removed. Their replacements are separate modules
behind their own extras: `frames2py.viewer` ([viewer.md](viewer.md)) and `frames2py.recorder`
([recorder.md](recorder.md)); paced replay is `frames2py.replay` ([adapters.md](adapters.md)).

---

## 6. Telemetry

```python
Telemetry(
    engine: Engine,
    poll_interval_ms: float = 1000.0,
    viewer: object | None = None,  # anything with frames_shown / frames_dropped
) -> Telemetry
```

**Methods:**
- `start() -> Telemetry`
- `stop() -> None`

**Properties:**
- `history: list[TelemetrySample]`
- `latest: TelemetrySample | None`

### 6.1 TelemetrySample

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

## 7. Kernel Protocol

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

## 8. Adapters

The file adapters (EVT 2.0 / 3.0, AEDAT 4.0, HDF5) are described in [adapters.md](adapters.md).

---

## 9. Validation

```python
from frames2py.core.types import validate_event_batch

events = validate_event_batch(arr)  # Raises TypeError/ValueError if invalid
```

Checks: 1-D structured array, C-contiguous, fields `t`, `x`, `y`, `p`.
