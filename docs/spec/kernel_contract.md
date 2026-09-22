# Kernel Contract

**Version:** 1.0  
**Status:** Formal specification for frames2py accumulation kernels.

---

## 1. Overview

A **kernel** transforms a stream of events into a 2-D (or 3-D) frame. It implements two operations:

1. **accumulate(events, state)**  -  Merge new events into internal state.
2. **snapshot(state, out) → meta**  -  Copy current state into output buffer, return metadata.

The contract is language-agnostic. Implementations may be written in Python (NumPy), C++ (pybind11), or any other backend.

---

## 2. Operations

### 2.1 accumulate(events, state)

| Requirement | Description |
|-------------|-------------|
| **Mutates state in-place** | All writes go to pre-allocated arrays inside `state`. |
| **Must not allocate** | No `malloc`, `new`, or dynamic array growth. |
| **Must not access shared state** | No locks, no global variables, no I/O. |
| **Idempotent for empty batch** | If `len(events) == 0`, no effect. |

**Parameters:**
- `events`: Contiguous array of events (see §4).
- `state`: Opaque accumulator state from `init_state()`.

### 2.2 snapshot(state, out) → meta

| Requirement | Description |
|-------------|-------------|
| **Copies state into out** | Writes to the pre-allocated `out` buffer. |
| **Returns metadata** | `SnapshotMeta` with `timestamp`, `seq`, `events_accumulated`, etc. |
| **Reset behavior** | Kernel-dependent. Some kernels reset after snapshot, others do not. |

**Parameters:**
- `state`: The accumulator state.
- `out`: Pre-allocated output buffer. Shape and dtype match kernel output (e.g. `(H, W)` or `(H, W, 2)`).

**Returns:** `SnapshotMeta` (or equivalent struct).

---

## 3. init_state(sensor_size, frame_dtype) → state

- Allocates and returns the initial accumulator state.
- Called once at engine construction.
- The returned object is passed to every subsequent `accumulate`, `snapshot`, and `reset` call.

---

## 4. reset(state)

- Zeros the accumulator without publishing a snapshot.
- Called by `Engine.reset()`.

---

## 5. Event Layout

Events are stored in a contiguous array with the following layout:

| Field | Type | Description |
|-------|------|-------------|
| `t` | `uint64` | Timestamp (microseconds) |
| `x` | `uint16` | X coordinate (column) |
| `y` | `uint16` | Y coordinate (row) |
| `p` | `uint8` | Polarity (0 = OFF, 1 = ON) |

**Byte layout (little-endian):**
```
t: 8 bytes | x: 2 bytes | y: 2 bytes | p: 1 byte  [per event]
```

**Total:** 13 bytes per event (or platform-specific struct packing).

---

## 6. Timestamp Monotonicity

- **Preferred** but **not required**.
- Kernels must not assume timestamps are monotonically increasing.
- Implementations should handle out-of-order timestamps correctly (e.g. `time_surface` uses `max` per pixel; `exp_decay` applies decay before add).

---

## 7. Coordinate Bounds

- Coordinates `(x, y)` should be within `[0, width-1]` and `[0, height-1]`.
- Kernels may **clip** out-of-range coordinates rather than failing.
- Out-of-bounds events may be silently dropped or clamped.

---

## 8. Built-in Kernels

### 8.1 event_count

| Property | Value |
|----------|-------|
| **Output shape** | `(height, width)` |
| **Output dtype** | `float32` |
| **Semantics** | Count of events per pixel since last snapshot. |
| **Resets on snapshot** | Yes |

### 8.2 polarity

| Property | Value |
|----------|-------|
| **Output shape** | `(height, width, 2)` |
| **Output dtype** | `float32` |
| **Semantics** | Channel 0 = ON events, Channel 1 = OFF events. |
| **Resets on snapshot** | Yes |

### 8.3 time_surface

| Property | Value |
|----------|-------|
| **Output shape** | `(height, width)` |
| **Output dtype** | `float64` |
| **Semantics** | Latest timestamp per pixel. Pixels with no events remain 0. |
| **Resets on snapshot** | No |

### 8.4 exp_decay

| Property | Value |
|----------|-------|
| **Output shape** | `(height, width)` |
| **Output dtype** | `float32` |
| **Semantics** | Exponential decay surface. Before each accumulate, multiply all pixels by decay factor; then add 1.0 at event locations. |
| **Resets on snapshot** | No |
| **Parameters** | `decay` (default 0.95)  -  multiplicative factor per accumulate. |

---

## 9. SnapshotMeta

Metadata returned by `snapshot()`:

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | `int64` | Latest event timestamp (µs) in this snapshot |
| `seq` | `int` | Publication sequence (may be overridden by engine) |
| `events_accumulated` | `int` | Total events contributing to this snapshot since reset |
| `events_dropped` | `int` | Events dropped since previous snapshot (set by engine) |
| `wall_time_ns` | `int64` | Wall-clock time when published |

---

## 10. Conformance

A conforming kernel must:

1. Implement `accumulate(events, state)` without allocation or shared-state access.
2. Implement `snapshot(state, out) → meta` copying state into `out`.
3. Accept events with layout `(t, x, y, p)`.
4. Not assume timestamp monotonicity.
5. Provide `init_state` and `reset`.
6. Expose `name` and `channels` (or equivalent).
