# Engine Semantics

**Version:** 1.0  
**Status:** Formal specification for frames2py accumulation engine.

---

## 1. Overview

The Engine is a **single-producer tap** that receives event batches, accumulates them into frame snapshots via a pluggable kernel, and publishes snapshots for consumers. The engine is designed so that the producer (ingest path) **never blocks** and has **no code path** to any consumer. There is no backpressure mechanism.

### 1.1 Three-Plane Architecture

```
Plane A: Ingest + Accumulate
    ingest(events) → ring buffer write → kernel accumulate → seqlock publish
    [Single producer thread, never blocks]

Plane B: Snapshot Bridge
    Seqlock-protected double buffer; lock-free publication

Plane C: Consumers
    Viewer, Recorder, Telemetry  -  each polls latest_snapshot() independently
```

---

## 2. Engine Ownership

The Engine owns the following components:

| Component | Description |
|-----------|-------------|
| **Ring buffer** | Chunk-based circular buffer for event batches. Pre-allocated slots, no heap allocation on write path. |
| **Kernel state** | Opaque accumulator state created by the kernel at init. Mutated only by `accumulate()` and `snapshot()`. |
| **Seqlock** | Double-buffered snapshot publication. Protects (frame, metadata) pair. |
| **Telemetry counters** | `events_ingested`, `events_dropped`, `chunks_dropped`, `snapshots_published`, `uptime_ns`. Monotonically non-decreasing. |

---

## 3. ingest()  -  Non-Blocking Guarantee

### 3.1 Contract

- **ingest(events)** is **non-blocking**. It must complete in bounded time.
- There exists **no code path** from `ingest()` to any consumer (viewer, recorder, telemetry).
- The engine **never** calls rendering code, display code, or I/O that could block (e.g. `cv2.imshow`, `VideoWriter.write`, file writes).
- If the ring buffer is full, the overflow policy is applied (evict oldest or discard newest). **No BLOCK mode** exists.

### 3.2 Steps (Logical Order)

1. **Validate**  -  Ensure `events` is a conforming `EventBatch` (1-D structured array, C-contiguous, fields `t`, `x`, `y`, `p`).
2. **Write**  -  Append events to the ring buffer. If overflow, evict chunks per policy. Update `events_dropped`, `chunks_dropped`.
3. **Drain**  -  Read all pending chunks from the ring buffer, concatenate into a single batch.
4. **Accumulate**  -  Call `kernel.accumulate(batch, state)`. Mutates `state` in-place.
5. **Publish**  -  If snapshot interval elapsed (or zero interval), call `kernel.snapshot(state, out)`, copy to seqlock buffer, increment seq, publish.
6. **Counters**  -  Update `events_ingested`, and if drops occurred, `events_dropped`, `chunks_dropped`.

### 3.3 Empty Batch

If `len(events) == 0`, `ingest()` returns immediately without modifying state or publishing.

### 3.4 Stopped Engine

If `running == False`, `ingest()` returns immediately without processing.

---

## 4. Consumers as Peers

- Consumers are **peers**. The engine has **zero knowledge** of consumers.
- Each consumer polls `latest_snapshot()` at its own rate (e.g. 30 Hz for viewer, 30 Hz for recorder, 1 Hz for telemetry).
- Consumers never block the engine. If a consumer falls behind, it skips frames.
- Consumers may run in separate threads. They do not hold any lock that the producer needs.

---

## 5. State Machine

The engine transitions through the following logical states during each `ingest()` call:

```
Idle
  ↓ (events arrive)
Ingesting      -  validate, write to ring buffer
  ↓
Accumulating   -  drain ring buffer, kernel.accumulate(batch, state)
  ↓
Publishing     -  kernel.snapshot(state, out), seqlock.end_write(meta)
  ↓
Idle
```

- **Idle**: No active ingest. Consumers may read the last published snapshot.
- **Ingesting**: Writing events to the ring buffer. Overflow evictions may occur.
- **Accumulating**: Kernel is merging events into state.
- **Publishing**: Copying state to seqlock buffer, incrementing seq.

The state machine is **implicit**  -  there is no explicit state variable. The transitions occur sequentially within a single `ingest()` call.

---

## 6. Thread Ownership

| Thread | Role | Calls |
|--------|------|-------|
| **Single producer** | Event source | `ingest(events)` |
| **Zero or more consumers** | Viewer, Recorder, Telemetry | `latest_snapshot()`, `stats`, `running` |

- Only one thread may call `ingest()` at a time. (Single-producer assumption.)
- Any number of threads may call `latest_snapshot()` concurrently.
- Consumers never call `ingest()`.

---

## 7. Lifecycle

### 7.1 Construction

```
Engine(sensor_size, kernel, buffer_capacity, chunk_size, overflow_policy, snapshot_interval_ms, frame_dtype)
```

- Allocates ring buffer (pre-allocated chunk slots).
- Allocates kernel state via `kernel.init_state(sensor_size, frame_dtype)`.
- Allocates seqlock with two frame buffers.
- Initializes counters to zero.
- `running = True`.

### 7.2 Ingest Loop

The producer repeatedly calls `ingest(events)` with batches from an event source (camera, file, network). Each call is non-blocking.

### 7.3 Reset

```
reset()
```

- Clears the ring buffer (discard all buffered events).
- Zeros kernel state via `kernel.reset(state)`.
- Resets seqlock (seq=0, buffers zeroed).
- Resets all telemetry counters to zero.
- Resets `_last_snapshot_ns`, `_start_ns`.
- Engine remains `running`. Ready for new ingestion.

### 7.4 Stop

```
stop()
```

- Sets `running = False`.
- Subsequent `ingest()` calls return immediately without processing.
- Consumers checking `running` should exit their poll loops.

### 7.5 Start (after Stop)

```
start()
```

- Sets `running = True`.
- Resets `_start_ns` to current monotonic time.
- Engine resumes accepting events.

---

## 8. Snapshot Interval

- **snapshot_interval_ms = 0**: Publish a snapshot on every `ingest()` call that produces data.
- **snapshot_interval_ms > 0**: Publish only when at least `snapshot_interval_ms` milliseconds have elapsed since the last publication.

---

## 9. Conformance

A conforming engine implementation must:

1. Ensure `ingest()` never blocks.
2. Never call consumer or rendering code from the ingest path.
3. Own ring buffer, kernel state, seqlock, and telemetry counters.
4. Expose `latest_snapshot()` for consumers to poll.
5. Implement the state machine transitions within each `ingest()` call.
6. Support single producer, zero or more consumer threads.
7. Support construction, reset, stop, and start as specified.
