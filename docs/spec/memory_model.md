# Memory Model and Threading Guarantees

**Version:** 1.0  
**Status:** Formal specification for frames2py concurrency.

---

## 1. Thread Roles

| Role | Count | Responsibility |
|------|-------|-----------------|
| **Producer** | 1 | Calls `ingest(events)`. Owns the ingest path. |
| **Consumers** | 0 or more | Call `latest_snapshot()`, read `stats`, check `running`. Viewer, Recorder, Telemetry. |

---

## 2. Shared State

There is **exactly one** shared mutable structure between producer and consumers:

| Structure | Description | Synchronization |
|-----------|-------------|-----------------|
| **Seqlock-protected snapshot** | (frame buffer, metadata) pair | Seqlock (sequence number + double buffering) |

### 2.1 No Other Shared Mutable State

- The **ring buffer** is accessed only by the producer. Consumers never touch it.
- The **kernel state** is accessed only by the producer. Consumers never touch it.
- **Telemetry counters** (`events_ingested`, `events_dropped`, etc.) are written only by the producer. Consumers read them. See §4 for ordering.

---

## 3. Lock Discipline

- **Consumers never acquire any lock that the producer also acquires.**
- The producer never blocks on a consumer-held lock.
- The seqlock is **lock-free**: no mutex. The writer uses a sequence number; readers use a read-copy-check pattern.

---

## 4. Memory Ordering

### 4.1 CPython (Phase 1 / Reference Implementation)

- The CPython **GIL (Global Interpreter Lock)** provides sufficient ordering for the reference implementation.
- With the GIL, a single Python process has sequential consistency for shared variables.
- The seqlock `seq` variable, frame buffers, and metadata are all visible across threads because Python bytecode execution is serialized by the GIL.

### 4.2 C++ Backend (Future / High-Performance)

For a C++ implementation (e.g. pybind11 kernels, native seqlock):

- **Writer (producer)**:
  - After writing the frame and metadata, use `std::atomic<uint64_t>` for `seq` with `memory_order_release` on the store that signals "published".
- **Reader (consumer)**:
  - Use `memory_order_acquire` when reading `seq` to establish a happens-before edge with the writer's release.
- This ensures that when a reader observes `seq` as even and stable, all writes to the frame and metadata are visible.

### 4.3 Formal Guarantee

- If a consumer observes `seq` as **even** and **unchanged** across its read (copy + re-check), then the frame and metadata are **mutually consistent** and reflect a complete write by the producer.

---

## 5. Consumer Polling Pattern

Consumers use a best-effort polling loop:

```
while running:
    result = engine.latest_snapshot()
    if result is not None:
        frame, meta = result
        # process frame
    sleep(interval)
```

- `latest_snapshot()` may return `None` if a write is in progress (odd seq) or if a torn read was detected.
- Consumers retry on the next poll. No blocking, no backpressure.

---

## 6. Producer-Only Structures

The following are **never** accessed by consumers:

- Ring buffer slots and indices
- Kernel state (accumulator buffers)
- Overflow counters (producer updates; consumers may read a snapshot via `stats`)

---

## 7. Conformance

A conforming implementation must:

1. Use a single producer thread for `ingest()`.
2. Allow zero or more consumer threads for `latest_snapshot()`.
3. Protect the shared snapshot with a seqlock (or equivalent lock-free mechanism).
4. Ensure no other shared mutable state between producer and consumers.
5. For CPython: rely on GIL ordering or document explicit synchronization.
6. For C++: use `std::atomic` with `memory_order_release` (writer) and `memory_order_acquire` (reader).
7. Ensure consumers never acquire a lock that the producer needs.
