# Build-Time Invariants

**Version:** 1.0  
**Status:** Mandatory rules for frames2py development and code review.

These 12 invariants must hold for every commit. They are checked during code review and CI. A conforming implementation must satisfy all of them.

---

## 1. ingest() Never Blocks

**Invariant:** `ingest()` must never block. No mutex, no condition variable, no I/O that could block, no backpressure.

**Spec reference:** [engine.md §3](spec/engine.md#3-ingest--non-blocking-guarantee)

**Explanation:** The entire value proposition of frames2py is that the event processing pipeline is never throttled by visualization. Blocking in `ingest()` would defeat this.

**Code review question:** Does any code path in `ingest()` acquire a lock, wait on a condition, or perform blocking I/O?

---

## 2. Engine Never Calls Rendering Code

**Invariant:** The engine must never call display, rendering, or consumer code. No `cv2.imshow`, no `VideoWriter.write`, no overlay drawing.

**Spec reference:** [engine.md §3.1](spec/engine.md#3-ingest--non-blocking-guarantee)

**Explanation:** Rendering runs in consumer threads. The engine is a tap; it publishes snapshots. Consumers pull. The engine has zero knowledge of consumers.

**Code review question:** Does the engine import or call `frames2py.viewer`, `frames2py.recorder` or `consumers/`?

---

## 3. Consumers Never Hold Locks Needed by Ingest

**Invariant:** Consumers must never acquire any lock that the producer (ingest path) also acquires.

**Spec reference:** [memory_model.md §3](spec/memory_model.md#3-lock-discipline)

**Explanation:** If a consumer held a lock the producer needs, the producer could block waiting for it. That would violate Invariant 1.

**Code review question:** Does the viewer, or Telemetry, acquire locks that the engine's `ingest()` path uses?

---

## 4. Backpressure Mode is "None" Permanently

**Invariant:** There is no BLOCK mode. When the ring buffer is full, the overflow policy (DROP_OLDEST or DROP_NEWEST) is applied. This is permanent.

**Spec reference:** [overflow.md §6](spec/overflow.md#6-no-block-mode)

**Explanation:** Blocking on full buffer would couple producer latency to consumer speed. frames2py explicitly rejects this design.

**Code review question:** Is there any code path that blocks or slows the producer when the buffer is full?

---

## 5. Ring Buffer Drops Whole Chunks

**Invariant:** Overflow eviction removes whole chunks atomically. Never evict individual events within a chunk.

**Spec reference:** [overflow.md §3](spec/overflow.md#3-whole-chunk-eviction)

**Explanation:** Whole-chunk eviction simplifies accounting (events_dropped = sum of evicted chunk sizes) and avoids partial-state corruption.

**Code review question:** Does the ring buffer ever evict a fraction of a chunk?

---

## 6. Snapshot Uses Seqlock, Not Mutex

**Invariant:** Snapshot publication uses a seqlock (sequence number + double buffering), not a mutex.

**Spec reference:** [snapshot.md](spec/snapshot.md)

**Explanation:** A mutex would allow a slow consumer to block the producer. The seqlock is lock-free: readers that detect a torn read simply retry on the next poll.

**Code review question:** Is there a mutex protecting the snapshot buffer?

---

## 7. The Viewer's Backend Is an Optional Extra

**Invariant:** The viewer's window backend (pyglet, `frames2py[viewer]`) is optional. The core library, and `frames2py.viewer.render()`, run without it; only `frames2py.viewer.run()` imports it.

**Spec reference:** [viewer.md](viewer.md)

**Explanation:** Users may run headless (e.g. recording-only, telemetry-only). The core must not depend on any GUI library.

**Code review question:** Does `import frames2py` or `import frames2py.viewer` import pyglet or any other display dependency?

---

## 8. Adapters Are Optional Extras with Lazy Imports

**Invariant:** Adapters (H5, AEDAT4, Prophesee, iniVation, UDP) are optional. Use lazy imports so the core has no hard dependency on vendor SDKs.

**Spec reference:** Architecture (README, adapters)

**Explanation:** Users may only need NumPy events. Vendor SDKs (metavision, dv-processing) are heavy and optional.

**Code review question:** Does the core import `h5py`, `metavision_core`, or `dv_processing` at module load time?

---

## 9. Kernel Interface is Frozen

**Invariant:** The kernel protocol (accumulate, snapshot, init_state, reset) is stable. New kernels may be added, but the interface must not change in breaking ways.

**Spec reference:** [kernel_contract.md](spec/kernel_contract.md)

**Explanation:** Kernels may be implemented in C++ or other languages. The contract is the ABI.

**Code review question:** Does any change alter the kernel method signatures or semantics?

---

## 10. Timestamp Monotonicity Never Assumed

**Invariant:** Kernels must not assume event timestamps are monotonically increasing. They must handle out-of-order timestamps correctly.

**Spec reference:** [kernel_contract.md §6](spec/kernel_contract.md#6-timestamp-monotonicity)

**Explanation:** Real-world event streams may have reordering (network, file formats, sensor quirks). Kernels use max-per-pixel or similar to handle this.

**Code review question:** Does any kernel assume `events["t"][i] >= events["t"][i-1]`?

---

## 11. Telemetry Counters Always Maintained

**Invariant:** The engine maintains `events_ingested`, `events_dropped`, `chunks_dropped`, `snapshots_published`, `uptime_ns` unconditionally  -  even when no Telemetry consumer is attached.

**Spec reference:** [engine.md §2](spec/engine.md#2-engine-ownership), [overflow.md §4](spec/overflow.md#4-counters)

**Explanation:** Any caller can inspect `engine.stats` at any time. Counters are not gated by consumer attachment.

**Code review question:** Are counters only updated when a Telemetry consumer is present?

---

## 12. Recording Never Inside Ingest

**Invariant:** The recorder (`frames2py.recorder`) is a sink the caller writes to next to `ingest()`. The engine never calls it or waits for it, and it starts no thread. `write()` does its compression and file I/O on the caller's thread.

**Spec reference:** [recorder.md](recorder.md)

**Explanation:** The engine's ingest path is the same with or without a recorder. A caller that records and ingests on one thread is bounded by both; the measured rates are in recorder.md.

**Code review question:** Does the engine ever call the recorder, or does the recorder add a hook, thread or queue?
