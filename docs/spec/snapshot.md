# Seqlock Snapshot Invariants

**Version:** 1.0  
**Status:** Formal specification for frames2py snapshot publication.

---

## 1. Overview

The snapshot bridge uses a **seqlock** (sequence lock) to publish (frame, metadata) pairs lock-free. A single writer (the engine) and multiple readers (viewer, recorder, telemetry) access the snapshot without mutexes.

---

## 2. Sequence Number Semantics

| `seq` value | Meaning |
|-------------|---------|
| **Even** | Published / idle. Safe to read. |
| **Odd** | Write in progress. Readers must retry. |

- `seq` is **monotonically non-decreasing**.
- Readers may **skip** values (observe seq 2, then 6) but **never regress** (observe 6, then 2).

---

## 3. Writer Protocol

1. **begin_write()**
   - Increment `seq` (→ odd).
   - Return the **inactive** buffer for the writer to fill.
   - The reader never reads from this buffer while it is being written.

2. **Write**
   - Fill the returned buffer with frame data.
   - Prepare metadata.

3. **end_write(meta)**
   - Store metadata.
   - Increment `seq` (→ even).
   - Toggle the active buffer index (ping-pong).

---

## 4. Reader Protocol

1. Read `seq` → `seq1`.
2. If `seq1` is **odd**, return `None` (write in progress).
3. If `seq1 == 0` and no data published yet, return `None`.
4. Copy frame from the **non-writing** buffer (the one the writer is not currently targeting).
5. Copy metadata.
6. Read `seq` again → `seq2`.
7. If `seq2 != seq1`, return `None` (torn read  -  writer overwrote during copy).
8. Return `(frame_copy, meta)`.

---

## 5. Double-Buffered Minimum

- At least **two** frame buffers are required.
- Writer writes to buffer A while readers read from buffer B, and vice versa.
- The writer and readers **never** access the same buffer simultaneously.

---

## 6. Formal Correctness Guarantee

**Invariant:** If a reader observes `seq` as **even** at the start of its copy and **unchanged** at the end, then the copied frame and metadata are **mutually consistent** and reflect a complete, atomic write by the producer.

**Proof sketch:**
- Even `seq` implies no write in progress.
- Unchanged `seq` implies no write completed during the copy (otherwise `seq` would have been incremented twice: odd→even for the new write).
- Double buffering ensures the reader reads from a buffer the writer is not modifying.

---

## 7. Return Value

- **Success:** `(frame_copy, meta)`. The frame is a **copy**; the caller owns it.
- **Failure:** `None`  -  either write in progress (odd seq) or torn read detected.

---

## 8. Reset

- `reset()` zeros both buffers, sets `seq = 0`, clears metadata.
- After reset, `try_read()` returns `None` until the first successful `end_write`.

---

## 9. Conformance

A conforming implementation must:

1. Use a sequence number with odd = writing, even = published.
2. Implement the reader protocol: check even, copy, re-check.
3. Ensure `seq` is monotonically non-decreasing.
4. Use at least double buffering.
5. Return a copy of the frame to readers (not a shared reference).
6. Provide the formal correctness guarantee above.
