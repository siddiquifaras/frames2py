# Overflow and Drop Semantics

**Version:** 1.0  
**Status:** Formal specification for frames2py ring buffer overflow.

---

## 1. Ring Buffer Structure

The ring buffer is **chunk-based**:

- **Slots**: A fixed, pre-allocated array of `capacity` slots.
- **Chunk size**: Each slot holds at most `chunk_size` events.
- **No heap allocation** on the write path. All slots are allocated at construction.

### 1.1 Layout

```
slots[0], slots[1], ..., slots[capacity-1]
counts[0], counts[1], ..., counts[capacity-1]   -  events actually stored per slot
```

- `write_idx`: Next slot to write.
- `read_idx`: Next slot to read (oldest unread).
- `size`: Number of occupied slots.

---

## 2. Overflow Eviction

When the buffer is full (`size == capacity`) and a new chunk must be written:

### 2.1 DROP_OLDEST

- **Evict** the oldest unread chunk (at `read_idx`).
- Advance `read_idx`, decrement `size`.
- Write the new chunk to `write_idx`, advance `write_idx`, increment `size`.
- **Atomicity**: The eviction and write are performed as a single logical operation per chunk. No partial evictions.

### 2.2 DROP_NEWEST

- **Discard** the incoming chunk. Do not write.
- Increment drop counters for the discarded events.
- Do not advance `read_idx` or `write_idx`.

---

## 3. Whole-Chunk Eviction

- Overflow evicts **whole chunks** atomically.
- **Never** evict individual events within a chunk.
- A chunk is either fully retained or fully evicted.

---

## 4. Counters

All counters are **monotonically non-decreasing** (except on explicit `reset()`).

| Counter | Description | Invariant |
|---------|-------------|-----------|
| `events_ingested` | Total events successfully written to the ring buffer | `events_ingested >= 0` |
| `events_dropped` | Total events lost due to overflow eviction | `events_dropped = Σ(event_count of each evicted chunk)` |
| `chunks_dropped` | Total chunk slots evicted | `chunks_dropped >= 0` |

### 4.1 events_dropped Semantics

- `events_dropped` is the **exact sum** of the actual event counts in each evicted chunk.
- **Not** `chunks_dropped * chunk_size`, because the last chunk in a batch may be partial (fewer than `chunk_size` events).

### 4.2 Example

- `capacity = 2`, `chunk_size = 50`
- Batch of 200 events → 4 chunks (50, 50, 50, 50)
- Only 2 slots available → evict 2 chunks to make room
- `events_dropped += 50 + 50 = 100`
- `chunks_dropped += 2`

---

## 5. Buffer Fill Ratio

```
buffer_fill_ratio = occupied_slots / total_slots
```

- Range: `[0.0, 1.0]`
- `occupied_slots` = number of slots currently holding data.
- `total_slots` = `capacity`.

---

## 6. No BLOCK Mode

- There is **no** blocking or backpressure mode.
- When the buffer is full, the overflow policy (DROP_OLDEST or DROP_NEWEST) is applied immediately.
- This is a **permanent constraint**. Blocking would re-introduce the latency coupling that frames2py exists to eliminate.

---

## 7. Batch Splitting

When a batch exceeds `chunk_size`, it is split into multiple chunks:

- Chunk 1: `events[0:chunk_size]`
- Chunk 2: `events[chunk_size:2*chunk_size]`
- ...
- Last chunk may be partial: `events[k*chunk_size:n]`

Each chunk is written (or dropped) independently. Eviction applies per chunk when the buffer is full.

---

## 8. Conformance

A conforming implementation must:

1. Use a chunk-based ring buffer with fixed pre-allocated slots.
2. Evict whole chunks atomically on overflow.
3. Maintain `events_ingested`, `events_dropped`, `chunks_dropped` as specified.
4. Ensure `events_dropped` equals the sum of actual event counts in evicted chunks.
5. Expose `buffer_fill_ratio` in `[0.0, 1.0]`.
6. **Never** implement a BLOCK mode.
