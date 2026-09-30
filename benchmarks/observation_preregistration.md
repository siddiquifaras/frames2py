# Observation architectures under consumer load: preregistration

**Status: approved protocol.** The user approved this design on 2026-09-30. When this text was
committed, none of the study existed yet: no harness, no calibration and no measurement. The
commit that adds it must be an ancestor of every commit that implements the study and of every
measurement (section 23). After it, the protocol changes only through the amendments of
section 24.

Base: `main` at `50a0d6e` (Frames2Py 1.0.0). Sources: `CLAUDE.md`, `.claude/decisions.md`,
`.claude/benchmarks.md`, `.claude/hill-climbing.md`, `.claude/rules/`, the code under
`src/frames2py/` and `benchmarks/`. Evidence labels: **CODE FACT** with a path and line, or
**INFERENCE**.

## 0. Repository facts that shape this protocol

Each of these differs from what a reader might assume about the study's design.

- **Validation is not Engine machinery.** Structural validation, the timestamp-range check,
  the bounds check and the watermark are the Accumulator's (CODE FACT
  `src/frames2py/_accumulator.py:75-88`; decisions.md 14). Every arm that uses the
  Accumulator pays for them. What the Engine adds beyond a hand-rolled reference swap is
  listed in 6.6. Validation is not on that list.
- **`plan.md` is `.claude/plan.md`.** It is an execution plan and sets no requirement for
  this study.
- **The Accumulator has no public window closure.** The Engine closes a window through the
  private `Accumulator._close_window()` (CODE FACT `src/frames2py/_engine.py:148`). The
  baselines use the public `Accumulator.reset()` instead. 6.6 gives the basis, and V1
  (section 22) checks that the two are equivalent on this study's inputs.
- **The Phase 7 viewer-impact harness produces non-monotonic timestamps.** The benchmark
  reuses a pool of batches whose event time advances by `batch_size` µs per batch. On each
  reuse it shifts them by only `batch_size / 20` µs per batch (CODE FACT
  `benchmarks/consumers.py:289`, `:303-304`; `benchmarks/workloads.py:33`). Timestamps
  therefore jump backwards at every pool wrap. This study does not use that construction
  (11.1). The existing results are unchanged and are not reinterpreted here.

## 1. Motivation

Frames2Py's architectural claim goes beyond fast kernels. It separates the parts of a live
system:

    high-rate event ingestion → maintained state → published observation → independent consumers

The v1 gate (decisions.md 16; benchmarks.md "v1 gate") shows that accumulation and
`Engine.ingest()` sustain the target rate. It does not show what the observation boundary is
worth under consumer load, compared with the other ways people connect maintained state to
consumers. That is the subject of this study.

This is not a study of whether Frames2Py is better. Each architecture gives different
guarantees, and a workload on which Frames2Py loses is a valid result that will be published.

## 2. Research question

> Given different observation requirements, what are the producer throughput, ingest latency
> and jitter, consumer freshness, memory, CPU use and data-loss characteristics of different
> observation architectures under increasing consumer load?

The output is a multidimensional description of where each architecture fits (section 18),
not a ranking and not a score.

The study compares complete practical architectures. Each arm uses its idiomatic delivery:
consumers of a queue are pushed each item through a condition variable, and consumers of a
latest-state arm poll at the publication interval (5, "Delivery mechanism"). A difference
between a queue arm and a latest-state arm therefore includes the difference in delivery. It
is never attributed to the data structure alone.

One comparison is singled out and specified in advance. It is the full Frames2Py Engine (arm
H) against a minimal hand-rolled latest-state reference swap under a lock (arm G). Both use
the same Accumulator and the same delivery, polling at the publication interval, so G against
H isolates what the Engine adds: its lifecycle, statistics, publication and snapshot
contract, at their cost. The study does not assume that the answer is a throughput advantage.

## 3. Questions the study may answer

The study is designed to bear on these questions. It does not promise to settle them all.

1. Is Frames2Py materially different from a competent reference-swap implementation (G vs H)?
2. Does the Engine's contract, lifecycle and statistics machinery impose measurable overhead
   (H vs G, H vs A at N = 0)?
3. How does a slow consumer affect producer throughput, latency and freshness in each
   architecture?
4. How does queue backlog affect state freshness?
5. How does memory behave under sustained consumer lag?
6. How does the GIL change the observed behaviour (CPython 3.11 vs 3.14t, CPU-bound vs
   sleeping consumers)? The NumPy-bound against pure-Python contrast needs W2 and W6, which
   belong to the follow-up study (20.3).
7. Does a latest-state architecture remain useful as consumer count grows?
8. When is a queue the correct abstraction instead?
9. Does raw-batch fan-out become prohibitively expensive because each consumer repeats the
   accumulation?
10. Is there evidence that future Frames2Py work should support another observation
    primitive, for example push notification (`wait_for_newer`, decisions.md 40) or an
    event-preserving subscription?

## 4. Hypotheses

These are hypotheses, stated before any measurement, not conclusions. Section 17 gives, for
each one, the observation that would support it and the observation that would contradict it.
Contradictory results are published with the same prominence as confirming ones.

The hypotheses are adapted to this study's scope (11.4) by two rules:
- A part of a hypothesis, or of its judgement rule, whose cells belong to the follow-up study
  (arm D or RE, workload W2 or W6, the `timestamp_decay` kernel, gzip, or an experiment S1 to
  S7) is moved to 20.3 as it was written. It is not reworded to fit.
- Where a judgement rule used N = 8 as the largest consumer count, it is restated at N = 4,
  the largest count here, with its threshold derived the same way. These rules are marked
  "adapted" in section 17.

- **H1 (inline coupling).** In arm A the producer's per-batch step contains every consumer's
  work, so the producer is not sustained whenever N × (consumer cost per observation) plus
  the accumulation load exceeds the publication interval.
- **H2 (backpressure).** In arm B (bounded, blocking) the producer is not sustained whenever
  the consumer's service time exceeds the publication interval. The publication rate falls
  to the consumer's service rate, and memory stays bounded.
- **H3 (drop policy).** Arm C keeps the producer sustained and memory bounded, and discards
  publications according to its policy. (Arm D's part is in 20.3.)
- **H4 (unbounded).** Arm E preserves every publication on the producer side. When consumers
  are slower than the cadence, queue depth, freshness age and memory grow roughly linearly
  with elapsed time.
- **H5 (copy under lock).** In arm F the producer's step latency tail grows with N, because
  consumers' copies hold the lock the producer needs. (The frame-size part, S3, and the
  lock-held variant FH, S5, are in 20.3.)
- **H6 (reference swap is strong).** Arm G is not distinguishable from arm H (16.3) on
  producer busy time per event, step latency p99 or freshness age in most state-observation
  cells.
- **H7 (meaning of a tie).** If H6 holds, the result is legitimate and informative. It would
  suggest that Frames2Py's value lies not in faster reference publication but in the complete
  event-state abstraction: validation shared through the Accumulator, lifecycle, statistics,
  snapshot semantics and the contract (6.6). H7 is an interpretation rule, fixed in advance,
  not a measurable prediction.
- **H8 (queues for preservation).** In the event-preservation class (P2), the queue-fed
  recorder arms record every event. The Frames2Py pattern (recording on the producer thread)
  also records every event, and the producer bears the recorder's cost. With Blosc it
  sustains 20M events/s. (The gzip part is in 20.3.)
- **H9 (Frames2Py for state observation).** Arm H keeps the producer sustained in every
  state-observation cell in which the producer's own CPU is not contended, with memory
  bounded and freshness age within two publication intervals. A slow consumer observes less
  often; the state it obtains is no older.
- **H10 (GIL).** On CPython 3.11, a pure-Python CPU-bound consumer (W5) raises the producer's
  step latency p99 by at least the interpreter switch interval (5 ms default) over its N = 0
  reference, in the latest-state arms (F, G, H), where consumer work runs outside the
  producer's step. On 3.14t with the GIL disabled it does not. (The W6 and W2 parts are in
  20.3.)
- **H11 (sleep vs CPU).** Sleeping consumers (W3) leave the latest-state producer's step
  latency within the noise band of N = 0. (The comparison with CPU-bound consumers of the
  same nominal duration, W2 and W6, and the queue-arm part are in 20.3.)
- **H12 (raw-batch fan-out).** In arm RB, total accumulation CPU grows about linearly with N.
  On 3.11 it stops being sustained at a smaller N than the finished-frame arm B does. (The RE
  part is in 20.3.)
- **H13 (push vs poll).** When consumers keep up (W1, N = 1), queue arms observe new state
  sooner after the producer step (post-step observation delay, section 9) than poll-based
  arms (F, G, H), whose delay is spread over the poll interval (median about 8 ms at 16 ms).
- **H14 (core placement).** Producer busy throughput (events per second of producer step
  time) is higher with consumer threads running than at N = 0. This repeats the Phase 7
  observation (benchmarks.md "Consumers", viewer impact) as a platform effect, not an
  architectural one. It is stated in advance so it is not misread as consumers speeding the
  producer up.

## 5. Controlled variables

**Identical in every arm:**

| Variable | Value |
|---|---|
| Accumulation implementation | `frames2py.Accumulator` from the installed Frames2Py 1.0.0 wheel. The Engine composes the same class internally (`src/frames2py/_engine.py:63`). No arm reimplements accumulation. |
| Kernel implementation and parameters | `frames2py.EventCount()`, which takes no parameters |
| Event contract | `EVENT_DTYPE`. Every batch passes the Accumulator's structural validation, range check and bounds check. |
| Input data | the same materialised batches, byte for byte, for every arm of a condition (11.1) |
| Sensor dimensions | per condition |
| Accumulation semantics | whole-call accumulation through the kernel's own strategy (decisions.md 13, 15) |
| Publication cadence rule | 6.3, identical to the Engine's |
| Producer loop, pacing and source | 6.2 |
| Consumer workloads | section 8, the same code in every arm |
| Instrumentation | the same per-batch and per-observation records in every arm (6.2, 21.4) |
| Process, runtime, environment | one run per process; same interpreters, packages and controls |

**Independent variables:** architecture (arm), consumer workload, consumer count N and
runtime. The kernel, offered rate, batch size, resolution, queue capacities, input source and
publication interval are fixed (11.3). Varying them belongs to the follow-up study (20.3).

**Deliberately different, and labelled as experimental variables:**

- **Accumulation location and multiplicity.** In RB each consumer accumulates the raw
  batches itself, using the same Accumulator implementation. N consumers perform N
  accumulations. The implementation is the same; its location and multiplicity differ. That
  is the variable being studied, not a violation of the control. RB is analysed as its own
  family (18.2).
- **Delivery mechanism.** Queue arms (B, C, E, RB) notify consumers through a condition
  variable (push). Latest-state arms (F, G, H) are polled at the publication cadence, as the
  Frames2Py docs instruct (`docs/content/consumers/writing-a-consumer.md`, "The pattern").
  Frames2Py v1 has no push API (decisions.md 40). Each arm's delivery is part of the
  architecture being compared (section 2). The effect is measured (section 9) and not hidden.
- **Guarantees.** Each arm keeps its own semantics (section 7). No arm is modified to imitate
  another's.

## 6. Architecture definitions

### 6.1 Terms

- **Batch k:** the k-th event array released by the source, 0-based.
- **Step:** the arm-specific work the producer does for one batch (6.4).
- **Publication:** a finished representation made available to consumers, with its watermark
  and a sequence number that starts at 1 and increases by 1.
- **Observation:** a consumer obtaining a state it will process. Its instant is O (section 9).
- **Work:** the consumer workload applied to one observation (section 8), from O to W.
- **Windowed kernel:** `event_count`, the only kernel in this study. (The running kernel
  `timestamp_decay` is in the follow-up study, 20.3.)

### 6.2 Common producer loop

Every arm runs this loop on one producer `threading.Thread`, started by the child process's
main thread. Only `step(batch)` differs between arms.

```text
wait until every consumer thread has signalled ready; T_start = perf_counter_ns()
for k = 0, 1, 2, ...:
    A_k = T_start + 1000 * (M_k - t_0 + 1)          # ns; scheduled availability of batch k
    if A_k > T_start + 15 s or stop.is_set(): break
    while (r = A_k - perf_counter_ns()) > 0:
        time.sleep(min(r, 1_000_000) / 1e9)         # sleeps only, in slices of at most 1 ms
    Q_k = perf_counter_ns(); cq_k = thread_time_ns()          # released
    batch = materialise(k)                                     # a fresh array (11.1)
    S_k = perf_counter_ns(); cs_k = thread_time_ns()
    step(batch)
    E_k = perf_counter_ns(); ce_k = thread_time_ns()
    p_k = published_sequence()
    append (k, Q_k, S_k, E_k, cq_k, cs_k, ce_k, p_k) to preallocated per-batch logs
```

- `M_k` is the largest timestamp in batches 0..k and `t_0` the smallest timestamp in batch
  0, both computed before the run. For the synthetic streams, `A_k = T_start + (k+1)·B/r`
  exactly (11.1).
- The pacing wait never spins. A spinning producer would hold the GIL on 3.11 and distort
  the effect being measured.
- `published_sequence()` is the arm's latest publication number, or 0 if there is none:
  - H: `s = engine.snapshot()`, then `s.meta.sequence`, or 0 if `s` is `None`. This is the
    only call the harness makes into arm H besides `ingest()`. It is public and takes no
    Frames2Py lock (CODE FACT `src/frames2py/_engine.py:101-104`,
    `src/frames2py/publish.py:80-81`).
  - A, B, C, E, F, G: the arm's own counter.
  - RB: always 0. This arm has no producer-side publication.
- Two things end the run:
  - the producer stops releasing batches at `T_start + 15 s`;
  - the monitor (6.8) sets `stop` at that time, or earlier on the memory ceiling (14.1).
- **Warm-up:** `[T_start, T_start + 5 s)`.
- **Measurement window W:** `[T0, T1) = [T_start + 5 s, T_start + 15 s)`, 10 s. `|W|` below
  is its length, 10 s.
- The source never drops a batch. A producer that falls behind releases overdue batches as
  soon as it can, and its lag is recorded (section 15). In a live system that backlog would
  sit, or be lost, upstream of the application (driver or SDK buffers). This study reports
  it as the implied upstream buffer (section 10) and does not model it.

### 6.3 Publication cadence (arms A, B, C, E, F and G)

Checked once per accumulation call, after accumulation, exactly as the Engine does it (CODE
FACT `src/frames2py/_engine.py:97-99`):

```text
now = time.monotonic_ns()
if last is None or now - last >= interval_ns:  publish; last = now
```

`interval_ns = snapshot_interval_ms * 1e6`. The interval is 16 ms. Arm H uses
`Engine(..., snapshot_interval_ms=interval)` and the Engine's own check. RB applies the same
rule on each consumer thread, after that consumer's accumulation (6.4).

### 6.4 Arms

Notation:
- `acc = frames2py.Accumulator(sensor_size, kernel)`.
- An item is the tuple `(frame, watermark, sequence)`.
- `publish_state()`, used by A, B, C, E, F and G:

```text
frame = acc.read()                  # a new array, the Accumulator's public read
w = acc.watermark
if kernel is windowed: acc.reset()  # window closure (6.6)
seq += 1
return (frame, w, seq)
```

Observation and work use the consumer loops of 6.5.

| Arm | Producer step | Consumers | Family |
|---|---|---|---|
| **A** inline | `acc.accumulate(b)`; if due: `item = publish_state()`, then for c = 1..N in order, observe and work on `item` on the producer thread | none: consumer work runs inside the step | finished frame, synchronous |
| **B** finished-frame queue, blocking | `acc.accumulate(b)`; if due: `item = publish_state()`; for c = 1..N: `q_c.put(item)`, waiting while full | one queue per consumer, capacity K_ff; queue loop | finished frame, fan-out |
| **C** finished-frame queue, drop oldest | as B; a full queue evicts its oldest item, then appends | as B | finished frame, fan-out |
| **E** finished-frame queue, unbounded | as B, no capacity | as B | finished frame, fan-out |
| **F** latest frame, copy under lock | `acc.accumulate(b)`; if due: `frame, w, s = publish_state()`; `with L: np.copyto(shared, frame); meta = (w, s)` | poll loop; per tick `with L:` if `meta.seq` is new, `np.copyto(private_c, shared)` and take `meta`; work on `private_c` outside the lock | latest state |
| **G** latest frame, reference swap under lock | `acc.accumulate(b)`; if due: `item = publish_state()`; `with L: slot = item` | poll loop; per tick `with L: item = slot`; if `item.seq` is new, work outside the lock | latest state |
| **H** Frames2Py | `engine.ingest(b)` on `Engine(sensor_size, kernel, snapshot_interval_ms=interval)` | poll loop; per tick `s = engine.snapshot()`; if `s.meta.sequence` is new, work on `s` | latest state |
| **H′** Frames2Py, A/A control | identical to H | identical to H | control (16.3) |
| **RB** raw-batch queue, blocking | for c = 1..N: `q_c.put(b)`, waiting while full | one queue per consumer, capacity K_raw; raw loop with its own `Accumulator` | raw batch, fan-out |

Arms D (drop newest), RE (raw-batch queue, unbounded) and FH (work under lock) are defined in
the follow-up study (20.3) and are not run here.

Details that apply across the table:

- **Shared items.** In A, B, C, E and G, one item, and so one frame, is created per
  publication. The same object goes to every consumer and every queue. No workload in this
  study copies the frame. Consumers treat shared frames as read-only.
- **F's buffers.** `shared` and each `private_c` are allocated once, before `T_start`. F
  allocates one frame per publication through `acc.read()`, as A, B, C, E and G do. It then
  copies once into `shared`, and once per observing consumer.
- **H's handoff.** The Engine publishes each snapshot as a fresh frame that it never writes
  again, stored with its metadata as one `Snapshot` in a one-element list (decisions.md 6,
  11). The handoff rests on two different things:
  - documented by CPython: single list-item reads and writes are atomic;
  - CPython 3.14 source-level behaviour, not a Python guarantee: a release store under the
    list's lock, and a sequentially consistent, reference-guarded load. With the GIL, the GIL
    orders them.

  Consumers take no Frames2Py lock. CPython itself locks the list during the store
  (decisions.md 6, 38).
- **RB batches.** The producer puts the same batch object into every consumer's queue. This
  is safe because the Accumulator does not modify its input: on the all-in-bounds path it
  returns the input array itself (CODE FACT `src/frames2py/_accumulator.py:90-96`),
  and the kernels only read fields.

**Event-preservation arms (P2).** Each keeps state with the Engine and records with
`frames2py.recorder`:
- The recorder is opened before `T_start` with `sensor_size` and `compression="blosc"`.
- It writes to `scratch/observation_study/tmp/<run id>.h5`.

| Arm | Producer step | Recorder | Basis |
|---|---|---|---|
| **EP-H** Frames2Py recorder pattern | `rec.write(b); engine.ingest(b)` | on the producer thread, inside the step | "call it from the producer's loop, next to `ingest()`" (`docs/content/consumers/writing-a-consumer.md:35`); `src/frames2py/recorder/__init__.py` docstring order |
| **EP-QB** raw-batch queue to a recorder thread, blocking | `q.put(b)` (capacity K_raw, waits while full), then `engine.ingest(b)` | own thread: `b = q.get(); rec.write(b)` | "call `write()` from a thread of your own" (`docs/content/data/recorder.md:156`) |
| **EP-QE** as EP-QB, unbounded | `q.put(b)`, then `engine.ingest(b)` | as EP-QB | as EP-QB |

After T1:
1. The producer stops.
2. EP-QB and EP-QE drain their queue to the recorder. The time from T1 until the queue is
   empty is the completion delay. It is capped at 120 s (14.1).
3. `rec.close()` runs untimed; its duration is recorded.
4. The file is read back with `frames2py.adapters.hdf5.open(path, group="events",
   sensor_size=...)`. Its events are hashed and compared with the re-materialised batches
   that were fed to the recorder (21.4).
5. The file is deleted.

### 6.5 Consumer loops

**Poll loop (F, G, H, H′).** This is the deadline pacing of `frames2py.viewer`'s loop
(CODE FACT `src/frames2py/viewer/_run.py:102-115`), reimplemented in the harness with
`perf_counter_ns`:

```text
deadline = perf_counter_ns(); last = None
while not stop.is_set():
    state = read_latest()                     # per arm, see the table
    if state is not None and state.seq != last:
        O = perf_counter_ns(); co = thread_time_ns()
        work(state)
        W = perf_counter_ns(); cw = thread_time_ns()
        log observation (O, W, co, cw, state.seq, state.watermark); last = state.seq
    count the poll
    deadline += poll_interval_ns; now = perf_counter_ns()
    if deadline > now: time.sleep((deadline - now) / 1e9)
    else: deadline = now                      # missed ticks are skipped, not repeated
```

`poll_interval` is the publication interval, 16 ms.

**Queue loop (B, C, E).** `item = q.get(timeout=0.1)`. On `None`, re-check `stop` and continue.
Otherwise observe and work. Items are processed in FIFO order. There is no conflation.

**Raw loop (RB).** Each consumer owns `acc_c = Accumulator(sensor_size, kernel)`:

```text
b = q.get(timeout=0.1); if b is None: continue
acc_c.accumulate(b)
while (b = q.get_nowait()) is not None: acc_c.accumulate(b)   # drain what is queued now
if due_c(): state = publish_state_c(); observe; work           # 6.3 rule on the consumer
```

Draining before observing is the competent pattern. A consumer that observed after every
batch would fall behind by construction.

**Recorder loop (EP-QB, EP-QE).** `b = q.get(timeout=0.1)`; if `b` is not `None`,
`rec.write(b)`.

Every consumer thread records `(perf_counter_ns, thread_time_ns)`:
- at its first loop iteration at or after T0;
- at its last loop iteration before T1;
- around every observation;
- in the raw loop, around each get-and-drain-and-accumulate block (accumulation CPU).

### 6.5.1 Queue implementation

All queue arms, finished-frame, raw and EP, use one harness class, `PolicyQueue(capacity,
policy)`. It mirrors the structure of the standard library's `queue.Queue`: one
`threading.Lock`, two `threading.Condition`s on it (`not_empty`, `not_full`) and a
`collections.deque`. The policies differ only in the full-queue branch of `put`:

| Policy | When full |
|---|---|
| `BLOCK` | `not_full.wait(0.1)` in a loop until there is space or `stop` is set. A put still waiting when `stop` is set is abandoned and logged `(consumer, seq)`. Its item is backlog at stop (7.4), never loss. |
| `DROP_OLDEST` | pop the oldest item, append the new one, log `(consumer, evicted seq)` |
| `DROP_NEWEST` | don't append, log `(consumer, rejected seq)`; arm D's policy, not run in this study |
| `UNBOUNDED` | never full |

- `get(timeout)` waits on `not_empty`. After removing an item it notifies `not_full`.
- `get_nowait()` returns `None` when the queue is empty.
- A harness test checks the class against `queue.Queue` for `BLOCK` and `UNBOUNDED` (FIFO
  order, no loss), and checks the drop policies' exact counts (V2).

**Capacities.** They are fixed here, not tuned:

| Queue | Capacity | Time depth at 100k / 20M/s / 16 ms |
|---|---|---|
| Finished frame, `K_ff` | 4 publications | 80 ms (publications are 20 ms apart: correction 1, 24) |
| Raw batch, `K_raw` | `ceil(64 ms / batch period)` | 13 batches = 65 ms |

The same capacities serve the EP arms' raw queue (EP-QB). Capacities for other conditions,
and the capacity variation of S4, are in the follow-up study (20.3).

### 6.6 What arm H adds beyond arm G

Both use the same Accumulator, so both pay for validation, the range check, the bounds check
and the watermark. Per `ingest()` the Engine additionally does the following (CODE FACT
`src/frames2py/_engine.py:77-99`, `:143-151`; `src/frames2py/publish.py:68-78`):
- the caller's thread identity;
- the lifecycle lock (uncontended in this study);
- the producer-ownership and running checks;
- `events_ingested` and the pending flag;
- the cadence check.

Per publication it additionally does:
- `np.empty` then the kernel read, the same work as `Accumulator.read()`;
- builds `SnapshotMeta`;
- `frame.flags.writeable = False` and `frame.view()`;
- builds a `Snapshot`;
- the list-slot store;
- `_close_window()`;
- counters.

Its consumers call `engine.snapshot()`, which returns the list item and takes no Frames2Py
lock. G's consumers take a `threading.Lock` for the reference instead.

Beyond cost, H provides contract features that G lacks. They are tabulated alongside the
numbers (18.1, T5):
- read-only published frames;
- producer-thread ownership enforcement;
- start, stop and reset semantics (stop publishes the pending window);
- the sequence across `reset()`;
- `EngineStats`;
- the fail-closed free-threaded runtime check;
- the documented basis for the memory ordering of the handoff (decisions.md 6).

**Window closure in A, B, C, E, F, G and RB.** For `EventCount`, `close_window` and `reset`
are the same kernel operation, `state.counts[...] = 0` (CODE FACT
`src/frames2py/kernels/_builtin.py:86-90`). `Accumulator.reset()` also sets the watermark to
`None` and the out-of-bounds count to 0 (`src/frames2py/_accumulator.py:106-110`). Every
window in this study is non-empty, and the synthetic stream is in timestamp order and in
bounds. Under those conditions the watermark each baseline publishes equals the Engine's. V1
checks this equality on the input used.

### 6.7 Competence requirements

Every baseline must meet these. They are reviewed against the code before any measurement
(21.3):

1. No redundant frame copies. A finished frame is made once per publication by
   `Accumulator.read()` and shared by reference. F copies exactly once into shared storage
   and once per observing consumer.
2. Lock scope is minimal: F holds its lock for the copy, G for the reference.
3. No consumer holds a lock while working.
4. Poll consumers pace at the publication interval with deadline pacing, skip missed ticks,
   and process only new sequences. This is Frames2Py's documented consumer pattern.
5. Queue consumers block on the condition and never sleep-poll.
6. Raw consumers drain queued batches before observing.
7. Producers never spin and never sleep outside the pacing wait.
8. No parameter is tuned per arm. Every parameter is fixed in this document.

### 6.8 Monitor

The child process's main thread is the monitor for every arm. It holds no arm state.

- **Every 100 ms** it samples:
  - `proc_pid_rusage(RUSAGE_INFO_V6)` of its own process, through ctypes, with the fields
    listed in section 10 (the same structure as `scratch/phase5/char2/rusage.py`);
  - `resource.getrusage(RUSAGE_SELF)`.
- **At T1** it sets `stop`.
- **Memory ceiling:** if `phys_footprint − (phys_footprint at T_start) > 4 GiB`, it sets
  `stop` early and records the outcome (14.1).
- **Shutdown:** after `stop` it joins every thread, with a grace period of 10 s.

### 6.9 Architectures not included

- **Shared work queue (competing consumers).**
  - Each item goes to exactly one consumer, so no consumer observes the stream or its state.
    For state observation that is a different semantics. For event preservation with one
    recorder it reduces to a fan-out queue with N = 1.
  - Its use case is load-balancing independent frame jobs across a worker pool (for example
    inference throughput). That is a different question, left to a follow-up (20.2).
  - It is excluded as not relevant to the research question, not for size.
- **Raw-batch queues with drop policies.** A dropped raw batch is permanently missing from
  that consumer's accumulated state, so the state no longer represents the stream. The
  finished-frame drop arms (C here, D in the follow-up study) measure the policies. This
  family is a follow-up.
- **Consumer-side conflation in finished-frame queues** (drain and keep only the newest).
  Its producer-side equivalent is drop-oldest with capacity 1 (C at K_ff = 1, in the
  follow-up study's S4).
- **Latest state with a condition-variable notification.** A condition variable on the
  producer path is excluded from Frames2Py by decisions.md 38. As a baseline it would add a
  push-latest variant rather than the minimal form of Frames2Py's own idea (G). Push
  observation is a follow-up (`wait_for_newer`, 20.2).
- **Cross-process, shared-memory and multiprocessing designs.** Out of scope (20.1).

## 7. Workload classes and semantic fairness

### 7.1 State observation (SO): experiment P1

- The consumer wants the current representation. Intermediate states need not all be
  observed. Freshness matters.
- Frames2Py is designed for this class.
- Arms A, B, C, E, F, G, H, H′ and RB run here.

### 7.2 Event preservation (EP): experiment P2

- Every event given to the producer must reach a durable recording, in order, exactly once,
  verified by reading the file back.
- Each architecture uses its idiomatic pattern:
  - Frames2Py: `recorder.write()` next to `ingest()` on the producer thread. This is the
    documented pattern. It is not a snapshot consumer, and its cost to the producer is a
    measured outcome.
  - Queue arms: a recorder thread fed by a raw-batch queue.
- Latest-state arms (F, G, H as a snapshot consumer) and the drop-policy arm (C) are not run
  in P2. They cannot meet the requirement by construction, and P1 already measures how much
  they skip or drop. Frames2Py's latest-state snapshots are never treated as equivalent to
  preserving events or batches.

### 7.3 Frame preservation

Between SO and EP there is a third requirement: every published frame reaches every
consumer. P1 measures it through publication coverage (section 10). The profile "SO-complete"
in 18.1 reports which arms meet it. Latest-state arms do not meet it by design. That fact is
reported as their semantics, not as data loss.

### 7.4 What is and isn't loss

| Category | Meaning | Where it occurs | Classification |
|---|---|---|---|
| Skipped publication | published, never observed by consumer c (a gap in the sequence c observed) | F, G, H, H′ | expected, by design |
| Unobserved window | a skipped or dropped publication of the windowed kernel; its events are in no state consumer c observed | F, G, H, H′, C | expected; quantified as the unobserved-event fraction |
| Declared drop | an item the queue's policy discarded | C (oldest) | expected, by policy |
| Backlog at stop | items queued, or batches not yet released, when the window ended | any arm | an outcome, not loss |
| Unexpected loss | any event, batch or item not accounted for by the rows above | any arm | invalid run (14.1) |

"Data loss" is used only for the last row. In SO, expected replacement of state is never
called data loss. A queue's preservation behaviour is never called unnecessary because
Frames2Py doesn't need it.

## 8. Consumer workloads

Every workload is the same function in every arm. It receives `(frame, watermark, sequence)`.
Arm H also receives the `Snapshot` itself. The work amount K5 is fixed by the calibration in
V5 (section 22), then frozen.

This study runs W1, W3 and W5 as consumer workloads, and the W4 class as the event
preservation experiment P2 with Blosc only. W2 and W6, and gzip in W4, are in the follow-up
study (20.3). Their definitions stay here so the follow-up study can refer to them.

| Id | Workload | Definition | Nominal cost | Resource character | Why it exists |
|---|---|---|---|---|---|
| W1 | rendering-like | `frames2py.viewer.render(snapshot)`, `scale=None`, output discarded. Baselines pass `frames2py.publish.Snapshot(frame, frames2py.SnapshotMeta(watermark=w, sequence=s))`. | whatever the real render costs, measured. Prior evidence at 1280x720: 4.4 to 12.5 ms depending on kernel, content and runtime (benchmarks.md "Consumers") | NumPy; reads the frame in place; allocates about 21 to 28 MiB of temporaries per render at 1280x720 | the real viewer consumer; fits the 16 ms cadence at N = 1 by prior evidence |
| W2 | CPU-bound, inference-like | `x = frame.astype(np.float32)`; then K2 times `np.multiply(x, 0.5, out=x); np.add(x, 1.0, out=x); np.sqrt(x, out=x)`; then `float(x.sum())` | 30 ms at 1280x720 on runtime A in isolation (calibrated) | NumPy ufunc loops on a private copy: CPU-bound, releases the GIL inside large ufunc loops, reacquires it between calls | compute near twice the cadence, of the kind NumPy or native inference does |
| W3 | sleep- or I/O-bound | hold the frame reference; `time.sleep(0.030)` | 30 ms | no CPU; GIL released | waiting on I/O, a device or the network while holding the state; the same nominal duration as W2, to isolate resource type from duration |
| W4 | recorder-like | the EP class (7.2): `frames2py.recorder`, Blosc by default, gzip as stress | depends on write size and compression (benchmarks.md "Recorder") | compression and file I/O | event preservation |
| W5 | pathologically slow | `a = s & 0x7FFFFFFF`; K5 times `a = (a * 1103515245 + 12345) & 0x7FFFFFFF` (pure Python) | 250 ms on runtime A in isolation (calibrated): about 15 publication intervals | pure-Python CPU; holds the GIL except at the switch interval | a consumer that monopolises the interpreter: a per-pixel Python loop, an accidental heavy callback. Slow and GIL-holding at once, labelled as such. |
| W6 | pure-Python, moderate | as W5 with K6 | 30 ms (calibrated) | pure-Python CPU; holds the GIL | isolates GIL-holding from duration: W2, W3 and W6 share the nominal 30 ms. W5 and W6 differ only in magnitude. |

- W2's sequence of operations is deterministic and does not overflow: the values converge
  towards the fixed point of `sqrt(0.5x + 1)`. `astype` always makes a new array, so W2
  never writes to a shared frame.
- The same K5 is used on both runtimes. Its cost on runtime B is measured and reported, not
  re-calibrated.
- CPU-bound and sleeping consumers are not assumed to be equivalent. W3 and W5 differ in
  both resource type and duration; the contrast at equal duration needs W2 and W6 (20.3).

## 9. Freshness and clock domains

### 9.1 Three clock domains

1. **Event time** `t`, µs, from the stream.
2. **Producer wall time**, `time.perf_counter_ns()` on the producer thread: A_k (scheduled,
   derived from T_start), Q_k, S_k and E_k (6.2).
3. **Consumer wall time**, `time.perf_counter_ns()` on consumer threads: O and W.

Domains 2 and 3 use the same process-wide monotonic clock. Event time is never subtracted
from wall time. It is used only to identify which batch an observed state's newest event came
from. The Engine's cadence reads `time.monotonic_ns()`, and so do the baselines (6.3). No metric
uses that clock.

### 9.2 Associating an observed state with its newest event

- Before the run, the harness records `m_k`, the largest timestamp in batch k, for every
  batch.
- During the run the producer records S_k and E_k for every batch (6.2). Every observation
  records O, W and the observed state's watermark `w`.
- After the run, offline, each observation is resolved to `k* = min{ k : m_k = w and S_k < O }`:
  the batch that contained the newest event in the observed state.
  - In every arm the watermark is the largest in-bounds timestamp in the observed state:
    - H: `snapshot.meta.watermark`;
    - A, B, C, E, F, G: `acc.watermark` at publication;
    - RB: the consumer's own `acc_c.watermark`.
  - For the synthetic streams, `m_k` is strictly increasing, so `k*` is unique.
  - A `w` that matches no batch is an integrity failure (14.1).
- Each ingest call's wall time is thus mapped to the newest event timestamp it contained,
  and consumers' watermarks are resolved through that map. The harness records it
  identically for every arm, and nothing is added to `src/`.

### 9.3 Freshness metrics

| Metric | Definition | Role |
|---|---|---|
| **Freshness age** | `O − S_{k*}` | primary; consumer observation time minus the producer wall time at which the batch holding the newest observed event entered the producer's step |
| End-to-end age | `O − A_{k*} = (Q_{k*} − A_{k*}) + (S_{k*} − Q_{k*}) + (O − S_{k*})` | adds the producer's own lag behind the offered schedule; essential for arms that stall the producer |
| Age at completion | `W − S_{k*}` | how old the state was when the consumer finished with it |
| Post-step observation delay | `O − E_{k*}` | signed; how long after the producer step that made the state available the consumer obtained it; shows push vs poll (H13) |

Where O is taken:
- A: at the consumer's inline call, inside the producer step. Consumer c's freshness includes
  the work of consumers 1..c−1, so A is also reported per consumer position.
- B, C, E: when `get()` returns the item.
- F: after the private copy completes and the lock is released.
- G: after the reference is taken and the lock released.
- H, H′: when `snapshot()` returns a new sequence.
- RB: after `acc_c.read()` returns.

Limitations, stated rather than hidden:
- **Publication instant.** The instant of publication inside `Engine.ingest()` cannot be
  observed through the public API. The protocol forbids hooks in `src/`. The post-step delay
  therefore uses the step end E_k for every arm. It is negative when a consumer obtains the
  state before the producer step returns:
  - queue arms with N > 1, while the producer is still putting to the other queues;
  - H, while `_close_window()` runs;
  - A always. For A it is reported as not applicable.
- **Poll interval.** Freshness in poll-based arms includes up to one poll interval of
  waiting. That is the documented observation pattern (decisions.md 40, consumer guidance),
  not an artefact.

## 10. Metrics

Conventions for every metric:
- Every metric is computed per run over the window W, unless its row says otherwise.
- Percentiles are nearest-rank (`benchmarks.measure.nearest_rank`).
- Consumer metrics are pooled over the run's consumers. Per-consumer values are kept as well.
- Cell values are the median of per-run values with their min-max range (section 16).
- "n/a" marks an arm where a metric has no meaning.

### 10.1 Producer

| Metric | Unit | Clock | Boundary | Aggregation per run | Notes |
|---|---|---|---|---|---|
| Offered rate | events/s | schedule | batches with A_k in W | Σ events / \|W\| | 20M/s for P1 and P2, by construction |
| Achieved rate | events/s | perf_counter | batches with E_k in W | Σ events / \|W\|; also the ratio to offered | |
| Step latency (the arm's `ingest` latency) | µs | perf_counter | S_k → E_k, steps with S_k in W | p50, p95, p99, max, with the sample count | A's step includes consumer work; RB's step is only the puts |
| Jitter | µs | perf_counter | as step latency | p99 − p50 of step latency | |
| Producer lag | ms | perf_counter | L_k = E_k − A_k, batches with A_k in W | median, p99, max; median over the window's last 1 s; least-squares slope against A_k, in ms/s | |
| Release lateness | ms | perf_counter | Q_k − A_k | median, p99, max | pacing overshoot plus backlog |
| Source backlog at T1 | batches, events | schedule | batches with A_k ≤ T1 not yet released | count | also the implied upstream buffer, events × 13 B |
| Producer off-CPU time | ms/s; µs per step | perf_counter and thread_time | (E_k − S_k) − (ce_k − cs_k) | sum / \|W\|; p99 per step | time inside the step not on CPU: GIL waits, lock waits, blocking puts, preemption. Uniform across arms. It replaces per-arm wait instrumentation, which H cannot have without `src/` hooks. |
| Excess step latency | µs | derived | cell p99(N) − cell p99(N = 0) for the same arm and runtime | cell level | A, F, G, H, H′ have N = 0 cells. B, C and E use A's N = 0 cell, which is the same work without queues. n/a for RB. |
| Producer busy fraction; busy throughput | fraction; events/s | perf_counter | Σ (E_k − S_k) over W | / \|W\|; events / Σ | |
| Producer CPU | cores | thread_time | Σ (ce_k − cq_k) over W, materialisation included | / \|W\| | |
| Accumulation-path CPU per event | ns/event | thread_time | every arm but RB: producer step CPU / events stepped. RB: Σ over consumers of accumulation CPU / mean events accumulated per consumer | value | comparable across families; in RB it grows with N by construction |
| Publication rate | 1/s | perf_counter | distinct p_k values first seen at a step in W | count / \|W\| | n/a for RB (see consumer observation rate) |
| Unexpected loss | events | accounting (21.4) | whole run | count | must be 0 (14.1) |

### 10.2 Consumer

| Metric | Unit | Clock | Boundary | Aggregation per run | Notes |
|---|---|---|---|---|---|
| Observation rate | 1/s | perf_counter | observations with O in W | count / \|W\|, per consumer and pooled | |
| Freshness age; end-to-end age; age at completion; post-step delay | ms | perf_counter | 9.3 | p50, p95, p99, max | post-step delay n/a for A |
| Processing latency | ms | perf_counter | O → W | p50, p95, p99, max | |
| Processing CPU and contention | ms; ratio | thread_time | cw − co; (W − O) / (cw − co) | p50, p99 | a ratio above 1 means time off CPU during work |
| Publication coverage | fraction | accounting | publications with step in W | (observed by c, or still queued for c at stop) / published; observed within W alone is reported too | 1 by construction in A, B, E, which 21.4 checks; n/a for RB, whose accounting is per batch (21.4) |
| Skipped publications | count | accounting | sequence gaps per consumer | sum | F, G, H, H′ |
| Declared drops | count | queue logs | per consumer | sum | C |
| Unobserved-event fraction | fraction | accounting | publications with step in W | events in windows neither observed by c nor queued for c at stop / events in all windows | a window's events are the batches after the previous publication step, up to and including its own; backlog at stop is not counted as unobserved (7.4) |
| Consumer CPU | cores | thread_time | per thread, first to last loop sample in W | / \|W\| | |
| Queue depth | items | reconstructed from logs | per queue: enqueued − dequeued − dropped, as a function of time | p50, max, slope in items/s | enqueue at the step's E_k, dequeue at O; no `qsize()` calls |
| Raw consumer achieved rate | events/s | perf_counter | events accumulated by c with the block's end in W | / \|W\| | RB |
| Polls | 1/s | | poll loop iterations in W | count / \|W\| | poll arms |

### 10.3 System

The source is the monitor's samples (6.8).

| Metric | Unit | Source | Aggregation per run |
|---|---|---|---|
| Peak physical footprint | MiB | `lifetime_max_phys_footprint` at the end; also `ru_maxrss` | value |
| Footprint trajectory and growth | MiB; MiB/s | `phys_footprint` at 10 Hz | series; least-squares slope over W; value at T1 minus value at T0 |
| Baseline footprint | MiB | `phys_footprint` at T_start | value (pool, interpreter, libraries) |
| Process CPU | cores | Δ(`user_time` + `system_time`) over W | / \|W\| |
| P-core share | fraction | Δ(`user_ptime` + `system_ptime`) / Δ(`user_time` + `system_time`) | value |
| Cycles per ns of CPU | 1/ns | Δ`cycles` / Δ CPU ns | value (an effective-frequency proxy) |
| Runnable time | ms/s | Δ`runnable_time` / \|W\| | value (time threads were runnable but not running) |
| Energy estimate | J; J per 10^9 events | Δ`energy_nj` over W | value; labelled as macOS's per-process estimate, not a power measurement |
| GC collections | count per generation | `gc.get_stats()` at T0 and T1 | delta |
| Consumer count, run duration | | request; wall time from process start to exit | value |

Not used, with the reason:
- `tracemalloc` in timed runs: it slows allocation unequally across arms. Temporary
  allocation per step is measured separately, untimed (V6).
- Queue `qsize()` polling: depth is reconstructed from the logs instead.
- A publication-to-observation latency for H: its publication instant is not observable
  (9.3).

## 11. Input conditions

### 11.1 Synthetic stream

- **Generator:** `benchmarks.workloads.Workload("uniform", sensor_size, B,
  seed=benchmarks.matrix.default_seed(sensor_size, B, "uniform"), event_rate_hz=r)`.
- **Contents:** uniform `x` and `y`, `p` in {0, 1}, every event in bounds, in timestamp
  order. Event i has `t = floor(i · 10^6 / r)` µs.
- **Event time runs at wall rate.** At 20M events/s, 20 events share each timestamp value
  and a 100k batch spans 5,000 µs. This differs from the gate stream (1 event per µs). It is
  chosen so that the schedule follows the timestamps (and, for the follow-up study's
  `timestamp_decay` cells, so that `tau_us` is real time).
- **Pool:** P = 6,400,000 / B distinct batches: 64 at 100k, 83.2 MB. They are
  generated before `T_start`.
- **Materialisation:**
  - Batch k is `pool[k mod P].copy()`, then `t += (k div P) · P · B · 10^6 / r` µs. The
    offset is an integer for the B and r used here.
  - The add also runs in the first cycle, with offset 0, so the cost is the same for every
    batch.
  - Materialisation happens on the producer thread before S_k, in every arm. It stands for a
    driver delivering a fresh buffer. Its cost is identical across arms and recorded (Q_k to
    S_k).
  - The pool itself is never modified, so a batch held in a queue can never change.
- **Distribution:** uniform only. It is the pessimistic locality case (plan.md, "Event
  distributions"). The real recording that supplies non-uniform spatial and temporal
  structure (S6) is in the follow-up study (20.3).
- **Integrity:** every run records the SHA-256 of the pool's bytes and the offset rule, and
  must match the value precomputed for the condition.

### 11.2 Real recording

The real recording (S6) is in the follow-up study (20.3). This study uses the synthetic
stream only.

### 11.3 Fixed conditions and why

| Condition | Value | Why |
|---|---|---|
| Resolution | 1280x720 | the largest gate frame (3.52 MiB at 4 bytes per pixel), so copy and memory effects are largest; every prior consumer measurement is at this size |
| Batch size | 100,000 events | a gate condition; 200 calls/s at 20M/s; the paced out-of-bounds check used it (benchmarks.md); within the range real readers yield (hill-climbing.md Experiment C) |
| Offered rate | 20M events/s | the project's target (decisions.md 16); paced single-producer evidence exists at this rate |
| Publication interval | 16 ms | the Engine's default; a display-rate cadence |
| Kernel | `event_count` (windowed) | the windowed semantics, where skipped and dropped publications leave events unobserved (7.4); the running kernel is in the follow-up study |
| Warm-up / window | 5 s / 10 s | 5 s as in the paced out-of-bounds check; 10 s gives 2,000 steps (20 samples in each run's p99 tail) and 500 publications per run (correction 1, 24) |
| Consumer counts | N ∈ {1, 4}, plus N = 0 for A, F, G, H, H′ | see below |
| Repetitions | 5 (P1, P2) | the gate's 5 for primary experiments |

**Why N ∈ {1, 4}.**
- The machine has 4 performance and 6 efficiency cores (`sysctl hw.perflevel0.logicalcpu` = 4,
  `hw.perflevel1.logicalcpu` = 6; read 2026-09-30).
- N = 1 is the single-consumer case. At N = 4 the producer and consumers are five runnable
  threads and first exceed the 4 performance cores. That boundary is expected to show up as a
  placement effect (P-core share is recorded). It is noted here so it is not read as an
  architectural discontinuity.
- N = 2 and N = 8 are in the follow-up study (20.3).

### 11.4 Experiments

Each run is one process and one cell.

**P1: state observation (primary; 5 repetitions).** 118 cells:

| Part | Arms | Workloads | N | Kernel | Runtimes | Cells |
|---|---|---|---|---|---|---|
| Main grid | A, B, C, E, F, G, H, H′, RB | W1, W3, W5 | 1, 4 | `event_count` | both | 108 |
| No-consumer reference | A, F, G, H, H′ | none | 0 | `event_count` | both | 10 |

H′ is the A/A control (16.3). It runs every cell H runs, interleaved with it in the same
passes. P1 uses the fixed conditions of 11.3 and K_ff = 4, K_raw = 13.

**P2: event preservation (primary; 5 repetitions).** 12 cells:

| Factor | Values |
|---|---|
| Arms | EP-H, EP-QB, EP-QE |
| Compression | `"blosc"` |
| State-observation consumers N_SO | 0, or 1 (W1 polling the Engine) |
| Runtime | 3.11.14, 3.14.2t |
| Fixed | `event_count`, 1280x720, 100k, 20M/s, 16 ms, K_raw = 13 |

Every cell set is fixed here. None is added, removed or chosen after results. No arm's rate
is lowered alone (section 15).

**Order and blocking.**
- A **pass** runs every cell of one experiment once, in an order drawn with a fixed seed:
  `random.Random(20261001 + 100·experiment_index + pass_index).shuffle`, applied to the
  cells in the order they are listed by the harness. Experiment indices: P1 1, P2 2.
- Runtimes and arms are interleaved within a pass. Consecutive passes give repetitions 1 to
  5.
- **Execution order:**
  1. validation (section 22);
  2. the calibration commit;
  3. P1 passes 1 to 5;
  4. P2 passes 1 to 5.
- Every pass is run. None is skipped or stopped because of its results. If the campaign ends
  early for an external reason, completed passes are reported and the rest are marked not
  run.

**Budget.** Estimated from the cells above, before any harness exists:
- A P1 run takes about 24 s: process start and imports (under 0.1 s warm on both study
  environments, measured during the feasibility check, 12), the pool and its digest, 15 s of
  stream, shutdown, integrity checks and log writing, the driver's checks, and a 5 s idle
  gap. This is the earlier draft's estimate of 34 s for a 25 s stream, less the 10 s by which
  the stream is shorter.
- A P2 run takes about 50 s: 15 s of stream, the drain, `close()`, reading back and hashing
  about 3 × 10^8 events, and the idle gap.

| Experiment | Runs | Machine time |
|---|---|---|
| P1 (118 cells × 5) | 590 | about 3.9 h |
| P2 (12 cells × 5) | 60 | about 0.8 h |
| **Campaign** | **650** | **about 4.8 h** |
| Validation V1 to V6, outside the campaign budget | about 110 | about 0.75 h |

One P1 pass is about 47 minutes. The campaign estimate is under 5 hours, so the approved
reduction rule (3 repetitions for arms A, B, C, E, F, RB and the EP arms if the estimate
exceeded 5 hours) does not apply. Every cell keeps 5 repetitions. Retried runs (14.1) add to
the actual machine time, which the report states.

## 12. Runtime conditions

| | Runtime A | Runtime B |
|---|---|---|
| Interpreter | CPython 3.11.14, the uv-managed build `cpython-3.11.14-macos-aarch64-none` | CPython 3.14.2 free-threaded, `cpython-3.14.2+freethreaded-macos-aarch64-none`, GIL disabled |
| NumPy | 2.4.6 | 2.4.6 (as in the gate; the lockfile's 2.5.3 is not used) |
| Status | performance-reference environment (decisions.md 17, terminology) | performance-reference environment; the only verified free-threaded minor (decisions.md 42) |
| Checks in every child | standard build (`Py_GIL_DISABLED` not 1), `gil_enabled` true in `benchmarks.environment.runtime()` (3.11 has no `sys._is_gil_enabled`) | `sysconfig` `Py_GIL_DISABLED` is 1 and `sys._is_gil_enabled()` is false, with NumPy, h5py and hdf5plugin imported; `PYTHON_GIL` unset |

**Packages.**
- **Environments:** new virtual environments at `scratch/observation_study/envs/py311` and
  `py314t`, built from exact pins with hashes:
  - `frames2py==1.0.0`, the PyPI wheel, SHA-256
    `fffb2fc09d1953b2485133026c92a414ba28f72980444d808da2ff8a0441f912` (progress.md, "1.0.0
    published and verified");
  - `numpy==2.4.6`, `h5py==3.16.0`, `hdf5plugin==7.1.0`. The last two are for the recorder
    and its read-back (P2).
  - pyglet is not installed. `render()` does not need it (decisions.md 53).
- **Versions frozen:** each environment's freeze is recorded.
- **Contingency:** if h5py or hdf5plugin cannot be installed or imported with NumPy 2.4.6 on
  runtime B with the GIL disabled, the study stops and reports it. Nothing is substituted
  without an amendment.
- **Feasibility check, 2026-09-30, before this text was committed (not a measurement).** Both
  environments were built from the hashed pins with `uv pip install --require-hashes
  --only-binary :all:`, frames2py from the wheel with the SHA-256 above. On runtime B,
  NumPy 2.4.6, h5py 3.16.0 (HDF5 2.0.0) and hdf5plugin 7.1.0 imported, `Py_GIL_DISABLED` was
  1, and `sys._is_gil_enabled()` was false before and after the imports and a recording.
  2,000,000 events written with `frames2py.recorder` (Blosc) and read back with
  `frames2py.adapters.hdf5` were byte-identical, on both runtimes. `frames2py.__file__` lay in
  each environment's site-packages. Record: `scratch/observation_study/feasibility/`.

**Runtime settings, recorded and left at defaults:**
- `sys.getswitchinterval()` (5 ms by default);
- the garbage collector enabled, with its thresholds;
- no `PYTHON*` tuning variables;
- `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and `BLOSC_NTHREADS`
  unset. Blosc then compresses on the calling thread (hill-climbing.md, "HDF5
  `BLOSC_NTHREADS`").
- `np.show_config()` is recorded. No workload calls BLAS.

The GC stays enabled because the consumers and the queue arms allocate, as real applications
do. Its collections are recorded (10.3).

**What the runtime comparison means.**
- On standard CPython, CPU-bound consumers compete with the producer for the GIL. That is an
  expected runtime effect, measured and reported. It is not a confound.
- On 3.14t with the GIL disabled, the architectures' behaviour under parallel Python
  execution can be observed more directly.
- The 3.14t results hold for CPython 3.14.2t with NumPy 2.4.6 on this machine. They say
  nothing about other free-threaded versions.

## 13. Environment controls

### 13.1 Machine

- Apple M4, MacBook Pro 14-inch (`hw.model` Mac16,1), 4 performance and 6 efficiency cores,
  16 GiB, macOS 15.7.7 (24G720), all read 2026-09-30.
- The campaign runs on this machine only.
- An OS update before the campaign is recorded. An OS or firmware change during the campaign
  stops it: results across OS versions are not pooled.

### 13.2 Operator checklist, before each session

The driver records the operator's confirmation with the session record. In an unattended
session (13.4) no operator confirms; the driver records that, and records each item it
cannot verify as unverified. It verifies what it can, and refuses to start if any of these
checks fails:
- AC power connected (`pmset -g batt`);
- Low Power Mode off (`pmset -g`);
- no denylisted process running (13.3);
- Docker down: `docker info` fails and no Docker process runs.

1. AC power connected, and Low Power Mode off. Lid open: a closed lid can sleep the machine.
2. Every application closed except one terminal:
   - IDEs (VS Code, Cursor) and browsers;
   - Docker Desktop, with the daemon down (`docker info` fails) and no containers;
   - virtual machines.
3. No development agents other than the launcher (13.4): no Codex or other agent process,
   in a terminal or inside an IDE, and no Claude Code session other than the one launching
   the campaign.
4. No other benchmark or build.
5. Time Machine automatic backup paused, no software update downloading, and
   `scratch/observation_study/` added to Spotlight's privacy exclusions.
6. The driver runs under `caffeinate -dimsu`, which keeps the display awake so timer
   behaviour doesn't change when it sleeps. Each run is also wrapped in the harness's own
   `benchmarks.power.hold_awake()`.
7. The repository is checked out at the measured commit on the study branch, with a clean
   tree.

### 13.3 Automatic checks

The driver process makes these checks, outside the child.

| When | Check | Refuse or invalidate |
|---|---|---|
| Session start | denylist: Docker Desktop and daemon, `com.docker.*`, `qemu*`, `codex`, Cursor, VS Code (`Code Helper`), other benchmark processes | refuse to start |
| Every run, start and end | `hold_awake()` record: full wake at start and end, `slept`, assertion confirmed and released | invalid if the machine slept or ended outside full wake |
| Every run, start and end | `pmset -g batt`: AC, battery percentage, charging state; Low Power Mode off | invalid if not on AC or Low Power Mode is on |
| Every run, start and end | `pmset -g therm`: no thermal warning level, performance warning level or CPU power status recorded | at start: wait up to 10 min, polling every 30 s, then invalid and the session pauses; at end: invalid |
| Every 5 s during a run | `ps -A -o pid=,ppid=,pcpu=,rss=,comm=` | invalid if any process other than the child, the driver and `ps` is at or above 10% of a core in two consecutive samples, or all such processes together are at or above 25% in two consecutive samples. `claude` processes, the launcher's included, are monitored like any other. |
| Every run, start and end | `sysctl vm.swapusage`; `vm_stat` swap-outs | invalid if swap-outs occurred during the run |
| Every run, start | free space on the scratch volume | P2: refuse below 20 GiB |
| Every run | `os.getloadavg()` | recorded |

**Recorded per session:** `sw_vers`, `hw.model`, `machdep.cpu.brand_string`, the
`hw.perflevel*` counts, memory size, `pmset -g` settings, uptime, and the operator's
confirmation.

**Recorded per run, from inside the child:**
- `benchmarks.environment.runtime()` and `capture()`: commit, tracked changes and untracked
  files;
- `frames2py.__version__` and `frames2py.__file__` (it must be the study environment's
  site-packages, version 1.0.0);
- the versions of NumPy, h5py and hdf5plugin, and NumPy's configuration;
- the switch interval and GC thresholds;
- the thread-count environment variables;
- `threading.active_count()` before and after;
- the SHA-256 of this preregistration file;
- the request.

### 13.4 Launcher and unattended execution

- **Launcher.** The campaign may be launched and supervised by a Claude Code session. While a
  run is in progress, that session makes no tool calls other than lightweight progress checks
  that read logs. Its process is subject to the per-run background-load check (13.3) like any
  other process. `claude` is therefore not on the session-start denylist, and it remains in
  per-run load monitoring.
- **Detached.** The driver is started detached from the launcher (under `caffeinate -dimsu`,
  with `nohup`), so the launcher's own activity is never part of the driver's process tree.
- **Progress.** The driver writes its progress after every run: the attempt's outcome, the
  pass and position reached, and the counts so far.
- **Resume.** If a pass is interrupted (the driver or the machine stopped), a restarted
  driver resumes it from its last completed run. The attempts already recorded are kept; the
  pass order is the seeded order of 11.4, so the remaining runs are the rest of that order.
  A pass re-run whole after an amendment (24) is a new pass record, never a resume.
- **Session summary.** When the driver stops, for any reason, it writes a session summary:
  the session record, the passes completed, the attempts per outcome class and flag, and why
  it stopped.

## 14. Run outcomes, validity and invalidity

### 14.1 Outcome classes

| Class | Cause | Effect |
|---|---|---|
| **VALID** | every check passed | used |
| VALID, with the flag NOT_SUSTAINED | the producer did not sustain the offered rate (15.2) | used; evidence, not an invalid run |
| VALID, with the flag MEMORY_CEILING | footprint growth passed 4 GiB; the run stopped early | used; metrics cover the stream up to the stop, and the stop time is reported |
| VALID, with the flag DRAIN_TIMEOUT | P2: the recorder's queue did not drain within 120 s after T1 | used; the written prefix is verified against the fed prefix; completeness is reported as not established |
| VALID, with the flag ARM_MEMORY_ERROR | `MemoryError` raised in an arm's code | used; an architecture outcome |
| **REFUSED** | unclean tree, preregistration hash mismatch, runtime or package mismatch, denylisted process at session start | the driver refuses to start or continue the session; nothing is measured |
| **INVALID_ENV** | power, sleep, DarkWake, thermal, AC or Low Power Mode, competing load, swap | re-queued at the end of the same pass, at most 2 retries; every attempt is kept |
| **INVALID_INTEGRITY** | unexpected loss (7.4); an accounting mismatch (21.4); an unresolvable watermark; a pool or input digest mismatch; a sequence observed out of order in a FIFO or non-monotonic in any arm; a P2 read-back mismatch | the campaign pauses for review; the run is not replaced until the cause is identified; the finding is reported whatever its cause (in arm H it would be a Frames2Py defect); in an unattended session, 14.2 applies |
| **HARNESS_FAILURE** | an exception in harness or baseline code other than `MemoryError`; a missing or truncated log; non-monotonic clock samples; a child exit without a result | the campaign stops; the harness is fixed through an amendment (section 24) before measurement resumes; the affected pass is re-run whole |
| **SHUTDOWN_TIMEOUT** | threads still alive 10 s after `stop` | the campaign pauses; reported; it is classified as an architecture outcome only if review attributes it to the arm (for example a deadlock in its synchronisation), otherwise as HARNESS_FAILURE |
| **CRASH** | the interpreter aborts or segfaults | the campaign stops for review |

Rules that follow:
- **Expected losses are outcomes.** Drops in C, skipped publications in latest-state
  arms, backlog at stop, and failure to sustain the offered rate are outcomes. None of them
  invalidates a run.
- **Unexpected loss invalidates.** Only unexpected loss does: loss in an arm whose policy
  declares none, loss that the declared policy doesn't explain, or corrupted input.
- **Missing repetitions:**
  - Primary cells need 5 valid repetitions. A cell with fewer after retries is reported as
    "n of 5 valid", described, and excluded from the distinguishability comparisons (16.3).

### 14.2 Amendments while unattended

In a session with no human available (13.4), only these amendments are permitted:
- **Harness defects.** A HARNESS_FAILURE, or an INVALID_INTEGRITY traced to harness code, may
  be fixed by a numbered, dated amendment (section 24) that names the defect and the fixing
  commit. The amendment and the fix are committed and pushed before the affected passes are
  re-run, whole.
- **Nothing else.** No other amendment is made in such a session. A situation this protocol
  does not cover stops the campaign for the user.
- **Frames2Py findings.** An INVALID_INTEGRITY attributable to Frames2Py itself (arm H, or
  the Engine in the EP arms) is a finding. The campaign stops and the finding is reported.
  `src/` is never modified.
- A CRASH or SHUTDOWN_TIMEOUT that cannot be traced to harness code stops the campaign.

## 15. Offered, achieved and sustainable rate

### 15.1 Definitions

- **Offered rate:** events scheduled per second (10.1). The same schedule applies to every
  arm of a condition.
- **Achieved rate:** events completed by the producer per second in W. For RB, also
  each consumer's accumulated rate.
- **Backlog:** the source backlog (batches scheduled but not yet released) plus queue depth,
  in items and events. The source backlog's implied buffer is events × 13 B. It is the
  memory an upstream buffer would need to hold for an arm that stalls its producer. Stalling
  arms therefore don't appear memory-cheap for having pushed their backlog upstream.
- **Queue depth, dropped data, skipped data:** 7.4 and 10.2.
- **Producer stall time:** producer off-CPU time and excess step latency (10.1).
- **Memory growth:** 10.3.

### 15.2 Sustained

A descriptive label per run, not a verdict on an architecture. A run is SUSTAINED when all
three hold:
1. the least-squares slope of producer lag L_k against A_k over W is at most 1.6 ms/s;
2. the median lag over the last 1 s of W is at most 16 ms;
3. the memory ceiling was not reached.

Otherwise it is NOT_SUSTAINED. The thresholds come from the publication interval:
- A slope of 1.6 ms/s accumulates one interval (16 ms) of backlog across the 10 s window.
- A slope at the threshold corresponds to processing about 0.16% slower than offered.
- The slope's standard error at 2,000 points and about 1 ms of lag noise is about
  0.008 ms/s (INFERENCE, from the regression formula: σ / (√n · |W| / √12)).
- The median-lag criterion (16 ms, one interval) does not depend on the window length.

The sustainable-rate ladder (S1) is in the follow-up study (20.3).

### 15.3 Not sustaining the offered rate

The run is recorded with everything the metrics show:
- achieved rate, lag trajectory, source backlog and implied buffer;
- what the consumers saw.

The offered rate is never lowered for that arm alone. A reduced-rate characterisation,
applied to every arm alike, is the follow-up study's S1 (20.3).

**Consumers keeping up.** How a consumer falls behind depends on the arm:

| Arm | How it shows |
|---|---|
| B | producer stall (NOT_SUSTAINED) |
| C | drops |
| E | queue-depth slope above 0.5 items/s and footprint growth |
| RB | producer stall |
| F, G, H | skipped publications |
| A | producer lag |

Each is reported in its own terms.

## 16. Statistical methodology

### 16.1 Units and aggregation

- **Replication:** the unit is the run: one process, one cell, one repetition.
- **Per-run values:** computed from the run's raw logs over W.
- **Per cell:** the median of the per-run values, with min and max. All per-run values are
  published.
- **Percentiles:** nearest rank, within a run. Consumer distributions are pooled across that
  run's consumers, and per-consumer summaries are kept. Sample counts are reported next to
  every percentile.
- **Outliers:** no outlier removal, trimming or winsorising. No valid run is discarded.
  Invalid runs are excluded only by the rules of section 14.
- **Uncertainty:** with 5 repetitions, uncertainty is shown as the range of per-run values.
  Distribution plots show every run's empirical CDF. No confidence intervals or p-values.

### 16.2 Seeds

- Workload seeds follow `benchmarks.matrix.default_seed`.
- Pass orders use 11.4's seeds.
- W5 starts from the sequence number. W1 and W3 have no randomness.
- No other randomness is involved.

### 16.3 Noise band and distinguishability

The noise band comes from the A/A control H′ against H: 14 cells (7 per runtime: W1, W3 and
W5 at N ∈ {1, 4}, and N = 0), the same code under two labels, measured interleaved in the
same passes. For each metric m in:
- producer busy time per event;
- step latency p99;
- freshness age p50 and p95;
- peak footprint;
- process CPU;

`band_m = the largest max(r, 1/r) over the 14 cells`, where r is the ratio of the H′ and H
cell medians. No floor is applied: the band is the A/A control's measured band. A cell pair
in which H or H′ has fewer than 5 valid runs is left out of the band, and the report says so.

Two P1 cells X and Y are **distinguishable** on metric m only if:
1. the ratio of their medians lies outside `[1/band_m, band_m]`; and
2. their per-run min-max ranges don't overlap.

Otherwise they are **not distinguishable at this study's resolution**. That is the tie
outcome for H6 and H7.

Both criteria are conservative: each alone can declare a pair not distinguishable, the band
is the largest A/A ratio over 14 cells, and five-run ranges overlap easily. The procedure
therefore favours the "not distinguishable" outcome that H6 predicts, and such an outcome is
weak evidence for H6. Results may say only "not distinguishable at this study's
resolution", never "equivalent", "the same" or "no difference".

Distinguishability is evaluated only for these comparisons, fixed here:
1. G against H, in every P1 cell pair where both exist;
2. F against G, in every P1 cell pair (H5);
3. each arm at N against the same arm at N = 0 (A, F, G, H);
4. runtime A against runtime B for the same arm, workload and N.

Everything else is described, not tested. No claim is made about multiple comparisons.

### 16.4 Secondary experiments

There are none in this study. The secondary experiments S1 to S7 are the follow-up study's
(20.3).

### 16.5 No aggregate

There is no overall score, no weighted index and no winner ranking. Tables list arms in the
fixed order A, B, C, E, F, G, H, H′, RB (and EP-H, EP-QB, EP-QE for P2), never sorted by a
metric.

## 17. Expected outcomes and how they would be judged

For each hypothesis: **supported (S)**, **contradicted (C)**, otherwise **inconclusive**.
"Every", "all" and "most" range over the valid cells of the stated experiment and runtime.

| H | Judged on | S if | C if |
|---|---|---|---|
| H1 | P1, arm A | A is NOT_SUSTAINED in every cell where N × (median processing latency) + b × 16 ms > 16 ms, and SUSTAINED in at least one cell where it is below. b is A's producer busy fraction in the N = 0 cell of the same runtime. | A is SUSTAINED in any cell with N × processing latency ≥ 32 ms |
| H2 | P1, arm B, W3, W5 | NOT_SUSTAINED; publication rate within ±20% of the consumer observation rate; footprint growth ≤ 0.5 MiB/s | SUSTAINED in any W3 or W5 cell at N = 1 |
| H3 | P1, arm C, W3, W5 | SUSTAINED; footprint growth ≤ 0.5 MiB/s; declared drops > 0 | C is NOT_SUSTAINED in any W3 cell (sleeping consumers, so no CPU contention) |
| H4 | P1, arm E, W3, W5 | queue-depth slope > 0.5 items/s and freshness age increasing through W (Spearman ρ of age against O > 0.8) in every cell; no drops | depth bounded (slope ≤ 0.5 items/s) with consumers slower than the cadence |
| H5 | P1, arm F | F's excess step latency p99 increases with N (N = 1 to N = 4) at 1280x720 | adapted: F's step p99 is not distinguishable (16.3) from G's at N = 4 for every workload and runtime |
| H6 | P1, G vs H | not distinguishable (16.3) on busy time per event, step p99 and freshness p50 in at least 75% of cell pairs, on each runtime | distinguishable on busy time per event or step p99 in more than 50% of pairs on a runtime (either direction is reported) |
| H8 | P2 | every run's read-back verifies, in every arm; EP-H SUSTAINED | EP-QB or EP-QE ends a run with DRAIN_TIMEOUT; or EP-H is NOT_SUSTAINED at N_SO = 0. A read-back mismatch is INVALID_INTEGRITY (14.1), not evidence for or against H8. |
| H9 | P1, arm H | SUSTAINED, footprint growth ≤ 0.5 MiB/s and freshness age p95 ≤ 32 ms (two publication intervals), in every W1 and W3 cell on both runtimes | NOT_SUSTAINED in any W1 or W3 cell at N ≤ 4 |
| H10 | P1, W5 | runtime A: excess step p99 ≥ 5 ms for W5 in F, G and H at every N measured; runtime B: < 5 ms | on runtime A, W5's excess step p99 is under 5 ms at N = 4 in both G and H |
| H11 | P1, W3 | latest-state arms (F, G, H): W3's excess step p99 not distinguishable from N = 0 | W3's excess p99 distinguishable from N = 0 in most latest-state cells |
| H12 | P1, RB and B | adapted: accumulation-path CPU per event (10.1) at N = 4 over N = 1 between 3 and 5; on runtime A with W1, RB NOT_SUSTAINED at a smaller N than B is NOT_SUSTAINED | adapted: that ratio < 2 at N = 4 |
| H13 | P1, W1, N = 1 | post-step delay p50: queue arms (B, C, E, RB) < 2 ms and poll arms (F, G, H) between 4 and 12 ms, on both runtimes | poll arms' p50 < 2 ms |
| H14 | P1, N = 0 vs N = 1, latest-state arms | busy throughput at N = 1 > at N = 0 in most cells, with P-core share higher at N = 1 | busy throughput at N = 1 ≤ at N = 0 in most cells |

Notes on the rules:
- **H5, adapted.** The draft's contradiction rule was at N = 8, the largest count there. It is
  restated at N = 4, the largest count here.
- **H10.** Its W5 part stays in this study. Its contradiction rule mirrors the draft's rule
  for W6, at N = 4 in G and H. The user wrote it on 2026-09-30, before any data existed. The
  W6 and W2 parts are in 20.3.
- **H12, adapted.** The draft's thresholds at N = 8 over N = 1 were 6 to 10 (0.75 to 1.25
  times N) for support and below 4 (half of N) for contradiction. Derived the same way at
  N = 4 they are 3 to 5 and below 2. "At a smaller N" compares, for each arm, the smallest
  N ∈ {1, 4} at which its cell is NOT_SUSTAINED, taken as infinite if there is none. The
  condition holds only if RB's is finite and smaller than B's.
- **H1.** "Of the same kernel" is dropped: there is one kernel.

H7 is the interpretation rule for H6. Whatever the outcome, contract features are reported
alongside the numbers (table T5 in 18.1).

## 18. Output artifacts

Every table and figure is produced from the raw logs by analysis code committed before the
campaign (21.2). No single "Frames2Py improvement percentage" appears anywhere, and none is
a headline.

The primary result is the full freshness distributions: per cell and per run, freshness age
p50, p95, p99 and max, and freshness over time within the window (T7, F15). The
decision-boundary table T1 is derived from them at fixed thresholds and is illustrative.

### 18.1 Tables

- **T1, decision boundary (illustrative).** For every P1 cell, grouped by (workload, N,
  runtime), the arms that meet each requirement profile, in fixed arm order. For every P2
  cell, the same for EP-events. SO-live is evaluated at three freshness thresholds, 32, 50 and
  100 ms, and the table is labelled illustrative: the thresholds are examples of
  requirements, not findings. Profiles:
  - **SO-live(θ), θ ∈ {32, 50, 100} ms:** producer SUSTAINED in ≥ 4 of 5 runs; the cell
    median of per-run freshness age p95 ≤ θ; footprint growth ≤ 0.5 MiB/s; no
    MEMORY_CEILING; unexpected loss 0.
  - **SO-complete:** SO-live's producer and memory criteria, plus publication coverage 1
    for every consumer.
  - **EP-events:** read-back verified; producer SUSTAINED in ≥ 4 of 5 runs; footprint
    growth ≤ 0.5 MiB/s, or no ceiling with the backlog drained within the cap.

  The profiles are descriptive requirement sets. Every metric behind them is published, so a
  reader can apply other thresholds.
- **T2, per-cell metrics.** Every metric of section 10, per cell: median, min to max, n
  valid.
- **T3, G against H.** Per cell: the ratios, `band_m` and the distinguishability outcome.
- **T4, outcomes.** Counts of each outcome class and flag, per arm, experiment and runtime.
- **T5, contract features.** A two-column table of what H and G each provide, from 6.6 (CODE
  FACT). It carries no numbers.
- **T6, environment.** Per session and per run: the records of 13.3.
- **T7, freshness distributions (primary).** Per P1 cell and per run: freshness age p50,
  p95, p99 and max with sample counts, pooled and per consumer; the same for end-to-end age
  and age at completion.

### 18.2 Figures

Each figure has a runtime A panel and a runtime B panel side by side, so every figure is
also a runtime comparison. The raw-batch family is drawn in its own panel or colour group,
never merged with the finished-frame arms. Arms keep one colour throughout.

| # | Figure | x | y | Facets |
|---|---|---|---|---|
| F1 | producer achieved rate vs consumer load | N (0, 1, 4) | achieved / offered, per-run points and median | workload |
| F2 | producer step latency p99 vs consumer load | N | p99, µs, log scale | workload |
| F3 | freshness vs consumer load | N | freshness age p50 and p95, ms, log scale | workload |
| F4 | freshness vs consumer processing time | median processing latency (W1, W3, W5) | freshness age p95 | N ∈ {1, 4} |
| F5 | memory vs elapsed time | time since T_start | `phys_footprint`, MiB | every arm at W3, N = 4 (all runs overlaid); the same for every P1 cell in the data release |
| F6 | queue depth vs time | time | depth, items | B, C, E, RB at W3 and W5, N = 4 |
| F7 | observation rate vs consumer cost | median processing latency | observations/s per consumer | N ∈ {1, 4} |
| F8 | drops, skips and unobserved events vs load | N | skipped per s, declared drops per s, unobserved-event fraction | workload |
| F9 | CPU vs consumer count | N | process CPU, producer CPU, Σ consumer CPU (cores); P-core share as a line | workload |
| F10 | runtime comparison | runtime A value | runtime B value, log-log with a diagonal | metrics: step p99, freshness p95, achieved / offered |
| F11 | G vs H | cell | ratio with the band shaded | metric |
| F13 | end-to-end age decomposition | arm | stacked medians: release lag, materialisation, freshness age | W3, W5; N = 4 |
| F14 | event preservation (P2) | arm | producer lag slope, recorder backlog, completion delay | N_SO |
| F15 | freshness distributions (primary) | freshness age, ms, log scale | empirical CDF, every run | arm × workload × N |
| F16 | freshness over time | O − T0, s | freshness age, ms | every arm at W1, W3 and W5, N = 4 |

F12 (sustainable rate) belongs to S1 and is in the follow-up study. Where F5, F6 and F13 used
W2, they use the workloads of this study instead. If a figure cannot be drawn without a
plotting dependency (21.2), the data table it would plot is produced in its place and the
report lists the figure as not drawn.

## 19. Relationship to existing evidence

This study's dataset contains only runs made under this protocol. None of the evidence below
is merged into it, reanalysed as part of it, or used to tune it. It motivated the questions
and a few parameter choices, cited where used.

| Evidence | What it measured | Why it is not this study's result |
|---|---|---|
| v1 gate (decisions.md 16; benchmarks.md "v1 gate"; `benchmarks/gate_preregistration.md`) | kernel and `ingest()` throughput back to back on a virtual clock, 150 cells, no consumers | no consumer load, no alternative architectures, virtual clock |
| Paced at 20M events/s (benchmarks.md) | single-producer CPU, footprint, busy throughput on a real clock, no consumers | no consumers; the paced/back-to-back gap it found is a covariate here (P-core share, H14) |
| Paced out-of-bounds reopen check (benchmarks.md) | lag slope and final lag for one Engine configuration | one arm, no consumers; its lag-slope method informed 15.2, with different thresholds derived here |
| Phase 5 one consumer, valid rerun (benchmarks.md; `scratch/phase5/consumer.py`, `preflight/consumer_summary.txt`) | busy throughput and p99 with 0 and 1 `frame.max()` reader, 2 s each, 3 runs | Frames2Py only; the 0-reader measurement always ran first (an order confound, recorded there); 2 s windows |
| Phase 7 viewer impact (benchmarks.md "Consumers") | a `render()` consumer's effect on a paced producer, 4 s windows, alternating order | Frames2Py only; its batch-reuse construction makes timestamps non-monotonic at pool wraps (section 0); its busy-throughput increase motivates H14 |
| Phase 4 M2 and M4, Experiment 4 (decisions.md 38, 40; hill-climbing.md) | handoff store times with readers; read and copy costs; polling costs | ran partly or entirely in DarkWake or sleep (benchmarks.md "Historical sleep forensics"); not clean timing evidence |
| Publisher and concurrency decisions (decisions.md 6, 8, 9, 11, 38, 39, 40) | design and correctness basis | define arm H; not measurements of this question |

## 20. Scope limits and follow-up studies

### 20.1 In scope

- In-process consumers only, as threads of one Python process.
- High-rate event streams, maintained event state, and the observation architectures of
  section 6.

### 20.2 Out of scope, possible future studies

None of these is part of this study or its data:
- cross-process consumers and multiprocessing;
- shared-memory snapshots;
- distributed and multi-machine systems;
- network transport and ROS transport benchmarking;
- GPU consumers and GPU transport;
- alternative event-camera accumulation algorithms;
- a push observation primitive (`wait_for_newer`, decisions.md 40);
- competing-consumer work queues;
- raw-batch drop policies;
- producer QoS or affinity control (hill-climbing.md, "Paced-mode producer placement");
- Linux platforms.

### 20.3 Planned follow-up study

Everything in this subsection was in the draft of this protocol and was moved out of this
study's scope on 2026-09-30, before any harness or measurement existed. It is a plan, not
part of this study or its data. It will be preregistered separately, in its own committed
protocol, before any of its measurements; that protocol may revise anything here. Nothing
below is run now.

**Moved out of this study:**
- the secondary experiments S1 to S7, with their cells as listed below;
- arms D (finished-frame queue, drop newest), RE (raw-batch queue, unbounded) and FH (work
  under lock, S5 only), defined below;
- workloads W2 and W6 (section 8), and so the K2 and K6 calibration of V5;
- the `timestamp_decay` kernel, `frames2py.TimestampDecay(tau_us=10_000.0)` (the gate
  parameter, `benchmarks/targets/v1.py`), and with it the overwritten-intermediate-state
  category of 7.4;
- consumer counts N = 2 and N = 8;
- 20 s measurement windows, with the draft's derived thresholds (0.8 ms/s lag slope, 40
  samples in the p99 tail, 1,250 publications per run);
- gzip compression in P2;
- the real recording of S6;
- the V1 inputs at 346x260, with `timestamp_decay`, and from the S6 recording.

**Arms, as the draft defined them (6.4):**

| Arm | Producer step | Consumers | Family |
|---|---|---|---|
| **D** finished-frame queue, drop newest | as B; a full queue rejects the new item | as B | finished frame, fan-out |
| **RE** raw-batch queue, unbounded | for c = 1..N: `q_c.put(b)` | as RB, no capacity | raw batch, fan-out |
| **FH** latest frame, work under lock (S5 only) | as F | poll loop; per tick `with L:` if `meta.seq` is new, work directly on `shared` while holding `L` | latest state; labelled anti-pattern (6.7) |

FH is the lock-held-consumption pattern that general concurrency guidance advises against. It
would appear only in S5, labelled as such. Its O is taken after the lock is acquired, before
work. With `poll_interval` for S7's interval 0: the batch period, as in the Phase 5 consumer
probe (`scratch/phase5/consumer.py`, CODE FACT). `K_raw` for other conditions: 10k at 20M/s:
128. 100k at 10, 40 and 80M/s: 7, 26 and 52. For the recording in S6:
`ceil(64 ms / mean batch period)`, computed from the input before any run. S4 varies K_ff
over {1, 16} and K_raw over {4, 52}, the same time-depth rule at 16 ms and 256 ms.

**The draft's experiments (its 11.4):**

**P1: state observation (primary; 5 repetitions).** 708 cells:

| Part | Arms | Workloads | N | Kernels | Runtimes | Cells |
|---|---|---|---|---|---|---|
| Main grid | A, B, C, D, E, F, G, H, RB, RE | W1, W2, W3, W5 | 1, 2, 4, 8 | both | both | 640 |
| No-consumer reference | A, F, G, H | none | 0 | both | both | 16 |
| A/A control | H′ | W1, and none at N = 0 | 0, 1, 4 | both | both | 12 |
| GIL contrast | A, B, C, D, E, F, G, H, RB, RE | W6 | 1, 4 | `event_count` | both | 40 |

P1 uses the fixed conditions of 11.3 and K_ff = 4, K_raw = 13.

**P2: event preservation (primary; 5 repetitions).** 24 cells:

| Factor | Values |
|---|---|
| Arms | EP-H, EP-QB, EP-QE |
| Compression | `"blosc"` (default), `"gzip"` |
| State-observation consumers | 0, or 1 (W1 polling the Engine) |
| Runtime | 3.11.14, 3.14.2t |
| Fixed | `event_count`, 1280x720, 100k, 20M/s, 16 ms |

**Secondary stress experiments (3 repetitions each).** Unless a row says otherwise, the
conditions are P1's.

| Id | Question | Arms | Varied | Held | Cells |
|---|---|---|---|---|---|
| S1 | sustainable rate | the 10 P1 arms | offered rate 10M, 20M, 40M, 80M/s | W1, N ∈ {1, 4}, `timestamp_decay` | 160 |
| S2 | per-call and per-item overhead | the 10 P1 arms | batch 10k (K_raw = 128) | W1, W3; N ∈ {1, 4}; `event_count` | 80 |
| S3 | frame size | the 10 P1 arms | 346x260 (0.34 MiB frames) | W1, W2; N ∈ {1, 4}; `event_count` | 80 |
| S4 | queue capacity | B, C, RB | K_ff ∈ {1, 16}; K_raw ∈ {4, 52} | W2; N ∈ {1, 4}; `event_count` | 24 |
| S5 | lock-held consumption | F, FH | | W1, W2; N ∈ {1, 4}; `event_count` | 16 |
| S6 | real recording | the 10 P1 arms | DSEC at 640x480, own rate (11.2) | W1; N ∈ {1, 4}; both kernels | 80 |
| S7 | high cadence | the 10 P1 arms | interval 0 ms (publish on every step); poll interval = batch period | W1, W3; N ∈ {1, 4}; `event_count` | 80 |

Every secondary cell set is fixed here. None is added, removed or chosen after results.
Where a reduced rate is needed to characterise an architecture, S1's ladder serves, and it
applies to every arm alike. No arm's rate is lowered alone (section 15).

**The draft's real recording (its 11.2, S6):**

- **Recording:** `dsec_thun_01_a_events_left.h5` from the registry in `tests/recordings.py`.
  `recordings.path()` checks its size and SHA-256. It is present locally.
- **Why this one:** it has the highest own event rate of the registry's recordings (13.84M
  events/s over its first 20M events; benchmarks.md "Consumers", recorder table) at 640x480.
  It is replayed at its own pace (speed 1.0), so the offered rate and its burstiness are the
  recording's, not controlled.
- **Decoding:** once, before the campaign, untimed:
  `frames2py.adapters.hdf5.open(path, group="events", t_offset="/t_offset",
  sensor_size=(640, 480), batch_size=100_000)`, concatenated. The event count must equal the
  registry's 131,482,728. The result is saved as `.npy` under `scratch/observation_study/input/`
  with its SHA-256 (1.71 GB). Each run loads it into memory with `np.load`.
- **Batches:** consecutive 100,000-event slices. The last, partial slice is kept.
- **Checks before any S6 run:**
  - every event is in bounds for 640x480;
  - batches are in timestamp order at batch level: `m_k < m_{k+1}` and
    `min t of batch k+1 ≥ m_k`.

  If either check fails, S6 stops for an amendment. Experiment C found no out-of-order or
  out-of-bounds events in the recordings (hill-climbing.md), but that is not assumed here.
- **Looping:** each pass over the recording is a cycle. Cycle c adds
  `c · (max t − min t + 1)` µs. Materialisation is as in 11.1.
- **Schedule:** `A_k` as in 6.2, from the running maximum `M_k`, at speed 1.0. The input's
  statistics are reported with the results, computed from the input before any run: mean
  rate and the distribution of per-batch time spans.

**Why N stops at 8 (the draft's rationale).**
- The machine has 4 performance and 6 efficiency cores (`sysctl hw.perflevel0.logicalcpu` = 4,
  `hw.perflevel1.logicalcpu` = 6; read 2026-09-30).
- At N = 8, the producer and eight consumers are nine runnable threads, within 10 cores, so
  on 3.14t every thread can run at once. Beyond N = 9 oversubscription is guaranteed, and the
  study would measure the OS scheduler rather than the architectures.
- The doubling steps show the shape of scaling.
- At N = 4 the producer and consumers first exceed the 4 performance cores. That boundary is
  expected to show up as a placement effect (P-core share is recorded). It is noted here so
  it is not read as an architectural discontinuity.

**Hypothesis parts moved here, as the draft wrote them.** H3's arm D part (D's freshness
worse than C's, because D serves the oldest retained items). H5's frame-size part (S3) and
its lock-held variant FH (S5), whose step latency grows with consumer work duration. H8's
gzip part (EP-H does not sustain 20M events/s with gzip). H9's W2 cells on runtime B. H10's
W6 part and its W2 contrast (at equal nominal cost, a NumPy-bound consumer, W2, has a smaller
effect than W6 on 3.11). H11's comparison with CPU-bound consumers of the same nominal
duration (W2, W6 may not stay within the noise band) and its queue-arm part (in queue arms,
W2 and W3 produce similar backlog, same service time, but different CPU use). H12's RE part.
The draft's judgement rows for these hypotheses, whole:

| H | Judged on | S if | C if |
|---|---|---|---|
| H3 | P1, arms C and D, W2, W3, W5 | SUSTAINED; footprint growth ≤ 0.5 MiB/s; declared drops > 0; D's freshness p50 > C's in every such cell | C or D is NOT_SUSTAINED in any W3 cell (sleeping consumers, so no CPU contention), or D's freshness p50 ≤ C's in most cells |
| H5 | P1 F; S3; S5 FH | F's excess step latency p99 increases with N at 1280x720, is smaller at 346x260 (S3) at the same N, and FH's increases with consumer cost (W2 > W1) | F's step p99 is not distinguishable (16.3) from G's at N = 8 for every workload, kernel and runtime |
| H8 | P2 | every run's read-back verifies with Blosc, in every arm; EP-H SUSTAINED with Blosc and NOT_SUSTAINED with gzip | EP-QB or EP-QE ends a Blosc run with DRAIN_TIMEOUT; or EP-H is NOT_SUSTAINED with Blosc at N_SO = 0; or EP-H is SUSTAINED with gzip. A read-back mismatch is INVALID_INTEGRITY (14.1), not evidence for or against H8. |
| H9 | P1, arm H | SUSTAINED, footprint growth ≤ 0.5 MiB/s and freshness age p95 ≤ 32 ms (two publication intervals), in every W1, W2 and W3 cell on runtime B and every W1 and W3 cell on runtime A | NOT_SUSTAINED in any W1 or W3 cell at N ≤ 4 |
| H10 | P1 W5, W6, W2 | runtime A: excess step p99 ≥ 5 ms for W5 and W6 in F, G and H at every N measured; runtime B: < 5 ms; on runtime A, W2's excess < W6's at N ∈ {1, 4} | on runtime A, W6's excess < 5 ms at N = 4 in both G and H |
| H11 | P1 W2, W3, W6 | latest-state arms: W3's excess step p99 not distinguishable from N = 0, while W2's or W6's is; unbounded queue arms (E, RE): W2 and W3 queue-depth slopes within ±20% of each other | W3's excess p99 distinguishable from N = 0 in most latest-state cells |
| H12 | P1 RB, RE | accumulation-path CPU per event (10.1) at N = 8 over N = 1 between 6 and 10; on runtime A with W1, RB or RE NOT_SUSTAINED, or with a depth slope > 0.5 items/s, at a smaller N than B and E are NOT_SUSTAINED or grow | that ratio < 4 at N = 8 |

**Draft outputs that belong to the follow-up study:** F12 (sustainable rate, S1) and the
kernel facets of F1 to F8.

## 21. Implementation constraints

### 21.1 The library as released

- No change to `src/` and no instrumentation hooks in it.
- The H arm uses the public API only (`Engine`, `ingest`, `snapshot`, `Snapshot`, `SnapshotMeta`).
- The baselines use only the public `Accumulator` (`accumulate`, `read`, `reset`,
  `watermark`), the public kernel classes and `frames2py.publish.Snapshot`.
- Consumers use `frames2py.viewer.render` and `frames2py.recorder`.
- The library comes from the Frames2Py 1.0.0 PyPI wheel (section 12). Every child asserts
  the version and that `frames2py.__file__` lies in the study environment's site-packages.
- The one exception is V1 (section 22). It uses the gate's virtual-clock substitution of
  `frames2py._engine.time` (`benchmarks/targets/v1.py`), in validation only, never in a
  measured run.

### 21.2 Code location and conventions

- **Harness:** `benchmarks/observation.py`, with helpers under `benchmarks/` as needed, and a
  CLI on the existing pattern: `python -m benchmarks observation {validate, calibrate, run,
  analyse}`, a JSON request on stdin to a worker process, the result on stdout.
- **Reuse:** `benchmarks.power.hold_awake`, `benchmarks.environment`,
  `benchmarks.measure.nearest_rank`, `benchmarks.workloads.Workload`,
  `benchmarks.matrix.default_seed`, and `benchmarks.targets.v1.reference` (V1).
- **Tests:** `tests/test_observation_benchmarks.py`, following `.claude/rules/testing.md`:
  harness behaviour against independent expectations, no change-detector tests.
- **Checks:** mypy strict and pyflakes on `benchmarks/`, as CI runs them (decisions.md 59).
- **Dependencies:** no new runtime dependency. A plotting or analysis dependency needs the
  user's approval before it is added.
- **Outputs:**
  - Raw outputs go to `scratch/observation_study/` (gitignored): raw logs per run as `.npz`,
    and a JSON document per run.
  - No result file is ever overwritten.
  - Publishing results in `benchmarks.md` or the docs needs the user's instruction (CLAUDE.md,
    "Handoff").
- **Freeze:** the analysis code (section 18) is written, tested on V4 output and committed
  with the harness, before the campaign.

### 21.3 Review before measurement

- Every arm is reviewed against its definition (6.4) and the competence requirements (6.7).
- The review finds and records the file and line of each arm's step and consumer loop.
- The measured commit contains the harness, the analysis and the calibration record, on a
  clean tree, with this preregistration's commit as an ancestor.

### 21.4 Integrity accounting, per run, after the window, untimed

| Scope | Check |
|---|---|
| Every arm | per-batch logs have no gap in k |
| Every arm | every observation's watermark resolves (9.2) |
| Every arm | each consumer's observed sequences increase strictly |
| Queue arms | FIFO order |
| H, H′, P2 arms | `engine.stats.events_ingested` = Σ events of the batches `ingest()` was called with |
| H, H′, P2 arms | `events_out_of_bounds` = 0 |
| Queue arms | per consumer: offered to the queue = enqueued + abandoned at stop; enqueued = dequeued + declared drops + remaining at stop |
| B, E, RB | declared drops = 0 |
| RB | per consumer: events enqueued = events accumulated + events remaining |
| RB | the final watermark equals `m` of the last accumulated batch |
| P2 | read-back SHA-256 equals the SHA-256 of the re-materialised batches whose put or `write()` completed, in order; a batch whose blocked put was abandoned at stop is backlog, not fed |

**Harness instrumentation:**
- Per batch: three `perf_counter_ns`, three `thread_time_ns`, one `published_sequence()`,
  and an append into preallocated storage.
- Per observation: two of each clock, and an append.
- Drop arms also log each drop.
- It is the same code in every arm. V3 measures its cost.

## 22. Validation stage

The validation stage runs after implementation and before the campaign. Its outputs check
the harness. They are not study results, except V3 and V6, which are reported as
characterisation. Nothing in the protocol may change because of them except through an
amendment (section 24).

V1, V2, V4 and V6 check function and take no timing measurements, so they may run in an
earlier session on a machine in normal use. V3 and V5 are timing measurements. They run
under the controls of section 13, in the campaign session, after the driver's environment
checks pass.

- **V1, semantic equivalence.**
  - Setup: single-threaded, in lockstep. 400 batches of this study's input (the synthetic
    1280x720 stream of 11.1, `event_count`). A virtual cadence clock advances by one batch
    period per step. Consumers are called after every step and record a copy of every state
    they observe.
  - The sequence of published (frame bytes, watermark, sequence) is identical, bit for bit,
    across A, B, C, E, F, G and H, at intervals 16 ms and 0 ms. Each consumer's windows in
    RB equal the Engine's under the same schedule.
  - The final states equal `benchmarks.targets.v1.reference`, exactly.
- **V2, harness tests.** `PolicyQueue` semantics against `queue.Queue` and exact drop counts;
  watermark-to-batch resolution, including ties; queue-depth reconstruction; the SUSTAINED
  classifier on constructed lag series; the outcome classifier; the accounting of 21.4.
- **V3, instrumentation overhead.**
  - Cells: A, G and H at N = 0; H, B and RB at N = 1 with W1, so the per-item recording of
    queue and raw consumers is covered too; `event_count`, both runtimes, 3 repetitions.
  - Full recording against aggregate-only recording (two clock reads per step).
  - Reported as the difference in busy time per event and step p99.
- **V4, dry run.** Every arm, W1, N = 1, one run each with a 5 s window, on both runtimes.
  Every harness, integrity and accounting check must pass. On a machine in normal use the
  environment checks of 13.3 are recorded and do not gate V4. The output only proves the
  harness runs, and is used to test the analysis code. It is not reported as data and
  nothing is tuned on it.
- **V5, calibration of K5.** (K2 and K6 belong to the follow-up study.)
  - Conditions: runtime A, no producer, one thread, a 1280x720 `uint32` frame.
  - Start with K = 1 and take the median of 50 calls as the unit cost. Set
    `K = round(target / unit)`. Verify that the median of 20 calls lies within ±3% of the
    target (250 ms). If it doesn't, set `K = round(K · target / median)`, for at most 5
    iterations. If it still misses, stop and report.
  - K5 and every calibration measurement are committed as the calibration record before the
    campaign. The same value is used everywhere.
- **V6, allocation.** `tracemalloc`, untimed. One short run per arm at N = 1 with W1. Reports
  the peak temporary bytes per producer step and per observation, as characterisation.

## 23. Preregistration timing

1. The draft was reviewed by the user and changed only at the user's direction.
2. The exact approved text is committed and pushed to the study branch
   (`docs/observation-study-preregistration`) before implementation. The pushed commit is
   the preregistration's timestamp. It is merged later, at the user's decision.
3. Only then may implementation begin: harness, analysis and tests. (The study environments
   were built before this commit, for the feasibility check of section 12. They contain no
   study code.)
4. Measurements, validation included, begin only after the implementation is complete,
   committed and reviewed (21.3). The preregistration commit must be an ancestor of the
   measured commit.
5. The driver records this file's SHA-256 in every document. It refuses to run if the file
   differs from its content at the measured commit, or if the tree is not clean.
6. The calibration record (V5) is committed before the first campaign run, so git shows
   that the work amounts predate the data.

Git history thus establishes that the protocol predates the results.

## 24. Amendment policy

- The protocol changes only through **numbered, dated amendments** appended to this file.
  Each is committed before any measurement it affects. Each states:
  - what changes, and why;
  - which data, if any, existed when it was written;
  - which cells it affects.
- **No amendment may be motivated by results of the cells it affects.** If a flaw is found
  after affected data exist:
  - those data are kept and labelled with the flaw;
  - the amendment fixes the protocol;
  - every affected cell is re-measured under it, in whole passes;
  - both sets are reported.
- **Harness defects** found during the campaign (HARNESS_FAILURE, a confirmed
  INVALID_INTEGRITY caused by the harness) are fixed by an amendment that names the defect
  and the commit, followed by a re-run of the affected passes. In an unattended session these
  are the only amendments permitted, and they are committed and pushed before the re-run
  (14.2).
- **Allowed without an amendment:**
  - corrections of typographical errors that change no parameter, rule or definition,
    committed and listed;
  - analysis additions labelled "not preregistered", reported separately and never replacing
    a preregistered analysis.
- **Deviations from this protocol** that occur anyway are reported in the results with
  their extent.

### 24.1 Listed corrections

Corrections of descriptive figures that change no parameter, rule or definition.

1. **2026-09-30, before V3, V5 and any campaign run.** Two figures derived from the publication
   interval were wrong. The cadence rule (6.3) is evaluated once per step, and steps are 5 ms
   apart, so with a 16 ms interval the first call that qualifies comes 20 ms after the last
   publication. The effective spacing is therefore 20 ms: 50 publications/s.
   - 11.3: "625 publications per run" is now 500 per 10 s window.
   - 6.5.1: K_ff's time depth, 4 publications, is now 80 ms, not 64 ms.
   - Unchanged: K_ff = 4, and the 64 ms time-depth rule that defines K_raw = 13 (6.5.1).
   - Unchanged: every threshold, rule and cell.
   - Data existing when this was written: validation output only (V1, V2, V4, V6). The V4 dry
     run showed about 50 publications/s.
   - The draft had the same error (1,250 per 20 s).

### 24.2 Amendments

1. **Amendment 1, 2026-09-30: a harness defect found in V3 (14.2).**
   - **Defect.** The integrity check that each queue's dequeue count equals what its
     consumer took (21.4) compared the count with the consumer's per-item key log.
     Aggregate-only instrumentation (V3, 22) does not keep that log, so the check failed even
     when nothing was lost. The second V3 run, `V3_B_W1_N1_B_aggregate_rep1`, was classified
     INVALID_INTEGRITY on `queue 0: dequeued 750, consumer took 0`: 750 items were enqueued,
     750 dequeued, none remained, and the consumer observed all 750.
   - **Fix.** Commit `866397348486e7f5d4e08b32ca2a61a8a933ca82` compares the dequeue count with the consumer's observation count,
     which both instrumentation modes keep. It adds a test of the accounting under
     aggregate-only instrumentation. Nothing else in the harness changes.
   - **What changes in the protocol.** Nothing. The check, its rule and every parameter,
     threshold and cell stay as written. Only the harness's implementation of the check is
     corrected.
   - **Data existing when this was written.** Validation only: V1, V2, V4 and V6 of Session 1,
     and the two V3 runs of `scratch/observation_study/validation/V3-20260930T151247Z/`. That
     directory holds one VALID run and the INVALID_INTEGRITY run above. Both are kept and
     labelled with this defect, and neither is used. No V5 or campaign run existed.
   - **Affected.** V3, which is re-run whole. V1, V2, V4 and V6 are unaffected: their runs
     used full instrumentation, under which the key log and the observation count agree, and
     they all passed.
2. **Amendment 2, 2026-09-30: a second harness defect found in V3 (14.2).**
   - **Defect.** The V3 stage declared V3 failed as soon as a run was INVALID_ENV. The
     protocol re-queues such a run at the end of its pass, at most twice (14.1). In the V3
     re-run under amendment 1 (`scratch/observation_study/validation/V3-20260930T153509Z/`),
     69 of 72 runs were VALID. Three were INVALID_ENV on background load, from 15:55 to
     15:58 UTC:
     - `V3_A_none_N0_B_aggregate_rep3`: WindowServer at or above 10%, and other processes
       together at 42.3% and 44.5%;
     - `V3_H_W1_N1_A_full_rep3`: other processes together at 37.7%;
     - `V3_H_W1_N1_B_full_rep3`: other processes together at 37.3%.
   - **Likely cause of the load (INFERENCE).** The launcher's own foreground activity: those
     minutes are when the launcher switched from background waits to a foreground wait. It
     makes only background waits from here on (13.4).
   - **Fix.** Commit `28f43df75cf84751d3eab469e4791146eae24b17` runs each V3 repetition as a pass under 14.1's retry rule. It adds
     a test of that rule. Nothing else changes.
   - **What changes in the protocol.** Nothing: rules, parameters and cells stay as written.
   - **Data existing when this was written.** Validation only: that V3 directory (72 runs,
     kept, not used) and the earlier directory under amendment 1. No V5 or campaign run
     existed.
   - **Affected.** V3, which is re-run whole.
3. **Amendment 3, 2026-09-30: a harness defect found when the campaign stopped (14.2).**
   - **What happened.** The machine lost AC power at 17:14 UTC during P1 pass 1, and every
     run after that was INVALID_ENV. Under the stop rule "environment checks keep failing",
     the launcher stopped the driver at about 17:20 UTC. A child had already started. It
     finished `P1_B_W5_N1_A_r0_p1_a2` on battery and wrote its result files after the driver
     had gone.
   - **Defect.** The driver had no way to record an attempt that finished after it stopped. The
     attempt had result files but no ledger entry and no environment checks around it.
     Resuming the pass would have reused its id and stopped on `FileExistsError`, and the
     attempt would have been missing from the ledger, where every attempt is kept (14.1).
   - **Fix.** Commit `a72c18e7b3f4a098551579178476bb69785a87a5`: on start, the driver records every run file that has no ledger
     entry as an attempt of its pass. It is INVALID_ENV, "the driver was interrupted during
     this attempt; the checks around it were not recorded", unless the run's own record shows
     a HARNESS_FAILURE, SHUTDOWN_TIMEOUT, CRASH or INVALID_INTEGRITY, which takes precedence.
     Its id is never reused. The fix adds tests.
   - **What changes in the protocol.** Nothing: rules, parameters and cells stay as written.
   - **Data existing when this was written.** V1 to V6, and P1 pass 1 of session
     `20260930T163756Z-55172e`: 124 attempts in the ledger (106 VALID, 18 INVALID_ENV on AC
     power) plus the unrecorded attempt above.
   - **Affected.** Only that attempt, which becomes attempt 2 of its cell, INVALID_ENV. No
     measured run is affected: the defect is in the driver's bookkeeping after an
     interruption, not in any code that measures. At the user's instruction (2026-09-30),
     P1 pass 1 resumes from its last completed run (13.4) rather than being re-run.
4. **Amendment 4, 2026-10-01: a harness defect found in P2 (14.2), introduced by amendment 1's
   fix.**
   - **What happened.** The first P2 run with no state-observation consumer,
     `P2_EP-QE_none_N0_A_r0_p1_a1`, was a HARNESS_FAILURE: `KeyError: 'c0_samples'` in the
     integrity accounting. The campaign stopped there, at 21:46 UTC on 2026-09-30.
   - **Defect.** Amendment 1's fix (`8663973`) read consumer i's observation count for every
     queue i. EP-QB and EP-QE at N_SO = 0 have a recorder queue and no consumer. The count is
     needed only for B, C and E, where every queue has its consumer.
   - **Fix.** Commit `602f9b3febc1b6113d6c0e0887f5858d4ed4fec4` reads the count only for B, C and E. It extends the P2 tests to
     N_SO = 0, which fail on the defective check. It raises the campaign revision to 1, so P2's
     passes run as new pass records.
   - **What changes in the protocol.** Nothing: rules, parameters and cells stay as written.
   - **Data existing when this was written.**
     - P1 passes 1 to 5, complete: 590 VALID runs (118 cells × 5), plus 34 INVALID_ENV
       attempts. They are unaffected: every P1 queue arm has a consumer per queue, and none
       failed the check.
     - P2 pass 1, revision 0: two VALID runs and the HARNESS_FAILURE above.
   - **Affected.** P2 pass 1, which is re-run whole at revision 1. Its revision-0 attempts are
     kept and not used. P2 passes 2 to 5 had not started and run at revision 1. The failed
     run's recording, `scratch/observation_study/tmp/P2_EP-QE_none_N0_A_r0_p1_a1.h5`, is kept
     unread: the harness stopped before its read-back.

