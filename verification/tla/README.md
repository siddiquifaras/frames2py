# TLA+ model of `wait_for_newer`

A model of the waiter registry and publication protocol of `Engine.wait_for_newer` (decisions.md 80), checked with
TLC. It is verification material for the repository, not part of the package: the wheel and the sdist never contain
it (`.github/scripts/check_dist.py`).

- `WaitForNewer.tla`: the protocol and its properties.
- `MC.tla`: the model values the configurations use.
- `correct_*.cfg`: the protocol as specified, at several bounds.
- `broken_*.cfg`: deliberately broken variants, which TLC must reject.

## What is modelled

- **Producer:** a program of publications and resets. Each publication stores the slot, appends a private marker to the
  registry, then pops and releases entries until it pops its own marker.
- **Waiters:** each one reads; registers a fresh locked lock; reads again; blocks until released or until it times out;
  removes its entry; and reads again. A call ends with a newer snapshot or, after a timeout, with the final read.
- **Memory:** the producer's slot stores go through a FIFO store buffer, and everyone else reads shared memory (the
  x86-TSO abstraction). That is the weakest model in which the store-buffering outcome, each side missing the other's
  write, can occur; ARM64 permits it too.
  - A registry operation is one atomic step, as CPython runs deque `append`, `popleft` and `remove` in the deque's
    critical section, in C.
  - The producer enters a registry operation only with an empty store buffer. In C11 terms, the slot store precedes
    the critical section's release, so a thread that later acquires the same mutex sees it. TLC doesn't check that
    step; it is a reading of the memory model.
- **Not modelled:** CPython's parking lot and the GIL; interrupted publications; `stop()`, which is a publication like
  any other or none; orderings weaker than TSO for other access pairs; and more than 3 waiters or 3 publications.

## Properties

| property | kind | meaning |
|---|---|---|
| `NoLostWakeup` | invariant | once the producer is done and its stores are visible, no waiter is blocked on an unreleased lock while a newer snapshot is published |
| `NoProtocolErrors` | invariant | no lock is released twice; a drain releases only entries registered before it began |
| `ProducerNeverWaits` | invariant | while the producer isn't done, one of its steps, or the flush of its own buffer, is enabled |
| `EventuallySettled` | liveness | with weak fairness on every step except timeouts, the producer finishes and every waiter returns or waits with nothing newer published |
| `TypeOK` | invariant | variables keep their types |

## Running TLC

TLC comes from `tla2tools.jar` and needs Java 11 or later. The results below used `tla2tools.jar` v1.7.4 (TLC 2.19,
SHA-256 `936a262061c914694dfd669a543be24573c45d5aa0ff20a8b96b23d01e050e88`) on Eclipse Temurin 21.

```sh
curl -fsSLO https://github.com/tlaplus/tlaplus/releases/download/v1.7.4/tla2tools.jar
cd verification/tla
java -XX:+UseParallelGC -cp ../../tla2tools.jar tlc2.TLC -workers auto -config correct_2w_2p.cfg MC.tla
```

TLC writes its state files under `states/` in the working directory; delete them afterwards.

## Results (2026-10-02, Apple M4)

| configuration | result | distinct states | time |
|---|---|---|---|
| `correct_1w_1p` | every property holds | 381 | <1 s |
| `correct_2w_2p` | every property holds | 617,604 | 10 s |
| `correct_2w_reset` | every property holds | 799,996 | 13 s |
| `correct_2w_3p` | every property holds | 40,158,941 | 11 min 38 s |
| `correct_3w_1p` | every property holds | 10,278,938 | 4 min 35 s |
| `correct_3w_2p` | not completed: stopped by hand after 42,325,897 states with 9,487,649 queued; no violation up to then; liveness not reached | - | ~17 min |
| `broken_fastpath_1w_1p` | `NoLostWakeup` violated | 68 | <1 s |
| `broken_fastpath_2w_2p` | `NoLostWakeup` violated | 1,258 | <1 s |
| `broken_fastpath_1w_1p_liveness` | `EventuallySettled` violated | 68 | <1 s |
| `broken_condvar_1w_1p` | `ProducerNeverWaits` violated | 32 | <1 s |
| `broken_untilempty_1w_1p` | `NoProtocolErrors` violated | 203 | <1 s |

The unlocked fast path's counterexample is the store-buffering lost wakeup:
1. The producer's store of publication 1 sits in its buffer.
2. Its unlocked size read finds the registry empty, so it skips the drain.
3. The waiter reads the old slot, registers, reads the old slot again, and blocks.
4. The buffer then flushes, and the producer finishes, leaving the waiter blocked.

The sequentially consistent interleaving tests (`tests/contract/test_wait_interleavings.py`) can't produce that trace.

These results check the protocol at these bounds under the stated abstraction. They are not a proof of the
implementation or of CPython's primitives.
