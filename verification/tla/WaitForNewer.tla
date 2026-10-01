--------------------------- MODULE WaitForNewer ---------------------------
(***************************************************************************)
(* The wait_for_newer protocol of decisions.md 80: a per-waiter lock in a   *)
(* registry deque, the producer's slot store, a private marker and a       *)
(* marker-bounded drain.                                                    *)
(*                                                                         *)
(* Memory: the producer's slot stores (publications and reset) go through  *)
(* a FIFO store buffer and reach shared memory in a separate Flush step;   *)
(* everyone else reads shared memory. That is the x86-TSO abstraction, the *)
(* weakest model in which the store-buffering outcome (each side misses    *)
(* the other's write) can occur; ARM64 permits it too. A registry critical *)
(* section is one atomic step, as CPython runs deque append, popleft and   *)
(* remove inside the deque's critical section and in C only. The producer  *)
(* may enter it only with an empty store buffer: the slot store comes      *)
(* before the critical section's release in program order, so an acquirer  *)
(* of the same mutex sees it (C11 synchronises-with; see README.md), and   *)
(* on x86 the locked instruction drains the buffer.                        *)
(*                                                                         *)
(* Sequences: publication k stores k; 0 is "no snapshot" (before the first  *)
(* publication, and after reset). A waiter with floor f wants slot > f, so *)
(* floor 0 is wait_for_newer(None).                                        *)
(*                                                                         *)
(* Variant selects the protocol:                                           *)
(*   "correct"     decisions.md 80                                         *)
(*   "fastpath"    the producer skips the registry when an unlocked size   *)
(*                 read sees it empty (the broken variant 80 names)        *)
(*   "condvar"     a waiter holds a registry lock while it reads and       *)
(*                 registers, and the producer takes it to drain           *)
(*   "untilempty"  the producer drains until the registry is empty, with   *)
(*                 no marker                                               *)
(***************************************************************************)
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS Waiters,   \* waiter ids (model values)
          Ops,       \* the producer's program: a sequence over {"pub", "reset"}
          Floor,     \* [Waiters -> Nat]: the floor of each waiter's first call
          Rounds,    \* calls per waiter; each later call's floor is the previous result
          Timeouts,  \* timeouts available to each waiter, taken nondeterministically
          Variant

None == "none"
Lock(id, w) == [kind |-> "lock", id |-> id, owner |-> w]
Marker(id) == [kind |-> "marker", id |-> id, owner |-> None]
NoEntry == [kind |-> "none", id |-> 0, owner |-> None]

VARIABLES
    slot, pbuf,               \* shared memory and the producer's store buffer
    reg, stamp, released,     \* registry, next entry id, ids of released waiter locks
    holder,                   \* "condvar" only: who holds the registry lock
    ppc, op, seq, cur, popped, drainStart,
    wpc, floor, myid, res, rounds, tleft, expired,
    errors                    \* set of strings: violations observed inside actions

vars == <<slot, pbuf, reg, stamp, released, holder, ppc, op, seq, cur, popped, drainStart,
          wpc, floor, myid, res, rounds, tleft, expired, errors>>
pvars == <<ppc, op, seq, cur, popped, drainStart>>
wvars == <<wpc, floor, myid, res, rounds, tleft, expired>>

Init ==
    /\ slot = 0 /\ pbuf = <<>>
    /\ reg = <<>> /\ stamp = 1 /\ released = {}
    /\ holder = None
    /\ ppc = "next" /\ op = 1 /\ seq = 0 /\ cur = 0 /\ popped = NoEntry /\ drainStart = 0
    /\ wpc = [w \in Waiters |-> "read1"]
    /\ floor = Floor
    /\ myid = [w \in Waiters |-> 0]
    /\ res = [w \in Waiters |-> 0]
    /\ rounds = [w \in Waiters |-> 1]
    /\ tleft = [w \in Waiters |-> Timeouts]
    /\ expired = [w \in Waiters |-> FALSE]
    /\ errors = {}

----------------------------------------------------------------------------
(* Memory system *)

Flush ==
    /\ pbuf /= <<>>
    /\ slot' = Head(pbuf)
    /\ pbuf' = Tail(pbuf)
    /\ UNCHANGED <<reg, stamp, released, holder, errors>> /\ UNCHANGED pvars /\ UNCHANGED wvars

----------------------------------------------------------------------------
(* Producer *)

NextOp ==
    /\ ppc = "next"
    /\ IF op > Len(Ops)
         THEN /\ ppc' = "done" /\ UNCHANGED <<pbuf, op, seq>>
         ELSE IF Ops[op] = "pub"
           THEN /\ pbuf' = Append(pbuf, seq + 1) /\ seq' = seq + 1 /\ ppc' = "regop"
                /\ UNCHANGED op
           ELSE /\ pbuf' = Append(pbuf, 0) /\ op' = op + 1      \* reset: stores None, wakes nobody
                /\ UNCHANGED <<seq, ppc>>
    /\ UNCHANGED <<slot, reg, stamp, released, holder, cur, popped, drainStart, errors>> /\ UNCHANGED wvars

\* The registry operation after the store: append the marker (or, in "untilempty", just start).
EnterDrain ==
    /\ pbuf = <<>>
    /\ IF Variant = "untilempty"
         THEN /\ drainStart' = stamp /\ UNCHANGED <<reg, stamp, cur>>
         ELSE /\ reg' = Append(reg, Marker(stamp)) /\ cur' = stamp /\ drainStart' = stamp
              /\ stamp' = stamp + 1
    /\ ppc' = "drain"

RegOp ==
    /\ ppc = "regop"
    /\ CASE Variant = "fastpath" ->
              \* Unlocked size read: a plain load, which doesn't wait for the store buffer.
              IF Len(reg) = 0
                THEN /\ ppc' = "next" /\ op' = op + 1
                     /\ UNCHANGED <<reg, stamp, cur, drainStart, holder>>
                ELSE EnterDrain /\ UNCHANGED <<op, holder>>
         [] Variant = "condvar" ->
              /\ holder = None /\ holder' = "P"
              /\ EnterDrain /\ UNCHANGED op
         [] OTHER -> EnterDrain /\ UNCHANGED <<op, holder>>
    /\ UNCHANGED <<slot, pbuf, released, seq, popped, errors>> /\ UNCHANGED wvars

DrainPop ==
    /\ ppc = "drain"
    /\ IF Variant = "untilempty" /\ reg = <<>>
         THEN /\ ppc' = "next" /\ op' = op + 1
              /\ UNCHANGED <<reg, popped, errors, holder>>
         ELSE LET h == Head(reg) IN
              /\ reg' = Tail(reg)
              /\ IF h.kind = "marker" /\ h.id = cur
                   THEN /\ ppc' = "next" /\ op' = op + 1 /\ UNCHANGED <<popped, errors>>
                        /\ holder' = IF Variant = "condvar" THEN None ELSE holder
                   ELSE /\ popped' = h /\ ppc' = "release" /\ UNCHANGED <<op, holder>>
                        /\ errors' = IF h.id >= drainStart THEN errors \cup {"drain past its start"}
                                                           ELSE errors
    /\ UNCHANGED <<slot, pbuf, stamp, released, seq, cur, drainStart>> /\ UNCHANGED wvars

DrainRelease ==
    /\ ppc = "release"
    /\ IF popped.id \in released
         THEN /\ errors' = errors \cup {"double release"} /\ UNCHANGED released
         ELSE /\ released' = released \cup {popped.id} /\ UNCHANGED errors
    /\ ppc' = "drain" /\ popped' = NoEntry
    /\ UNCHANGED <<slot, pbuf, reg, stamp, holder, op, seq, cur, drainStart>> /\ UNCHANGED wvars

Producer == NextOp \/ RegOp \/ DrainPop \/ DrainRelease

----------------------------------------------------------------------------
(* Waiters *)

\* End a call with result v (v = 0 with expired: a timeout's None, floor unchanged).
Finish(w, v, newer) ==
    /\ floor' = [floor EXCEPT ![w] = IF newer THEN v ELSE @]
    /\ expired' = [expired EXCEPT ![w] = FALSE]
    /\ IF rounds[w] < Rounds
         THEN /\ rounds' = [rounds EXCEPT ![w] = @ + 1] /\ wpc' = [wpc EXCEPT ![w] = "read1"]
         ELSE /\ wpc' = [wpc EXCEPT ![w] = "done"] /\ UNCHANGED rounds

Read1(w) ==
    /\ wpc[w] = "read1"
    /\ Variant = "condvar" => holder = None
    /\ IF slot > floor[w]
         THEN /\ Finish(w, slot, TRUE) /\ UNCHANGED holder
         ELSE IF expired[w]
           THEN /\ Finish(w, 0, FALSE) /\ UNCHANGED holder
           ELSE /\ wpc' = [wpc EXCEPT ![w] = "register"]
                /\ holder' = IF Variant = "condvar" THEN w ELSE holder
                /\ UNCHANGED <<floor, rounds, expired>>
    /\ UNCHANGED <<slot, pbuf, reg, stamp, released, errors, myid, res, tleft>> /\ UNCHANGED pvars

Register(w) ==
    /\ wpc[w] = "register"
    /\ reg' = Append(reg, Lock(stamp, w))
    /\ myid' = [myid EXCEPT ![w] = stamp]
    /\ stamp' = stamp + 1
    /\ wpc' = [wpc EXCEPT ![w] = "read2"]
    /\ UNCHANGED <<slot, pbuf, released, holder, errors, floor, res, rounds, tleft, expired>>
    /\ UNCHANGED pvars

Read2(w) ==
    /\ wpc[w] = "read2"
    /\ IF slot > floor[w]
         THEN /\ res' = [res EXCEPT ![w] = slot] /\ wpc' = [wpc EXCEPT ![w] = "removeret"]
         ELSE /\ wpc' = [wpc EXCEPT ![w] = "block"] /\ UNCHANGED res
    /\ holder' = IF Variant = "condvar" THEN None ELSE holder
    /\ UNCHANGED <<slot, pbuf, reg, stamp, released, errors, floor, myid, rounds, tleft, expired>>
    /\ UNCHANGED pvars

Wake(w) ==
    /\ wpc[w] = "block"
    /\ myid[w] \in released
    /\ wpc' = [wpc EXCEPT ![w] = "remove"]
    /\ UNCHANGED <<slot, pbuf, reg, stamp, released, holder, errors, floor, myid, res, rounds, tleft, expired>>
    /\ UNCHANGED pvars

TimeOut(w) ==
    /\ wpc[w] = "block"
    /\ myid[w] \notin released
    /\ tleft[w] > 0
    /\ tleft' = [tleft EXCEPT ![w] = @ - 1]
    /\ expired' = [expired EXCEPT ![w] = TRUE]
    /\ wpc' = [wpc EXCEPT ![w] = "remove"]
    /\ UNCHANGED <<slot, pbuf, reg, stamp, released, holder, errors, floor, myid, res, rounds>>
    /\ UNCHANGED pvars

Without(s, id) == SelectSeq(s, LAMBDA e : e.id /= id)

Remove(w) ==   \* the finally: remove its own entry if a publication hasn't taken it, then read again
    /\ wpc[w] = "remove"
    /\ reg' = Without(reg, myid[w])
    /\ wpc' = [wpc EXCEPT ![w] = "read1"]
    /\ UNCHANGED <<slot, pbuf, stamp, released, holder, errors, floor, myid, res, rounds, tleft, expired>>
    /\ UNCHANGED pvars

RemoveReturn(w) ==   \* the re-read after registering found a newer snapshot
    /\ wpc[w] = "removeret"
    /\ reg' = Without(reg, myid[w])
    /\ Finish(w, res[w], TRUE)
    /\ UNCHANGED <<slot, pbuf, stamp, released, holder, errors, myid, res, tleft>>
    /\ UNCHANGED pvars

WaiterStep(w) == Read1(w) \/ Register(w) \/ Read2(w) \/ Wake(w) \/ Remove(w) \/ RemoveReturn(w)

Next == Flush \/ Producer \/ \E w \in Waiters : WaiterStep(w) \/ TimeOut(w)

\* Weak fairness on everything except timeouts, which are optional nondeterminism.
Spec == Init /\ [][Next]_vars /\ WF_vars(Flush) /\ WF_vars(Producer)
        /\ \A w \in Waiters : WF_vars(WaiterStep(w))

----------------------------------------------------------------------------
(* Properties *)

TypeOK ==
    /\ slot \in Nat /\ \A i \in 1..Len(pbuf) : pbuf[i] \in Nat
    /\ ppc \in {"next", "regop", "drain", "release", "done"}
    /\ \A w \in Waiters : wpc[w] \in {"read1", "register", "read2", "block", "remove", "removeret", "done"}

Quiescent == ppc = "done" /\ pbuf = <<>>

\* No lost wakeup: once the producer has finished and its stores are visible, no waiter sits
\* blocked on an unreleased lock while a snapshot newer than its floor is published.
NoLostWakeup ==
    Quiescent => \A w \in Waiters :
        ~(wpc[w] = "block" /\ myid[w] \notin released /\ slot > floor[w])

\* Each popped entry is released once, and only entries registered before the drain began.
NoProtocolErrors == errors = {}

\* The producer is never unable to move because of a waiter: whenever it isn't done, one of its
\* own steps, or the flush of its own store buffer, is enabled.
ProducerNeverWaits == ppc /= "done" => (ENABLED Producer \/ ENABLED Flush)

\* Every waiter that should observe a publication eventually proceeds: eventually the producer
\* is done and every waiter has returned or is blocked with nothing newer published.
Settled(w) == \/ wpc[w] = "done"
              \/ wpc[w] = "block" /\ myid[w] \notin released /\ slot <= floor[w]
EventuallySettled == <>[](Quiescent /\ \A w \in Waiters : Settled(w))
=============================================================================
