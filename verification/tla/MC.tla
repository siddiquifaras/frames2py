---------------------------------- MODULE MC ----------------------------------
EXTENDS WaitForNewer
CONSTANTS w1, w2, w3
OneWaiter == {w1}
TwoWaiters == {w1, w2}
ThreeWaiters == {w1, w2, w3}
OnePub == <<"pub">>
TwoPubs == <<"pub", "pub">>
PubResetPub == <<"pub", "reset", "pub">>
ThreePubs == <<"pub", "pub", "pub">>
\* w1 waits for any publication; w2 for one past the first; w3 for any.
Floors(ws) == [w \in ws |-> IF w = w2 THEN 1 ELSE 0]
FloorOne == Floors(OneWaiter)
FloorTwo == Floors(TwoWaiters)
FloorThree == Floors(ThreeWaiters)
================================================================================
