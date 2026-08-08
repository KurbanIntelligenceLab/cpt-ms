# Status — two decisions need the team, everything else proceeds

## Decisions that need input

### 1. Compute: the 8-GPU node does not exist (Sync-A budget)

Case B holds. All six registered SSH targets are single-GPU vast.ai boxes — two RTX 5090s
and four RTX 4090s — and every one is attached to another active project. Two aliases point
at endpoints their own provider notes record as torn down, so the live count is four, none
of them available to this project. Nothing has been rented.

The plan sizes the bake-off at 4 variants x 5 seeds x 2 backbones, roughly 40 runs, and
assumes an 8-GPU node clears it in days. On one box those 40 runs serialize.

Three options, in the order I would consider them:

1. Rent one RTX 5090 dedicated to this project and serialize. Model sizes are modest at
   10-25M parameters, so memory is not the constraint; throughput is. Matches the existing
   pattern on this account.
2. Rent several single-GPU boxes and shard by seed or variant. Recovers most of the assumed
   parallelism at proportionally higher spend, with a separate environment build per box.
3. Reduce the matrix — fewer seeds or one backbone. This weakens the statistical claims the
   plan makes at 5 seeds with confidence intervals, and it interacts with pre-registered
   thresholds, so it should not be taken silently.

My recommendation is option 1, then decide between serialize and shard once the timing
prototype gives a per-run wall-clock. That measurement is cheap and converts the choice into
arithmetic.

This also gates the CP2 overhead benchmark, whose threshold is pre-registered against the
training setting. It cannot be sized until the training hardware is chosen.

### 2. B1 serialization: charge balance is sound but nearly inert

Measured, not assumed: on the CIF character backbone, family (i) charge balance has a mean
usable fraction of 0.002 — active over roughly the last two tokens of a 300-800 token
sequence — because CIF spells out composition in the atom-site loop at the very end. Family
(ii) density is fine there at 0.670, since the cell block sits in the header. On the
structured backbone both are healthy (0.785 and 0.480).

A projection active over 0.2% of positions will not produce a measurable internalization
curve, which is what CP0 reads. Three options:

1. Reorder the B1 serialization so composition precedes the cell block. Makes B1 behave like
   B2; costs a deviation from standard CIF field order, which matters only if B1 is meant to
   be directly comparable to published CIF language models.
2. Scope family (i) to B2 and run B1 on family (ii) only. Clean; costs the charge-balance
   claim on one of two backbones.
3. Accept the near-terminal signal and report it. Honest, but produces a null that is a fact
   about the serialization rather than about the method.

My recommendation is 1 if the published-model comparison is not load-bearing, otherwise 2.
The question for the team is exactly that: is B1's comparability to published CIF models
load-bearing for the paper?

Whichever is chosen, the CP0 active-multiplier threshold must be pre-registered per backbone
and per family, since a constraint's usable fraction bounds the gradient it can contribute.

## Completed since the last update

The projection layer is built and verified. Gradients agree with cvxpylayers to 2.0e-6
relative across five constraint configurations and with a step-swept finite-difference
oracle to 4.6e-10. Forward residuals at most 7.8e-11 in 4-5 Newton iterations. 200
randomized Jacobian rank checks, zero deficient.

The flat-dual line-search defect is recorded as a named finding with its mechanism, not as a
changelog entry, and the regression test is confirmed load-bearing — it fails against the
reverted fix at offsets 1e-8 and 1e-9 and passes when restored. The pre-flight harness runs
the same near-optimal case on every launch.

The run ledger and pre-flight harness are in place. The ledger is append-only and
line-delimited, stores no results so it cannot disagree with the artifacts, and tolerates a
truncated final line from a hard kill. The harness runs six checks and reports all of them
rather than stopping at the first, including a scan for runs left in `started` from a
previous crash.

Carry-forwards: the base-rate pair is now stated per row in the sweep power table. Version
control is deferred at the user's direction; artifacts record an explicit unversioned
sentinel rather than a fabricated hash.

## Proceeding without input

A3's induced sequence distribution against the exponential tilt, on the synthetic task
first, where both targets are exact. That is the comparison the plan's A3-vs-A6 contrast
rests on, and it does not depend on either decision above: the synthetic task has its own
serialization and needs no GPU.

The B1 decision blocks only the crystal-backbone half of that comparison. The compute
decision blocks the CP2 benchmark and the bake-off.
