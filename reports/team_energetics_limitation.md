# The energetics step cannot reach our generated samples, and why

Raised because it changes what the paper can claim about physical plausibility, and the
reason is structural rather than a tooling gap that more compute would close.

## Summary

The ML-potential energy screen is built and calibrated well — held-out agreement of 21
meV/atom against MP-20's own DFT labels. Neither it nor DFT can be applied to our
generated samples. The serialization the model produces is composition plus cell with no
atomic positions, and both an ML potential and a DFT calculation require positions.
Obtaining them is crystal structure prediction, a separate problem the model does not
attempt. Three position-free screens are built instead, and the measured data reference
turns out to matter more than the screens: real MP-20 structures score 0.715 on charge
neutrality, so any generated arm must be reported against that ceiling rather than against
1.0.

**Correction to an earlier version of this note.** It stated that no DFT engine was
available in this environment. That was asserted without testing the plan's own designated
engine: I had probed for `vasp_std`, `pw.x` and `castep.serial` and for ML-potential
packages, but never for GPAW. GPAW **is** available. It installs from conda-forge on the
compute box and runs: version 25.7.0 with libxc 7.1.2, PAW datasets, MPI and OpenMP
enabled, and a PBE calculation on Si diamond (2 atoms, 300 eV cutoff, 3x3x3 k-points)
completes in 3.2 s. A local source build does fail on macOS/arm64 with a compiler-flag
error, but the box is where DFT would run and there it works.

The substitution of an ML potential for DFT is therefore **not** justified by engine
unavailability. It is justified only for reference structures, where DFT labels already
exist in MP-20 and recomputing them buys nothing. For generated samples the blocker is
the absence of atomic positions, which blocks DFT and the ML potential identically. The
availability of a working DFT engine does not change that, and no claim below depended on
the incorrect statement.

## What was built and what it is good for

`scripts/crystal/energy_screen.py` calibrates CHGNet against MP-20's DFT labels. CHGNet
predicts total energy per atom, so formation energy requires elemental references; those
are fitted by least squares on one half of a 3,000-structure sample and evaluated on the
other.

| Quantity | Value |
|---|---|
| MAE | 0.0210 eV/atom |
| RMSE | 0.0298 eV/atom |
| Bias | +0.0015 eV/atom |
| Pearson r | 0.99958 |
| Label spread | 1.0265 eV/atom |
| Held-out structures | 1,456 |
| Excluded for rare elements | 44 |
| Element references | 79, full rank |

MAE is 2.0% of the label spread and below the 30 meV/atom window conventionally used for
near-hull stability. The artifact carries `is_dft: false` and a test asserts it, so no
downstream claim can drift into calling this DFT. The surrogate is used for these
reference structures because MP-20 already carries DFT labels for them and recomputing
them buys nothing — not because DFT is unavailable, which is what an earlier version of
this note wrongly claimed.

This screen is sound for reference structures. It is not applicable to our generated
samples.

## Why it does not apply to generated samples

The generated token stream is `<BOS> <CELL> len len len ang ang ang <END_CELL> <COMP>
el cnt ... <END_COMP> <EOS>` — six cell parameters and a composition. There are no Wyckoff
positions and no fractional coordinates, by design: the constraint the projection steers is
a function of composition and cell, and nothing in the method requires positions.

An energy model requires them. Three routes were considered and all fail for reasons worth
recording:

**Solve for positions.** Placing atoms given a composition and cell is crystal structure
prediction. The model does not attempt it and adding it would mean evaluating a different
system than the one the paper describes.

**Relax from random positions.** The resulting energy is dominated by the initialization
rather than by the generated composition and cell, so the measurement would not be
sensitive to the quantity under test. Two arms differing only in their steering would be
compared through a noise channel that swamps the difference.

**Match to a known structure prototype.** This imports plausibility from the match rather
than measuring the sample, and would score a generated composition by how closely it
resembles something already in the database — which is precisely the coverage question the
A4 comparison exists to separate from accuracy.

## What is measurable, and the reference that changes the reporting

`scripts/crystal/plausibility_screen.py` runs three screens computable from composition and
cell alone: SMACT charge neutrality, electronegativity ordering among charge-balanced
assignments, and packing plausibility as volume per atom against a window taken from the
data's own 1st and 99th percentiles, [8.9, 46.3] A^3/atom.

The load-bearing result is the reference. Scoring 800 real MP-20 structures through the
identical path:

| Screen | Real MP-20 structures |
|---|---|
| Charge balanced | 0.7150 |
| Electronegativity ordered | 0.6400 |
| Packing plausible | 0.9775 |

Real materials fail the charge screen 28.5% of the time, because oxidation-state
enumeration covers only common states. Any generated arm reported against an assumed
ceiling of 1.0 would look bad for a reason having nothing to do with the model, and an arm
scoring 0.70 would be indistinguishable from the data — which is the outcome that matters
and is invisible without the reference in hand. Every screened rate is therefore reported
alongside this measured reference, never as a bare fraction.

## Consequence for the paper

The plausibility claim available is composition-level and cell-level, measured against a
data reference: whether steering toward a density target damages chemical plausibility
relative to what the data itself achieves. That is a real question and the screens answer
it.

The claim not available is thermodynamic stability of generated structures. Obtaining it
requires either extending the serialization to include atomic positions — which changes the
method and the constraint form, not just the evaluation — or a structure-prediction step
whose output would be the object evaluated rather than ours. Both are larger than an
evaluation addition, and the scope section states plainly that thermodynamic stability is
not assessed.

## Verification

Sixteen tests across `tests/test_energy_screen.py` (9) and
`tests/test_plausibility_screen.py` (7). The energy tests assert the artifact declares
itself not-DFT, that agreement is reported on a held-out split, that the reference fit is
not rank-deficient, that MAE is small relative to the label spread rather than in absolute
terms alone, and that a missing element reference returns nothing rather than silently
scoring as zero. Two were added with the GPAW correction above: the disclaimer must not
claim DFT is unavailable and must name GPAW, and it must name the real blocker for
generated samples. The plausibility tests assert the data reference is measured and strictly
below 1.0, that the energy limitation is recorded in the artifact itself, that a known
balanced composition passes and an all-cation composition fails, and that an impossibly
dense cell fails regardless of its chemistry.
