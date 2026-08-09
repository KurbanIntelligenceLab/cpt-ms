# WS-2 — Per-step determinability of both Regime-E families, both backbones

Standing note, rebuilt on change. Feeds CP0 and CP4. All numbers are read from
`results/WS2_family_determinability.json`, emitted by
`scripts/crystal/determinability.py`.

## Resolution: the reference ordering dissolves the B1 problem

The blocking question — whether charge balance can carry a steering sweep on the CIF
character backbone — is answered, and the answer is yes, at no cost in convention.

The reference model's own serialization places the complete cell composition in the
`data_` tag on the first line. CrystaLLM's prompt construction builds
`f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"`, where `comp_str`
is the full formula with per-element counts, elements ordered by electronegativity
(`bin/make_prompt_file.py::get_prompt`). Composition is therefore complete before any
lattice or site token is emitted.

Measured under that ordering, family (i) rises from a mean usable fraction of 0.002 to
**0.949**, determined at token 13-21 of 245-409. Family (ii) remains healthy at 0.490.
Both constraints are usable over most of the sequence on both backbones.

This is the reverse of a deviation. Matching the reference ordering is matching the format
the published model was trained on; the earlier 0.002 figure was measured on a naive CIF
layout that no reference implementation uses. No B1-versus-B2 scoping decision is needed,
and no departure from convention is required.

| Backbone | Family (i) charge | Family (ii) density |
|---|---|---|
| B1 naive CIF layout | 0.002 | 0.670 |
| B1ref reference ordering | 0.949 | 0.490 |
| B2 typed fields | 0.785 | 0.480 |

## Summary

The plan's family (ii) hedge is instantiable per step on both backbones, but the
measurement inverts the concern that motivated it. The worry was that density would be
undefined per step on B1 because it depends on lattice parameters at scattered positions.
In fact CIF emits the cell block in its header, so on B1 the volume is determined at token
161 of 323–773, giving family (ii) a mean usable fraction of 0.670. The family that is
near-terminal on B1 is family (i), charge balance, at a mean usable fraction of **0.002** —
because composition on a CIF character stream is spelled out in the atom-site loop at the
very end. The hedge holds; the thing it was hedging against is the problem on the other
family.

## What "determined" means

The projection layer needs a constraint state that is a pure function of the prefix, and
family (ii) needs more than that. Density is

    rho = (1/V) * sum_e n_e * m_e

a ratio of two incrementally emitted quantities, and the expectation of a ratio is not the
ratio of expectations, so rho is not a linear functional of the next-token distribution
while V is still open. Once V is prefix-determined, 1/V is a known constant and rho becomes
exactly linear in the remaining element counts. The determinability position is therefore
the position at which the cell volume closes, and the usable fraction is the share of
sequence positions at or after it.

The same logic applied to family (i) gives the position at which the composition block
closes.

## Measurements

Six representative structures, 2–5 elements and 4–20 sites.

| Backbone | Family | Mean usable fraction | Range | Determined at |
|---|---|---|---|---|
| B2 (fields) | (i) charge | 0.785 | 0.750–0.828 | token 4–7 of 16–35 |
| B2 (fields) | (ii) density | 0.480 | 0.312–0.600 | token 11–14 of 16–35 |
| B1 (CIF chars) | (i) charge | 0.002 | 0.001–0.003 | token 322–772 of 323–773 |
| B1 (CIF chars) | (ii) density | 0.670 | 0.502–0.794 | token 161 of 323–773 |

The asymmetry is structural, not a formatting artifact. A CIF header carries the cell
parameters and the atom-site loop carries the composition, so a character stream learns
the volume early and the composition last. A structured field stream emits composition
first by construction, so it learns the composition early and the volume second.

## Consequences

**Family (ii) is a real hedge, not a nominal one.** It is instantiable per step on both
backbones and covers the majority of positions on B1, which was the case in doubt. The
plan's Risk 4 mitigation stands.

**Charge balance on B1 is sound but nearly useless as a training signal.** At a mean usable
fraction of 0.002, family (i) on a CIF character stream contributes gradient over roughly
the last two tokens of a 300–800 token sequence. The constraint is not wrong there; it is
simply almost never active. Three options, and this is a design decision rather than a
measurement:

1. Reorder the B1 serialization so the composition precedes the cell block. This makes B1
   behave like B2 and costs a deviation from standard CIF field order, which matters if
   the backbone is meant to be comparable to published CIF language models.
2. Scope family (i) to B2 and run B1 on family (ii) only. Clean, and it costs the
   charge-balance claim on one of two backbones.
3. Accept the near-terminal signal and report it. Honest, but a projection active over
   0.2% of positions will not produce a measurable internalization curve, which is what
   CP0 reads.

**Superseded.** The three options below were framed before the reference ordering was
measured. Option 1 turned out to require no deviation at all — the reference implementation
already places composition first — so the choice is moot and no scoping is needed. The
options are retained because the pre-registration derived from them, and the record of why
a threshold was set should stay visible.

My recommendation at the time was 1 if the B1 comparison to published CIF models was not
load-bearing, otherwise 2. What should not happen is discovering this at CP4 through a null result on
one backbone.

**The determinability position belongs in the pre-registration.** A constraint's usable
fraction bounds how much gradient it can contribute, so a CP0 active-multiplier threshold
must be stated per backbone and per family. A converged multiplier fraction of near zero on
B1 family (i) would be the expected outcome of the serialization, not evidence about the
method.

## A note on character-stream numerics

A character backbone determines a numeric field only when its literal terminates, not when
its last digit is emitted: the prefix `_cell_length_a 5.12` does not determine `a`, because
the next character may extend it to `5.123`. The value commits on the delimiter. This
one-token lag is not a rounding detail — it is the reason a character backbone can never
activate a numeric constraint at the same position a field backbone can, and the parser
implements the delimiter rule rather than the last-digit rule.

## Verification

Fifteen tests in `tests/test_parser_determinability.py`, each checking a property by a route
independent of the implementation: the cubic cell against its exact volume of 125 cubic
angstroms and a triclinic cell against the closed-form expression; prefix-purity by
re-running each truncated prefix from scratch and comparing states; each determinability
position against a brute-force scan confirming no earlier prefix suffices; and the
Regime-S mask against monotone shrinkage. A closing marker on an incomplete lattice block
is asserted not to determine a volume, so a truncated stream cannot report a cell it never
emitted. All pass, within a suite of 136.
