# WS-2 — Parse validity: grammar masking vs unconstrained decoding

Standing note, rebuilt on change. Feeds CP4. All numbers are read from
`results/WS2_parse_validity.json`, emitted by `scripts/crystal/parse_validity.py`.

## Summary

The structural half of the parse-validity claim, established without a trained model.

**Grammar masking removes syntactic invalidity completely: 500 of 500 masked samples parse,
against 0 of 500 unmasked.** Judged by pymatgen's `CifParser`, which shares no code with the
emitter. This is a property of the mask rather than of any model, which is why it is measurable
now and why it holds by construction rather than statistically.

The same measurement quantifies the scope boundary. Of the masked samples that parse, 47.0%
yield a coherent structure; the rest are rejected for physical incoherence — near-degenerate
cell angles that parse perfectly and then fail to define a lattice. So the honest statement is
two numbers, not one: masking closes the syntactic failure mode entirely and leaves the semantic
one entirely open.

## Results

| Arm | Syntactic parse rate | Structure rate | Structure given syntactic | Distinct formulas |
|---|---|---|---|---|
| masked | 500/500 = 1.000 | 0.470 | 0.470 | 218 |
| unmasked | 0/500 = 0.000 | 0.000 | — | 0 |

Unmasked failures are all `NoDataBlocks`: the sampler emits characters with no parseable CIF
structure at all. Masked failures are all semantic, arising after a successful parse.

The unmasked arm is deliberately not a straw man. Both arms draw from the same character
proposal distribution, weighted by character frequencies measured in grammar-valid CIFs, so the
unconstrained sampler has the best per-character information available. It still produces
nothing parseable, because validity is a global property of the token sequence that
per-character likelihood does not enforce. That is the argument for a mask rather than a better
character model.

## The mask leaves real freedom

"Valid by construction" would be worthless if it meant one possible output. Measured over 80,228
positions:

| Quantity | Value |
|---|---|
| Vocabulary size | 63 |
| Mean allowed characters per step | 4.14 |
| Median allowed characters per step | 1 |
| Maximum allowed characters per step | 37 |
| Fraction of steps fully forced | 0.706 |

Roughly 71% of positions are forced — those are the literal tag names and fixed separators the
format requires, where there is genuinely one legal character. The remaining 29% carry the
content: formula, six cell parameters, element symbols, and coordinates, with up to 37 choices
at a step. The 500 masked samples produced 218 distinct reduced formulas and site counts across
the full permitted range, so the constrained decoder explores rather than replaying one file.

## Defect found, and it was the measurement not the grammar

The first run reported the masked arm at 47.0%, which would have falsified the claim outright.
Investigating rather than adjusting the claim: every failing sample parsed as a CIF, and
substituting right angles for the near-degenerate ones converted failure to success without
touching the syntax. The grammar was sound; the measurement was conflating two questions.

Judgement is now split. `parses` asks whether the text parses as a CIF, which is what masking
guarantees. `builds_structure` asks whether a coherent structure comes out, which masking does
not address. Reporting only the second charged the grammar for semantic failures it never
claimed to prevent — the precise error the scope caveat was written to avoid, committed in the
code that was meant to demonstrate the scope.

A separate defect was found earlier by the same oracle: pymatgen raises `KeyError` without an
`_atom_site_label` column. Reading the CIF specification had not surfaced it; running the
emitter's own output through an independent parser did. That is the argument for a third-party
soundness oracle over a self-consistency check.

## What this does and does not support

Supported: a grammar-constrained decoder cannot emit a syntactically invalid CIF, so the
syntactic failure mode is eliminated rather than reduced. Reported as an A1-versus-A2
comparison this is a large positive delta requiring no Regime E, no steering sweep, and no DFT.

Not supported: any claim that masking improves structural quality. 53% of syntactically valid
masked samples fail to yield a structure here. Nor does this predict a specific published
model's parse-failure rate, whose losses that source attributes partly to prompt handling. The
comparison isolates the grammar's contribution by holding the character proposal fixed across
arms.

## Verification

Eighteen tests in `tests/test_cif_grammar.py`: soundness at twelve independent seeds, each
asserted individually so a single failure is a defect rather than a shortfall; non-triviality
via branching factor and sample diversity; that the mask is strictly narrower than the
vocabulary at every position of a complete run, so a grammar that stops constraining partway
cannot pass; termination in the `done` phase; the `_atom_site_label` regression; that the
syntactic and semantic rates are measured separately and do not collapse onto each other; and
that substituting right angles converts a semantic failure into a success, pinning the specific
mechanism behind the gap.
