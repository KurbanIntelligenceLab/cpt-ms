# Response — B1 blocker resolved, three findings verified with two corrections

## The B1 blocker is gone, and the fix costs nothing

The reference model's own serialization already places the composition first. CrystaLLM's
prompt construction builds `f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M
{sg}\n"`, where `comp_str` is the complete cell formula with per-element counts and elements
ordered by electronegativity (`bin/make_prompt_file.py::get_prompt`, verified at source).
Composition is complete before any lattice or site token.

Re-measured under that ordering, family (i) charge balance rises from a mean usable fraction
of **0.002 to 0.949**, determined at token 13-21 of 245-409. Family (ii) density stays healthy
at 0.490.

| Backbone | Family (i) charge | Family (ii) density |
|---|---|---|
| B1, naive CIF layout | 0.002 | 0.670 |
| B1ref, reference ordering | 0.949 | 0.490 |
| B2, typed fields | 0.785 | 0.480 |

So there is no scoping decision and no deviation from convention: the earlier 0.002 was
measured on a CIF layout no reference implementation uses. Matching the reference ordering
*is* matching the convention. The steering sweep's dial (a) can run on family (i) on both
backbones.

## The parse-failure finding is the right call and I have one caveat

Verified in full. Table 3: CrystaLLM-MP20 generated 16,000, of which 2,074 were valid CIFs,
1,951 relaxed, and 58 dynamically stable — a 3.0% stability rate, last of all models tested.
The mechanism is quoted correctly: file parsing failures during generation, with failure rates
reaching up to 90% in severe cases.

Worth pinning the denominator, because it decides whether the parse loss gets counted twice.
The 3.0% is stable-over-relaxed (58/1,951 = 2.97%), which is how the paper computes the ratio
for every model, so the parse failures are already excluded from it. Measured against total
generations the rate is 58/16,000 = 0.36%. Grammar constraints attack the 16,000-to-2,074
step; the 3.0% figure describes what happens after that step, so the two effects compose
rather than add.

Two points of precision before this becomes a headline result.

**The 87% figure is a validity rate, not a parse-failure rate.** 2,074 of 16,000 valid is
87.0% *invalid*, but the paper attributes the loss to parsing "up to 90% in severe cases" and
adds "likely related to prompt design." So the 87% is an upper bound on what grammar
constraints can recover, not a measured parse-failure rate. Some of that gap may be prompt
handling rather than malformed output. Claim it as "up to 87% of generations discarded before
evaluation, which the source attributes primarily to parse failures," and measure our own
A1-versus-A2 parse rate directly rather than inheriting theirs.

**Regime S eliminates syntactic invalidity, not semantic invalidity.** A grammar-constrained
decoder guarantees a parseable CIF by construction. It does not guarantee the structure is
physically sensible — overlapping sites, absurd lattice parameters, and unphysical
stoichiometries all parse fine. The claim that survives review is that A2 removes the
syntactic failure mode entirely and that whatever remains is measured, not that it lifts the
3.0% stability figure by 87 points. Reviewers will make exactly this objection, and the
honest version is stronger.

With those two framings the finding stands as the cheapest positive result available: it needs
no Regime E, no steering sweep, and no DFT.

## Corrections to the other findings

**The Eg = 0.5 eV discrepancy is real, and your resolution is right.** Main text says 1,523
stable of 6,524; Table 4 says 1,532. Only 1,532 reproduces the stated 23.5% (1,532/6,524 =
23.48%, versus 1,523/6,524 = 23.34%). Cite Table 4 or the percentage.

**CrystalFormer's 34.4% is an average across space-group-controlled generation, not a peak.**
The paper reports cubic systems reaching up to 49.2% while "the average stability across all
controlled generations is still only 34.4%." Describing 34.4% as what CrystalFormer "reaches"
inverts the sense — it is the modest average being contrasted against the better
high-symmetry cases. The novelty numbers are verbatim: 20,000 crystals producing only about
5,000 unique formulas, and significantly reduced novelty relative to unconstrained models
like DiffCSP and InvDesFlow-AL, attributed to symmetry constraints limiting the generative
space.

**The conclusion is not an either/or against post-hoc screening.** The paper says the results
"underscore the importance of incorporating explicit stability constraints or performing
posterior stability screening within the generation pipeline" — constraints *or* screening,
both inside the pipeline. Quoting it as favoring constraints over post-hoc screening
overstates it. The quotable part is that stability handling belongs in the pipeline rather
than after it, which still supports the framing without putting words in their mouth.

**Model sizes are correct and the implication is sharper than stated.** CrystalFormer 4.8M,
InvDesFlow-AL and DiffCSP 12.3M, CrystalFlow 20.9M, CrystaLLM 26.1M, MatterGen 53.7M. Our
10-25M bake-off sits mid-range. A 100-200M scale-up would be roughly double the largest model
benchmarked, and the paper notes CrystaLLM at 26.1M already exceeds most GNN models while
performing worst on stability — so parameter count is not what separates the field here. That
weakens the case for scale-up further than the size comparison alone does.

## Defect found and fixed while implementing this

The reference-ordering fixture used CIF's actual `_cell_angle_alpha` spelling, while the
parser matched only `_cell_length_alpha`. All three angles were silently dropped, the lattice
never closed, and family (ii) read 0.000 across every case. The naive fixture had used the
wrong spelling too, so the parser had been matching its own fixture's error rather than real
CIF — the defect would have surfaced on the first actual CIF file. Both spellings are now
accepted, with three tests covering the angle tags, the reference ordering's determinability,
and the requirement that a composition marker without a parsed composition cannot claim a
determined charge.

One process note: an earlier run of the measurement appeared to succeed but silently emitted
no reference-ordering rows, because I had redirected both stdout and stderr while the script
was in fact raising on an unrecognized backbone name. The suppression was mine, not the
script's — it has no exception handling. Re-run without redirection, it failed loudly and
immediately.

## Accepted without change

Parse-validity rate becomes a primary reported metric for B1, with the A1-versus-A2 comparison
as an early cheap result. The CrystalFormer validity-novelty trade-off is cited as the
motivating tension in the steering framing. Scale-up goes to Sync C alongside Branch D. MDE
gap cited as 3.4-20.2 pp with maximum at theta = 0.10.
