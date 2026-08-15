# Redundancy sweep of the arXiv manuscript

`template/main.tex`, measured with the float/caption sweep and its companion audits.
Audit table: `reports/redundancy_sweep_audit.csv`.

## What the caption sweep found: almost nothing

Caption-body duplication was already near zero, because the manuscript was drafted under
the rule that no number appears in both a caption and the body. The worst float measured
1.9% shared five-word runs, and no float shared a single number between its caption and
the paragraphs citing it. Reporting a clean result is the honest outcome here rather than
manufacturing cuts.

| float | caption words | duplication | shared numbers | caption-only |
|---|---|---|---|---|
| `fig:charge` | 59 | 0.0% | none | none |
| `fig:frontier` | 36 | 0.0% | none | none |
| `fig:pareto` | 44 | 0.0% | none | none |
| `tab:concentration` | 56 | 1.9% | none | none |
| `si:tab:perseed` | 40 | 0.0% | none | 0, 1.0, 10.405 |
| `si:tab:grid` | 44 | 0.0% | none | none |

The three caption-only values in the per-seed table are the published ceiling (10.405),
the validity constant (1.0) and a row label. All three are reading keys a reader needs while
looking at the table, which the sweep's own guidance says to keep, and both derived values
were verified against `results/WS4_retargeting_seed_replication.json` rather than assumed.

## What the density proxy found, and where it was a false positive

Six main-text paragraphs exceeded five real numbers per hundred words. Testing the proxy
rather than trusting it — checking how many of each paragraph's numeric tokens actually
appear in a float — cleared five of the six:

| paragraph | density | tokens in a float | verdict |
|---|---|---|---|
| related work, crystal generation | 8.5/100w | 0 of 7 | publication years, not float cells |
| retargeting accuracy | 8.4/100w | 7 of 13 | REAL: recited the table beside it |
| comparator reaches its target | 7.4/100w | 0 of 4 | the arm's own convergence figures |
| synthetic task setting | 7.2/100w | 0 of 5 | task dimensions, reproducibility infrastructure |
| best-of-n | 7.1/100w | 1 of 10 | sample counts, not cells |
| serialization | 5.4/100w | 0 of 5 | position counts stated once |

## The one real finding, and what it exposed

The retargeting paragraph recited the mean and standard deviation printed in the table
beside it. Fixing it exposed a worse defect the sweep was not looking for: the table was
captioned "per-seed retargeting summary" and contained no per-seed values at all — only the
aggregate the body was reciting.

The table now holds the three seeds, and the body states the comparison instead of the
cells. Every migrated value was verified against
`results/WS4_retargeting_seed_replication.json` before the body copy was deleted, in that
order: twelve values checked, and the mean of the three per-seed errors recomputed to
confirm it equals the artifact's stated mean (0.6497).

The migration also made a real finding visible that the aggregate had hidden. Seed 0's
backbone sits at 22.769 against 8.381 and 8.692 for the other two, so its lower error is
an excursion rather than the reading to take. The caption now says so.

## Citations: six references cited twice

Every bibliographic key was cited exactly once after the sweep, from six that were cited
twice. In each case the duplicate was an anonymous citation in the introduction or
discussion plus a named citation in related work; the named occurrence stayed and the
anonymous one was replaced by a section pointer, which is the direction the rule specifies.

Two entries were cited nowhere while the SI named CHGNet in prose. That is a missing
citation rather than dead weight, and both were wired up.

## Reachability, which this pass broke and then fixed

De-duplicating created exactly the failure the sweep's guidance warns about. Removing
values on the grounds that a table holds them left two SI tables cited nowhere in the main
text, and trimming a solver sentence orphaned an SI section. A third section had never been
called out at all. All four now have main-text pointers at existing call sites, and the SI
sections were renumbered so declaration order matches order of first mention, which the
drafting brief requires.

## Em dashes

All 31 were rewritten, none replaced by a swapped punctuation mark: parentheticals became
commas or colons where the aside was short, and separate sentences where the clause was
long enough to bury the main verb.

## Result

Main text ends on page 10 including figures, tables and references, with the template's
class, font size, margins and spacing untouched. Zero undefined references, zero citation
warnings, zero overfull boxes. Main text 3,827 words, SI 1,229 words.

`tests/test_manuscript_consistency.py` pins eleven of these properties, including the
concentration factor being quoted against the optimum rather than the unconstrained arm,
the per-seed table containing seeds, SI callout order, one citation per reference, and the
absence of em dashes. Verified load-bearing by drifting three of them — the variance factor,
the SI order, and an inserted em dash — each rejected, then reverted.
