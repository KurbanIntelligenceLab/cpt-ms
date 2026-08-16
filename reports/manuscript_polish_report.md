# Manuscript polish: five passes, in dependency order

`template/main.tex` plus `template/sections/*.tex`. Every pass ran sequentially, never in
parallel, because each rewrites the prose the next one measures.

## Before and after

| measure | before | after |
|---|---|---|
| main-text pages | 10 | 12 (limit 12) |
| main-text words | 3,838 | 4,589 |
| SI words | 1,229 | 1,396 |
| abstract words | 274 | 265 |
| main-text sections | 9 | 6 |
| labels referenced more than once | 9 | 0 |
| em dashes in prose | 0 | 0 |
| caption/body duplication (worst float) | 0.0% | 0.0% |
| table widths (% of text block) | 67.9 / 68.7 / 71.2 | 100.0 / 100.0 / 100.0 |

## Order, and why it is not the order requested

The requested order was abstract, citations, intro/related, architecture, redundancy. Three
dependencies force a different one.

Architecture runs first because it moves whole sections, and reordering changes which
sentences echo the abstract and which mention of a float is first. The abstract runs second,
before the introduction, because `intro-related-work-optimize` forbids sentences echoed from
the abstract: polishing the abstract afterwards silently reintroduces the echo the intro pass
just removed. Citations run after both, because restructuring moves citations across sections
and changes first-named position. The redundancy sweep runs last, since every earlier pass
rewrites the prose it measures against.

Citations and redundancy also cannot run concurrently: both rewrite cross-references in one
file, and two writers lose one set of edits with no error on either side.

## What each pass changed

**Section architecture.** Four separate empirical sections became one "Experiments and
results" with a shared-setup subsection first and one subsection per experiment. The
conclusion was rewritten in fresh words, carrying limitations then future work. Internal run
codes were replaced with descriptive names throughout prose and table rows.

**Abstract.** 274 words to 265. Two result values kept, both re-read from their
artifacts rather than from the old abstract's rounding. The p-value, the pre-registered
margin, the training-cost multiple and the unglossed dataset code were removed; every dropped
statistic survives in the results section or SI.

**Introduction and related work.** Related work stayed a separate section, as instructed. The
introduction went from citing nothing at all -- it deferred the entire prior-work gap to the
next section -- to citing prior work in five of seven paragraphs, and from two near-duplicate
sentences against the abstract to zero. Related work was reordered to mirror the introduction
one-to-one in the same order, with two paragraphs expanded so each adds detail.

**Citations.** Six keys cited twice, four floats or equations referenced twice, one figure
referenced never, and three sections over-referenced: all resolved. The never-referenced
figure turned out to be a subfigure parent whose panels are each cited once, so the unused
label was deleted rather than a pointer invented.

**Redundancy sweep.** Caption/body duplication was already zero on all seven floats, reported
as clean rather than converted into cuts. One real finding: a supplementary table was
reachable but unexplained where it lives, so a paragraph was written for it.

## Judgement calls, stated because they are what needs review

The density proxy flagged four paragraphs and testing it cleared all four. The closest call
was the retargeting paragraph, where six of ten numeric tokens appear somewhere in a float --
the signature of a recitation. None of its substantive numbers are in the table: the matches
are a count of models, an exponent, and a value that appears only there. It stays.

Measuring table width needed the same scepticism. The widest horizontal rule on every page is
the template's page-header rule, which appears on pages with no table at all. Taking it as the
reference would have scored every table against page furniture and reported all three as
already full width.

Two rules genuinely conflicted. Trimming a supplementary section to one reference made it
unreachable from the main text; the pointer was restored and the other mention rewritten as
prose. And the citation-coverage rule wanted a citation on a sentence where every candidate
key was already cited exactly once, so the class is named and the survey section pointed at
instead, which is what that skill's own guidance prescribes.

## Structure

`main.tex` is now a skeleton that inputs one file per section plus a single
`sections/supplementary.tex` after `\appendix`. The split produces byte-identical output.

Three Makefile bugs surfaced from it, each of which yields a tarball rather than an error: the
packaging target copied neither `sections/` nor the figures now referenced from those files,
and the clean target's globs reached only the project root.

## Bibliography

arXiv has processed `.bib` into `.bbl` itself since November 2025, so the `.bib` is the file
the submission needs. Verified by deleting `main.bbl` from the extracted tarball and building
from `.bib` alone: all 22 citations resolved. The `.bbl` still ships, because it is small and
covers a build with no BibTeX pass.

## Verification

Clean compile: zero errors, zero undefined references, zero citation warnings, zero overfull
boxes, 22 bibliography entries resolved. Every supplementary section reachable exactly once
and printed in callout order. 288 tests pass.
