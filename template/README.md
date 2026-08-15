# Kurban Intelligence Lab — arXiv preprint template

A branded LaTeX template for KIL preprints. It targets **pdfLaTeX + BibTeX**, because
that is what arXiv's AutoTeX actually runs — no XeLaTeX, no LuaLaTeX, no Biber. The
logo is drawn in TikZ, so the title page needs no image files at all.

```bash
make          # main.pdf
make arxiv    # arxiv-submission.tar.gz — upload this
make blind    # main-blind.pdf — double-blind version
make clean
```

---

## The part that matters: affiliations

Different papers have different co-authors. You should never have to renumber a
superscript by hand, and you should never ship a paper listing an institution that no
longer has an author at it.

So institutions are declared **once**, in `affiliations.tex`, and referenced **by key**:

```latex
\Author{Can Polat}{hbku}
\Author{Alex Rivera}{tamu}
\Author[corresponding=hkurban@hbku.edu.qa]{Hasan Kurban}{hbku, tamu}
```

Numbers are assigned **in order of first use**. Institutions nobody cites simply do not
appear. Delete a co-author and everything below them renumbers itself.

The same unedited `affiliations.tex` gives you:

| Paper | You write | You get |
|---|---|---|
| KIL alone | `{hbku}` | ¹ HBKU |
| KIL + Texas A&M | `{hbku}`, `{tamu}` | ¹ HBKU, ² TAMU |
| KIL + Indiana | `{hbku}`, `{iub}` | ¹ HBKU, ² Indiana |

Adding a new collaborator is one line in `affiliations.tex`, forever:

```latex
\DeclareAffiliation{qcri}{Qatar Computing Research Institute, Doha, Qatar}
```

**`\Author` options:** `equal` · `corresponding=email` · `orcid=0000-...` · `footnote=...`
Equal-contribution asterisks and the correspondence line appear only if some author
actually uses them.

> Check the address strings in `affiliations.tex` once against how HBKU, TAMU and IU
> want to be cited. They're my best reconstruction, not gospel.

---

## Options

```latex
\usepackage[font=libertinus, citestyle=numeric]{kurbanlab}
```

| Option | Values | Default |
|---|---|---|
| `font` | `libertinus`, `times`, `charter` | `libertinus` |
| `citestyle` | `numeric`, `authoryear` | `numeric` |
| `paper` | `letterpaper`, `a4paper` | `letterpaper` |
| `status` | any text; `status={}` hides the badge | `Preprint` |
| `blind` | strips authors, affiliations, logo, acks, metadata | off |
| `review` | line numbers, looser leading, visible `\todo` | off |
| `nologo` | title page without the lockup | off |
| `nomath` | skip the math font package | off |

Every option and combination is compile-tested.

### Blind mode is not magic

It removes authors, affiliations, the logo, acknowledgments, funding, contributions,
the running head, and the PDF metadata. It **cannot** remove:

- repo and dataset URLs (including in your availability statement),
- self-citations phrased as "in our previous work",
- named internal clusters or grant numbers.

The package prints a warning saying exactly this on every blind build. `main.tex` shows
the pattern for URLs:

```latex
\ifbool{KIL@blind}{\url{https://anonymous.4open.science/r/XXXXXX}}
                  {\url{https://github.com/KurbanIntelligenceLab/...}}
```

Read the blind PDF once as a reviewer before you upload it.

---

## arXiv checklist

`make arxiv` handles all of this, but so you know *why*:

- [x] **`\pdfoutput=1` in the first five lines** of `main.tex` — forces pdfLaTeX. Don't move it.
- [x] **Ship `main.bbl`, not `references.bib`.** arXiv does not run BibTeX. If you forget the `.bbl`, your citations render as `[?]`.
- [x] **No `\write18`/shell-escape**, no `minted`, no `-shell-escape` anything.
- [x] **Figures as PDF** (vector). `figures/make-figure.py` already emits PDF.
- [x] **Strip `.aux`, `.log`, `.out`, `.synctex.gz`.**
- [x] The logo needs no files — it's TikZ. `logo/kil-lockup.pdf` is shipped as a fast path and falls back to redrawing itself if absent.

After you get the identifier, add `\ArxivID{2607.xxxxx}` and it appears in the footer.
The lab name in the footer links to `\LabURL` (default: the GitHub org).

---

## Files

```
main.tex           your paper
affiliations.tex   institution registry — copy into every paper
kurbanlab.sty      the style package (brand, layout, affiliation engine)
references.bib     bibliography
Makefile           build / arxiv / blind / clean
logo/              TikZ vector master + build script → 8 variants, PDF + PNG
figures/           make-figure.py — brand-matched matplotlib, replace with real figures
```

**Logo:** `cd logo && sh build-logo.sh` regenerates every variant (lockup, stacked,
mark, mono, white, glow) from one source. In text, `\kilmark[12pt]` and
`\killockup[34pt]` draw the mark inline at any size.

**Brand colours** (usable anywhere: `\textcolor{kilviolet}{...}`):
`kilink` `#0D1526` · `kilviolet` `#6C5CF6` · `killilac` `#B49BFB` ·
`kilslate` `#5C6C87` · `kilmist` `#F4F2FE` · `killine` `#E3DFF8` · `killink` `#4C3ECF`

---

## Writing helpers

`\best{}` `\runnerup{}` `\gain{12}` for results tables · `kilkey` / `kilbox` / `kilrepro`
callout boxes · `\todo{}` (visible only in `review`) · `\eg` `\ie` `\etal` ·
`\R` `\E` `\vect{}` `\mat{}` · `\cref{}` for all cross-references.

**Theorem environments.** All share one counter, so the paper reads
`Theorem 1 ... Lemma 2 ... Definition 3` in one sequence, and `\cref` resolves each
by its own name (`\cref{as:x}` prints `Assumption 2`, not `Theorem 2`).

The head speaks the same visual language as the rest of the paper: the number is
violet, like every section number, and an optional title is introduced by the lilac
`|` bar, the same separator that labels every figure, table, algorithm, and listing.
Name in Montserrat SemiBold navy, title in Montserrat Medium slate, so the head reads
in three levels instead of one flat line.

- *Formal, to-be-proved results* set their body in italic:
  `theorem` `lemma` `proposition` `corollary` `conjecture` `claim`.
- *Definitional / structural* statements stay upright:
  `definition` `assumption` `condition` `property` `problem` `question`
  `example` `fact` `observation` `notation`.
- *Asides* get a quieter, all-slate head with a hairline bar: `remark`.
- Every one takes an optional title: `\begin{theorem}[Expressivity]...`.

**Proofs.** `proof` closes with a violet QED square. Name it with the optional
argument, `\begin{proof}[Proof of \cref{thm:x}]`. Use `proofsketch` for the argument
you give in the body and finish in the appendix (no QED square, on purpose).

**Restating a result.** Every environment has a starred, unnumbered twin
(`theorem*`, `lemma*`, `definition*`, `remark*`, ...) plus a bare `note`. Use these
to repeat a named result — e.g. before its full proof in the appendix — without
minting a second number:
`\begin{theorem*}[Restatement of \cref{thm:x}]...\end{theorem*}`.

Back matter: `acknowledgments`, `contributions`, `funding`, `availability`, `conflicts` —
the first three vanish in blind mode automatically.

## For students: your first paper, in order

1. Copy this folder. Rename nothing.
2. Put your title, `\Subtitle`, `\Keywords`, and `\RunningTitle` in `main.tex`.
3. Write the author block with `\Author{Name}{keys}` — keys come from
   `affiliations.tex`, and you never type an affiliation number.
4. Write. When you need a pattern — subfigure, decimal-aligned table, merged cells,
   table footnote — **Appendix B of the compiled PDF is a pattern library**; copy it.
5. Figures: edit `figures/make-figure.py` or export your own **PDFs** (never PNG
   screenshots of plots) into `figures/`.
6. Every citation goes in `references.bib`. Get entries from the arXiv page's
   "Export BibTeX" or from doi.org — Google Scholar entries are often wrong.
7. `make` to build. **Fix every warning**; this template compiles with zero, so any
   warning you see is yours.
8. Circulating a draft? Build with `review` — line numbers (equations included),
   visible `\todo{}` notes, and a compile timestamp in the footer so nobody comments
   on a stale printout.
9. Submitting double-blind? `make blind`, then **read the PDF as a reviewer** — the
   build log tells you exactly what blind mode cannot anonymize for you.
10. Submitting to arXiv? `make arxiv` and upload the tarball. Done.

### Already there so you don't have to set it up

**Units & numbers** (`siunitx`): `\qty{3.2}{\electronvolt}`,
`\qty{28}{\milli\electronvolt\per\angstrom}`, ranges `\qtyrange{2}{8}{\kelvin}`,
and `S[table-format=1.3]` columns that align tables on the decimal point.

**Chemistry** (`mhchem`): `\ce{TiO2}`, `\ce{Zn(NO3)2.6H2O}` — correct subscripts,
no math-mode juggling.

**Math you'd otherwise fake**: `\argmax`, `\argmin`, `\Tr`, `\diag`, `\rank`,
`\Var`, `\Cov`, `\KL`, `\sign`, plus auto-sizing `\norm{x}`, `\abs{x}`,
`\inner{u}{v}`, `\floor{x}`, `\ceil{x}` (star them for `\left...\right` sizing).

**Paste-proofing**: text copied from PDFs and Word arrives full of Unicode that
pdfLaTeX normally rejects — μ, λ, ±, ≤, →, 10⁻³, H₂O, 25 °C. The template maps ~75
of these to what you meant, so pasting a sentence doesn't stop your build.

**Code listings, lab-grade**: Python is the default language (with `self`, f-strings,
`match`/`case`, `async` all recognized); keywords set in brand violet, strings in teal,
numeric literals in amber — strings deliberately do *not* share the hyperlink color.
Long lines wrap with a visible ↪ marker, pasted unicode inside code (µ, λ, ≤, →)
survives, quotes stay straight for copy-paste (`upquote`), and `(*@ ... @*)` escapes
to LaTeX mid-listing for callouts. Other languages: `\begin{lstlisting}[language=bash]`.

**Floats behaving**: `\FloatBarrier` pins figures before a section boundary;
`\tabnote{...}` puts a quiet footnote under a table; subfigures are set up and
demonstrated in Appendix B.

**Several corresponding authors**: give `corresponding=email` to as many authors as
apply; every email is listed and mailto-linked.

**Names never split**: an author's first and last name (and their superscripts) can
never be separated by a line break, no matter how long the author list gets.

### Don'ts (each one has cost us a desk-reject or an hour)

- Don't hand-type affiliation superscripts. Ever. That's the engine's job.
- Don't `\vspace{-8pt}` your way around float placement — use `[t]`, `[h]`, and
  `\FloatBarrier`.
- Don't screenshot plots. Vector PDFs or nothing.
- Don't ignore a warning because the paper "looks fine". Zero is the baseline.
- Don't edit `kurbanlab.sty` for one paper. If something's missing, it's missing for
  every paper — fix it once, at the lab level.

## Overleaf

The project is Overleaf-ready as-is: upload the folder as a **zip**
(`kurban-latex-template.zip`), set the compiler to **pdfLaTeX** (Menu → Compiler) if
it isn't already, and Overleaf's latexmk runs the pdflatex/bibtex cycle for you — the
included `.latexmkrc` does the same for local `latexmk -pdf main`. All packages used
are in Overleaf's full TeX Live.

Honesty note: this was verified against a full TeX Live 2023, the same vintage arXiv
runs, but not on Overleaf's servers themselves. If anything misbehaves there, it will
be within the first compile — check the compiler setting first.

## Troubleshooting

**`Command \Bbbk already defined`**, **`File binhex.tex not found`**, or
**`Font TS1/ntxtlf ... not loadable`** — a lean TeX install is missing pieces `newtx`
needs (`texlive-plain-generic`, `tex-gyre`). The template works around the first and
falls back to `mathptmx` where it can, but the real fix is:
`apt install texlive-plain-generic texlive-fonts-extra tex-gyre`. Overleaf and full
TeX Live installs have all of this already.

**`BibTeX: I couldn't open database file references.bib`** — `references.bib` isn't
next to `main.tex`. Two common ways to get here: files were downloaded individually
(folder contents don't travel with single files — use the zip), or you opened an
older arXiv tarball, which shipped only the `.bbl`. Fixes: put `references.bib`
beside `main.tex`, or just build with the template's `.latexmkrc`, which skips
BibTeX when no `.bib` exists and trusts the shipped `.bbl` (citations still
resolve). Current tarballs include both the `.bib` and the `.latexmkrc`.

**Montserrat missing** — falls back to Lato, then Helvetica. Install
`texlive-fonts-extra` for the real thing (it's the wordmark face).
