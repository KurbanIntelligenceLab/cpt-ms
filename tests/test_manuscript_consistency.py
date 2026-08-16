"""The manuscript's numbers must match the artifacts they cite.

A figure quoted in a caption, a table cell and a body sentence is three chances to drift,
and the failure mode this suite is built around is prose that no longer matches a rerun.
Each test below recomputes a load-bearing value from its artifact and asserts the
compiled source still carries it. A rounding convention change is a legitimate reason for
one of these to fail; a rerun that moved a number is the reason they exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / "template" / "main.tex"
BIB = ROOT / "template" / "references.bib"
RESULTS = ROOT / "results"

pytestmark = pytest.mark.skipif(not TEX.exists(), reason="manuscript source not present")


def _p(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())["payload"]


def _assembled() -> str:
    """main.tex with every \\input inlined.

    The manuscript is split across sections/*.tex, so main.tex on its own is a preamble
    and a list of includes -- reading it directly makes every prose assertion below pass
    vacuously or fail spuriously depending on which way the check runs. Resolve the
    includes so the tests see the document a reader sees.
    """
    text = TEX.read_text()

    def inline(match: "re.Match[str]") -> str:
        rel = match.group(1).strip()
        for candidate in (rel, rel + ".tex"):
            path = TEX.parent / candidate
            if path.exists():
                return path.read_text()
        raise AssertionError(f"\\input{{{rel}}} does not resolve from {TEX.parent}")

    for _ in range(3):  # inputs may nest
        text, n = re.subn(r"\\input\{([^}]+)\}", inline, text)
        if not n:
            break
    assert r"\input{" not in text, "unresolved \\input remains after inlining"
    return text


def _src() -> str:
    return re.sub(r"\s+", " ", _assembled())


def _main_text() -> str:
    return re.sub(r"\s+", " ", _assembled().split(r"\appendix")[0])


def test_concentration_numbers_match_their_artifacts():
    """The paper's central claim, recomputed from the two artifacts that hold it."""
    a2 = _p("A2_training_summary.json")["arms"]
    a6 = _p("WS7_a6_tilt_comparator.json")
    src = _src()

    # These arrays are [mean, sd] pairs, NOT per-seed lists. Averaging them yields a
    # number that is not any real quantity -- a mistake made once already.
    a3_var = a2["A3_deployed"]["charge_variance"][0]
    a3_viol = a2["A3_deployed"]["violation_rate"][0]
    tilt_var = a6["reference"]["tilt"]["charge_variance"]
    tilt_viol = a6["reference"]["tilt"]["hard_violation_rate"]
    data_var = a6["reference"]["base_S_conditioned"]["charge_variance"]

    for value in (f"{a3_var:.3f}", f"{tilt_var:.3f}", f"{data_var:.3f}"):
        assert value in src, f"charge variance {value} missing from the manuscript"

    # The claim is stated against the OPTIMUM, not against the unconstrained arm.
    assert f"{tilt_var / a3_var:.2f}" in src, (
        "the variance factor must be quoted against the sequence-level optimum; "
        "quoting it against the mask-only arm answers a different question"
    )
    assert f"{tilt_viol / a3_viol:.2f}" in src, "per-sequence violation ratio missing"

    # Anti-concentration: the optimum raises variance above the data it started from.
    assert tilt_var > data_var > a3_var, (
        "the ordering projection < data < optimum is the mechanism; if a rerun "
        "reversed it, every concentration sentence in the paper is wrong"
    )
    assert f"{100 * (tilt_var / data_var - 1):.1f}" in src


def test_equivalence_numbers_come_from_the_preregistration():
    eq = _p("WS4_equivalence.json")["equivalence"]
    src = _src()
    assert f"{eq['tost_p']:.4f}" in src
    assert f"{eq['margin']:.3f}" in src
    prereg = ROOT / "preregistration" / "WS4_equivalence_margin.md"
    assert prereg.exists(), "the margin must live in a pre-registration file"


def test_per_seed_table_carries_per_seed_values():
    """A table captioned 'per-seed' must hold seeds, not just their aggregate."""
    ps = _p("WS4_retargeting_seed_replication.json")["per_seed"]
    src = _src()
    for seed, row in ps.items():
        assert f"{row['sub_ceiling_mae']:.4f}" in src, (
            f"seed {seed}'s sub-ceiling MAE is missing; the table is labelled per-seed"
        )
        assert f"{row['backbone_mean']:.3f}" in src, f"seed {seed} backbone missing"


def test_direction_asymmetry_counts_match_the_grid():
    """The failing cells must be counted from the artifact, not described from memory."""
    cols = _p("WS5_two_constraint_grid.json")["summary"]["fixed_valence_columns"]
    rows = [r for c in cols for r in c["rows"]]
    up = [r for r in rows if r["asked_direction"] == "up"]
    down = [r for r in rows if r["asked_direction"] == "down"]
    up_ok = sum(r["moved_correct_direction"] for r in up)
    down_ok = sum(r["moved_correct_direction"] for r in down)
    src = _src()
    assert f"${down_ok}$ of $" in src or f"{down_ok} of {len(down)}" in src
    assert f"${up_ok}$ of" in src or f"{up_ok} of {len(up)}" in src, (
        f"the manuscript must state {up_ok} of {len(up)} for the 'asked up' cells"
    )


def test_every_si_section_is_called_out_from_the_main_text():
    """An SI section a reader cannot reach from the main text is unreachable, not optional."""
    tex = _assembled()
    cite = r"\\(?:ref|cref|Cref|autoref)\{([^}]*)\}"
    main = tex.split(r"\appendix")[0]
    cited = {k.strip() for m in re.finditer(cite, main) for k in m.group(1).split(",")}
    labels = [
        lab for lab in re.findall(r"\\label\{(si:[^}]+)\}", tex)
        if not lab.startswith("si:tab")
    ]
    missing = [lab for lab in labels if lab not in cited]
    assert not missing, f"SI sections with no main-text callout: {missing}"


def test_si_callouts_are_in_order_of_first_appearance():
    tex = _assembled()
    cite = r"\\(?:ref|cref|Cref|autoref)\{([^}]*)\}"
    main = tex.split(r"\appendix")[0]
    first = {}
    for m in re.finditer(cite, main):
        for k in m.group(1).split(","):
            first.setdefault(k.strip(), m.start())
    declared = [
        lab for lab in re.findall(r"\\label\{(si:[^}]+)\}", tex)
        if not lab.startswith("si:tab")
    ]
    assert declared == sorted(declared, key=lambda l: first.get(l, 10**9)), (
        "SI sections must be numbered in the order the main text first calls them out"
    )


def test_no_bibliographic_key_is_cited_twice():
    """One citation per reference, at its first named appearance."""
    src = _assembled()
    keys = [k.strip() for m in re.finditer(r"\\citep?\{([^}]*)\}", src)
            for k in m.group(1).split(",")]
    dupes = {k for k in keys if keys.count(k) > 1}
    assert not dupes, f"keys cited more than once: {sorted(dupes)}"


def test_every_bib_entry_is_cited_and_every_citation_resolves():
    src = _assembled()
    entries = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text()))
    cited = {k.strip() for m in re.finditer(r"\\citep?\{([^}]*)\}", src)
             for k in m.group(1).split(",")}
    assert not (cited - entries), f"cited but not in the bibliography: {sorted(cited - entries)}"
    assert not (entries - cited), f"in the bibliography but never cited: {sorted(entries - cited)}"


def _no_comments(text: str) -> str:
    """Blank LaTeX comment bodies, preserving line structure.

    Comments matter here because affiliations.tex rules its sections off with `% ---
    home ---...`, and those dashes are invisible to a reader. A check that counts them
    reports 49 em dashes on a manuscript that has none in prose.
    """
    return re.sub(r"(?<!\\)%[^\n]*", "%", text)


def test_no_em_dashes():
    """House style: an em dash is a parenthetical the author did not resolve."""
    prose = _no_comments(_assembled())
    found = [
        " ".join(prose[max(0, m.start() - 60) : m.start() + 40].split())
        for m in re.finditer(r"---", prose)
    ]
    assert not found, f"em dashes must be rewritten, not inserted: {found[:3]}"


def test_no_forward_reference_to_an_equation():
    src = _assembled()
    for lab in re.findall(r"\\label\{(eq:[^}]+)\}", src):
        decl = src.find(r"\label{" + lab + "}")
        refs = [m.start() for m in re.finditer(re.escape(lab), src) if m.start() != decl + 7]
        assert all(r > decl for r in refs), f"{lab} is referenced before it is stated"


def test_figures_referenced_by_the_manuscript_exist_as_vector_pdfs():
    src = _assembled()
    for rel in re.findall(r"\\kilgraphics(?:\[[^\]]*\])?\{([^}]+)\}", src):
        path = TEX.parent / rel
        assert path.exists(), f"{rel} is referenced but absent"
        assert path.suffix == ".pdf", f"{rel} must be vector PDF for arXiv, not {path.suffix}"
