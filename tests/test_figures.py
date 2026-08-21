"""Figures must exist, be referenced by artifact marker, and carry captions whose numbers
agree with the artifacts they plot.

Two failure modes motivated this file. A figure referenced by bare filename breaks whenever
two artifacts share a name, and an image path that resolved to an absolute local path at
authoring time is unusable to anyone else — both happened here. And a caption is prose like
any other, so its numbers drift from the data exactly as body text does.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "development" / "figures"
RESULTS = ROOT / "development" / "docs" / "paper" / "results.md"

from _devtree import requires_dev_tree

pytestmark = requires_dev_tree


def _text() -> str:
    if not RESULTS.exists():
        pytest.skip("results.md absent")
    return RESULTS.read_text()


def _payload(name: str) -> dict:
    p = ROOT / "results" / name
    if not p.exists():
        pytest.skip(f"{name} absent")
    return json.loads(p.read_text())["payload"]


def test_referenced_figures_exist_on_disk():
    """Every figure the paper embeds must be a file in figures/."""
    expected = ["frontier_mechanism.png", "composition_failure.png", "amortization.png",
                "inference_compute.png"]
    missing = [f for f in expected if not (FIGS / f).exists()]
    assert not missing, f"referenced figures absent from figures/: {missing}"


def test_figures_are_embedded_by_artifact_marker_not_by_path():
    """A bare filename breaks on name collision; an absolute path breaks for everyone else."""
    txt = _text()
    for m in re.finditer(r"!\[([^\]]*)\]\(([^)]*)\)", txt):
        alt, target = m.group(1), m.group(2)
        assert not target.startswith("/"), (
            f"figure {alt!r} is embedded by absolute path {target[:60]!r}; use an artifact "
            f"marker so the reference resolves outside this machine"
        )
        assert target.startswith("{{artifact:art_"), (
            f"figure {alt!r} is embedded as {target[:60]!r} rather than by artifact marker"
        )
        assert re.fullmatch(r"\{\{artifact:art_[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                            r"[0-9a-f]{4}-[0-9a-f]{12}\}\}", target), (
            f"figure {alt!r} marker is malformed: {target!r}"
        )


def test_every_embedded_figure_has_a_caption():
    txt = _text()
    n_embeds = len(re.findall(r"!\[[^\]]*\]\(\{\{artifact:", txt))
    n_caps = len(re.findall(r"\*\*Figure \d+\.\*\*", txt))
    assert n_caps >= n_embeds, (
        f"{n_embeds} figures embedded but only {n_caps} captions found"
    )


def test_frontier_caption_numbers_match_the_artifact():
    rows = _payload("steering_frontier_rows.json")["rows"]
    by: dict[float, list[dict]] = {}
    for v in rows.values():
        by.setdefault(v["target"], []).append(v)
    ts = sorted(by)

    def ag(t: float, f: str) -> float:
        return float(np.mean([r[f] for r in by[t]]))

    ceiling = ag(max(ts), "deployed_density")
    rep = np.array([ag(t, "infeas_init") / ag(t, "infeas") for t in ts])
    txt = re.sub(r"\s+", " ", _text())
    assert f"{ceiling:.2f}" in txt, "the caption's ceiling disagrees with the artifact"
    assert f"{rep.max():.0f}" in txt and f"{rep.min():.2f}" in txt, (
        "the caption's repair-factor extremes disagree with the artifact"
    )


def test_composition_caption_numbers_match_the_artifact():
    p = _payload("two_constraint_grid.json")
    rows = [r for r in p["rows"] if r.get("complete")]
    if not rows:
        pytest.skip("no complete grid row")
    r = rows[0]
    txt = re.sub(r"\s+", " ", _text())
    assert f"{r['pass_through'] * 100:.1f}%" in txt
    assert f"{r['control_valence']:.3f}" in txt, (
        "the caption's unconstrained control disagrees with the artifact"
    )


def test_partial_rows_are_labelled_as_partial():
    """A partial row plotted beside a complete one must say so.

    Driven by the artifact rather than by a hardcoded row: whichever rows are incomplete must
    be disclosed, and a row that has since COMPLETED must no longer be described as partial.
    """
    art = ROOT / "results" / "two_constraint_grid.json"
    if not art.exists():
        pytest.skip("grid artifact absent")
    rows = json.loads(art.read_text())["payload"]["rows"]
    txt = re.sub(r"\s+", " ", _text())
    for r in rows:
        label = f"density target {r['density_target']}"
        if label not in txt:
            continue
        if not r.get("complete"):
            assert f"{r['n_cells']} of 3" in txt or "partial" in txt, (
                f"{label} is incomplete ({r['n_cells']} of 3 cells) but is plotted and "
                f"described as a peer of the complete rows"
            )
        else:
            assert f"{r['n_cells']} of 3 cells" not in txt, (
                f"{label} is now complete but the text still describes it as partial"
            )
