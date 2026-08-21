"""Every cited work must be pointed at in prose, and every prose citation must resolve.

A reference list nobody points at is not a citation, and a name in prose with no entry is
worse. These tests keep the two halves in correspondence, and pin the two attributions that
external review got wrong or that are easy to get wrong.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RW = ROOT / "development" / "docs" / "paper" / "related_work.md"

from _devtree import requires_dev_tree

pytestmark = requires_dev_tree

# Surname -> (venue string that must appear in the entry, must-appear-in-prose)
EXPECTED = {
    "Khalifa": "ICLR",
    "Kruszewski": "ACL",
    "Ahmed": "NeurIPS",
    "Loula": "ICLR",
    "Zeni": "Nature",
    "Antunes": "Nature Communications",
    "Gruver": "ICLR",
}


def _split() -> tuple[str, str]:
    if not RW.exists():
        pytest.skip("related_work.md absent")
    txt = RW.read_text()
    if "## References" not in txt:
        pytest.skip("no reference section yet")
    body, refs = txt.split("## References", 1)
    return body, refs


def test_every_reference_is_cited_in_prose():
    body, refs = _split()
    missing = [k for k in EXPECTED if k not in body]
    assert not missing, (
        f"these works are in the reference list but never pointed at in prose: {missing}"
    )


def test_every_prose_citation_has_an_entry():
    body, refs = _split()
    missing = [k for k in EXPECTED if k not in refs]
    assert not missing, f"cited in prose with no reference entry: {missing}"


def test_venues_are_as_the_publisher_states_them():
    """Secondary sources mis-venue papers; these were checked against publisher records."""
    _, refs = _split()
    for surname, venue in EXPECTED.items():
        block = next((b for b in refs.split("\n\n") if b.strip().startswith(surname)), None)
        assert block, f"no reference block for {surname}"
        assert venue in block, (
            f"{surname} entry does not state the verified venue {venue!r}: {block[:120]!r}"
        )


def test_the_two_systems_are_findable_by_name():
    """GeLaTo and Ctrl-G are referred to by system name in prose; a reader must be able to
    connect those names to the author-year entries."""
    body, refs = _split()
    for sysname in ("GeLaTo", "Ctrl-G"):
        assert sysname in body, f"{sysname} not discussed in prose"
        assert sysname in refs, (
            f"{sysname} is used in prose but the reference list gives only author names, so "
            f"the citation cannot be resolved by a reader"
        )


def test_mattergen_is_positioned_against_the_composition_result():
    """A materials reviewer will raise joint property conditioning; the text must meet it."""
    body, _ = _split()
    txt = re.sub(r"\s+", " ", body)
    assert "MatterGen" in txt
    assert "joint" in txt and "composition" in txt, (
        "MatterGen is cited but not positioned against the composition result, which is the "
        "comparison a materials reviewer will make"
    )


def test_the_scale_limitation_is_stated_where_the_external_models_are_cited():
    """Citing CrystaLLM and MatterGen invites a scale comparison; state it rather than let a
    reviewer notice it."""
    body, _ = _split()
    txt = re.sub(r"\s+", " ", body)
    assert "smaller" in txt and (
        "state-of-the-art" in txt or "state of the art" in txt), (
        "the backbone's scale relative to the cited external models is not stated"
    )
