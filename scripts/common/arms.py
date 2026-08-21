"""Canonical names for the experimental arms, and a resolver for legacy spellings.

The arms were once called A1..A7. The codes said nothing to a reader, they were
never defined where a reader would look, and A5 never existed -- so the sequence
itself invited the question "what happened to A5?". Worse, the codes drifted:
`results/training_summary.json` spells the mask-only arm `A2_masked` while
`results/charge_marginals.json` spells it `A2_mask_only`, and two test modules
looked for the short forms `masked` and `mask_only` that neither artifact uses.
Both of those tests failed for that reason alone.

So this module is the single place the vocabulary is defined. New artifacts are
emitted under CANONICAL names. Artifacts already committed keep whatever they were
stamped with -- their payload checksum covers those strings, and rewriting them
would break provenance to fix a spelling -- so readers go through `resolve()` or
`pick()`, which understand every historical form.
"""

from __future__ import annotations

from typing import Iterable, Mapping, TypeVar

#: canonical name -> what the arm actually is
ARMS: dict[str, str] = {
    "unconstrained": "backbone decoder: no grammar mask, no projection",
    "mask_only": "grammar mask only, no expectation projection",
    "projected_backbone": "the projection-trained model read without projection at inference",
    "projected": "grammar mask plus trained-through projection, one active constraint",
    "projected_cell_first": "as projected, cell-first static-channel variant",
    "conditional": "conditional-training baseline, retrained on target-filtered MP-20",
    "comparator": "sequence-level comparator trained by MLE on exact tilt samples",
    "joint": "projection under two simultaneous constraints",
}

#: every historical spelling -> canonical name
LEGACY: dict[str, str] = {
    "A1": "unconstrained",
    "A1_unconstrained": "unconstrained",
    "A2": "mask_only",
    "A2_masked": "mask_only",
    "A2_mask_only": "mask_only",
    "masked": "mask_only",
    "A3": "projected",
    "A3_deployed": "projected",
    "A3_backbone": "projected_backbone",
    "A3_f2": "projected_cell_first",
    "A4": "conditional",
    "A6": "comparator",
    "A7": "joint",
}


def resolve(name: str) -> str:
    """Canonical name for any historical or current spelling.

    Raises rather than guessing: an unrecognised arm name is a bug in the caller or
    a new arm that belongs in ARMS, and silently passing it through would let a
    typo read as a missing measurement.
    """
    if name in ARMS:
        return name
    try:
        return LEGACY[name]
    except KeyError:
        raise KeyError(
            f"unknown arm {name!r}; add it to ARMS if it is new, or to LEGACY if it is "
            f"a spelling used by an artifact already on disk") from None


_V = TypeVar("_V")


def pick(mapping: Mapping[str, _V], arm: str) -> _V:
    """Read `arm` out of an artifact keyed by whatever spelling it was stamped with.

    Committed artifacts carry a payload checksum over their own key strings, so they
    cannot be renamed in place. This finds the one key in `mapping` that resolves to
    the same arm, and fails loudly listing what was actually there -- the failure
    mode this replaces was a bare KeyError that named neither the arm nor the file.
    """
    want = resolve(arm)
    for key in mapping:
        try:
            if resolve(key) == want:
                return mapping[key]
        except KeyError:
            continue  # not an arm key at all (e.g. "tilt", "data_S_conditioned")
    raise KeyError(f"no key for arm {want!r} among {sorted(mapping)}")


def names(*arms: str) -> Iterable[str]:
    """Canonical names for several arms at once."""
    return tuple(resolve(a) for a in arms)


def normalize(mapping: Mapping[str, _V]) -> dict[str, _V]:
    """Re-key an artifact's arm-indexed dict to canonical names.

    Apply this once where an artifact is loaded, and everything downstream can use
    canonical names as plain literals instead of threading `pick()` through every
    access. Keys that are not arm names (`tilt`, `data_S_conditioned`, ...) pass
    through untouched, so this is safe on mixed dicts.
    """
    out: dict[str, _V] = {}
    for key, value in mapping.items():
        try:
            out[resolve(key)] = value
        except KeyError:
            out[key] = value
    return out
