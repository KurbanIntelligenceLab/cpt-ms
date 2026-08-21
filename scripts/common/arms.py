"""Canonical names for the experimental arms, and a resolver for legacy spellings."""

from __future__ import annotations

from typing import Iterable, Mapping, TypeVar

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
    """Canonical name for any historical or current spelling."""
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
    """Read `arm` out of an artifact keyed by whatever spelling it was stamped with."""
    want = resolve(arm)
    for key in mapping:
        try:
            if resolve(key) == want:
                return mapping[key]
        except KeyError:
            continue
    raise KeyError(f"no key for arm {want!r} among {sorted(mapping)}")


def names(*arms: str) -> Iterable[str]:
    """Canonical names for several arms at once."""
    return tuple(resolve(a) for a in arms)


def normalize(mapping: Mapping[str, _V]) -> dict[str, _V]:
    """Re-key an artifact's arm-indexed dict to canonical names."""
    out: dict[str, _V] = {}
    for key, value in mapping.items():
        try:
            out[resolve(key)] = value
        except KeyError:
            out[key] = value
    return out
