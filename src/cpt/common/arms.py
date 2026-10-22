from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TypeVar

ARMS = {
    "unconstrained": "decoder without grammar masking or projection",
    "mask_only": "grammar-masked decoder",
    "projected": "decoder trained through the density or charge projection",
    "projected_backbone": "projection-trained decoder evaluated with masking alone",
    "projected_cell_first": "cell-first projection-trained decoder",
    "conditional": "masked decoder trained on a target-filtered dataset",
    "comparator": "decoder trained on samples from the exact sequence tilt",
}


def resolve(name: str) -> str:
    if name not in ARMS:
        raise KeyError(f"Unknown model configuration: {name!r}")
    return name


Value = TypeVar("Value")


def pick(mapping: Mapping[str, Value], arm: str) -> Value:
    return mapping[resolve(arm)]


def names(*arms: str) -> Iterable[str]:
    return tuple(resolve(arm) for arm in arms)


def normalize(mapping: Mapping[str, Value]) -> dict[str, Value]:
    return dict(mapping)
