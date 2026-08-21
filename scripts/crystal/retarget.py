"""Zero-shot retargeting: deploy one trained checkpoint at targets it never saw.

THE CLAIM THIS TESTS. The constraint target is an ARGUMENT to the dual solve, not a
training-time label, so one training run should cover a family of targets at inference.
The conditional baseline A4 needs one training run per target. If a checkpoint trained
at 5.0 g/cm^3 can be deployed at 1.5 or 10.0 without retraining, the method buys
amortized multi-target control, which is an axis no baseline here competes on.

WHAT IS HELD FIXED. Only `cfg.density_target` changes between deployments. The weights,
the vocabulary, the grammar mask, the sampling temperature and the seed are identical,
so any difference in achieved density is attributable to the deployment target alone.
The checkpoint records the target it TRAINED at, which this script never overwrites --
`trained_density_target` and `deploy_target` are separate fields in every row, because
the diagonal of the retargeting matrix (train == deploy) is the existing result and must
stay distinguishable from the off-diagonal.

WHY THE BACKBONE READING IS ALSO REPORTED. A deployed reading alone cannot say whether
retargeting works through the layer or because the backbone happens to sit near the new
target. Both readings are emitted at every cell, as everywhere else in this project.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import load_split
from scripts.train.train_crystal import (
    ELEMENTS,
    CrystalDecoder,
    CrystalTrainConfig,
    CrystalVocab,
    GrammarMask,
    sample,
    structural_metrics,
)

SCRIPT = "scripts/crystal/retarget.py"

# The ten frontier targets, so the matrix columns line up with the published frontier.
FRONTIER_TARGETS = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0, 13.0, 16.0, 20.0]


def load_checkpoint(path: str | Path, device) -> tuple[torch.nn.Module, dict]:
    """Rebuild the model from a checkpoint without touching the training data.

    The geometry (vocab size, max_len) comes from the checkpoint rather than from a
    fresh data load, so a retargeting run cannot silently disagree with the model it
    is deploying. A mismatch raises instead of being padded or truncated into place.
    """
    ck = torch.load(Path(path), map_location="cpu", weights_only=False)
    cfg = CrystalTrainConfig(**ck["train_config"])
    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    if len(V) != ck["vocab_size"]:
        raise ValueError(
            f"vocab size {len(V)} rebuilt from checkpoint disagrees with the stored "
            f"{ck['vocab_size']}; the serialization changed since this checkpoint"
        )
    model = CrystalDecoder(len(V), ck["max_len"] + 2, cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    if "n_parameters" in ck and n_par != ck["n_parameters"]:
        raise ValueError(
            f"rebuilt model has {n_par} parameters against the stored "
            f"{ck['n_parameters']}; architecture drift"
        )
    return model, ck


def retarget(ckpt: str | Path, targets=None, n_samples: int | None = None,
             seed: int = 0, smoke: bool = False) -> dict:
    """Deploy one checkpoint at each target. Returns rows, one per deployment."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_checkpoint(ckpt, dev)
    trained_at = float(ck["trained_density_target"])
    cfg = CrystalTrainConfig(**ck["train_config"])
    if n_samples is not None:
        cfg.n_eval_samples = int(n_samples)
    if smoke:
        cfg.n_eval_samples = min(cfg.n_eval_samples, 256)

    V = CrystalVocab(ck.get("elements", ELEMENTS), order=ck["vocab_order"])
    gm = GrammarMask(V, order=ck["vocab_order"])
    max_len = int(ck["max_len"])

    # Scoring reference is the FULL split, identical to training, so a retargeted
    # reading is comparable to the frontier row at the same target.
    train = load_split("train", limit=(2000 if smoke else None))
    data_density = np.array([c.density for c in train])
    from collections import Counter
    _el: Counter = Counter()
    for c in train:
        for e, k in c.formula.items():
            _el[e] += k
    _tot = sum(_el.values())
    COV = dict(train_formulas={tuple(sorted(c.formula.items())) for c in train},
               full_element_freq={e: k / _tot for e, k in _el.items()})

    rows, t0 = [], time.time()
    for tgt in (targets or FRONTIER_TARGETS):
        cfg.density_target = float(tgt)
        rng = np.random.default_rng(seed)
        bseq, bdiag = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                             use_mask=True, project_at_inference=False,
                             max_len=max_len + 2, rng=rng)
        bb = structural_metrics(bseq, V, gm, cfg, **COV) | bdiag
        rng = np.random.default_rng(seed)
        dseq, ddiag = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                             use_mask=True, project_at_inference=True,
                             max_len=max_len + 2, rng=rng)
        dp = structural_metrics(dseq, V, gm, cfg, **COV) | ddiag
        dens = dp["mean_density"]
        rows.append({
            "trained_density_target": trained_at,
            "deploy_target": float(tgt),
            "is_diagonal": abs(float(tgt) - trained_at) < 1e-9,
            "deployed_density": dens,
            "deployed_error": (abs(dens - float(tgt)) if dens is not None else None),
            "backbone_density": bb["mean_density"],
            "validity": dp["structural_validity_rate"],
            "density_computable": dp["density_computable_rate"],
            "infeasible_step_fraction": dp["per_step_infeasible_fraction"],
            "novel_formula_fraction": dp.get("novel_formula_fraction"),
            "unique_formula_fraction": dp.get("unique_formula_fraction"),
            "distinct_elements": dp.get("distinct_elements"),
            "n_samples": cfg.n_eval_samples,
            "seed": seed,
        })
        print(f"  train {trained_at:>6.3f} -> deploy {float(tgt):>6.3f}  "
              f"density {dens if dens is None else round(dens, 3)}  "
              f"err {rows[-1]['deployed_error'] if dens is None else round(rows[-1]['deployed_error'], 3)}  "
              f"valid {dp['structural_validity_rate']:.4f}", flush=True)

    return {
        "checkpoint": str(ckpt),
        "trained_density_target": trained_at,
        "train_config": asdict(cfg),
        "data_density_mean": float(data_density.mean()),
        "device": str(dev),
        "rows": rows,
        "wall_clock_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, action="append",
                    help="checkpoint path; repeat for several training targets")
    ap.add_argument("--targets", default=None,
                    help="comma-separated deployment targets (default: ten frontier targets)")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="write raw JSON here instead of emitting")
    a = ap.parse_args()

    tg = [float(x) for x in a.targets.split(",")] if a.targets else None
    payload = {
        "question": "Does a checkpoint trained at one density target steer to targets "
                    "it never trained for, with no retraining?",
        "matrix": [retarget(c, targets=tg, n_samples=a.n_samples, seed=a.seed,
                            smoke=a.smoke) for c in a.ckpt],
    }
    if a.out:
        Path(a.out).write_text(json.dumps(payload, indent=1))
        print(a.out)
    else:
        print(emit("zero_shot_retargeting" + ("_smoke" if a.smoke else ""),
                   payload,
                   config={"n_checkpoints": len(a.ckpt), "smoke": a.smoke,
                           "targets": tg or FRONTIER_TARGETS},
                   seeds=[a.seed], script=SCRIPT))
