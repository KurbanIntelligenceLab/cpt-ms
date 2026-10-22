from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path

from cpt.common.artifacts import emit, load


def collect_runs(inputs: Path) -> list[dict]:
    runs = []
    for path in sorted(inputs.glob("crystal_*.json")):
        record = load(path.stem, inputs)
        runs.append(record["payload"])
    if not runs:
        raise ValueError("No crystal training records found")
    keys = [(run["train_config"]["arm"], run["train_config"]["density_target"], run["train_config"]["seed"]) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate model/target/seed records")
    return runs


def aggregate_sweep(inputs: Path, output: Path) -> Path:
    rows = {}
    for run in collect_runs(inputs):
        cfg, final = run["train_config"], run["final"]
        if cfg["arm"] != "projected":
            continue
        deployed = final["deployed"]
        if not isinstance(deployed["mean_density"], (int, float)):
            raise ValueError("Run has no computable deployed density")
        key = f"density{cfg['density_target']:g}_seed{cfg['seed']}"
        rows[key] = {"target": cfg["density_target"], "seed": cfg["seed"],
                     "backbone_density": final["mean_density"],
                     "deployed_density": deployed["mean_density"],
                     "deployed_sd": deployed["density_sd"],
                     "validity": deployed["structural_validity_rate"],
                     "computable": deployed["density_computable_rate"]}
    if not rows:
        raise ValueError("No projected training records found")
    return emit("steering_frontier_rows", {"rows": rows},
                config={"targets": sorted({row["target"] for row in rows.values()})},
                seeds=sorted({row["seed"] for row in rows.values()}),
                script="src/cpt/reproduce.py", results_dir=output)


def aggregate_conditional(inputs: Path, output: Path) -> Path:
    rows = {}
    for run in collect_runs(inputs):
        cfg, final = run["train_config"], run["final"]
        arm = cfg["arm"]
        if arm not in {"projected", "conditional"}:
            continue
        observed = final["deployed"] if arm == "projected" else final
        prefix = "dep" if arm == "projected" else "bb"
        row = {"arm": arm, "target": cfg["density_target"], "seed": cfg["seed"]}
        fields = {"density": "mean_density", "novel": "novel_formula_fraction",
                  "uniq": "unique_formula_fraction", "valid": "structural_validity_rate",
                  "elem": "distinct_elements", "ent": "element_entropy_nats"}
        for name, field in fields.items():
            row[f"{prefix}_{name}"] = observed[field]
        rows[f"{arm}_density{cfg['density_target']:g}_seed{cfg['seed']}"] = row
    if not rows:
        raise ValueError("No projected or conditional training records found")
    per_target = {}
    for target in sorted({row["target"] for row in rows.values()}):
        per_target[str(float(target))] = {}
        for arm in ("projected", "conditional"):
            selected = [row for row in rows.values() if row["target"] == target and row["arm"] == arm]
            if not selected:
                raise ValueError(f"Missing {arm} runs for target {target}")
            prefix = "dep" if arm == "projected" else "bb"
            errors = [abs(row[f"{prefix}_density"] - target) for row in selected]
            per_target[str(float(target))][arm] = {
                "err": statistics.mean(errors), "err_sd": statistics.pstdev(errors),
                "nov": statistics.mean(row[f"{prefix}_novel"] for row in selected),
                "unq": statistics.mean(row[f"{prefix}_uniq"] for row in selected),
                "val": statistics.mean(row[f"{prefix}_valid"] for row in selected),
                "ele": statistics.mean(row[f"{prefix}_elem"] for row in selected),
                "ent": statistics.mean(row[f"{prefix}_ent"] for row in selected),
                "dens": statistics.mean(row[f"{prefix}_density"] for row in selected),
                "n": len(selected),
            }
    return emit("conditional_baseline", {"runs": rows, "per_target": per_target},
                config={"targets": sorted(per_target), "error_sd": "population"},
                seeds=sorted({row["seed"] for row in rows.values()}),
                script="src/cpt/reproduce.py", results_dir=output)


def audit_data(data_dir: Path, output: Path) -> Path:
    import numpy as np
    import pandas as pd

    from cpt.crystal.model import CrystalDecoder, CrystalTrainConfig
    from cpt.crystal.mp20 import ELEMENTS, CrystalVocab, parse_cif

    splits = {}
    seen_ids = set()
    seen_cifs = set()
    retained = []
    for split in ("train", "val", "test"):
        path = data_dir / f"mp20_{split}.csv"
        frame = pd.read_csv(path)
        ids = set(frame["material_id"])
        cifs = set(frame["cif"])
        splits[split] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "rows": len(frame), "duplicate_ids": len(frame) - len(ids),
                         "duplicate_cif_strings": len(frame) - len(cifs),
                         "ids_shared_with_preceding_splits": len(ids & seen_ids),
                         "cifs_shared_with_preceding_splits": len(cifs & seen_cifs)}
        seen_ids.update(ids)
        seen_cifs.update(cifs)
        if split == "train":
            retained = [item for cif in frame["cif"] if (item := parse_cif(cif)) is not None]
    vocab = CrystalVocab(ELEMENTS)
    longest = max(len(vocab.encode(item)) for item in retained)
    model = CrystalDecoder(len(vocab), longest + 2, CrystalTrainConfig())
    volumes = np.array([item.volume / item.n_atoms for item in retained])
    result = {"splits": splits, "retained_training_records": len(retained),
              "count_clipped_records": sum(any(count > vocab.N_CNT for count in item.formula.values()) for item in retained),
              "vocabulary_size": len(vocab), "maximum_sequence_length": longest,
              "model_position_slots": longest + 2,
              "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
              "packing_window": [float(np.percentile(volumes, 1)), float(np.percentile(volumes, 99))],
              "duplicate_check_scope": "Exact identifiers and CIF strings; not structural equivalence."}
    return emit("materials_inputs", result, config={"serialization": "cell_first"}, seeds=[],
                script="src/cpt/reproduce.py", results_dir=output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["sweep", "conditional", "data", "summary"])
    parser.add_argument("--inputs", type=Path, default=Path("results"))
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "data":
        result = audit_data(args.data, args.output)
    elif args.mode == "sweep":
        result = aggregate_sweep(args.inputs, args.output)
    elif args.mode == "conditional":
        result = aggregate_conditional(args.inputs, args.output)
    else:
        from cpt.verify import summaries

        result = emit("summary", summaries(args.inputs),
                      config={"seed_uncertainty": "sample standard deviation", "target_weighting": "equal"},
                      seeds=[], script="src/cpt/reproduce.py", results_dir=args.output)
    print(result)
