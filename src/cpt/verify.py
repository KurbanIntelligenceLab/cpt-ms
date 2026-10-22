from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from cpt.common.artifacts import load


def payload(directory: Path, name: str) -> dict:
    return load(name, directory)["payload"]


def close(actual: float, expected: float, tolerance: float = 1e-10) -> bool:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"Numerical mismatch: {actual} versus {expected}")
    return True


def summaries(directory: Path) -> dict:
    """Aggregate all reported settings with the manuscript's metric definitions."""
    selection = {}
    for name in ("best_of_n", "hybrid_selection"):
        data = payload(directory, name)
        aggregates = {}
        for count in data["n_values"]:
            rows = [row for row in data["rows"] if row["n"] == count]
            diversity = [row for row in data["diversity"] if row["n"] == count]
            if len(rows) != 10 or len(diversity) != 10:
                raise ValueError("Selection grid must include ten targets per candidate count")
            aggregates[str(count)] = {
                "candidate_error": statistics.mean(row["mean_best_error"] for row in rows),
                "unique_fraction": statistics.mean(row["unique_formula_fraction"] for row in diversity),
            }
        selection[name] = aggregates
    reduction = 1 - selection["hybrid_selection"]["16"]["candidate_error"] / selection["best_of_n"]["16"]["candidate_error"]

    matrix = payload(directory, "zero_shot_retargeting")["matrix"]
    reuse = [next(group for group in matrix if group["trained_density_target"] == 10)]
    reuse.extend(payload(directory, f"retarget_t10_s{seed}")["matrix"][0] for seed in (1, 2))
    sweep = list(payload(directory, "steering_frontier_rows")["rows"].values())
    targets = sorted({row["target"] for row in sweep})
    per_target = {}
    for target in targets:
        observed = [row for row in sweep if row["target"] == target]
        per_target[str(target)] = {
            "seeds": sorted(row["seed"] for row in observed),
            "mean_response_error": statistics.mean(abs(row["deployed_density"] - target) for row in observed),
        }
    lower = [target for target in targets if target <= 10]
    reuse_errors = [statistics.mean(abs(row["deployed_density"] - row["deploy_target"])
                                   for row in group["rows"] if row["deploy_target"] <= 10)
                    for group in reuse]
    retargeting = {
        "per_target": per_target,
        "completed_target_specific_runs": len(sweep),
        "missing_target_specific_runs": [[target, seed] for target in targets for seed in (0, 1)
                                         if not any(row["target"] == target and row["seed"] == seed for row in sweep)],
        "lower_target_reuse_errors": reuse_errors,
        "lower_target_reuse_mean": statistics.mean(reuse_errors),
        "lower_target_reuse_seeds_one_two": statistics.mean(reuse_errors[1:]),
        "lower_target_specific_mean": statistics.mean(per_target[str(target)]["mean_response_error"] for target in lower),
    }

    finite = {}
    for arm in ("mask_only", "projected"):
        rows = [payload(directory, f"train_{arm}_seed{seed}")["final"] for seed in range(3)]
        if arm == "projected":
            rows = [row["deployed"] for row in rows]
        finite[arm] = {field: {"mean": statistics.mean(row[field] for row in rows),
                              "sample_sd": statistics.stdev(row[field] for row in rows)}
                       for field in ("mean_charge", "charge_variance", "violation_rate")}
    comparator = payload(directory, "tilt_comparator")["runs"]
    finite["comparator"] = {field: {"mean": statistics.mean(row[field] for row in comparator),
                                  "sample_sd": statistics.stdev(row[field] for row in comparator)}
                           for field in ("mean_charge", "charge_variance", "hard_violation_rate", "kl_from_tilt")}
    finite["exact"] = payload(directory, "synthetic_ground_truth")["distributions"]

    controls = payload(directory, "conditional_baseline")
    conditional = {}
    for arm in ("projected", "conditional"):
        by_target = []
        for target, saved in controls["per_target"].items():
            rows = [row for row in controls["runs"].values() if row["arm"] == arm and row["target"] == float(target)]
            prefix = "dep" if arm == "projected" else "bb"
            values = {
                "err": statistics.mean(abs(row[f"{prefix}_density"] - float(target)) for row in rows),
                "nov": statistics.mean(row[f"{prefix}_novel"] for row in rows),
                "unq": statistics.mean(row[f"{prefix}_uniq"] for row in rows),
            }
            for field, value in values.items():
                close(value, saved[arm][field])
            by_target.append(values)
        conditional[arm] = {field: statistics.mean(row[field] for row in by_target) for field in ("err", "nov", "unq")}
    return {"selection": selection, "candidate_error_reduction": reduction,
            "retargeting": retargeting, "finite_task": finite, "conditional": conditional}


def numerical_checks() -> dict:
    """Check interior projection, its derivative and closing-step moment identities."""
    import numpy as np
    import torch
    from scipy.optimize import brentq
    from scipy.special import softmax

    from cpt.layer.projection import project, project_differentiable

    def independent(logits, features, budget):
        multiplier = brentq(lambda v: softmax(logits + v * features) @ features - budget,
                            -200.0, 200.0, xtol=5e-15)
        return softmax(logits + multiplier * features)

    rng = np.random.default_rng(0)
    maximum = 0.0
    for _ in range(40):
        z = rng.normal(size=6)
        phi = rng.uniform(-2, 2, 6)
        b = phi.min() + rng.uniform(0.2, 0.8) * np.ptp(phi)
        q = independent(z, phi, b)
        u = q * (phi - b)
        jacobian = np.diag(q) - np.outer(q, q) - np.outer(u, u) / (q @ (phi - b) ** 2)
        h = 2e-5
        fd = np.column_stack([(independent(z + h * e, phi, b) - independent(z - h * e, phi, b)) / (2 * h)
                              for e in np.eye(6)])
        maximum = max(maximum, float(np.max(np.abs(jacobian - fd))))
        close(float(q @ phi), float(b), 1e-10)
        logits = torch.tensor(z[None, :], dtype=torch.float64)
        features = torch.tensor(phi[None, None, :], dtype=torch.float64)
        budget = torch.tensor([[b]], dtype=torch.float64)
        mask = torch.ones_like(logits, dtype=torch.bool)
        result = project(logits, features, budget, mask)
        if not result.converged.item():
            raise ValueError("Interior projection did not converge")
        close(float(np.max(np.abs(result.q.detach().numpy()[0] - q))), 0.0, 1e-8)
    close(maximum, 0.0, 1e-8)
    logits = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    features = torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    budget = torch.tensor([[0.5]], dtype=torch.float64)
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = project(logits, features, budget, mask)
    unrolled = torch.autograd.grad(result.q[0, 0], logits)[0]
    implicit_logits = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    implicit = project_differentiable(implicit_logits, features, budget, mask)
    gradient = torch.autograd.grad(implicit[0, 0], implicit_logits)[0]
    close(float(unrolled[0, 0]), 0.25)
    close(float(unrolled[0, 1]), -0.25)
    close(float(gradient.abs().max()), 0.0, 1e-8)
    weights = np.array([0.2, 0.5, 0.3])
    prefixes = np.array([-2.0, 1.0, 3.0])
    increments = np.array([[1.0, 4.0], [-3.0, 2.0], [-4.0, 1.0]])
    probabilities = np.array([[0.25, 0.75], [0.8, 0.2], [0.4, 0.6]])
    target = 1.2
    means = np.sum(probabilities * increments, axis=1)
    variances = np.sum(probabilities * (increments - means[:, None]) ** 2, axis=1)
    delta = means - (target - prefixes)
    totals = prefixes[:, None] + increments
    joint = weights[:, None] * probabilities
    mean = np.sum(joint * totals)
    variance = np.sum(joint * (totals - mean) ** 2)
    close(float(mean - target), float(weights @ delta))
    close(float(variance), float(weights @ variances + weights @ ((delta - weights @ delta) ** 2)))
    close(float(np.sum(joint * (totals - target) ** 2)), float(weights @ variances + weights @ delta ** 2))
    return {"interior_cases": 40, "max_jacobian_error": maximum,
            "early_stop_gradient": unrolled.tolist(), "implicit_gradient": gradient.tolist(),
            "closing_step_identities": True}


def verify_checkpoints(directory: Path) -> int:
    """Check distributed model files against their manifest and rebuilt architecture."""
    from cpt.common.artifacts import file_sha256
    from cpt.crystal.retarget import load_checkpoint

    folder = directory / "checkpoints"
    manifest = json.loads((folder / "manifest.json").read_text())
    entries = manifest["checkpoints"]
    if not entries or len({entry["file"] for entry in entries}) != len(entries):
        raise ValueError("Checkpoint manifest must contain unique model files")
    for entry in entries:
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Checkpoint manifest requires a filename")
        path = folder / entry["file"]
        if path.stat().st_size != entry["bytes"] or file_sha256(path) != entry["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path.name}")
        model, checkpoint = load_checkpoint(path, "cpu")
        observed = {
            "trained_density_target": checkpoint["trained_density_target"],
            "seed": checkpoint["train_config"]["seed"],
            "training_steps": checkpoint["train_config"]["steps"],
            "vocabulary_size": checkpoint["vocab_size"],
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "tokenizer": checkpoint["tokenizer"],
        }
        if any(entry[field] != value for field, value in observed.items()):
            raise ValueError(f"Checkpoint metadata mismatch: {path.name}")
    return len(entries)


def verify(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    listed = {entry["file"] for entry in manifest["records"]}
    actual = {path.name for path in directory.glob("*.json") if path.name != "manifest.json"}
    if not listed or listed != actual:
        raise ValueError("Result files do not match the nonempty evidence manifest")
    for entry in manifest["records"]:
        path = directory / entry["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"File checksum mismatch: {path.name}")
        load(path.stem, directory)
    result = summaries(directory)
    close(result["candidate_error_reduction"], 0.7052262228301325)
    close(result["selection"]["best_of_n"]["16"]["candidate_error"], 0.9741564897466605)
    close(result["selection"]["hybrid_selection"]["16"]["candidate_error"], 0.28715578803716246)
    if result["retargeting"]["completed_target_specific_runs"] != 18:
        raise ValueError("Expected 18 available target-specific runs")
    if result["retargeting"]["missing_target_specific_runs"] != [[13.0, 0], [16.0, 1]]:
        raise ValueError("Unexpected missing target-specific runs")
    close(round(result["retargeting"]["lower_target_reuse_mean"], 4), 0.6497)
    close(round(result["retargeting"]["lower_target_specific_mean"], 4), 0.6620)
    close(round(result["finite_task"]["projected"]["charge_variance"]["mean"], 3), 1.547)
    funnel = payload(directory, "generated_plausibility")["funnel"]
    if (funnel["n_sampled"], funnel["n_built_composition_cell"]) != (2048, 2031):
        raise ValueError("Chemical-screen denominators differ")
    diagnostics = payload(directory, "induced_vs_tilt")["per_step_diagnostics"]
    if (diagnostics["nodes_visited"], diagnostics["per_step_infeasible_count"]) != (16537, 6140):
        raise ValueError("Prefix counts differ")
    widths = payload(directory, "smc_tuned")
    if len(widths["rows"]) != 90:
        raise ValueError("Expected all 90 particle width/target settings")
    width_errors = {width: statistics.mean(row["mean_error"] for row in widths["rows"] if row["sigma"] == width)
                    for width in widths["sigmas"]}
    best = min(width_errors, key=width_errors.get)
    close(best, widths["global_best_fixed_sigma"]["sigma"])
    close(width_errors[best], widths["global_best_fixed_sigma"]["mean_error_over_targets"])
    particles = payload(directory, "smc_diversity")
    if len(particles["rows"]) != 7 or any(row["n_valid"] != 100 for row in particles["rows"]):
        raise ValueError("Particle evaluation grid or denominators differ")
    close(statistics.mean(row["mean_error"] for row in particles["rows"]), particles["mean_error_over_targets"])
    close(statistics.mean(row["unique_formula_fraction"] for row in particles["rows"]), particles["mean_unique_formula_fraction_over_targets"])
    frontier = payload(directory, "steering_frontier")["frontier"]
    if [row["charge_target"] for row in frontier] != [-6, -4, -2, 0, 2, 4, 6]:
        raise ValueError("Finite-task target grid differs")
    for row in frontier:
        if row["induced_mass_exactly_on_target"] <= row["tilt_mass_exactly_on_target"]:
            raise ValueError("Finite-task target-hit comparison differs")
    example = payload(directory, "projection_example")
    unnormalized = [p * math.exp(example["multiplier"] * mass)
                    for p, mass in zip(example["masked_probabilities"], example["masses_amu"], strict=True)]
    normalizer = sum(unnormalized)
    for value, expected in zip(unnormalized, example["projected_probabilities"], strict=True):
        close(value / normalizer, expected)
    close(sum(q * mass for q, mass in zip(example["projected_probabilities"], example["masses_amu"], strict=True)), example["local_target_mass"])
    inputs = payload(directory, "materials_inputs")
    if (inputs["retained_training_records"], inputs["count_clipped_records"],
            inputs["vocabulary_size"], inputs["model_parameters"]) != (26910, 39, 884, 5199220):
        raise ValueError("Materials input/model audit differs from Methods")
    if payload(directory, "summary") != result:
        raise ValueError("Saved summary differs from current numerical aggregation")
    return {"verified_records": len(listed), "verified_checkpoints": verify_checkpoints(directory),
            "summaries": result, "mathematics": numerical_checks()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", default="", help="optional numerical summary JSON")
    args = parser.parse_args()
    report = verify(args.results)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text)
