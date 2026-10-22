# Reusable density control for inverse materials design with autoregressive models

Code, numerical results, and a trained density-5 checkpoint for the paper.

Can Polat¹ · Mustafa Kurban²˒³ · Erchin Serpedin¹ · Hasan Kurban⁴

¹ Department of Computer and Electrical Engineering, Texas A&M University, College Station, TX, USA

² Department of Electrical and Computer Engineering, Texas A&M University at Qatar, Doha, Qatar

³ Department of Prosthetics and Orthotics, Ankara University, Ankara, Turkey

⁴ College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar

Correspondence: kurbanm@ankara.edu.tr; hkurban@hbku.edu.qa

## Abstract

Inverse materials design requires generating property-targeted candidate structures without repeatedly retraining generative models or sacrificing compositional variety. Here, we present a cell-first mass-budgeting framework that applies local minimum-Kullback–Leibler moment projections during autoregressive decoding, dynamically guiding element choices toward requested target densities using pre-emitted cell volumes. Retargeting a single trained model across ten density requests demonstrates robust checkpoint reusability without parameter updates. Across candidate selection pools of size sixteen, local density steering reduces average candidate density error by 70.5%. Selecting among sixteen steered candidates achieves nearly identical density accuracy to sixty-four unsteered candidates while expanding the unique composition fraction from 22.8% to 37.1%. Analytical enumeration over an exact sequence space further demonstrates that per-step moment control concentrates total additive-feature outcomes while clarifying global variance behavior. These results demonstrate that reusable, representation-aligned decoding control significantly enhances candidate selection efficiency while balancing property precision and compositional diversity.

## Layout

```text
src/cpt/      numerical methods, training, sampling, aggregation and verification
results/      supporting JSON records and their integrity manifest
  checkpoints/  supplied model weights and checkpoint manifest
data/         local input datasets; gitignored
development/  manuscripts, figure/table generators, tests and other private work; gitignored
```

Saved numerical outputs are collected in `results/`. Local inputs belong in `data/`; working files and generated outputs belong in `development/`.

## Install and verify

Python 3.11 or later is required. Run these commands from the extracted package's root directory with uv installed (macOS/Linux shell):

```sh
uv sync --locked --no-dev
source .venv/bin/activate
python -m cpt.verify --results results
```

The verification command checks the result manifest, payload/configuration checksums, numerical summaries, the supplied checkpoint hash and architecture, the projection derivative and closing-step identities. It runs directly on the supplied results without training models or loading input datasets.

Successful verification prints a JSON report with `"verified_records": 24` and `"verified_checkpoints": 1` and exits with status 0. Installation needs network access; verification then runs locally. Keep the environment activated for every command below. On Windows PowerShell, activate it with `.venv\Scripts\Activate.ps1` and set environment variables with `$env:NAME="value"`.

For materials training and chemical screening, use `uv sync --locked --no-dev --extra crystal`. The optional `baselines` extra supplies CVXPY/CVXPYLayers for `python -m cpt.layer.verify_gradients`; install it with `uv sync --locked --no-dev --extra baselines` (add `--extra crystal` to keep both). A wheel contains the code; supply the accompanying `results/` directory explicitly when using the wheel outside this checkout.

## Results

`results/` contains JSON outputs for training, sampling, selection, chemical screening and numerical checks. `manifest.json` indexes the records and their checksums; `summary.json` collects numerical summaries.

Load a record and verify its integrity with:

```python
from pathlib import Path
from cpt.common.artifacts import load

record = load("summary", results_dir=Path("results"))
print(record["payload"])
```

## Paper results

The supplied JSON records support the following manuscript displays. Run `python -m cpt.reproduce summary --inputs results --output development/runs/summaries` to recompute the numerical summaries. Figure/table generation scripts and tests remain in `development/`.

| Manuscript display | Numerical records in `results/` |
|---|---|
| Figure 1: cell-first density control | `projection_example.json`, `materials_inputs.json` |
| Figure 2: checkpoint reuse | `zero_shot_retargeting.json`, `retarget_t10_s1.json`, `retarget_t10_s2.json`, `steering_frontier_rows.json` |
| Figure 3: candidate selection | `best_of_n.json`, `hybrid_selection.json` |
| Figure 4: finite-task moments | `synthetic_ground_truth.json`, `induced_vs_tilt.json`, `tilt_comparator.json`, `train_mask_only_seed*.json`, `train_projected_seed*.json` |
| Supplementary chemical screens | `plausibility_reference.json`, `generated_plausibility.json` |
| Supplementary target-filtered control | `conditional_baseline.json` |
| Supplementary particle guidance | `smc_tuned.json`, `smc_diversity.json` |
| Supplementary charge-target sweep | `steering_frontier.json` |

Across the ten density targets, projected best-of-16 has mean candidate error 0.2872 g/cm³ versus 0.9742 g/cm³ for unprojected best-of-16, a 70.5% reduction. Projected best-of-16 and unprojected best-of-64 have similar mean errors (0.2872 and 0.2850 g/cm³), with unique composition fractions of 37.1% and 22.8%, respectively.

## Supplied checkpoint

`results/checkpoints/projected_d5.0.pt` is the density-5, seed-0 checkpoint used for candidate selection and chemical screening (`A3_d5.0.pt` in the manuscript). It contains 5,199,220 parameters, was trained for 6,000 steps, and occupies 20,821,887 bytes (20.8 MB). Its tokenizer uses 884 tokens, including 512 length bins over [2, 46] Å and 256 angle bins over [5, 175] degrees. The checkpoint manifest records its SHA-256 and tokenizer configuration.

Load it without input datasets:

```python
from cpt.crystal.retarget import load_checkpoint

model, checkpoint = load_checkpoint("results/checkpoints/projected_d5.0.pt", "cpu")
print(checkpoint["trained_density_target"])
```

Run a small candidate-selection check without input datasets:

```sh
CPT_RESULTS_DIR=development/runs/checkpoint_check python -m cpt.crystal.best_of_n --ckpt results/checkpoints/projected_d5.0.pt --smoke
```

The supplied checkpoint is included in Git and the source distribution. Additional `.pt`, `.pth`, and `.ckpt` files remain gitignored, including files placed under `results/checkpoints/`.

## Input data

Obtain the `train.csv`, `val.csv` and `test.csv` splits from [CDVAE's MP-20 directory](https://github.com/txie-93/cdvae/tree/main/data/mp_20). Save them as `data/mp20_train.csv`, `data/mp20_val.csv` and `data/mp20_test.csv`. Keep the supplied splits and CIF strings unchanged. The data source and its attribution remain those documented by CDVAE and the Materials Project.

Audit the local input files and model dimensions with:

```sh
python -m cpt.reproduce data --data data --output development/runs/input_audit
```

Compare the resulting hashes with `results/materials_inputs.json`. `CPT_DATA_DIR` overrides the local input directory for training and screening. `data/` is gitignored and excluded from distributions.

## Recompute numerical summaries

```sh
python -m cpt.reproduce summary --inputs results --output development/runs/summaries
```

This writes numerical JSON only. Records encode nonfinite values as `{"nonfinite": "nan"}`, `{"nonfinite": "+inf"}` or `{"nonfinite": "-inf"}`; JSON `null` means unavailable. Use `cpt.common.artifacts.load` to read records and check their integrity.

## Training and sampling

Run commands below in the installed environment. Use a fresh output directory so new computations do not overwrite the saved results:

```sh
export CPT_RESULTS_DIR=development/runs/current
export CPT_DATA_DIR=data
```

Finite-task masked/projected training uses 4,000 steps and seeds 0, 1 and 2:

```sh
for arm in mask_only projected; do
  for seed in 0 1 2; do
    python -m cpt.train.train_synthetic --arm "$arm" --seed "$seed" --steps 4000
  done
done
python -m cpt.synthetic.compute_ground_truth
python -m cpt.synthetic.train_comparator --steps 4000
python -m cpt.synthetic.induced_distribution
python -m cpt.synthetic.steering_frontier
```

The comparator defaults to all three seeds. Synthetic training writes separate seed-specific files.

Materials training uses 6,000 steps. For a density-5 projected checkpoint:

```sh
python -m cpt.train.train_crystal --arm projected --density-target 5 --seed 0 --steps 6000 --ckpt results/checkpoints/projected_density5_seed0_retrained.pt
```

Adjust `--density-target`, `--seed` and `--steps` for additional runs. Use the supplied checkpoint for sampling and selection:

```sh
python -m cpt.crystal.retarget --ckpt results/checkpoints/projected_d5.0.pt --n-samples 4096 --seed 0
python -m cpt.crystal.best_of_n --ckpt results/checkpoints/projected_d5.0.pt --pool 1024 --seed 0
python -m cpt.crystal.hybrid_selection --ckpt results/checkpoints/projected_d5.0.pt --pool 512 --seed 0
python -m cpt.crystal.smc_tune --ckpt results/checkpoints/projected_d5.0.pt --K 16 --runs 16 --seed 0
python -m cpt.crystal.smc_diversity --ckpt results/checkpoints/projected_d5.0.pt --K 16 --sigma 0.5 --trials 100 --seed 0
python -m cpt.crystal.plausibility_screen --limit 800
python -m cpt.crystal.generated_plausibility --ckpt results/checkpoints/projected_d5.0.pt --n-samples 2048
```

Retargeting accepts repeated `--ckpt` arguments for a matrix and `--out` for an explicit checksummed JSON output file, for example `--out development/runs/current/retarget_t10_s1.json`. Choose a distinct output path for each run. Sampling seeds control both NumPy and PyTorch; the recorded training configuration remains the checkpoint's original configuration. Selection uses the full ten-target grid by default. The particle-diversity command uses the seven targets listed in its help; width selection uses all ten targets and the nine recorded widths.

For the target-filtered control, train both arms into a dedicated output directory:

```sh
for arm in conditional projected; do
  for target in 3 5 8 10; do
    for seed in 0 1; do
      CPT_RESULTS_DIR=development/runs/controls python -m cpt.train.train_crystal --arm "$arm" --density-target "$target" --seed "$seed" --steps 6000
    done
  done
done
```

To aggregate a density-target sweep, save projected training runs under `CPT_RESULTS_DIR=development/runs/target_specific`. Aggregate each directory separately:

```sh
python -m cpt.reproduce conditional --inputs development/runs/controls --output development/runs/control_summary
python -m cpt.reproduce sweep --inputs development/runs/target_specific --output development/runs/target_summary
```

Chemical screens require the `crystal` extra. Run `cpt.crystal.plausibility_screen` first to create the reference-screen record in `CPT_RESULTS_DIR`, then run `cpt.crystal.generated_plausibility` with a trained checkpoint.

Each command provides its available options through `--help`, for example:

```sh
python -m cpt.train.train_crystal --help
python -m cpt.crystal.retarget --help
python -m cpt.reproduce --help
```
