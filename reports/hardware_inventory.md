# Compute inventory — resolving the 8-GPU node assumption

Reported as requested before any rental decision. **Case B holds: none of the six SSH
targets is the 8-GPU node the plan assumes.** This is a Sync-A budget conversation, not a
per-task decision.

## What the six targets actually are

Every registered target is a single-GPU vast.ai box. All are attached to other active
projects.

| Target | GPU | VRAM | Cores | RAM | Project |
|---|---|---|---|---|---|
| gpu-5090-vlm-project-1 | RTX 5090 | 32.6 GB | 24 | 61 GB | CoCr, persistent, marked keep |
| gpu-5090-vlm-project-3 | RTX 5090 | 32.6 GB | 128 | 516 GB | CoCr ICLR/A4 branch |
| gpu-cliffordstf | RTX 4090 | 24.5 GB | 192 | 1032 GB | CliffordSTF |
| gpu-cliffordstf-tuned | RTX 4090 | 24.5 GB | 128 | 516 GB | CliffordSTF |
| stemgym-gpu | RTX 4090 | 24.5 GB | 256 | 516 GB | STEMGym (notes record the box as destroyed) |
| mlxc-experiments | RTX 4090 | 24.5 GB | 12 | 64 GB | MLXC-DFTB (notes record the box as destroyed) |

Two aliases point at endpoints their own provider notes record as torn down, so the live
count is four, none of them ours. A standing policy on this account prohibits any agent
session from destroying or re-keying a box, and several belong to concurrently running
sessions; scope discipline means this project touches none of them.

## What the plan assumed

The resource note states that all training fits one 8-GPU node, and sizes the bake-off at
4 variants x 5 seeds x 2 backbones, roughly 40 runs at 10-25M parameters on MP-20 (~45k
structures), hours per run, clearing in days on that node.

No such node exists in the current inventory. The bake-off on a single 4090 or 5090 is the
same 40 runs serialized rather than parallelized across 8 GPUs.

## Two coupled decisions, both Sync-A

**The CP2 overhead benchmark.** The threshold is pre-registered against the training
setting, so the benchmark must run on the GPU class A2 actually trains on. Running it
anywhere else produces a number requiring a disclosed addendum re-run rather than a weaker
measurement. It cannot be sized until the training hardware is chosen.

**A2 training and the bake-off.** Model sizes are modest — 10-25M parameters is small for a
32 GB card — so a single 5090 is not memory-constrained. The constraint is throughput
across roughly 40 runs. Options, in the order I would consider them:

1. Rent one 5090 for the project and serialize. Simplest, matches the existing pattern on
   this account, and the per-run cost is low. Wall-clock for the full bake-off scales as
   the number of runs rather than dividing across devices.
2. Rent several single-GPU boxes and shard by seed or variant. Recovers most of the
   parallelism the plan assumed, at proportionally higher hourly spend, and each box needs
   its own environment build.
3. Reduce the matrix. Fewer seeds or one backbone, which weakens the statistical claims the
   plan makes at 5 seeds with confidence intervals.

Option 3 interacts directly with pre-registered thresholds and should not be taken
silently.

## Recommendation

Rent one RTX 5090 for CPT, dedicated to this project so no other session's scope discipline
is violated, and decide between serialize and shard after the timing prototype gives a
per-run wall-clock. That measurement is cheap and turns the choice between options 1 and 2
into arithmetic rather than a guess.

The gradient verification already completed needs no GPU and no rental. The overhead
benchmark waits for this decision.

## Practical notes carried from provider documentation

- RTX 5090 is sm_120. Torch images built for CUDA 12.4 report `is_available() == True` on
  the card and then fail every kernel launch. Verify with an actual device operation, never
  with the availability flag. The image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel` ships
  a working torch for this architecture.
- Register the SSH key at the vast account level before renting, so it is written to
  authorized_keys at boot. Attaching a key to an already-running container does not
  reliably propagate.
- Use a project-specific key file rather than the shared one; several projects share this
  account and a shared key file breaks whichever alias was registered first.
