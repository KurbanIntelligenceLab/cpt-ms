# Resumption state — verified against the repo, the artifact export, and the vast.ai account

Every fact below was checked against a live source in this session: the exported session JSON,
the working tree at `/Users/jp/cpt-ms`, a full pytest run, and the vast.ai instances API.
Items that could not be verified are marked as such rather than assumed.

## The session being resumed

| Field | Value |
|---|---|
| Conversation | WS-1: Theory and Projection Layer |
| Root frame | `6e4d788e-3479-4dac-a72e-c0c10229a991` |
| Project | `proj_d880caac4375` (the current session runs under `proj_2efd85a4711b`) |
| Frames | 138 — 1 orchestrator, 137 reviewer frames; 136 completed, 1 cancelled, 1 error |
| Tokens | 447,912,691 in / 2,751,987 out |
| Cost | $371.69 |
| Span | 2026-08-07 20:19 UTC to 2026-08-11 19:34 UTC |
| Genuine user turns | 41 |

## Repository state

Working tree at `/Users/jp/cpt-ms` is clean. HEAD is `845012c`, "Sparse-target runs: A4 keeps
its accuracy and loses its coverage", on branch `jp/dev`.

The full suite passes: **257 tests collected**, all passing when `git init` is available; 5 provenance tests skip
in sandboxes that block it in the `cpt-proj`
environment. The transcript's last count was 185 with 5 skips; the five are conditional on
optional artifacts and on git being usable, and all of those conditions hold locally.

Present: 61 files in `results/`, 20 test modules, 19 standing notes in `reports/` besides this one, 6
pre-registration documents, 4 figures, and the three MP-20 CSV splits (56 MB) in `data/`.

`paper/` holds three drafted sections — `scope.md`, `methods.md`, `related_work.md`. There is no
introduction, results, or discussion section.

## Artifact coverage

The export lists 86 artifacts. All 56 files in `~/Downloads/artifacts` match export entries, and
they are the reports, pre-registrations, figures, and team documents. Of the 30 export entries not
in that folder, 25 are per-run JSONs that are present in `results/`. Five are not on disk anywhere
in the repo: `baseline.json`, `frontier.json`, `speedup_sweep.json`, `mp20_data.tar.gz`, and
`10.1038_s41467-024-54639-7.pdf`. The first three are intermediate handoff files whose contents
were folded into named artifacts; the tar is regenerable from `data/`; the PDF is a reference copy.

## Where the work stopped

The WS-3 conditional-training baseline is complete: 16 runs across four targets and two seeds,
plus 4 sparse-target runs at 13.0 and 16.0 and a refusal at 20.0. `WS3_conditional_baseline_notes.md`
is at v2 with the sparse regime measured, and the finding it lands on is that A4 is more accurate
than A3 at every target it can run, while coverage collapses with slice size — correlation 0.981
between log training-set size and unique-formula fraction.

## Open items

**Remote sync is unverified.** The local remote-tracking ref for `origin/jp/dev` sits at `87c24ec`,
recorded as pushed on 2026-08-08, leaving 36 commits ahead of it. The transcript reports later
pushes, including `845012c`. This session cannot confirm the remote: no GitHub credential is
configured, and the sandbox's git has `GIT_SSH_COMMAND=/bin/false` with no DNS for github.com.
Resolution requires either a GitHub credential or a check run outside the sandbox.

**The rented GPU box is running and idle.** Instance `47219544`, label `cpt-a2-training`, RTX 5090
on `ssh7.vast.ai:19544`, image `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel`, 32 cores, 126 GB RAM,
120 GB disk. Billing $0.445/hour, up 118.6 hours, roughly $52.8 accrued. GPU utilization reads 0.0.
It is not registered as a compute provider in this session — the only registered target is
`ssh:criticality-wm`, whose own provider notes record its instance as destroyed. Reaching the box
again needs an `~/.ssh/config` alias with the project key. The standing instruction on this account
is that no box is destroyed, stopped, or rebooted without an explicit user instruction.

**Two team decisions from `team_status.md` are still open.** The compute sizing question — the plan
sizes the bake-off at 4 variants x 5 seeds x 2 backbones and assumes an 8-GPU node that does not
exist — and the B1 serialization question, where family (i) charge balance has a usable fraction of
0.002 on the CIF character backbone. The dial-assignment pre-registration has since made density
the primary dial on all three backbones, which narrows the second question to how charge is
reported on B1 rather than whether the headline experiment can run.

**The bake-off matrix has not been run.** Crystal runs cover arms A1 through A4 at three seeds on a
single backbone; `train_crystal.py` exposes `--arm` and `--steps` with no backbone selector. The
plan's five-seed, two-backbone matrix would need both more runs and a backbone flag.

**Four pre-registration documents carry stale status lines.** `CP2_overhead_budget.md`,
`WS2_dial_assignment.md`, `CP0_fallback_policy.md`, and `CP0_actionable_window.md` each read
"draft, not yet committed" while naming a trigger that has already fired — the CP2 benchmark ran
(`results/CP2_overhead_benchmark.json`), the steering sweep ran (`results/WS1_steering_frontier.json`,
`results/WS3_steering_frontier_rows.json`), and A3 trained (`results/A2_train_A3_seed*.json`,
`results/WS3_crystal_A3_seed*.json`). `CP4_violation_rate_MDE.md` names a CP4 sampling script that
has no corresponding module in `scripts/`, so its draft status is accurate.
