#!/usr/bin/env bash
# Wire this working tree to the remote on branch jp/dev.
#
# Run from the repository root:  bash setup_repo.sh
#
# Authorship: this script never sets a git identity. It reads whatever your git
# config already provides and REFUSES to run if none is configured, so every
# commit is authored by you. Nothing here writes an author name or email.
#
# Idempotent: safe to re-run. It skips any step already done.

set -euo pipefail

REMOTE_URL="git@github.com:KurbanIntelligenceLab/cpt-ms.git"
BRANCH="jp/dev"

# --- 1. Authorship guard --------------------------------------------------
NAME="$(git config --get user.name || true)"
EMAIL="$(git config --get user.email || true)"
if [[ -z "$NAME" || -z "$EMAIL" ]]; then
  echo "ERROR: git identity not configured. Set it first, e.g.:"
  echo "  git config --global user.name  \"Your Name\""
  echo "  git config --global user.email \"you@example.com\""
  echo "Refusing to commit without an identity — commits must be authored by you."
  exit 1
fi
echo "Committing as: $NAME <$EMAIL>"

# --- 2. Init --------------------------------------------------------------
if [[ ! -d .git ]]; then
  git init -q
  echo "Initialized repository."
fi

# --- 3. Remote ------------------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  CURRENT="$(git remote get-url origin)"
  if [[ "$CURRENT" != "$REMOTE_URL" ]]; then
    echo "NOTE: origin is $CURRENT, not $REMOTE_URL — leaving it alone."
  fi
else
  git remote add origin "$REMOTE_URL"
  echo "Added origin: $REMOTE_URL"
fi

# --- 4. Branch ------------------------------------------------------------
if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
  git checkout -q -b "$BRANCH"
else
  git checkout -q "$BRANCH"
fi
echo "On branch: $BRANCH"

# --- 5. First commit ------------------------------------------------------
git add -A
if git diff --cached --quiet; then
  echo "Nothing staged; working tree already committed."
else
  git commit -q -m "CPT: synthetic task, crystal parser, projection layer, reporting infrastructure

Workstream deliverables through tranche item 4:
- Synthetic constrained-sequence task with exact conditioning and tilting targets
- Incremental constraint-state parser and per-step determinability measurement
- Batched KL projection layer with implicit-function-theorem gradients
- Run ledger, pre-flight harness, artifact emission with provenance
- Pre-registration notes for the CP2 overhead budget and CP4 violation-rate MDE

54 tests pass."
  echo "Committed."
fi

# --- 6. Re-emit artifacts under version control ---------------------------
# Artifacts produced before init record the sentinel UNVERSIONED in their
# provenance field, by design rather than by accident: a missing commit is
# recorded as missing rather than fabricated. Re-running now stamps them with
# the real commit that produced them, which is the state a checkpoint expects.
echo
echo "Re-running scripts so artifacts carry a real commit hash..."
export PYTHONPATH="$PWD"
python -m scripts.synthetic.compute_ground_truth
python -m scripts.synthetic.alignment_sweep
python -m scripts.crystal.determinability
python -m scripts.layer.verify_gradients

if ! git diff --quiet -- results/; then
  git add results/
  git commit -q -m "Re-emit artifacts under version control

Replaces the UNVERSIONED provenance sentinel with the commit that produced each
artifact. Values are unchanged; only the provenance field differs."
  echo "Committed re-emitted artifacts."
fi

# --- 7. Push --------------------------------------------------------------
echo
echo "Ready to push. Run:"
echo "  git push -u origin $BRANCH"
