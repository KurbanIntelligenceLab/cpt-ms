"""Run ledger (WS-7).

Every training or evaluation run appends one immutable record here. The ledger is
the join key between a claim in a report and the run that produced it: a number
that cannot be traced to a ledger entry does not go in a paper.

Design constraints, all of which come from how this gets misused rather than how
it gets used:

  Append-only.      Records are never edited in place. A correction is a new
                    record that supersedes an earlier one by id, so the history
                    of a claim stays visible.

  Line-delimited.   One JSON object per line. A crashed run leaves a truncated
                    final line rather than an unparseable file, and concurrent
                    appends from separate processes interleave safely because
                    each write is a single line under O_APPEND.

  Self-describing.  Each record carries the config hash, seeds, git commit, and
                    environment. Reproducing a run requires the ledger and the
                    repository, not a memory of what was passed on the command
                    line.

The ledger deliberately does NOT store results. Results are artifacts; the ledger
stores what produced them and which artifacts they are. Keeping numbers out of it
prevents the ledger and the artifact from disagreeing.
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from scripts.common.artifacts import REPO_ROOT, _git_commit, config_hash

LEDGER_PATH = REPO_ROOT / "results" / "run_ledger.jsonl"
SCHEMA_VERSION = "1.0"


@dataclass
class RunRecord:
    """One entry. Immutable once written."""

    run_id: str
    kind: str                      # "train" | "eval" | "benchmark" | "analysis"
    script: str
    status: str                    # "started" | "completed" | "failed" | "superseded"
    config: dict[str, Any]
    config_hash: str
    seeds: list[int]
    git_commit: str
    started_utc: str
    schema_version: str = SCHEMA_VERSION
    finished_utc: str | None = None
    artifacts: list[str] = field(default_factory=list)
    supersedes: str | None = None
    notes: str = ""
    environment: dict[str, str] = field(default_factory=dict)

    def as_line(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)


def _environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", ""),
    }


def start_run(
    kind: str,
    script: str,
    config: dict[str, Any],
    seeds: list[int],
    *,
    supersedes: str | None = None,
    notes: str = "",
    ledger_path: Path | None = None,
) -> RunRecord:
    """Open a run and append its 'started' record."""
    if kind not in ("train", "eval", "benchmark", "analysis"):
        raise ValueError(f"unknown run kind {kind!r}")
    rec = RunRecord(
        run_id=uuid.uuid4().hex[:16],
        kind=kind,
        script=script,
        status="started",
        config=config,
        config_hash=config_hash(config),
        seeds=seeds,
        git_commit=_git_commit(),
        started_utc=datetime.now(timezone.utc).isoformat(),
        supersedes=supersedes,
        notes=notes,
        environment=_environment(),
    )
    _append(rec, ledger_path)
    return rec


def finish_run(
    rec: RunRecord,
    status: str,
    artifacts: list[str] | None = None,
    notes: str = "",
    ledger_path: Path | None = None,
) -> RunRecord:
    """Append the terminal record for a run.

    The started record is left in place rather than rewritten: the pair is the
    audit trail, and a run that started but never finished must remain visible
    as exactly that.
    """
    if status not in ("completed", "failed", "superseded"):
        raise ValueError(f"terminal status must be completed/failed/superseded, got {status!r}")
    done = RunRecord(
        **{
            **asdict(rec),
            "status": status,
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": artifacts or [],
            "notes": (rec.notes + " | " + notes).strip(" |") if notes else rec.notes,
        }
    )
    _append(done, ledger_path)
    return done


def _append(rec: RunRecord, ledger_path: Path | None = None) -> None:
    path = Path(ledger_path) if ledger_path is not None else LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    # O_APPEND with a single write keeps concurrent writers from interleaving
    # partial lines.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(rec.as_line() + "\n")


def read_ledger(ledger_path: Path | None = None) -> list[dict[str, Any]]:
    """Read all records, tolerating a truncated final line from a crashed run."""
    path = Path(ledger_path) if ledger_path is not None else LEDGER_PATH
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            # A truncated tail is expected after a hard kill; earlier records
            # remain valid and are what matters.
            continue
    return out


def latest_status(ledger_path: Path | None = None) -> dict[str, dict[str, Any]]:
    """Current state of each run_id, last record winning."""
    state: dict[str, dict[str, Any]] = {}
    for rec in read_ledger(ledger_path):
        state[rec["run_id"]] = rec
    return state


def unfinished(ledger_path: Path | None = None) -> list[dict[str, Any]]:
    """Runs that started and never reached a terminal status.

    These are the entries that silently corrupt a results table: a run that
    crashed mid-way may still have written partial artifacts.
    """
    return [r for r in latest_status(ledger_path).values() if r["status"] == "started"]
