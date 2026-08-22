"""Provenance stamping must describe the CODE, not the output directory."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from cpt.common.artifacts import _git_commit


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=20,
        env={"HOME": str(repo), "PATH": "/usr/bin:/bin",
             "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
             "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"},
    )


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A minimal repo laid out like this project, or skip if git is unusable."""
    r = tmp_path / "proj"
    (r / "src" / "cpt").mkdir(parents=True)
    (r / "results").mkdir()
    (r / "src" / "cpt" / "mod.py").write_text("x = 1\n")
    if _git(r, "init", "-q").returncode != 0:
        pytest.skip("git init unavailable in this environment")
    _git(r, "add", "-A")
    if _git(r, "commit", "-q", "-m", "initial").returncode != 0:
        pytest.skip("git commit unavailable in this environment")

    import cpt.common.artifacts as art
    monkeypatch.setattr(art, "REPO_ROOT", r)
    return r


def test_clean_source_gives_a_clean_commit(repo):
    assert not _git_commit().endswith("-dirty")
    assert _git_commit() != "UNVERSIONED"


def test_untracked_results_do_not_dirty_provenance(repo):
    """The ordering rule: an artifact must not dirty its own provenance."""
    (repo / "results" / "out.json").write_text('{"a": 1}')
    assert not _git_commit().endswith("-dirty"), (
        "an emitted artifact dirtied the commit that describes the code"
    )


def test_modified_source_does_dirty_provenance(repo):
    """The scoping must not be so loose that real source drift goes unrecorded."""
    (repo / "src" / "cpt" / "mod.py").write_text("x = 2\n")
    assert _git_commit().endswith("-dirty")


def test_new_untracked_source_dirties_provenance(repo):
    (repo / "src" / "cpt" / "extra.py").write_text("y = 3\n")
    assert _git_commit().endswith("-dirty")


def test_reports_and_figures_do_not_dirty_provenance(repo):
    (repo / "reports").mkdir()
    (repo / "reports" / "note.md").write_text("# note\n")
    assert not _git_commit().endswith("-dirty")


def test_stale_artifact_is_rejected(tmp_path):
    """A stale file must not pass for a fresh measurement."""
    import os
    import time

    from cpt.common.artifacts import assert_fresh, emit

    emit("probe", {"v": 1}, config={}, seeds=[], script="t", results_dir=tmp_path)
    assert_fresh("probe", max_age_s=3600, results_dir=tmp_path)

    old = time.time() - 7200
    os.utime(tmp_path / "probe.json", (old, old))
    with pytest.raises(ValueError, match="failed silently"):
        assert_fresh("probe", max_age_s=3600, results_dir=tmp_path)


def test_missing_artifact_is_distinguished_from_stale(tmp_path):
    from cpt.common.artifacts import assert_fresh

    with pytest.raises(FileNotFoundError):
        assert_fresh("never_emitted", results_dir=tmp_path)


def test_sentinel_when_not_a_repository(tmp_path, monkeypatch):
    """No git means UNVERSIONED, never a fabricated hash."""
    import cpt.common.artifacts as art
    bare = tmp_path / "nogit"
    bare.mkdir()
    monkeypatch.setattr(art, "REPO_ROOT", bare)
    assert _git_commit() == "UNVERSIONED"
