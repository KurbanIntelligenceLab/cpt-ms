"""Parse validity: grammar-masked vs unconstrained decoding (WS-2 / CP4).

The structural half of the parse-validity claim, and it needs no trained model.

The claim under test is narrow and worth stating precisely: a decoder restricted
to a grammar's per-step legal set cannot emit a syntactically invalid file, so the
syntactic failure mode is removed by CONSTRUCTION rather than reduced. That is a
property of the mask, not of any particular model, which is why it is measurable
now.

What this does NOT claim. It says nothing about whether the structures are
physically sensible — overlapping sites, absurd lattice parameters, and
unphysical stoichiometries all parse. It also does not predict the parse-failure
rate of a specific published model, whose losses are attributed partly to prompt
handling. The comparison here is between two decoding procedures over the same
character distribution, which isolates the grammar's contribution.

Design. Both arms sample from the same character-level proposal distribution, so
the only difference is whether the per-step legal set is enforced:

  MASKED     sample only from the grammar's allowed set at each step
  UNMASKED   sample from the full vocabulary, stop at the end marker or a length
             cap, and let the file be whatever it is

Judgement is by pymatgen's CifParser, which shares no code with the emitter. A
masked sample that fails to parse is a defect in the grammar, reported as such.

The unmasked arm is deliberately given a favourable proposal: characters are
weighted by their frequency in valid CIFs, so it is not a uniform-random baseline
that would fail trivially. The point is that even a well-shaped unconstrained
sampler emits unparseable output, because validity is a global property that
per-character likelihood does not enforce.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.cif_grammar import (
    GrammarState,
    allowed,
    full_vocabulary,
    render,
    step,
)

SCRIPT = "scripts/crystal/parse_validity.py"
MAX_CHARS = 900


def reference_char_distribution(n: int = 400, seed: int = 0) -> dict[str, float]:
    """Character frequencies in grammar-valid CIFs.

    Used to shape the UNMASKED arm's proposal so the comparison is against a
    plausible character model rather than a uniform sampler. Built from masked
    samples, which is legitimate here: it gives the unconstrained arm the best
    per-character information available.
    """
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    for _ in range(n):
        s, out = GrammarState(), []
        for _ in range(MAX_CHARS):
            a = allowed(s)
            if not a:
                break
            ch = a[rng.integers(len(a))]
            out.append(ch)
            s = step(s, ch)
        counts.update(out)
    total = sum(counts.values())
    return {c: k / total for c, k in counts.items()}


def sample_masked(rng: np.random.Generator) -> tuple[str, bool]:
    """Sample with the grammar mask applied at every step."""
    s, out = GrammarState(), []
    for _ in range(MAX_CHARS):
        a = allowed(s)
        if not a:
            return render(out), True          # reached `done`
        ch = a[rng.integers(len(a))]
        out.append(ch)
        s = step(s, ch)
    return render(out), s.phase == "done"


def sample_unmasked(rng: np.random.Generator, vocab: str,
                    probs: np.ndarray) -> tuple[str, bool]:
    """Sample from the full vocabulary with no grammar constraint."""
    out = []
    for _ in range(MAX_CHARS):
        ch = vocab[rng.choice(len(vocab), p=probs)]
        if ch == "\x03":
            return render(out), True
        out.append(ch)
    return render(out), False                  # hit the length cap


def parses(text: str) -> tuple[bool, str]:
    """SYNTACTIC judgement: does the text parse as a CIF at all?

    This is what grammar masking claims to guarantee, and it must be measured
    separately from structure construction. Building a structure additionally
    requires the values to be physically coherent — a cell with 8.9-degree angles
    parses perfectly and then fails to yield a lattice. Conflating the two would
    charge the grammar for semantic failures it never claimed to prevent, and an
    earlier version of this script did exactly that: it reported the masked arm at
    47% when every one of its samples was syntactically valid.
    """
    from pymatgen.io.cif import CifParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parser = CifParser.from_str(text)
            blocks = list(parser._cif.data.keys())
        except Exception as e:                  # noqa: BLE001 - any failure is a failure
            return False, type(e).__name__
    if not blocks:
        return False, "NoDataBlocks"
    return True, ""


def builds_structure(text: str) -> tuple[bool, str]:
    """SEMANTIC judgement: does a coherent structure come out?

    Reported alongside the syntactic rate, never instead of it. The gap between
    the two is the honest scope boundary of Regime S: masking removes syntactic
    invalidity by construction and leaves semantic invalidity entirely open.
    """
    from pymatgen.io.cif import CifParser

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            st = CifParser.from_str(text).parse_structures(primitive=False)
        except Exception as e:                  # noqa: BLE001
            return False, type(e).__name__
    if not st:
        return False, "NoStructures"
    return True, ""


def main(smoke: bool = False) -> Path:
    n = 40 if smoke else 500
    rng_m = np.random.default_rng(11)
    rng_u = np.random.default_rng(11)

    vocab = full_vocabulary()
    freq = reference_char_distribution(n=40 if smoke else 400)
    # Unseen characters get a small floor so the proposal has full support.
    w = np.array([freq.get(c, 1e-4) for c in vocab], dtype=np.float64)
    probs = w / w.sum()

    arms: dict[str, dict] = {}
    for name, sampler in (("masked", lambda r: sample_masked(r)),
                          ("unmasked", lambda r: sample_unmasked(r, vocab, probs))):
        rng = rng_m if name == "masked" else rng_u
        ok_n, struct_n, terminated_n = 0, 0, 0
        fails: Counter[str] = Counter()
        sem_fails: Counter[str] = Counter()
        lengths, formulas, site_counts = [], set(), []
        for _ in range(n):
            text, terminated = sampler(rng)
            terminated_n += int(terminated)
            lengths.append(len(text))
            ok, kind = parses(text)
            if not ok:
                fails[kind] += 1
                continue
            ok_n += 1
            built, skind = builds_structure(text)
            if not built:
                sem_fails[skind] += 1
                continue
            struct_n += 1
            from pymatgen.io.cif import CifParser
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st = CifParser.from_str(text).parse_structures(primitive=False)[0]
            formulas.add(st.composition.reduced_formula)
            site_counts.append(len(st))
        arms[name] = {
            "n_samples": n,
            "n_syntactically_valid": ok_n,
            "syntactic_parse_rate": ok_n / n,
            "n_structure_built": struct_n,
            "structure_rate": struct_n / n,
            "structure_rate_given_syntactic": (struct_n / ok_n) if ok_n else None,
            "n_terminated_properly": terminated_n,
            "mean_length_chars": float(np.mean(lengths)),
            "syntactic_failure_kinds": dict(fails.most_common()),
            "semantic_failure_kinds": dict(sem_fails.most_common()),
            # Non-triviality: a constrained decoder that emits one file is not a
            # decoder. Distinct formulas and site-count spread show the masked arm
            # still explores.
            "distinct_formulas_among_parsed": len(formulas),
            "site_count_range": ([min(site_counts), max(site_counts)]
                                 if site_counts else None),
        }

    # Branching factor: how much freedom the mask leaves, averaged over positions.
    rng_b = np.random.default_rng(5)
    branch, singleton = [], 0
    for _ in range(40 if smoke else 200):
        s = GrammarState()
        for _ in range(MAX_CHARS):
            a = allowed(s)
            if not a:
                break
            branch.append(len(a))
            singleton += int(len(a) == 1)
            ch = a[rng_b.integers(len(a))]
            s = step(s, ch)

    payload = {
        "arms": arms,
        "delta_syntactic_parse_rate": (arms["masked"]["syntactic_parse_rate"]
                                      - arms["unmasked"]["syntactic_parse_rate"]),
        "delta_structure_rate": (arms["masked"]["structure_rate"]
                                 - arms["unmasked"]["structure_rate"]),
        "mask_freedom": {
            "vocabulary_size": len(vocab),
            "mean_allowed_chars_per_step": float(np.mean(branch)),
            "median_allowed_chars_per_step": float(np.median(branch)),
            "max_allowed_chars_per_step": int(np.max(branch)),
            "fraction_of_steps_forced": singleton / len(branch),
            "positions_measured": len(branch),
        },
        "claim_scope": (
            "Grammar masking removes the SYNTACTIC failure mode by construction. "
            "It does not address semantic validity: overlapping sites, absurd "
            "lattice parameters, and unphysical stoichiometries parse without "
            "complaint and are present in the masked arm's own output. The "
            "comparison isolates the grammar's contribution by giving both arms "
            "the same character proposal distribution."
        ),
        "judge": "pymatgen.io.cif.CifParser, independent of the emitter",
    }

    return emit(
        "parse_validity" + ("_smoke" if smoke else ""),
        payload,
        config={"n_samples": n, "max_chars": MAX_CHARS, "smoke": smoke},
        seeds=[11, 5],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
