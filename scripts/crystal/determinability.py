"""Measure per-step determinability of both Regime-E families on both backbones.

Emits `results/WS2_family_determinability.json`, the artifact that decides whether
the plan's family (ii) hedge against ionic-scoping risk is instantiable per step
or only at the sequence level.

The two backbones emit the same structure with different token granularity:

  B2  typed fields:      <COMP> el:Li:2 el:O:1 <END_COMP> lat:a:... <END_LATTICE> site:... <EOS>
  B1  CIF characters:    the same content spelled out character by character, in
                         CIF's own field order, where numeric values occupy many
                         tokens and field boundaries are not marked by the vocabulary

Sequence lengths are modeled from the structure of a real CIF rather than
sampled from MP-20, which requires the dataset; the script is written so that
swapping in real MP-20 token streams changes only `build_sequences`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.actionability import profile as actionability_profile
from scripts.crystal.family_ii import family_i_profile, family_ii_profile

SCRIPT = "scripts/crystal/determinability.py"


def build_b2(n_elements: int, n_sites: int) -> list[str]:
    """Structured-field token stream.

    Uses real element symbols. An earlier version emitted placeholders (`el:E0:1`)
    which carry no charge and no mass, so the charge statistic was identically zero
    and the density statistic was zero regardless of the cell — which made every
    lattice token read as non-actionable for density and collapsed the two
    families onto the same figure. Same defect class as the naive B1 fixture.
    """
    elements = ["Na", "Cl", "Mg", "O", "Fe"][:max(n_elements, 1)]
    toks = ["<COMP>"]
    toks += [f"el:{el}:{i+1}" for i, el in enumerate(elements)]
    toks += ["<END_COMP>"]
    for k, v in (("a", 5.0), ("b", 5.0), ("c", 5.0),
                 ("alpha", 90.0), ("beta", 90.0), ("gamma", 90.0)):
        toks.append(f"lat:{k}:{v}")
    toks += ["<END_LATTICE>"]
    toks += [f"site:S{i}" for i in range(n_sites)]
    toks += ["<EOS>"]
    return toks


def build_b1_reference(n_elements: int, n_sites: int) -> list[str]:
    """B1 in the reference model's standardized ordering (CrystaLLM).

    CrystaLLM's prompt construction emits `data_{comp_str}` as the FIRST line,
    where comp_str is the full cell composition with element counts (e.g.
    `data_Zn1Cu1Te1Se1`), elements sorted by electronegativity. Source:
    `bin/make_prompt_file.py::get_prompt`, which builds
    `f"data_{comp_str}\\n{block}\\n_symmetry_space_group_name_H-M {sg}\\n"`.

    This matters for family (i). On the naive CIF ordering the composition is
    recoverable only from the atom-site loop at the end of the file, so charge
    balance is determined within the last few tokens. In the reference ordering
    the complete composition is present in the data_ tag before anything else,
    which is the ordering the published model was actually trained on — so
    matching it is not a deviation from convention, it IS the convention.
    """
    toks: list[str] = []
    elements = ["Na", "Cl", "Mg", "O", "Fe"][:n_elements]
    per = max(1, n_sites // n_elements)

    # data_ tag carrying the full cell composition, character by character.
    comp_str = "".join(f"{el}{per}" for el in elements)
    for ch in f"data_{comp_str}\n":
        toks.append(ch)
    # The composition is complete at this point: emit the marker the parser
    # keys on, so the constraint state closes here rather than at the loop.
    toks.append("<COMP_COMPLETE>")

    for tag, val in [("_cell_length_a", "5.0000"), ("_cell_length_b", "5.0000"),
                     ("_cell_length_c", "5.0000"), ("_cell_angle_alpha", "90.0000"),
                     ("_cell_angle_beta", "90.0000"), ("_cell_angle_gamma", "90.0000")]:
        for ch in f"{tag} {val}\n":
            toks.append(ch)
    toks.append("<END_LATTICE>")

    for ch in "loop_\n_atom_site_type_symbol\n_atom_site_fract_x\n":
        toks.append(ch)
    for i in range(n_sites):
        el = elements[i % n_elements]
        for ch in f"{el} 0.{i:04d}\n":
            toks.append(ch)
    return toks


def build_b1(n_elements: int, n_sites: int) -> list[str]:
    """CIF-character stream, in CIF's own field order.

    The ordering is the point. A real CIF emits the cell parameters (_cell_length_a
    and friends) in its header, but the ATOM SITE LOOP — which carries the
    composition — comes last and is by far the longest section. So on B1 the volume
    is known early but the composition is not known until nearly the end, which is
    the reverse of B2's profile and the reason family (ii) behaves differently on
    the two backbones.

    Character counts per field are representative of real CIF formatting: a
    numeric cell parameter renders as roughly 8 characters, and a site line as
    roughly 40.
    """
    toks: list[str] = []
    toks += list("data_structure\n")
    # CIF header: cell parameters come first.
    for k, v in (("a", 5.0), ("b", 5.0), ("c", 5.0),
                 ("alpha", 90.0), ("beta", 90.0), ("gamma", 90.0)):
        toks += list(f"_cell_length_{k} {v:.4f}\n")
    toks.append("<END_LATTICE>")
    # Atom site loop: composition is spelled out here, at the end.
    toks += list("loop_\n_atom_site_label\n_atom_site_type_symbol\n")
    toks.append("<COMP>")
    # Real element symbols, not placeholders. An earlier version emitted `E0`,
    # `E1` labels, which parse to an empty composition — so the family (i)
    # statistic evaluated to a meaningless zero and the measurement described the
    # fixture rather than the format.
    elements = ["Na", "Cl", "Mg", "O", "Fe"][:max(n_elements, 1)]
    for i in range(n_sites):
        el = elements[i % len(elements)]
        toks += list(f"{el}{i} {el} 0.5000 0.5000 0.5000\n")
    toks.append("<END_COMP>")
    toks.append("<EOS>")
    return toks


def build_sequences() -> dict[str, list[tuple[str, list[str]]]]:
    """Representative sequences at MP-20-like sizes (2-5 elements, 4-20 sites)."""
    cases = [(2, 4), (2, 8), (3, 8), (3, 16), (4, 12), (5, 20)]
    return {
        "B2": [(f"{e}el_{s}site", build_b2(e, s)) for e, s in cases],
        "B1": [(f"{e}el_{s}site", build_b1(e, s)) for e, s in cases],
        "B1ref": [(f"{e}el_{s}site", build_b1_reference(e, s)) for e, s in cases],
    }


def main(smoke: bool = False) -> Path:
    seqs = build_sequences()
    if smoke:
        seqs = {k: v[:2] for k, v in seqs.items()}

    rows = []
    for backbone, cases in seqs.items():
        for name, toks in cases:
            for family, fn in (("i_charge", family_i_profile),
                               ("ii_density", family_ii_profile)):
                p = fn(toks, backbone).as_dict()
                # Determinability and actionability answer different questions and
                # are reported separately: a constraint can be computable from the
                # prefix yet frozen, admitting no projection. CP0's threshold is
                # set on activity within the ACTIONABLE window, since a working
                # layer confined to a few positions reads low when averaged over
                # the whole sequence.
                a = actionability_profile(toks, backbone, family).as_dict()
                p |= {
                    "case": name,
                    "family": family,
                    "actionable_fraction": a["actionable_fraction"],
                    "n_actionable": a["n_actionable"],
                    "first_actionable": a["first_actionable"],
                    "last_actionable": a["last_actionable"],
                }
                rows.append(p)

    def agg(backbone: str, family: str) -> dict:
        sel = [r for r in rows if r["backbone"] == backbone and r["family"] == family]
        fr = [r["usable_fraction"] for r in sel]
        ac = [r["actionable_fraction"] for r in sel]
        return {
            "n_cases": len(sel),
            "min_usable_fraction": min(fr),
            "max_usable_fraction": max(fr),
            "mean_usable_fraction": sum(fr) / len(fr),
            "mean_actionable_fraction": sum(ac) / len(ac),
            "min_actionable_fraction": min(ac),
            "max_actionable_fraction": max(ac),
        }

    summary = {
        f"{b}_{f}": agg(b, f)
        for b in ("B1", "B1ref", "B2")
        for f in ("i_charge", "ii_density")
    }

    # Verdicts are DERIVED from the computed summary, never hand-written. A
    # narrative sentence stored beside its own data can drift from it silently
    # when the computation changes; generating it from the numbers makes that
    # impossible by construction.
    def verdict(backbone: str, family: str) -> dict:
        a = summary[f"{backbone}_{family}"]
        mean = a["mean_usable_fraction"]
        sel = [r for r in rows if r["backbone"] == backbone and r["family"] == family]
        undetermined = [r for r in sel if r["first_determined_position"] is None]
        if undetermined:
            band = "never determined"
        elif mean < 0.05:
            band = "near-terminal"
        elif mean < 0.35:
            band = "late"
        else:
            band = "usable over most of the sequence"
        act = a["mean_actionable_fraction"]
        return {
            "band": band,
            "mean_usable_fraction": mean,
            "mean_actionable_fraction": act,
            "actionable_band": (
                "frozen" if act < 1e-9
                else "narrow" if act < 0.10
                else "moderate" if act < 0.35
                else "wide"
            ),
            "n_cases_never_determined": len(undetermined),
            "determined_positions": sorted(
                {r["first_determined_position"] for r in sel}
            ),
            "statement": (
                f"{backbone} family {family}: determinable {band} "
                f"(mean usable fraction {mean:.3f}), actionable window "
                f"{'frozen' if act < 1e-9 else 'narrow' if act < 0.10 else 'moderate' if act < 0.35 else 'wide'} "
                f"(mean actionable fraction {act:.4f}) over {a['n_cases']} cases; "
                f"{len(undetermined)} case(s) never determined."
            ),
        }

    payload = {
        "rows": rows,
        "summary": summary,
        "verdict": {
            f"{b}_{f}": verdict(b, f)
            for b in ("B1", "B1ref", "B2")
            for f in ("i_charge", "ii_density")
        },
    }

    return emit(
        "WS2_family_determinability" + ("_smoke" if smoke else ""),
        payload,
        config={"cases": [c for c, _ in seqs["B2"]], "smoke": smoke},
        seeds=[],
        script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    print(main(smoke=ap.parse_args().smoke))
