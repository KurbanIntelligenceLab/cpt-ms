"""MP-20 loading and tokenization for the crystal arms (WS-3).

Serializes each structure into the B1ref token order established in WS-2 — the
reference implementation places the complete cell composition in the opening data
tag, which is what makes the composition-dependent constraints determinable near the
start of the file rather than in the trailing atom-site loop.

Tokenization is at the CIF-line level rather than per character. The determinability
and actionability measurements were made against a token stream, and per-character
decoding would multiply sequence length by roughly forty for no change in the
constraint structure. What matters for the projection is that composition tokens
appear before the constraint must be evaluated, which this preserves.

The density dial is primary here, per `preregistration/WS2_dial_assignment.md`: it
has strictly more actionable room than charge on every backbone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

# Elements ordered by atomic number so an index maps to a mass and a plausible
# oxidation state without a lookup table at decode time.
ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
    "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
    "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
    "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu",
]
ATOMIC_MASS = {  # u; only what MP-20 needs, filled from standard values
    "H":1.008,"He":4.003,"Li":6.94,"Be":9.012,"B":10.81,"C":12.011,"N":14.007,"O":15.999,
    "F":18.998,"Ne":20.180,"Na":22.990,"Mg":24.305,"Al":26.982,"Si":28.085,"P":30.974,
    "S":32.06,"Cl":35.45,"Ar":39.948,"K":39.098,"Ca":40.078,"Sc":44.956,"Ti":47.867,
    "V":50.942,"Cr":51.996,"Mn":54.938,"Fe":55.845,"Co":58.933,"Ni":58.693,"Cu":63.546,
    "Zn":65.38,"Ga":69.723,"Ge":72.630,"As":74.922,"Se":78.971,"Br":79.904,"Kr":83.798,
    "Rb":85.468,"Sr":87.62,"Y":88.906,"Zr":91.224,"Nb":92.906,"Mo":95.95,"Tc":98.0,
    "Ru":101.07,"Rh":102.906,"Pd":106.42,"Ag":107.868,"Cd":112.414,"In":114.818,
    "Sn":118.710,"Sb":121.760,"Te":127.60,"I":126.904,"Xe":131.293,"Cs":132.905,
    "Ba":137.327,"La":138.905,"Ce":140.116,"Pr":140.908,"Nd":144.242,"Pm":145.0,
    "Sm":150.36,"Eu":151.964,"Gd":157.25,"Tb":158.925,"Dy":162.500,"Ho":164.930,
    "Er":167.259,"Tm":168.934,"Yb":173.045,"Lu":174.967,"Hf":178.49,"Ta":180.948,
    "W":183.84,"Re":186.207,"Os":190.23,"Ir":192.217,"Pt":195.084,"Au":196.967,
    "Hg":200.592,"Tl":204.38,"Pb":207.2,"Bi":208.980,"Po":209.0,"At":210.0,"Rn":222.0,
    "Fr":223.0,"Ra":226.0,"Ac":227.0,"Th":232.038,"Pa":231.036,"U":238.029,"Np":237.0,
    "Pu":244.0,
}

_CELL = re.compile(r"_cell_length_([abc])\s+([\d.]+)")
_ANG = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d.]+)")
_VOL = re.compile(r"_cell_volume\s+([\d.]+)")
_SUM = re.compile(r"_chemical_formula_sum\s+'([^']+)'")


@dataclass
class Crystal:
    formula: dict[str, int]
    lengths: tuple[float, float, float]
    angles: tuple[float, float, float]
    volume: float

    @property
    def mass(self) -> float:
        return sum(ATOMIC_MASS[e] * n for e, n in self.formula.items())

    @property
    def density(self) -> float:
        """g/cm^3. 1 u/A^3 = 1.66053906660 g/cm^3."""
        return self.mass / self.volume * 1.66053906660

    @property
    def n_atoms(self) -> int:
        return sum(self.formula.values())


def parse_cif(cif: str) -> Crystal | None:
    """Pull the fields the constraints need. Returns None if any is missing."""
    lengths, angles = {}, {}
    for m in _CELL.finditer(cif):
        lengths[m.group(1)] = float(m.group(2))
    for m in _ANG.finditer(cif):
        angles[m.group(1)] = float(m.group(2))
    mv, ms = _VOL.search(cif), _SUM.search(cif)
    if len(lengths) != 3 or len(angles) != 3 or not mv or not ms:
        return None
    formula: dict[str, int] = {}
    for tok in ms.group(1).split():
        m = re.fullmatch(r"([A-Z][a-z]?)(\d*)", tok)
        if not m or m.group(1) not in ATOMIC_MASS:
            return None
        formula[m.group(1)] = formula.get(m.group(1), 0) + int(m.group(2) or 1)
    return Crystal(
        formula=formula,
        lengths=(lengths["a"], lengths["b"], lengths["c"]),
        angles=(angles["alpha"], angles["beta"], angles["gamma"]),
        volume=float(mv.group(1)),
    )


def load_split(split: str, limit: int | None = None) -> list[Crystal]:
    df = pd.read_csv(DATA / f"mp20_{split}.csv")
    if limit:
        df = df.iloc[:limit]
    out = []
    for cif in df["cif"]:
        c = parse_cif(cif)
        if c is not None:
            out.append(c)
    return out


# ---------------------------------------------------------------- tokenization

def _bucket(x: float, lo: float, hi: float, n: int) -> int:
    """Uniform bucket index, clamped to range."""
    if not np.isfinite(x):
        return 0
    return int(np.clip((x - lo) / (hi - lo) * n, 0, n - 1))


class CrystalVocab:
    """B1ref token order: composition first, then cell, then a length marker.

    Composition-first is the ordering the reference implementation uses and the one
    WS-2 measured: it is what makes the density constraint determinable early enough
    for a per-step projection to act on it.
    """

    # Resolution chosen by measurement, not convention. The density dial is the
    # quantity the projection steers, so discretization error must sit well below
    # the steering effects being resolved. Median relative error on MP-20 train:
    # 64/32 -> 4.4%, 128/64 -> 2.4%, 256/128 -> 1.1%, 512/256 -> 0.6%. At 256/128
    # the vocabulary is 500 tokens and the error is small enough not to confound;
    # doubling again buys 0.5 points for a 77% larger vocabulary.
    N_LEN, N_ANG, N_CNT = 256, 128, 16

    def __init__(self, elements: list[str]):
        self.elements = elements
        self.tokens: list[str] = ["<BOS>", "<COMP>", "<END_COMP>", "<CELL>", "<END_CELL>", "<EOS>"]
        self.tokens += [f"el:{e}" for e in elements]
        self.tokens += [f"cnt:{i+1}" for i in range(self.N_CNT)]
        self.tokens += [f"len:{i}" for i in range(self.N_LEN)]
        self.tokens += [f"ang:{i}" for i in range(self.N_ANG)]
        self.index = {t: i for i, t in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, c: Crystal) -> list[int]:
        t = ["<BOS>", "<COMP>"]
        for e, n in sorted(c.formula.items()):
            t += [f"el:{e}", f"cnt:{min(n, self.N_CNT)}"]
        t += ["<END_COMP>", "<CELL>"]
        for L in c.lengths:
            t.append(f"len:{_bucket(L, 2.0, 30.0, self.N_LEN)}")
        for a in c.angles:
            t.append(f"ang:{_bucket(a, 20.0, 160.0, self.N_ANG)}")
        t += ["<END_CELL>", "<EOS>"]
        return [self.index[x] for x in t]

    def decode_density(self, ids: list[int]) -> float | None:
        """Density implied by a token sequence, or None if it is incomplete.

        Uses bucket centres, so this is the density the model can actually target
        given the discretization — not the exact value of the source structure.
        """
        formula, lens, angs, cur_el = {}, [], [], None
        for i in ids:
            tok = self.tokens[i]
            if tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                formula[cur_el] = formula.get(cur_el, 0) + int(tok[4:])
                cur_el = None
            elif tok.startswith("len:"):
                j = int(tok[4:]); lens.append(2.0 + (j + 0.5) * (30.0 - 2.0) / self.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:]); angs.append(20.0 + (j + 0.5) * (160.0 - 20.0) / self.N_ANG)
        if len(lens) != 3 or len(angs) != 3 or not formula:
            return None
        a, b, cc = lens
        al, be, ga = np.radians(angs)
        # Triclinic cell volume.
        v2 = (1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
              + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
        if v2 <= 0:
            return None
        vol = a * b * cc * np.sqrt(v2)
        mass = sum(ATOMIC_MASS[e] * n for e, n in formula.items())
        return mass / vol * 1.66053906660
