"""MP-20 loading and tokenization for the crystal arms."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / "data"

ELEMENTS = [
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br",
    "Kr","Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te",
    "I","Xe","Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm",
    "Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
    "Fr","Ra","Ac","Th","Pa","U","Np","Pu",
]
VALENCE = {
    "H":1,"He":2,"Li":1,"Be":2,"B":3,"C":4,"N":5,"O":6,"F":7,
    "Ne":8,"Na":1,"Mg":2,"Al":3,"Si":4,"P":5,"S":6,"Cl":7,"Ar":8,
    "K":1,"Ca":2,"Sc":3,"Ti":4,"V":5,"Cr":6,"Mn":7,"Fe":8,"Co":9,
    "Ni":10,"Cu":11,"Zn":12,"Ga":13,"Ge":14,"As":15,"Se":16,"Br":17,"Kr":18,
    "Rb":1,"Sr":2,"Y":3,"Zr":4,"Nb":5,"Mo":6,"Tc":7,"Ru":8,"Rh":9,
    "Pd":10,"Ag":11,"Cd":12,"In":13,"Sn":14,"Sb":15,"Te":16,"I":17,"Xe":18,
    "Cs":1,"Ba":2,"La":3,"Ce":4,"Pr":5,"Nd":6,"Pm":7,"Sm":8,"Eu":9,
    "Gd":10,"Tb":11,"Dy":12,"Ho":13,"Er":14,"Tm":15,"Yb":16,"Lu":17,"Hf":18,
    "Ta":19,"W":20,"Re":21,"Os":22,"Ir":23,"Pt":24,"Au":25,"Hg":26,"Tl":27,
    "Pb":28,"Bi":29,"Po":30,"At":31,"Rn":32,"Fr":1,"Ra":2,"Ac":3,"Th":4,
    "Pa":5,"U":6,"Np":7,"Pu":8,
}


ATOMIC_MASS = {
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


def _bucket(x: float, lo: float, hi: float, n: int) -> int:
    if not np.isfinite(x):
        return 0
    return int(np.clip((x - lo) / (hi - lo) * n, 0, n - 1))


class CrystalVocab:
    """B1ref token order: composition first, then cell, then a length marker."""

    N_LEN, N_ANG, N_CNT = 512, 256, 16

    ANG_LO, ANG_HI = 5.0, 175.0
    LEN_LO, LEN_HI = 2.0, 46.0

    def __init__(self, elements: list[str], order: str = "cell_first"):
        if order not in ("cell_first", "comp_first"):
            raise ValueError(f"unknown order {order!r}")
        self.order = order
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
        """Serialize. `order` selects composition-first (B1ref) or cell-first."""
        comp = []
        for e, n in sorted(c.formula.items()):
            comp += [f"el:{e}", f"cnt:{min(n, self.N_CNT)}"]
        cell = []
        for L in c.lengths:
            cell.append(f"len:{_bucket(L, self.LEN_LO, self.LEN_HI, self.N_LEN)}")
        for a in c.angles:
            cell.append(f"ang:{_bucket(a, self.ANG_LO, self.ANG_HI, self.N_ANG)}")
        if self.order == "cell_first":
            t = ["<BOS>", "<CELL>"] + cell + ["<END_CELL>", "<COMP>"] + comp + ["<END_COMP>", "<EOS>"]
        else:
            t = ["<BOS>", "<COMP>"] + comp + ["<END_COMP>", "<CELL>"] + cell + ["<END_CELL>", "<EOS>"]
        return [self.index[x] for x in t]

    @staticmethod
    def cell_volume(lengths, angles) -> float | None:
        """Triclinic cell volume in A^3, or None if the angles are incoherent."""
        a, b, cc = lengths
        al, be, ga = np.radians(angles)
        v2 = (1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
              + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
        if v2 <= 0:
            return None
        return float(a * b * cc * np.sqrt(v2))

    def decode_density(self, ids: list[int]) -> float | None:
        """Density implied by a token sequence, or None if it is incomplete."""
        formula, lens, angs, cur_el = {}, [], [], None
        for i in ids:
            tok = self.tokens[i]
            if tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                formula[cur_el] = formula.get(cur_el, 0) + int(tok[4:])
                cur_el = None
            elif tok.startswith("len:"):
                j = int(tok[4:]); lens.append(self.LEN_LO + (j + 0.5) * (self.LEN_HI - self.LEN_LO) / self.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:]); angs.append(self.ANG_LO + (j + 0.5) * (self.ANG_HI - self.ANG_LO) / self.N_ANG)
        if len(lens) != 3 or len(angs) != 3 or not formula:
            return None
        a, b, cc = lens
        al, be, ga = np.radians(angs)
        v2 = (1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
              + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
        if v2 <= 0:
            return None
        vol = a * b * cc * np.sqrt(v2)
        mass = sum(ATOMIC_MASS[e] * n for e, n in formula.items())
        return mass / vol * 1.66053906660

    def decode_weighted_density(self, ids: list[int],
                                table: dict[str, float]) -> float | None:
        """Density of any additive per-element quantity, in units of table-value per A^3."""
        formula, lens, angs, cur_el = {}, [], [], None
        for i in ids:
            tok = self.tokens[i]
            if tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                formula[cur_el] = formula.get(cur_el, 0) + int(tok[4:])
                cur_el = None
            elif tok.startswith("len:"):
                j = int(tok[4:]); lens.append(self.LEN_LO + (j + 0.5) * (self.LEN_HI - self.LEN_LO) / self.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:]); angs.append(self.ANG_LO + (j + 0.5) * (self.ANG_HI - self.ANG_LO) / self.N_ANG)
        if len(lens) != 3 or len(angs) != 3 or not formula:
            return None
        a, b, cc = lens
        al, be, ga = np.radians(angs)
        v2 = (1 - np.cos(al) ** 2 - np.cos(be) ** 2 - np.cos(ga) ** 2
              + 2 * np.cos(al) * np.cos(be) * np.cos(ga))
        if v2 <= 0:
            return None
        vol = a * b * cc * np.sqrt(v2)
        return sum(table.get(e, 0.0) * n for e, n in formula.items()) / vol
