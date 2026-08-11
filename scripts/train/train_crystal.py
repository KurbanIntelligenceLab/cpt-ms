"""Crystal A2: the three arms on MP-20 with a real backbone (WS-3).

The synthetic task established the mechanism against exact enumerable targets. This
carries it to MP-20, where the targets are approximate and the backbone is a real
model trained on real structures. Two conventions carry over unchanged, per the
review:

  TWO READINGS FOR A3. The backbone reading applies the mask only and reports what
  the network internalized; the deployed reading applies the same projection the
  model trained with and is the system anyone would run. Reporting either alone
  misleads in opposite directions, and the gap between them IS the internalization
  result.

  PROBABILITY-WEIGHTED INFEASIBLE-STEP FRACTION. An unweighted mean over a fixed
  evaluation set is model-independent and therefore not comparable across arms.

What necessarily differs from the synthetic task: the sequence space is not
enumerable, so every distributional quantity is estimated from samples rather than
computed exactly. Sample sizes are stated with each figure and the same draw count
is used for every arm.

The steered quantity is DENSITY, primary on all backbones per
`preregistration/WS2_dial_assignment.md`.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from collections import Counter

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.common.artifacts import emit
from scripts.crystal.mp20 import ATOMIC_MASS, ELEMENTS, CrystalVocab, load_split
from scripts.layer.projection import project

SCRIPT = "scripts/train/train_crystal.py"


@dataclass
class CrystalTrainConfig:
    arm: str = "A2"
    d_model: int = 256
    n_layer: int = 6
    n_head: int = 8
    steps: int = 6000
    batch: int = 128
    lr: float = 3e-4
    warmup: int = 300
    seed: int = 0
    eval_every: int = 1000
    n_eval_samples: int = 4096
    density_target: float = 5.0     # g/cm^3, set from the data below the median
    train_limit: int | None = None
    order: str = "cell_first"      # density needs the volume before the composition
    expected_atoms: float = 10.4    # mean atoms per MP-20 structure
    channel: str = "static"   # "static" = formulation 2; "prefix" = formulation 3
    # A4: conditional-training baseline. Train the masked model on the SUBSET of MP-20
    # within `filter_tol` of the target instead of applying any constraint at decode
    # time. This is the "just retrain on filtered data" objection, and it is the
    # baseline a reviewer reaches for first, so it gets the same budget as A2/A3 and
    # its own arm rather than a footnote. Steps are held equal, which is deliberately
    # generous: the filtered set is smaller, so A4 sees each structure more often.
    filter_tol: float = 0.5


class CrystalDecoder(nn.Module):
    def __init__(self, vocab: int, max_len: int, cfg: CrystalTrainConfig):
        super().__init__()
        self.tok = nn.Embedding(vocab, cfg.d_model)
        self.pos = nn.Embedding(max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_head, 4 * cfg.d_model,
            batch_first=True, norm_first=True, dropout=0.0)
        self.blocks = nn.TransformerEncoder(layer, cfg.n_layer)
        self.head = nn.Linear(cfg.d_model, vocab)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        m = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        return self.head(self.blocks(h, mask=m, is_causal=True))


class GrammarMask:
    """Regime S for the crystal serialization: which tokens are legal next.

    Prefix-closed by construction — the legal set depends only on which block is
    open and what has been emitted — which is what makes the mask trainable without
    a straight-through estimator, exactly as on the synthetic task.
    """

    def __init__(self, V: CrystalVocab, max_elements: int = 6, order: str = "cell_first"):
        self.V = V
        self.order = order
        self.max_elements = max_elements
        self.el = np.array([t.startswith("el:") for t in V.tokens])
        self.cnt = np.array([t.startswith("cnt:") for t in V.tokens])
        self.len_ = np.array([t.startswith("len:") for t in V.tokens])
        self.ang = np.array([t.startswith("ang:") for t in V.tokens])
        self.i = {t: V.index[t] for t in
                  ("<BOS>","<COMP>","<END_COMP>","<CELL>","<END_CELL>","<EOS>")}

    def masks_for(self, ids: np.ndarray) -> np.ndarray:
        """Legal-token mask at every position of every sequence in the batch.

        `out[r, t]` is the set of tokens legal AT position t, i.e. after consuming
        ids[r, :t]. So the state must be advanced by token t-1 before writing row t.
        Writing the mask before consuming the current token instead describes what
        may follow position t-1, which rejects the data's own sequences.
        """
        n, L = ids.shape
        out = np.zeros((n, L, len(self.V)), dtype=bool)
        for r in range(n):
            state, n_el, n_len, n_ang, seen = "bos", 0, 0, 0, set()
            for t in range(L):
                if t > 0:
                    tok = self.V.tokens[ids[r, t - 1]]
                    if tok == "<COMP>":
                        state = "want_el"
                    elif tok.startswith("el:"):
                        seen.add(tok[3:]); state = "want_cnt"
                    elif tok.startswith("cnt:"):
                        n_el += 1; state = "want_el"
                    elif tok == "<END_COMP>":
                        state = "eos" if self.order == "cell_first" else "want_cell"
                    elif tok == "<CELL>":
                        state = "want_len"
                    elif tok.startswith("len:"):
                        n_len += 1; state = "want_len" if n_len < 3 else "want_ang"
                    elif tok.startswith("ang:"):
                        n_ang += 1; state = "want_ang" if n_ang < 3 else "want_end_cell"
                    elif tok == "<END_CELL>":
                        state = "want_comp" if self.order == "cell_first" else "eos"
                m = out[r, t]
                if state == "bos":
                    m[self.i["<CELL>" if self.order == "cell_first" else "<COMP>"]] = True
                elif state == "want_el":
                    m[self.el] = True
                    for e in seen:
                        m[self.V.index[f"el:{e}"]] = False       # no repeats
                    if n_el >= 1:
                        m[self.i["<END_COMP>"]] = True
                    if n_el >= self.max_elements:
                        m[self.el] = False
                        m[self.i["<END_COMP>"]] = True
                elif state == "want_cnt":
                    m[self.cnt] = True
                elif state == "want_cell":
                    m[self.i["<CELL>"]] = True
                elif state == "want_comp":
                    m[self.i["<COMP>"]] = True
                elif state == "want_len":
                    m[self.len_] = True
                elif state == "want_ang":
                    m[self.ang] = True
                elif state == "want_end_cell":
                    m[self.i["<END_CELL>"]] = True
                else:
                    m[self.i["<EOS>"]] = True
        return out


def mass_channel(ids: np.ndarray, V: CrystalVocab) -> np.ndarray:
    """Per-token mass contribution GIVEN the prefix. Shape (n, vocab).

    The constraint channel is prefix-dependent here, which the synthetic task did
    not require. An element token `el:E` contributes nothing on its own — the mass
    it adds is decided by the count token that follows, as ATOMIC_MASS[E] * n. So
    at a count position with pending element E the channel value of `cnt:n` is
    ATOMIC_MASS[E] * n, and at an element position the channel is the element's own
    mass as a proxy for the mass about to be committed.

    A fixed per-token channel makes every count position carry mass zero, so the
    attainable range collapses to {0} and any nonzero budget is infeasible there.
    That is a property of the serialization, not of the constraint.
    """
    n = ids.shape[0]
    out = np.zeros((n, len(V)), dtype=np.float32)
    cnt_idx = [(i, int(t[4:])) for i, t in enumerate(V.tokens) if t.startswith("cnt:")]
    el_mass = np.array([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                        for t in V.tokens], dtype=np.float32)
    for r in range(n):
        pending = None
        for t in range(ids.shape[1]):
            tok = V.tokens[ids[r, t]]
            if tok.startswith("el:"):
                pending = tok[3:]
            elif tok.startswith("cnt:"):
                pending = None
        if pending is not None:
            m = ATOMIC_MASS.get(pending, 0.0)
            for i, k in cnt_idx:
                out[r, i] = m * k
        else:
            out[r] = el_mass
    return out


def _element_position(ids: np.ndarray, V: CrystalVocab) -> np.ndarray:
    """True where the NEXT token is an element symbol.

    With the static mass channel these are the only positions at which the
    constraint is actionable: everywhere else every legal token has channel value
    zero, so no substitution can move the constrained quantity.
    """
    n = ids.shape[0]
    out = np.zeros(n, dtype=bool)
    for r in range(n):
        state = None
        for t in range(ids.shape[1]):
            tok = V.tokens[ids[r, t]]
            if tok == "<COMP>":
                state = "el"
            elif tok.startswith("el:"):
                state = "cnt"
            elif tok.startswith("cnt:"):
                state = "el"
            elif tok in ("<END_COMP>", "<EOS>"):
                state = None
        out[r] = state == "el"
    return out


def _mass_budget(ids: np.ndarray, V: CrystalVocab, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Remaining mass to reach the target density, and remaining element slots.

    Returns (budget_u, slots). Budget is NaN until the cell block closes, because
    until then the volume — and therefore the total mass a target density implies —
    is not determined. That is the actionability boundary WS-2 measured, appearing
    here as an explicit precondition rather than as a silent wrong answer.

    The denominator is remaining ATOMS, not remaining element tokens. The static
    channel gives an element token its ATOMIC mass — a per-atom quantity — while the
    mass it actually commits is that atomic mass times the count that follows. So
    dividing the budget by element slots demands, on MP-20, about three times the
    mass the data's own next element carries (median 177u demanded against 59u
    actual), and a third of positions demand more than the heaviest available
    element. Dividing by remaining atoms puts the target on the same per-atom footing
    as the channel.

    The atom count cannot be known exactly (the model decides it), so it is estimated
    from the data mean. An estimate is sound here because the budget is re-derived at
    every step from the mass actually emitted, so an over- or under-estimate corrects
    itself on the next token rather than accumulating.
    """
    n, L = ids.shape
    budget = np.full(n, np.nan, dtype=np.float64)
    slots = np.zeros(n, dtype=np.float64)
    for r in range(n):
        lens, angs, mass, n_el, cell_closed = [], [], 0.0, 0, False
        cur_el = None
        for t in range(L):
            tok = V.tokens[ids[r, t]]
            if tok.startswith("len:"):
                j = int(tok[4:]); lens.append(V.LEN_LO + (j + 0.5) * (V.LEN_HI - V.LEN_LO) / V.N_LEN)
            elif tok.startswith("ang:"):
                j = int(tok[4:]); angs.append(V.ANG_LO + (j + 0.5) * (V.ANG_HI - V.ANG_LO) / V.N_ANG)
            elif tok == "<END_CELL>":
                cell_closed = len(lens) == 3 and len(angs) == 3
            elif tok.startswith("el:"):
                cur_el = tok[3:]
            elif tok.startswith("cnt:") and cur_el:
                k = int(tok[4:])
                mass += ATOMIC_MASS.get(cur_el, 0.0) * k
                n_el += k          # atoms committed, not element tokens
                cur_el = None
        if cell_closed:
            vol = V.cell_volume(lens, angs)
            if vol:
                total = cfg.density_target * vol / 1.66053906660
                budget[r] = max(total - mass, 0.0)
                slots[r] = max(cfg.expected_atoms - n_el, 1)
    return budget, slots


def pad_batch(seqs: list[list[int]], pad: int) -> np.ndarray:
    L = max(len(s) for s in seqs)
    a = np.full((len(seqs), L), pad, dtype=np.int64)
    for i, s in enumerate(seqs):
        a[i, :len(s)] = s
    return a


@torch.no_grad()
def sample(model, V, gm, cfg, device, n, *, use_mask: bool, project_at_inference: bool,
           max_len: int, rng) -> tuple[list[list[int]], dict]:
    """Autoregressive generation. Returns token sequences and diagnostics."""
    model.eval()
    ids = torch.full((n, 1), V.index["<BOS>"], dtype=torch.long, device=device)
    done = torch.zeros(n, dtype=torch.bool, device=device)
    n_infeasible = torch.zeros(n, device=device)
    n_projected = torch.zeros(n, device=device)
    n_degenerate = torch.zeros(n, device=device)
    eos = V.index["<EOS>"]
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=device, dtype=torch.float32)

    for _ in range(max_len - 1):
        logits = model(ids)[:, -1, :]
        if use_mask:
            # The mask needed is the one for the token about to be APPENDED, i.e.
            # at position len(ids). masks_for returns the mask AT each existing
            # position, so index -1 gives the mask for the token already emitted and
            # every token is generated twice. Pad by one column and take the last.
            cur = ids.cpu().numpy()
            padded = np.concatenate([cur, np.zeros((cur.shape[0], 1), dtype=cur.dtype)], axis=1)
            mk = torch.from_numpy(gm.masks_for(padded)[:, -1, :]).to(device)
            # A finished row has no legal token; let it emit EOS so the batch is
            # rectangular without corrupting the mask semantics for live rows.
            mk[done] = False
            mk[done, eos] = True
            logits = logits.masked_fill(~mk, float("-inf"))

        if project_at_inference:
            # RUNNING MASS BUDGET, not a fixed mass per token.
            #
            # Density is mass/volume, so a target density fixes a total mass budget
            # rho*V/1.66054 once the cell is known. Cell-first ordering makes V
            # determined before any element token, so the per-step target is the
            # remaining mass spread over the remaining element slots -- the same
            # running-budget form the synthetic task validated.
            #
            # Targeting a constant density value on the mass channel instead (the
            # first version of this code) asks for 5 u per token, lighter than
            # lithium, and drives the composition toward the lightest elements.
            B, Vsz = logits.shape
            live = mk if use_mask else torch.ones_like(logits, dtype=torch.bool)
            budget, slots = _mass_budget(ids.cpu().numpy(), V, cfg)
            budget = torch.from_numpy(budget).to(device).float()
            slots = torch.from_numpy(slots).to(device).float()
            # ACTIONABLE POSITIONS ONLY, per preregistration/CP0_actionable_window.md.
            # With the static channel, only element positions carry nonzero mass: at
            # a count or structural position every legal token has channel value 0,
            # so the attainable range is {0} and any nonzero budget is infeasible by
            # construction. Counting those as infeasible steps measures the
            # serialization, not the constraint -- it was inflating the reported
            # infeasible fraction from 0.331 to 0.52 without the projection ever
            # having had a decision to make there.
            el_next = torch.from_numpy(_element_position(ids.cpu().numpy(), V)).to(device)
            active = (slots > 0) & torch.isfinite(budget) & ~done & el_next
            if cfg.channel == "prefix":
                ch = torch.from_numpy(mass_channel(ids.cpu().numpy(), V)).to(device)
            else:
                ch = mass_t.unsqueeze(0).expand(B, Vsz)
            A = ch.unsqueeze(1).contiguous()
            attain = torch.where(live, A[:, 0, :], torch.full_like(logits, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            per_step = torch.where(active, budget / slots.clamp_min(1), lo)
            feas = active & (per_step >= lo - 1e-6) & (per_step <= hi + 1e-6)
            b = per_step.clamp(min=lo, max=hi).unsqueeze(-1)
            res = project(logits, A, b, live)
            q = torch.where(res.converged.unsqueeze(-1) & feas.unsqueeze(-1),
                            res.q, torch.softmax(logits, -1))
            n_infeasible += (active & ~feas).float()
            n_projected += feas.float()
        else:
            q = torch.softmax(logits, dim=-1)

        # Sanitize before sampling, and fall back on a row that has no mass left.
        #
        # `clamp_min(0).nan_to_num()` was not enough and it failed intermittently:
        # clamp leaves NaN as NaN, nan_to_num then maps it to 0, so an all-NaN row
        # becomes an all-ZERO row and multinomial raises a device-side assert. Two
        # runs of twenty died this way, on different seeds at different targets --
        # seed-dependent rather than target-dependent, which is the signature of a
        # rare degenerate solve rather than of an unreachable target.
        #
        # A row can lose all mass legitimately: where the projection is asked for a
        # near-degenerate distribution the solved q can underflow across every legal
        # token. The masked softmax is always a valid distribution at those
        # positions, so it is the correct fallback, and rows falling back are counted
        # rather than silently repaired.
        q = torch.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
        bad = q.sum(-1) <= 0
        if bool(bad.any()):
            safe = torch.softmax(logits, dim=-1)
            safe = torch.nan_to_num(safe, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0)
            # If even the masked softmax is degenerate the row is finished; emit EOS.
            still_bad = bad & (safe.sum(-1) <= 0)
            q = torch.where(bad.unsqueeze(-1), safe, q)
            if bool(still_bad.any()):
                onehot = torch.zeros_like(q)
                onehot[:, eos] = 1.0
                q = torch.where(still_bad.unsqueeze(-1), onehot, q)
            n_degenerate += bad.float()
        nxt = torch.multinomial(q, 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, eos), nxt)
        ids = torch.cat([ids, nxt.unsqueeze(-1)], dim=1)
        done = done | (nxt == eos)
        if bool(done.all()):
            break

    model.train()
    seqs = [[int(t) for t in row if True] for row in ids.cpu().numpy()]
    tot = (n_infeasible + n_projected).clamp_min(1)
    return seqs, {
        "per_step_infeasible_fraction": float((n_infeasible / tot).mean()),
        "mean_steps_projected": float(n_projected.mean()),
        # Rows that lost all probability mass and fell back to the masked softmax.
        # Reported rather than hidden: a nonzero rate means the solve is degenerating
        # somewhere and the fallback is doing work the projection should be doing.
        "degenerate_row_rate": float((n_degenerate / tot).mean()),
        "sequences_with_any_degenerate_step": float((n_degenerate > 0).float().mean()),
    }


def structural_metrics(seqs, V, gm, cfg, train_formulas=None,
                       full_element_freq=None) -> dict:
    """Validity, density, and DISTRIBUTIONAL COVERAGE of generated sequences.

    Accuracy at the target is not sufficient to compare arms. A model trained only on
    structures near the target can hit it precisely while generating a narrow slice of
    chemistry; a steered model reaches the target from the full distribution. Coverage
    is therefore measured alongside accuracy, and the two are reported together:

      distinct_elements / element_entropy  -- chemical breadth of the samples
      unique_formula_fraction             -- collapse onto repeated compositions
      novel_formula_fraction              -- formulas absent from the TRAINING set
      element_tv_vs_full_data             -- SHIFT from the full data's chemistry

    The last is NOT a diversity measure and must not be read as one. Any arm hitting a
    target away from the data mean must shift its chemistry to get there, so both arms
    have a large TV by construction, and the two can order either way: measured across
    targets 3.0 and 5.0 the conditional arm had the larger TV at one and the smaller at
    the other. What TV does establish is reproduction: the conditional arm's TV tracks
    the TV between its own filtered training slice and the full data to within 0.004
    (0.3919 vs 0.3836 at target 3.0; 0.2028 vs 0.1988 at 5.0), which is what
    reproducing a slice looks like. Diversity is carried by the unique- and
    novel-formula fractions, which do not confound the two.

    A high-accuracy, low-coverage arm and a lower-accuracy, high-coverage arm are
    different propositions, and a single error figure cannot distinguish them.
    """
    dens, valid, complete = [], 0, 0
    el_counter: Counter = Counter()
    formulas: list[tuple] = []
    for s in seqs:
        toks = [V.tokens[i] for i in s]
        ok = (toks.count("<COMP>") == 1 and toks.count("<END_COMP>") == 1
              and toks.count("<CELL>") == 1 and toks.count("<END_CELL>") == 1
              and sum(t.startswith("len:") for t in toks) == 3
              and sum(t.startswith("ang:") for t in toks) == 3
              and sum(t.startswith("el:") for t in toks)
              == sum(t.startswith("cnt:") for t in toks)
              and sum(t.startswith("el:") for t in toks) >= 1)
        valid += ok
        if ok:
            f = {}
            els = [t[3:] for t in toks if t.startswith("el:")]
            cnts = [int(t[4:]) for t in toks if t.startswith("cnt:")]
            for e, k in zip(els, cnts):
                f[e] = f.get(e, 0) + k
                el_counter[e] += k
            formulas.append(tuple(sorted(f.items())))
        d = V.decode_density(s) if ok else None
        if d is not None and np.isfinite(d) and 0 < d < 100:
            complete += 1
            dens.append(d)
    d = np.asarray(dens)
    return {
        "n_samples": len(seqs),
        "structural_validity_rate": valid / len(seqs),
        "density_computable_rate": complete / len(seqs),
        "mean_density": float(d.mean()) if len(d) else None,
        "median_density": float(np.median(d)) if len(d) else None,
        "density_sd": float(d.std()) if len(d) else None,
        "density_error_vs_target": (float(d.mean() - cfg.density_target)
                                    if len(d) else None),
        **_coverage(el_counter, formulas, train_formulas, full_element_freq),
    }


def _coverage(el_counter, formulas, train_formulas, full_element_freq) -> dict:
    """Chemical breadth of a sample, and its departure from the reference chemistry."""
    tot = sum(el_counter.values())
    if not tot:
        return {"distinct_elements": 0, "element_entropy_nats": None,
                "element_tv_vs_full_data": None, "unique_formula_fraction": None,
                "novel_formula_fraction": None}
    freq = {e: k / tot for e, k in el_counter.items()}
    ent = -sum(p * np.log(p) for p in freq.values() if p > 0)
    tv = None
    if full_element_freq:
        keys = set(freq) | set(full_element_freq)
        tv = 0.5 * sum(abs(freq.get(k, 0.0) - full_element_freq.get(k, 0.0)) for k in keys)
    uniq = len(set(formulas)) / len(formulas) if formulas else None
    novel = None
    if train_formulas is not None and formulas:
        novel = sum(f not in train_formulas for f in formulas) / len(formulas)
    return {
        "distinct_elements": len(freq),
        "element_entropy_nats": float(ent),
        "element_tv_vs_full_data": (float(tv) if tv is not None else None),
        "unique_formula_fraction": (float(uniq) if uniq is not None else None),
        "novel_formula_fraction": (float(novel) if novel is not None else None),
    }


def main(smoke: bool = False, arm: str = "A2", steps: int | None = None) -> Path:
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CrystalTrainConfig(arm=arm, steps=steps or (300 if smoke else 6000))
    if smoke:
        cfg.eval_every, cfg.n_eval_samples, cfg.train_limit = 150, 512, 2000

    train = load_split("train", limit=cfg.train_limit)
    # A4 conditional training: restrict the TRAINING SET to structures near the target
    # and apply no decode-time constraint. The full-data density statistics are kept
    # separately, because every arm must be scored against the same reference — scoring
    # A4 against its own filtered mean would flatter it by construction.
    full_train = list(train)
    full_density = np.array([c.density for c in train])
    n_full = len(train)
    if cfg.arm == "A4":
        train = [c for c in train
                 if abs(c.density - cfg.density_target) <= cfg.filter_tol]
        # Floor of 20, not 100. An earlier floor of 100 was a guess, and it was about
        # to carry a claim that A4 "starves" at sparse targets -- asserting a failure
        # without running it. The sparse regime is now measured; below 20 structures
        # there is genuinely nothing to train on and the refusal is itself the result.
        if len(train) < 20:
            raise ValueError(
                f"A4 filter at target {cfg.density_target} +/- {cfg.filter_tol} left "
                f"{len(train)} structures of {n_full}; too few to train on"
            )
    V = CrystalVocab(ELEMENTS, order=cfg.order)
    gm = GrammarMask(V, order=cfg.order)
    enc = [V.encode(c) for c in train]
    max_len = max(len(e) for e in enc)
    data = pad_batch(enc, V.index["<EOS>"])
    # Scoring reference is ALWAYS the full data, never the arm's own training subset:
    # A4 trains on a filtered slice, and scoring it against that slice's mean would
    # flatter it by construction. For A1/A2/A3 the two are identical.
    data_density = full_density
    # Reference chemistry for coverage metrics: element frequencies over the FULL split
    # so every arm is compared against the same distribution, and the formula set of
    # THIS arm's training data so novelty means "not in what this model saw".
    _full_el: Counter = Counter()
    for c in full_train:
        for e, k in c.formula.items():
            _full_el[e] += k
    _tot_el = sum(_full_el.values())
    full_element_freq = {e: k / _tot_el for e, k in _full_el.items()}
    train_formulas = {tuple(sorted(c.formula.items())) for c in train}
    COV = dict(train_formulas=train_formulas, full_element_freq=full_element_freq)

    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = CrystalDecoder(len(V), max_len + 2, cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / cfg.warmup))
    mass_t = torch.tensor([ATOMIC_MASS.get(t[3:], 0.0) if t.startswith("el:") else 0.0
                           for t in V.tokens], device=dev, dtype=torch.float32)

    # Initialization baseline for the A3 arm. The decisive quantity is whether the
    # probability-weighted infeasible-step fraction FALLS through training, as it did
    # 19.9x on the synthetic task. Measuring it only at convergence cannot answer
    # that, so it is measured at step 0 against the identical untrained weights.
    init_ref = None
    if cfg.arm == "A3":
        s0, d0 = sample(model, V, gm, cfg, dev, cfg.n_eval_samples, use_mask=True,
                        project_at_inference=True, max_len=max_len + 2, rng=rng)
        init_ref = structural_metrics(s0, V, gm, cfg, **COV) | d0 | {"step": 0}
        print(f"  step     0 (init) infeasible {d0['per_step_infeasible_fraction']:.4f} "
              f"density {init_ref['mean_density'] if init_ref['mean_density'] else float('nan'):.3f}",
              flush=True)

    history, t0 = [], time.time()
    for step in range(1, cfg.steps + 1):
        idx = rng.integers(0, len(data), cfg.batch)
        x = torch.from_numpy(data[idx]).to(dev)
        inp, tgt = x[:, :-1], x[:, 1:]
        logits = model(inp)

        if cfg.arm != "A1":
            mk = torch.from_numpy(gm.masks_for(x.cpu().numpy())[:, 1:, :]).to(dev)
            logits = logits.masked_fill(~mk, float("-inf"))

        if cfg.arm == "A3":
            B, L, Vs = logits.shape
            flat = logits.reshape(B * L, Vs)
            mkf = mk.reshape(B * L, Vs)
            # Same running-budget target as inference, computed per prefix. Training
            # and deployment must solve the identical problem, or the layer is
            # trained against a constraint the deployed system never applies.
            bud = np.empty((B, L)); slt = np.empty((B, L))
            chan = np.empty((B, L, Vs), dtype=np.float32)
            xc = x.cpu().numpy()
            for t in range(L):
                bb_, ss_ = _mass_budget(xc[:, :t + 1], V, cfg)
                bud[:, t], slt[:, t] = bb_, ss_
                if cfg.channel == "prefix":
                    chan[:, t] = mass_channel(xc[:, :t + 1], V)
            if cfg.channel == "prefix":
                A = torch.from_numpy(chan).to(dev).reshape(B * L, 1, Vs).contiguous()
            else:
                A = mass_t.view(1, 1, Vs).expand(B * L, 1, Vs).contiguous()
            attain = torch.where(mkf, A[:, 0, :], torch.full_like(flat, float("nan")))
            lo = torch.nan_to_num(attain, nan=float("inf")).min(-1).values
            hi = torch.nan_to_num(attain, nan=float("-inf")).max(-1).values
            bt = torch.from_numpy(bud).to(dev).float().reshape(-1)
            st = torch.from_numpy(slt).to(dev).float().reshape(-1)
            elp = np.empty((B, L), dtype=bool)
            for t in range(L):
                elp[:, t] = _element_position(xc[:, :t + 1], V)
            ept = torch.from_numpy(elp).to(dev).reshape(-1)
            act = (st > 0) & torch.isfinite(bt) & ept
            b = torch.where(act, bt / st.clamp_min(1), lo).clamp(min=lo, max=hi).unsqueeze(-1)
            res = project(flat, A, b, mkf)
            q = torch.where((res.converged & act).unsqueeze(-1), res.q,
                            torch.softmax(flat, -1))
            logp = torch.log(q.clamp_min(1e-30)).reshape(B, L, Vs)
        else:
            logp = torch.log_softmax(logits, dim=-1)

        loss = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if step % cfg.eval_every == 0 or step == cfg.steps:
            seqs, diag = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                                use_mask=(cfg.arm != "A1"), project_at_inference=False,
                                max_len=max_len + 2, rng=rng)
            m = structural_metrics(seqs, V, gm, cfg, **COV)
            m |= {"step": step, "loss": float(loss.item()),
                  "elapsed_s": round(time.time() - t0, 1)} | diag
            if cfg.arm == "A3":
                sd, dg = sample(model, V, gm, cfg, dev, cfg.n_eval_samples,
                                use_mask=True, project_at_inference=True,
                                max_len=max_len + 2, rng=rng)
                m["deployed"] = structural_metrics(sd, V, gm, cfg, **COV) | dg
                # COMPENSATION TEST. If the backbone biases its unaided predictions
                # to offset a projection that reliably pulls one way, the offset
                # should scale with the pull. Both are recorded per target so the
                # sweep can regress one on the other.
                if m["mean_density"] is not None and m["deployed"]["mean_density"] is not None:
                    m["projection_pull"] = m["mean_density"] - m["deployed"]["mean_density"]
                    m["backbone_offset_vs_data"] = m["mean_density"] - float(data_density.mean())
            history.append(m)
            print(f"  step {step:5d} loss {m['loss']:.4f} valid {m['structural_validity_rate']:.4f} "
                  f"density {m['mean_density'] if m['mean_density'] else float('nan'):.3f}",
                  flush=True)

    payload = {
        "arm": cfg.arm,
        "train_config": asdict(cfg),
        "device": str(dev),
        "gpu": (torch.cuda.get_device_name(0) if dev.type == "cuda" else None),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "vocab_size": len(V),
        "n_train_structures": len(train),
        "max_len": max_len,
        "training_set": {
            "n_structures": len(train), "n_full_split": n_full,
            "fraction_of_split": len(train) / n_full,
            "filtered": cfg.arm == "A4",
            "filter_tol": (cfg.filter_tol if cfg.arm == "A4" else None),
            "train_density_mean": float(np.mean([c.density for c in train])),
        },
        "data_density": {
            "mean": float(data_density.mean()), "median": float(np.median(data_density)),
            "sd": float(data_density.std()),
        },
        "history": history,
        "final": history[-1],
        "init_reference": init_ref,
        "wall_clock_s": round(time.time() - t0, 1),
    }
    return emit(
        f"WS3_crystal_{cfg.arm}" + ("_smoke" if smoke else ""),
        payload,
        config={"arm": cfg.arm, "steps": cfg.steps, "smoke": smoke},
        seeds=[cfg.seed], script=SCRIPT,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", default="A2", choices=["A1", "A2", "A3", "A4"])
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    print(main(smoke=a.smoke, arm=a.arm, steps=a.steps))
