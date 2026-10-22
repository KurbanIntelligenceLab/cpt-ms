from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from cpt.crystal.mp20 import CrystalVocab


@dataclass
class CrystalTrainConfig:
    arm: str = "mask_only"
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
    density_target: float = 5.0
    train_limit: int | None = None
    order: str = "cell_first"
    expected_atoms: float = 10.4
    channel: str = "static"
    filter_tol: float = 0.5
    valence_target: float = 0.48


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
        """Legal-token mask at every position of every sequence in the batch."""
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
                        m[self.V.index[f"el:{e}"]] = False
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
