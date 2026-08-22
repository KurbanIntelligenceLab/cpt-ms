from __future__ import annotations

from cpt.common.artifacts import emit

SCRIPT = "src/cpt/related/external_conditioning_survey.py"


def build_payload() -> dict:
    candidates = {
        "mattergen": {
            "arxiv_id": None,
            "venue": "Nature 639, 624-632 (2025)",
            "doi": "10.1038/s41586-025-08628-5",
            "conditions_on": (
                "a broad range of scalar and categorical property constraints "
                "(including density-adjacent targets), via adapter modules "
                "fine-tuned per property on top of a base diffusion model"
            ),
            "released_checkpoint": True,
            "checkpoint_source": "https://github.com/microsoft/mattergen",
            "density_targetable_natively": False,
            "density_targeting_verdict": (
                "requires selecting or fine-tuning a property-specific adapter; "
                "no adapter for mass density is among the officially released "
                "fine-tuned checkpoints, so a fair density comparison would need "
                "a new adapter trained by us on a labelled density set, which is "
                "exactly the per-target retraining cost this paper targets"
            ),
            "cost_structure": "one fine-tuning run per target family, each requiring its own labelled dataset",
            "already_cited_in_manuscript": True,
        },
        "crystalgrw": {
            "arxiv_id": "2501.08998",
            "venue": "arXiv preprint (cond-mat.mtrl-sci), v3 Aug 2025",
            "doi": "10.48550/arXiv.2501.08998",
            "conditions_on": (
                "point group class, via classifier-free guidance on a "
                "categorical label; the released conditional checkpoint is "
                "trained specifically for point-group conditioning, not for any "
                "continuous scalar property"
            ),
            "released_checkpoint": True,
            "checkpoint_source": "https://github.com/trachote/crystalgrw (pretrained_alexmp20_pointgroups_2025-04-18)",
            "density_targetable_natively": False,
            "density_targeting_verdict": (
                "the released conditional checkpoint conditions on point group, "
                "a categorical label, not a continuous scalar; density targeting "
                "would require retraining the classifier-free-guidance head on a "
                "density-labelled set, which the released artifacts do not provide"
            ),
            "cost_structure": "one classifier-free-guidance head retrained per conditioning variable",
            "already_cited_in_manuscript": False,
        },
        "unigenx": {
            "arxiv_id": "2503.06687",
            "venue": "arXiv preprint (cs.LG), v2 Aug 2025",
            "doi": "10.48550/arXiv.2503.06687",
            "conditions_on": (
                "task-specific numeric and symbolic tokens steering a "
                "conditional diffusion head, demonstrated on multi-property "
                "crystal targets (evaluated with MatterSim) and on molecule and "
                "protein property targets"
            ),
            "released_checkpoint": False,
            "checkpoint_source": "code repository exists (github.com/Nobody-Zhang, per author bio); "
            "no released pretrained materials checkpoint or weights were located "
            "as of this verification",
            "density_targetable_natively": None,
            "density_targeting_verdict": (
                "no released checkpoint was found to test density targeting "
                "against; the paper's own crystal experiments target other "
                "properties (via MatterSim-evaluated multi-property tasks), and "
                "density is not reported as a demonstrated target"
            ),
            "cost_structure": "unified backbone with task-specific conditioning tokens; "
            "per-property demonstrations in the paper still required separate "
            "task setups rather than a single reusable checkpoint retargeted "
            "post hoc",
            "already_cited_in_manuscript": False,
        },
        "materium": {
            "arxiv_id": "2512.07486",
            "venue": "arXiv preprint (cs.LG), Dec 2025",
            "doi": "10.48550/arXiv.2512.07486",
            "conditions_on": (
                "multiple properties as prepended conditioning tokens, "
                "including density and space group as fundamental properties "
                "and band gap and magnetic density as practical targets, "
                "trained jointly with NaN-embedding substitution for missing "
                "conditions during batched training"
            ),
            "released_checkpoint": False,
            "checkpoint_source": (
                "no code repository or released checkpoint was located under "
                "the authors' institution (Fraunhofer SCAI) or on arXiv/GitHub "
                "at the time of this verification"
            ),
            "density_targetable_natively": True,
            "density_targeting_verdict": (
                "density is one of the properties the paper's own conditional "
                "model was trained and evaluated on, so density targeting is "
                "native to the method as described; however no checkpoint or "
                "code was located, so no independent measurement is possible, "
                "and the paper itself was flagged by the reviewer as known "
                "only through third-party summaries prior to this verification"
            ),
            "cost_structure": "single model trained jointly across several conditioning properties, "
            "reported as trained in a few hours on one GPU; no per-target "
            "retraining reported in the paper, but this cannot be independently "
            "confirmed without located code",
            "already_cited_in_manuscript": False,
        },
    }

    conclusion = {
        "measured_comparison_possible": False,
        "reason": (
            "of the four candidates with a verifiable primary record, one "
            "(MatterGen) has a released checkpoint but no released density "
            "adapter; one (CrystalGRW) has a released checkpoint conditioned "
            "on a categorical label (point group), not density; one (UniGenX) "
            "has no released pretrained materials checkpoint; and one "
            "(Materium) natively supports density conditioning in its own "
            "reported experiments but has released neither code nor weights "
            "at the time of this verification. No candidate offers a released, "
            "density-conditioned checkpoint that could be run locally without "
            "training or fine-tuning an external model, which the task scope "
            "excludes. This is reported as a documented negative rather than a "
            "measured comparison."
        ),
        "scope_reminder": (
            "structural, stability and S.U.N. metrics are additionally not "
            "comparable in either direction: this project's serialization "
            "(CrystalVocab, src/cpt/crystal/mp20.py) carries only el:, cnt:, "
            "len:, ang: token families and no atomic-position tokens, so no "
            "structure-level metric can be computed for this paper's own "
            "generated samples regardless of what an external baseline reports"
        ),
    }

    return {"candidates": candidates, "conclusion": conclusion}


def main() -> None:
    payload = build_payload()
    config = {
        "scope": "density-only external conditioning survey, W6/W11/Q6",
        "candidates_surveyed": sorted(payload["candidates"].keys()),
    }
    path = emit(
        "external_conditioning_survey",
        payload,
        config=config,
        seeds=[],
        script=SCRIPT,
    )
    print(path)


if __name__ == "__main__":
    main()
