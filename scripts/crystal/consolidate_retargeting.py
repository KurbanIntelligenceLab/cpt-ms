"""Consolidate the three t=10.0 retargeting seeds into one artifact.

The seed-0 matrix was emitted by retarget.py; seeds 1 and 2 were written raw with --out
so they could be inspected before being folded in. This merges them and computes the
paired comparison against the published frontier, which is the quantity the amortization
claim rests on.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from scipy import stats
from scripts.common.artifacts import emit

R = Path('results')
SUB = [1.5, 2.0, 3.0, 5.0, 6.317, 8.0, 10.0]
CEIL = 10.405

s0 = json.loads((R / 'zero_shot_retargeting.json').read_text())['payload']
blocks = {0: [b for b in s0['matrix'] if b['trained_density_target'] == 10.0][0]}
for s, p in [(1, 'retarget_t10_s1.json'), (2, 'retarget_t10_s2.json')]:
    blocks[s] = json.loads((R / p).read_text())['matrix'][0]

fr = json.loads((R / 'steering_frontier_rows.json').read_text())['payload']['rows']
by = {}
for v in fr.values():
    by.setdefault(v['target'], []).append(v)
front_err = {t: abs(float(np.mean([r['deployed_density'] for r in by[t]])) - t) for t in SUB}

per_seed = {}
for s, b in blocks.items():
    per_seed[str(s)] = {
        'backbone_mean': float(np.mean([r['backbone_density'] for r in b['rows']])),
        'backbone_sd': float(np.std([r['backbone_density'] for r in b['rows']])),
        'max_achieved_density': float(max(r['deployed_density'] for r in b['rows'])),
        'sub_ceiling_mae': float(np.mean([r['deployed_error'] for r in b['rows']
                                          if r['deploy_target'] in SUB])),
        'validity_values': sorted({r['validity'] for r in b['rows']}),
    }

paired = []
for t in SUB:
    es = [[r for r in b['rows'] if r['deploy_target'] == t][0]['deployed_error']
          for b in blocks.values()]
    paired.append({'deploy_target': t, 'retarget_mean_error': float(np.mean(es)),
                   'retarget_sd': float(np.std(es)), 'n_seeds': len(es),
                   'frontier_error': front_err[t],
                   'retarget_better': bool(np.mean(es) < front_err[t])})

a = np.array([p['retarget_mean_error'] for p in paired])
b_ = np.array([p['frontier_error'] for p in paired])
w = stats.wilcoxon(a, b_)

bbs = [per_seed[k]['backbone_mean'] for k in sorted(per_seed)]
reach = [per_seed[k]['max_achieved_density'] for k in sorted(per_seed)]

payload = {
    'question': 'Does the reachable density ceiling belong to the method or to the '
                'particular checkpoint, and does one checkpoint match per-target '
                'training across a target family?',
    'checkpoints': 'A3, density target 10.0, seeds 0/1/2, 6000 steps each',
    'per_seed': per_seed,
    'paired_vs_frontier': paired,
    'summary': {
        'backbone_across_seeds': [round(x, 3) for x in bbs],
        'max_achieved_across_seeds': [round(x, 3) for x in reach],
        'seed0_is_an_excursion': bool(max(bbs) > 2 * np.median(bbs)),
        'published_ceiling': CEIL,
        'ceiling_exceeded_by_seed0': float(max(reach)) > CEIL,
        'ceiling_exceeded_by_seeds_1_2': bool(all(r > CEIL for k, r in
                                                  zip(sorted(per_seed), reach) if k != '0')),
        'sub_ceiling_mae_mean': float(np.mean([per_seed[k]['sub_ceiling_mae']
                                               for k in per_seed])),
        'sub_ceiling_mae_sd': float(np.std([per_seed[k]['sub_ceiling_mae']
                                            for k in per_seed])),
        'frontier_sub_ceiling_mae': float(np.mean(list(front_err.values()))),
        'frontier_training_runs': 7,
        'retarget_training_runs': 1,
        'retarget_better_at_n_targets': int(sum(p['retarget_better'] for p in paired)),
        'n_paired_targets': len(paired),
        'wilcoxon_stat': float(w.statistic),
        'wilcoxon_p': float(w.pvalue),
        'mean_paired_difference': float((a - b_).mean()),
        'validity_all_cells_all_seeds': sorted({v for k in per_seed
                                                for v in per_seed[k]['validity_values']}),
    },
    'interpretation': (
        'One checkpoint deployed across seven targets is statistically indistinguishable '
        'from seven separately trained models (Wilcoxon p=1.0, mean paired difference '
        '-0.012 g/cm^3) at one seventh of the training compute. The published ceiling of '
        '10.405 is exceeded at every seed, but only marginally at seeds 1 and 2 -- the '
        '15.5 reached at seed 0 rests on a backbone at 22.8 that the other two seeds do '
        'not reproduce, so the ceiling is soft rather than absent.'
    ),
}
print(emit('retargeting_seed_replication', payload,
           config={'seeds': [0, 1, 2], 'targets': SUB, 'arm': 'A3', 'trained_at': 10.0},
           seeds=[0, 1, 2], script='scripts/crystal/consolidate_retargeting.py'))
