"""Verify bundled source arrays and the six archived candidate audit records."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
audit = json.loads((ROOT / 'STAGE_VIEW_AUDIT.json').read_text(encoding='utf-8'))
with np.load(ROOT / 'REAL_STAGE_LAYOUTS.npz', allow_pickle=False) as saved:
    for key, expected in audit['bundled_array_sha256'].items():
        assert hashlib.sha256(saved[key].tobytes()).hexdigest() == expected, key
    assert saved['nodes'].tolist() == list(range(48))
    edges = {tuple(sorted(e)) for e in saved['edges'].tolist()}
    assert len(edges) == 67 and all(u != v and 0 <= u < 48 and 0 <= v < 48 for u, v in edges)
    assert set(saved['communities'].tolist()) == set(range(6))
with (ROOT / 'CASE_SELECTION_AUDIT.csv').open(encoding='utf-8', newline='') as stream:
    rows = list(csv.DictReader(stream))
assert len(rows) == 60
best_alpha, best_score = 0., 0.
for alpha in [.0625, .125, .25, .5, .75, 1.]:
    trial = [r for r in rows if float(r['alpha']) == alpha]
    assert len(trial) == 10
    feasible = all(r['violation'] == 'False' for r in trial) and all(r['at_least_one_improvement'] == 'True' for r in trial)
    assert all((r['feasible'] == 'True') == feasible for r in trial)
    score = float(trial[0]['score'])
    assert all(float(r['score']) == score for r in trial)
    if feasible and (score > best_score + 1e-12 or (abs(score-best_score) <= 1e-12 and alpha > best_alpha)):
        best_alpha, best_score = alpha, score
assert best_alpha == audit['selected_alpha'] == 1.
assert abs(best_score-float(audit['case_record']['score'])) < 1e-10
print('PASS: bundled array hashes, graph identities, six archived candidate audits, selected alpha=1.0')
