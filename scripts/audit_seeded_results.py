"""Audit saved fold metrics against the manuscript without changing either."""
from itertools import combinations, product
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'results' / 'all_kfold_results.pkl'
TEX = ROOT / 'manuscript' / 'cas-sc-sample.tex'
df = pd.read_pickle(DATA)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'PR-AUC']
assert df.shape[0] == 70 and not df.duplicated(['Model', 'Fold']).any()
assert df.groupby('Model')['Fold'].apply(lambda x: sorted(x) == [1, 2, 3, 4, 5]).all()
assert np.isfinite(df[metrics]).all().all() and df[metrics].ge(0).all().all() and df[metrics].le(1).all().all()
mat = df.pivot(index='Fold', columns='Model', values='Recall')
assert mat.shape == (5, 14) and not mat.isna().any().any()
stat, p = friedmanchisquare(*mat.to_numpy().T)
# Independent Friedman rank calculation with within-fold tie correction.
ranks = np.array([rankdata(row) for row in mat.to_numpy()])
n, k = mat.shape
ties = sum(sum(c**3 - c for c in np.unique(row, return_counts=True)[1]) for row in mat.to_numpy())
manual = (12 * np.sum(ranks.sum(axis=0)**2) / (n*k*(k+1)) - 3*n*(k+1)) / (1 - ties/(n*(k**3-k)))
assert np.isclose(stat, manual)
rows = []
for a, b in combinations(mat.columns, 2):
    diff = (mat[a] - mat[b]).to_numpy()
    diff = diff[diff != 0]  # Wilcox convention: discard zero differences.
    r = rankdata(abs(diff))
    wplus = r[diff > 0].sum()
    observed = min(wplus, r.sum() - wplus)
    possible = [np.dot(bits, r) for bits in product([0, 1], repeat=len(diff))]
    exact = np.mean([min(v, r.sum()-v) <= observed for v in possible])
    _, pv = wilcoxon(diff, method='exact', alternative='two-sided')
    assert np.isclose(pv, exact), (a, b, pv, exact)
    rows.append([a, b, (2*wplus-r.sum())/r.sum(), pv])
out = pd.DataFrame(rows, columns=['Model 1', 'Model 2', 'r', 'p_raw'])
order = np.argsort(out.p_raw.to_numpy(), kind='stable')
adjusted = np.empty(len(out))
adjusted[order] = np.minimum(1, np.maximum.accumulate((len(out)-np.arange(len(out))) * out.p_raw.to_numpy()[order]))
out['p_adj'] = adjusted
out['Sig'] = adjusted <= .05
assert len(out) == 91 and out.p_raw.min() == .0625 and (out.p_adj == 1).all()
out = out.sort_values(['r'], key=lambda s: s.abs(), ascending=False, kind='stable')
out.to_csv(ROOT / 'results' / 'verified_pairwise_recall.csv', index=False)
print(f'VERIFIED: Friedman chi2(13)={stat:.8f}; p={p:.10g}; W={stat/65:.8f}')
print(out.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
tex = TEX.read_text(encoding='utf-8')
active = '\n'.join(l for l in tex.splitlines() if not l.lstrip().startswith('%'))
print('\nCURRENT MANUSCRIPT OMNIBUS CLAIMS:')
for line in active.splitlines():
    if '43.79' in line or '0.674' in line:
        print(re.findall(r'.{0,25}(?:43\.79|0\.674).{0,25}', line))
labels = {'K-Nearest Neighbors':'KNN', 'Decision Trees':'Decision Tree', 'Support Vector Machines':'SVM', 'Adaptive Boosting':'Ada Boost', 'Extreme Gradient Boosting':'XGBoost', 'Light Gradient Boosting':'LightGBM', 'Categorical Boosting':'CatBoost', 'Multi-layer Perceptron':'MLP', 'Convolutional Neural Networks':'CNN'}
block = active.split(r'\label{tab:model_comparison}')[1].split(r'\end{table}')[0]
print('\nTABLE 4 DIFF (mean +/- sample SD, percent):')
for line in block.splitlines():
    cells = line.split('&')
    name = labels.get(cells[0].strip(), cells[0].strip())
    if name not in mat.columns:
        continue
    g = df[df.Model == name]
    old = [tuple(re.findall(r'\d+\.\d+', c)) for c in cells[1:5]]
    new = [(f'{g[c].mean()*100:.2f}', f'{g[c].std()*100:.2f}') for c in metrics[:4]]
    print(name, 'UNCHANGED' if old == new else f'{old} -> {new}')
    print(f'  ADD AP: {g["PR-AUC"].mean():.4f} +/- {g["PR-AUC"].std():.4f}')
app = active.split(r'\label{tab:app_statistical_comparisons}')[1].split(r'\end{longtable}')[0]
print('\nAPPENDIX PAIRWISE DIFF:')
changed = 0
for line in app.splitlines():
    c = [v.strip().replace('AdaBoost', 'Ada Boost') for v in line.split('&')]
    if len(c) != 6 or c[0] not in mat.columns or c[1] not in mat.columns:
        continue
    row = out[(out['Model 1'] == c[0]) & (out['Model 2'] == c[1])].iloc[0]
    new = [f'{row[v]:.4f}' for v in ['r', 'p_raw', 'p_adj']]
    old = [f'{float(v):.4f}' for v in c[2:5]]
    if old != new:
        changed += 1
        print(f'{c[0]} / {c[1]}: {old} -> {new}')
print(f'{changed}/91 appendix pairs changed; no Holm-significant pairs.')
print('VALIDATION PASSED: complete folds, independent Friedman, exhaustive Wilcoxon, Holm bound.')
