"""Validate the revised LaTeX source and all four updated numerical tables.

This is a source/data check, not a substitute for a TeX build.
"""
from pathlib import Path
import re
import json
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, t

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / 'manuscript' / 'cas-sc-sample.tex'

def uncomment(text):
    return re.sub(r'(?<!\\)%[^\n]*', '', text)

def syntax_errors(text):
    text = uncomment(text)
    errors, stack = [], []
    for match in re.finditer(r'(?<!\\)[{}]', text):
        char = match.group()
        line = text.count('\n', 0, match.start()) + 1
        if char == '{':
            stack.append(line)
        elif stack:
            stack.pop()
        else:
            errors.append(f'unmatched closing brace at line {line}')
    errors.extend(f'unmatched opening brace at line {line}' for line in stack)
    stack = []
    for m in re.finditer(r'\\(begin|end)\{([^}]+)\}', text):
        action, env = m.groups()
        if action == 'begin':
            stack.append(env)
        elif not stack or stack[-1] != env:
            errors.append(f'environment mismatch: end {env}, stack {stack}')
        else:
            stack.pop()
    if stack:
        errors.append(f'unclosed environments: {stack}')
    for m in re.finditer(r'\\textcolor\b', text):
        if not re.match(r'\\textcolor\s*\{[^{}]+\}\s*\{', text[m.start():]):
            errors.append('invalid textcolor arguments at offset '+str(m.start()))
    return errors

raw = TEX.read_text(encoding='utf-8')
active = uncomment(raw)
errors = syntax_errors(raw)
backup = ROOT / 'results' / 'cas-sc-sample.before_seeded_update.tex'
old_errors = syntax_errors(backup.read_text(encoding='utf-8'))
print('SOURCE syntax findings:', errors or 'none')
print('PRE-EDIT syntax findings:', old_errors or 'none')
assert not errors, 'Source syntax findings require review.'
labels = re.findall(r'\\label\{([^}]+)\}', active)
assert all(n == 1 for n in Counter(labels).values()), 'Duplicate active LaTeX labels.'

names = {'K-Nearest Neighbors':'KNN', 'Decision Trees':'Decision Tree', 'Support Vector Machines':'SVM', 'Adaptive Boosting':'Ada Boost', 'Extreme Gradient Boosting':'XGBoost', 'Light Gradient Boosting':'LightGBM', 'Categorical Boosting':'CatBoost', 'Multi-layer Perceptron':'MLP', 'Convolutional Neural Networks':'CNN'}
df = pd.read_pickle(ROOT / 'results' / 'all_kfold_results.pkl')
pairs = pd.read_csv(ROOT / 'results' / 'verified_pairwise_recall.csv')

def table(label, end='table'):
    return active.split('\\label{'+label+'}', 1)[1].split('\\end{'+end+'}', 1)[0]

def rows(block, count):
    output = []
    for line in block.splitlines():
        cells = [c.strip() for c in line.split('&')]
        if cells[0] not in names and cells[0] not in set(df.Model):
            continue
        assert len(cells) == count, (count, line)
        assert line.rstrip().endswith(r'\\'), line
        output.append(cells)
    return output

block = table('tab:model_comparison')
assert r'\begin{tabular}{lccccc}' in block
r = rows(block, 6)
assert len(r) == 14
for cells in r:
    model = names.get(cells[0], cells[0])
    g = df[df.Model == model]
    for cell, metric in zip(cells[1:], ['Accuracy','Precision','Recall','F1 Score','PR-AUC']):
        scale, places = (1, 4) if metric == 'PR-AUC' else (100, 2)
        expected = [f'{g[metric].mean()*scale:.{places}f}', f'{g[metric].std()*scale:.{places}f}']
        assert re.findall(r'\d+\.\d+', cell) == expected, (model, metric, cell, expected)
print('PASS: Table 4, all 140 mean/SD values and six columns.')

r = rows(table('tab:recall_ci'), 4)
assert len(r) == 14
previous = 1
for cells in r:
    model = names.get(cells[0], cells[0]); g = df[df.Model == model].Recall
    m, sd = g.mean(), g.std(); margin = t.ppf(.975, 4)*sd/np.sqrt(5)
    assert m <= previous
    previous = m
    expected = [f'{v:.4f}' for v in [m, sd, m-margin, m+margin]]
    assert re.findall(r'\d+\.\d+', ' '.join(cells[1:])) == expected, (model, expected)
print('PASS: Table 5, 56 mean/SD/CI entries and sort order.')

for label, end, count in [('tab:posthoc_top','table',8), ('tab:app_statistical_comparisons','longtable',91)]:
    r = rows(table(label, end), 6)
    assert len(r) == count
    assert [tuple(c[:2]) for c in r] == list(pairs[['Model 1','Model 2']].head(count).itertuples(index=False, name=None)), 'Pairwise row order differs from verified CSV.'
    seen = set()
    for cells in r:
        a, b = cells[:2]
        assert (a,b) not in seen
        seen.add((a,b))
        row = pairs[(pairs['Model 1']==a) & (pairs['Model 2']==b)].iloc[0]
        expected = [f'{row[c]:.4f}' for c in ['r','p_raw','p_adj']]
        assert cells[2:5] == expected, (a,b,cells,expected)
        assert cells[5] == r'No \\'
    print(f'PASS: {label}, {count} verified pairwise rows and six columns.')
mat = df.pivot(index='Fold', columns='Model', values='Recall')
stat,p = friedmanchisquare(*mat.to_numpy().T)
assert active.count(f'chi^2(13) = {stat:.2f}') == 2
assert active.count(f'{stat/65:.3f}') >= 2
assert active.count(r'p = 8.817\times10^{-5}') == 2
for stale in ['43.79','0.674','86.22','86.93','78.88','exactly reproducible']:
    assert stale not in active, stale
print('PASS: omnibus statistics in both sections and no obsolete neural/seed claims.')
roc_pngs = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{(\w+_roc\.png)\}', active)
assert roc_pngs == ['LR_roc.png','svm_roc.png','knn_roc.png','dt_roc.png','rf_roc.png','gb_roc.png','ada_roc.png','xg_roc.png','light_roc.png','cat_roc.png','stack_roc.png','mlp_roc.png','cnn_roc.png','tab_roc.png'], roc_pngs
assert active.count(r'\ref{fig:roc_subfigures}') == 2
for stale in ['historical illustrations', 'retained from the earlier evaluation',
              'Panels are limited to the shallow learners',
              'omitted because per-instance probability scores from the updated seeded evaluation were not retained',
              'the TabNet panel is omitted', 'no certified ROC curve can be drawn',
              'the TabNet curve is absent']:
    assert stale not in active, stale
for required in ['AUC $= 0.9101$', 'AUC $= 0.8798$', 'non-deterministic in the current software environment',
                 'certified seeded replications', 'TabNet', 'ROC-AUC $= 0.7867$',
                 'illustrative single run under the identical protocol']:
    assert required in active, required
tabroc = json.loads((ROOT / 'results' / 'tabnet_roc_fold1.json').read_text(encoding='utf-8'))
assert f"{tabroc['roc_auc_fold1']:.4f}" == '0.7867' and tabroc['status'].startswith('illustrative')
print('PASS: ROC appendix retains the 11 classical/ensemble panels, the certified MLP/CNN replications and the TabNet illustrative panel with its non-certification stated.')
for stale in ['no random seed was fixed', 'without a fixed random', 'likewise unseeded', 'JobLevel[inc]']:
    assert stale not in active, stale
for required in ['seed=42', 'state=42', 'rng(42)', 'features=12']:
    assert required in active, required
print('PASS: XAI-stage seeds documented and unseeded wording removed.')

feas = pd.read_csv(ROOT / 'results' / 'feasible_sets_by_N.csv')
expected_sizes = {32: 6, 16: 4, 14: 4, 12: 4, 10: 3, 8: 3}
assert dict(zip(feas.N, feas.Size)) == expected_sizes, dict(zip(feas.N, feas.Size))
print('PASS: feasible-set sizes vary with N exactly as reported (6, 4, 4, 4, 3, 3).')
for stale in ['identical by construction', 'identical across all six values',
              'selects the same three features', 'are identical for all six candidate',
              'Frequencies are identical across all six', 'twelve panels are therefore identical']:
    assert stale not in active, stale
for required in ['27 distinct recommendation sets', 'occurs only 12', '1 - 4/23 = 0.8261',
                 r'\mathcal{F}^*_N$ holds three features at $N \in \{8,10\}$']:
    assert required in active, required
print('PASS: feasible-set variation reported consistently; retired invariance wording gone.')

stale_c7_9 = [
    'increasing job level',
    "an increase in 'JobLevel'",
    'including job level, shift structure and overtime',
    "fewer 'Shift' assignments showed higher retention",
    'Working shifts for an individual can be increased or decreased',
]
for stale in stale_c7_9:
    assert stale not in active, stale
for required in [
    'job involvement, shift structure, overtime workload and daily pay rate',
    'shift-category reassignment, daily pay-rate adjustment and overtime reduction',
    'labels [inc] and [dec] denote movement up or down the encoded ordinal scale',
    '19.8\\% of category-0',
    'JobInvolvement changes persist at lower frequency',
    'no sign reversal',
]:
    assert required in active, required
print('PASS: comments 7-9 corrections present; retired JobLevel/Shift wording gone.')

# ---- Comments 13-14: ablation reporting and design-consequence framing ----
assert '0.8696' not in active, 'stale 0.8696 sparsity value still present'
assert 'is higher for LSC-XAI (0.8261) than for either standalone method' in active
step5_row = ('5. FS + Box-Cox + SMOTENC & $87.47 \\pm 1.25\\%$ & $72.54 \\pm 1.62\\%$'
             ' & $83.11 \\pm 1.34\\%$ & $76.05 \\pm 1.72\\%$')
assert step5_row in active, 'Table 7 Step 5 row must equal the Table 4 LR values'
assert 'Logistic Regression & 87.47 $\\pm$ 1.25\\%' in active
for required in [
    'using the Logistic Regression base model selected in Section~5.4',
    'All steps use the Logistic Regression base model selected in Section~5.4',
    'roughly 40 attrition cases',
    'all below their Step 1 means',
    'rather than an empirical advantage',
    'rather than constituting measured advantages',
    'do not establish that this combination is strictly necessary',
]:
    assert required in active, required
print('PASS: comments 13-14 corrections present (5-fold ablation with named model; '
      'design-consequence framing; no stale 0.8696).')

# ---- Round 4: five-fold LSC-XAI replication integrated (all-folds rerun) ----
for stale in ['which was likewise run on this fold']:
    assert stale not in active, stale
for required in [
    'was first run on this fold and then replicated across all five folds',
    'In the five-fold replication of Section~5.5, the same a-priori threshold',
    'Five-fold replication.',
    '455, 455, 449, 440, 440 and 427 counterfactual change-sets',
    '154 correctly classified attrition instances',
    'OverTime and Shift are feasible in all 30 fold--threshold cells',
    "17 cells whose fold's model retains it",
    '330.05 [312, 348]', '198.90 [179, 219]', '44.90 [32, 59]', '43.19 [31, 56]',
    '342.97 [326, 359]', '61.00 [47, 75]',
    'the DailyRate emphasis of the Fold-1 recommendations is fold-specific',
    'LSC-XAI five-fold replication',
    'run\\_explainability\\_all\\_folds.py',
    'mean pairwise Jaccard similarity of $\\mathcal{F}^*_N$ across the five folds',
    'rises from 0.56 at $N \\in \\{16, 32\\}$ to 0.60 at $N = 14$, 0.72 at the a-priori '
    '$N = 12$ and 0.80 at $N \\in \\{8, 10\\}$',
    'the per-fold feasible sets form a nested chain',
    'overtime reduction ranks second in folds 2--4 and never below third in any fold',
    'two fold-5 instances per threshold, four at $N = 8$, admit no counterfactual',
]:
    assert required in active, required
manifest = json.loads((ROOT / 'results' / 'xai_run_manifest_all_folds.json').read_text(encoding='utf-8'))
assert manifest['no_cf_skips_per_fold_per_n'] == {
    '1': {}, '2': {}, '3': {}, '4': {},
    '5': {'32': 2, '16': 2, '14': 2, '12': 2, '10': 2, '8': 4}}
assert manifest['n_true_positives_per_fold'] == {'1': 31, '2': 30, '3': 30, '4': 32, '5': 31}

from itertools import combinations as _comb
common = pd.read_csv(ROOT / 'results' / 'all_folds_common_features_by_N.csv')
mj = {}
for n in (32, 16, 14, 12, 10, 8):
    fs = {f: set(common[(common.N == n) & (common.Fold == f)].iloc[0].Feasible_Set.split('; '))
          for f in range(1, 6)}
    mj[n] = sum(len(fs[a] & fs[b]) / len(fs[a] | fs[b])
                for a, b in _comb(range(1, 6), 2)) / 10
assert abs(mj[12] - 0.7167) < 1e-3 and abs(mj[10] - 0.8) < 1e-3 and abs(mj[8] - 0.8) < 1e-3
assert abs(mj[32] - 0.5643) < 1e-3 and abs(mj[16] - 0.5579) < 1e-3 and abs(mj[14] - 0.6) < 1e-3
cfsets = pd.read_csv(ROOT / 'results' / 'all_folds_cf_recommendation_sets.csv')
cfreq = pd.read_csv(ROOT / 'results' / 'all_folds_cf_feature_frequencies.csv')
tot = cfsets[cfsets.Fold == 'pooled'].groupby('N').Frequency.sum()
assert [tot[n] for n in (32, 16, 14, 12, 10, 8)] == [455, 455, 449, 440, 440, 427], tot.to_dict()
p12 = cfsets[(cfsets.N == 12) & (cfsets.Fold == 'pooled')].reset_index(drop=True)
assert len(p12) == 21 and p12.Frequency.sum() == 440
assert p12.iloc[0]['Recommendation_Set'] == 'Shift[inc]' and p12.iloc[0]['Frequency'] == 159
assert p12.iloc[1]['Recommendation_Set'] == 'OverTime[dec], Shift[inc]' and p12.iloc[1]['Frequency'] == 92
f12 = cfreq[(cfreq.N == 12) & (cfreq.Fold == 'pooled')].set_index('Feature')
for feat, mean, lo, hi in [('Shift[inc]', '330.05', '312', '348'),
                           ('OverTime[dec]', '198.90', '179', '219'),
                           ('JobInvolvement[inc]', '44.90', '32', '59'),
                           ('DailyRate[inc]', '43.19', '31', '56')]:
    row = f12.loc[feat]
    assert f'{row.Mean_Frequency:.2f}' == mean and f'{row.Lower_CI:.0f}' == lo and f'{row.Upper_CI:.0f}' == hi, (feat, row.to_dict())
f10 = cfreq[(cfreq.N == 10) & (cfreq.Fold == 'pooled')].set_index('Feature')
assert f"{f10.loc['Shift[inc]'].Mean_Frequency:.2f}" == '342.97'
assert f"{f10.loc['JobInvolvement[inc]'].Mean_Frequency:.2f}" == '61.00'
for frag in ['330.05 & 312 & 348', '198.90 & 179 & 219', '44.90 & 32 & 59',
             '43.19 & 31 & 56', '37.90 & 27 & 49', '11.05 & 5 & 18',
             '3.02 & 0 & 7', '1.08 & 0 & 3']:
    assert frag in active, frag
for frag in ['Shift[inc]                                         & 159',
             'OverTime[inc], Shift[inc]                          & 1']:
    assert frag in active, frag
assert active.count('tab:cf_freq_pooled') >= 3
assert active.count('tab:all_folds_feasible') >= 2
for cell in ['6 (93)', '2 (86)', '2 (71)', '4 (96)']:
    assert cell in active, cell
assert 'af_top_12_recomm.png' in active and 'af_top_12_freq.png' in active
print('PASS: five-fold replication integrated; pooled numbers match results/all_folds_*.csv.')

assets = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', active)
missing = [a for a in assets if not (TEX.parent/a).exists()]
print('MISSING FIGURE PATHS:', missing or 'none')
assert not missing
print('PASS: referenced image files exist (image numerical provenance is a separate check).')
print('FINAL SOURCE/DATA VALIDATION PASSED. PDF build NOT performed: TeX engine unavailable.')
