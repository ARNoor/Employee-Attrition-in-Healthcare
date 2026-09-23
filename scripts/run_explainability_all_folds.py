"""All-folds LSC-XAI (Sections 5.4-5.5 across folds 1-5, pooled).

Repeats the seeded Fold-1 pipeline of run_explainability.py for every fold and
pools the counterfactual change-sets across the five folds before drawing the
twelve top-N panels (N = 32, 16, 14, 12, 10, 8). Per fold the retained feature
space differs (23/23/17/25/30 features), so C_N and F*_N are computed per fold
against that fold's own model; the figures show pooled counts, the CSVs keep the
per-fold detail. No model is retrained; every fold's Logistic Regression is
verified against results/all_kfold_results.pkl (5e-7 tolerance) first.

Seeds are identical to run_explainability.py (LIME 42 fresh explainer per
instance; SHAP permutation 42; DiCE random_seed 42 + global numpy/stdlib 42 with
sample_size 1000; bootstrap default_rng(42)). The stability study, the single-
instance LIME figure and the SHAP beeswarm are not part of this run.
"""
from pathlib import Path
import json
import pickle
import random
import sys
import warnings
from collections import Counter

from raiutils.exceptions import UserConfigValidationException

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
ALL_FOLDS_DIR = ROOT / 'all_folds_figures'
ARTIFACT = RESULTS / 'fold_artifacts_LR.pkl'
KFOLD = RESULTS / 'all_kfold_results.pkl'

SEED = 42
FEASIBLE = ['OverTime', 'Shift', 'JobInvolvement', 'JobLevel',
            'EnvironmentSatisfaction', 'YearsSinceLastPromotion',
            'DailyRate', 'HourlyRate', 'MonthlyRate']
FEASIBLE_CORE = ['JobInvolvement', 'Shift', 'OverTime']
N_GRID = [32, 16, 14, 12, 10, 8]
N_LIME_PER_INSTANCE = 12
TOTAL_CFS = 3
CF_SAMPLE_SIZE = 1000
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95


def load_fold(artifacts, reference, fold):
    """Load one fold's artifact and verify it against all_kfold_results.pkl."""
    fa = artifacts[fold]
    features = list(fa['retained_features'])
    X_train = fa['X_train_resampled'][features].reset_index(drop=True)
    y_train = np.asarray(fa['y_train_resampled']).reshape(-1)
    X_test = fa['X_test_scaled'][features].reset_index(drop=True)
    y_test = fa['y_test'].reset_index(drop=True).to_numpy()
    model = fa['trained_model']

    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    observed = {'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro'),
                'Recall': recall_score(y_test, y_pred, average='macro'),
                'F1 Score': f1_score(y_test, y_pred, average='macro'),
                'PR-AUC': average_precision_score(y_test, proba)}
    stored = reference[(reference.Model == 'Logistic Regression')
                       & (reference.Fold == fold)].iloc[0]
    for metric, value in observed.items():
        assert abs(value - stored[metric]) < 5e-7, (fold, metric, value, stored[metric])
    not_retained = sorted(set(FEASIBLE) - set(features))
    print(f'Fold {fold}: PASS reproduction gate ({len(features)} retained features, '
          f'{not_retained or "no umbrella members dropped"}).')
    return features, X_train, y_train, X_test, y_test, model, y_pred, not_retained

# ---------------------------------------------------------------- LIME / SHAP (verbatim)

def build_lime_explainer(X_train, features, seed):
    """Fresh explainer per use: LimeTabularExplainer holds a stateful RandomState."""
    from lime import lime_tabular
    return lime_tabular.LimeTabularExplainer(
        X_train.values,
        feature_names=features,
        class_names=['No Attrition', 'Attrition'],
        mode='classification',
        random_state=seed,
    )


def top_features(explanation, num_features):
    ranked = explanation.as_list()[:num_features]
    return [name for name, _ in ranked], [weight for _, weight in ranked]


def run_lime(X_train, X_test, y_test, y_pred, model, features):
    """Explain every correctly predicted attrition instance (true positives)."""
    counts = Counter()
    indices = [i for i in range(len(X_test)) if y_test[i] == 1.0 and y_pred[i] == 1.0]
    for i in indices:
        explainer = build_lime_explainer(X_train, features, SEED)
        explanation = explainer.explain_instance(
            X_test.iloc[i].values, model.predict_proba, num_features=N_LIME_PER_INSTANCE)
        names, _ = top_features(explanation, N_LIME_PER_INSTANCE)
        for name in names:
            counts[name] += 1
    print(f'  LIME: {len(indices)} true positives, {len(counts)} local feature strings.')
    return counts, indices


def run_shap(X_test, model, features):
    """Permutation SHAP with a fixed seed; the API seeds numpy internally."""
    import shap
    explainer = shap.Explainer(model.predict, X_test.values,
                               feature_names=features,
                               algorithm='permutation', seed=SEED)
    return explainer(X_test.values)


def extract_base_feature_name(lime_feature_string):
    """'MonthlyRate <= 841.70' -> 'MonthlyRate' (notebook_raw.py:1520)."""
    import re
    for match in re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', lime_feature_string):
        if not match.replace('_', '').isdigit():
            return match
    return lime_feature_string


def top_lime_base_names(sorted_counts, n):
    names, seen = [], set()
    for raw, _ in sorted_counts:
        base = extract_base_feature_name(raw)
        if base not in seen:
            names.append(base)
            seen.add(base)
        if len(names) == n:
            break
    return names


def top_shap_names(shap_values, features, n):
    importances = np.abs(shap_values.values).mean(axis=0)
    order = np.argsort(-importances)
    return [features[i] for i in order[:n]]


def intersect(sorted_counts, shap_values, features, n):
    """LIME-SHAP intersection C_N (the cross-method agreement gate)."""
    return sorted(set(top_lime_base_names(sorted_counts, n))
                  & set(top_shap_names(shap_values, features, n)))

# ------------------------------------------------------- DiCE / statistics (verbatim)

def build_dice(X_train, y_train, model, features):
    """DiCE data/model interfaces built exactly as in the notebook."""
    import dice_ml
    frame = pd.concat([X_train, pd.Series(y_train, name='Attrition')], axis=1)
    continuous = [c for c in X_train.columns if X_train[c].nunique() > 10]
    data_dice = dice_ml.Data(dataframe=frame, continuous_features=continuous,
                             outcome_name='Attrition')
    model_dice = dice_ml.Model(model=model, backend='sklearn')
    return dice_ml.Dice(data_dice, model_dice, method='random')


def counterfactuals_for_instance(explainer, query, features_to_vary, original):
    """Generate and annotate one instance's counterfactuals (Fold-1 protocol).

    Returns (change_records, columns_touched, skipped_flag). A query for which
    DiCE finds no counterfactual at all raises UserConfigValidationException;
    that instance is reported as skipped instead of aborting the fold.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    try:
        cf = explainer.generate_counterfactuals(
            query.astype(object), total_CFs=TOTAL_CFS, desired_class='opposite',
            features_to_vary=features_to_vary, random_seed=SEED,
            sample_size=CF_SAMPLE_SIZE)
    except UserConfigValidationException:
        return [], set(), 1
    example = cf.cf_examples_list[0]
    if example.final_cfs_df is None or example.final_cfs_df.empty:
        return [], set(), 0
    frame = example.final_cfs_df.copy()
    frame = frame.drop(columns=[c for c in ['Attrition'] if c in frame.columns])
    frame = frame.apply(pd.to_numeric, errors='coerce')
    changes = []
    for row in range(len(frame)):
        record = []
        for col in frame.columns:
            if col not in original.index:
                continue
            value = frame.iloc[row][col]
            base = float(original[col])
            if pd.isna(value) or value == base:
                continue
            record.append(f'{col}[inc]' if value > base else f'{col}[dec]')
        if record:
            changes.append(record)
    return changes, set(frame.columns), 0


def run_counterfactuals(explainer, X_test, y_test, y_pred, indices, features_to_vary):
    """Batch counterfactual generation over the fold's evaluated true positives."""
    per_instance, varied, skipped = [], set(), 0
    for i in indices:
        changes, requested, skip = counterfactuals_for_instance(
            explainer, X_test.iloc[i:i + 1], features_to_vary, X_test.iloc[i])
        varied |= requested
        skipped += skip
        if changes:
            per_instance.extend(changes)
    return per_instance, varied, skipped


def bootstrap_stats(per_instance):
    """Mean frequency and percentile CI over bootstrapped resamples of instances."""
    if not per_instance:
        return pd.DataFrame(columns=['Feature', 'Mean_Frequency', 'Lower_CI', 'Upper_CI'])
    rng = np.random.default_rng(SEED)
    flat = [f for sub in per_instance for f in sub]
    features = sorted(set(flat))
    n = len(per_instance)
    draws = {f: [] for f in features}
    for _ in range(N_BOOTSTRAP):
        pick = rng.integers(0, n, n)
        counts = Counter(f for i in pick for f in per_instance[i])
        for f in features:
            draws[f].append(counts.get(f, 0))
    rows = []
    for f in features:
        values = np.array(draws[f])
        rows.append({'Feature': f, 'Mean_Frequency': values.mean(),
                     'Lower_CI': np.percentile(values, (1 - CONFIDENCE) / 2 * 100),
                     'Upper_CI': np.percentile(values, (1 + CONFIDENCE) / 2 * 100)})
    return pd.DataFrame(rows).sort_values('Mean_Frequency', ascending=False)


def recommendation_counts(per_instance):
    counts = Counter(', '.join(sorted(changes)) for changes in per_instance)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return pd.DataFrame(ranked, columns=['Recommendation_Set', 'Frequency'])


def bar_figure(labels, values, title, xlabel, path, figsize=(20, 10), errors=None):
    color = 'skyblue' if errors is not None else 'steelblue'
    fig, ax = plt.subplots(figsize=figsize)
    if errors is None:
        ax.barh(labels, values, color=color)
    else:
        ax.barh(labels, values, xerr=errors, capsize=5, color=color)
    ax.set_ylabel('Features')
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def fig_tag(n):
    """N = 32 is the full retained-feature space, tagged 'all' in the figures."""
    return 'all' if n == 32 else n

# --------------------------------------------------------------------- all-folds run

def fold_pipeline(fold, artifacts, reference):
    """Full LIME/SHAP/DiCE pipeline for one fold; returns the per-fold record."""
    features, X_train, y_train, X_test, y_test, model, y_pred, not_retained = \
        load_fold(artifacts, reference, fold)

    lime_counts, indices = run_lime(X_train, X_test, y_test, y_pred, model, features)
    shap_values = run_shap(X_test, model, features)
    ranked_lime = lime_counts.most_common()
    common = {n: intersect(ranked_lime, shap_values, features, n) for n in N_GRID}
    feasible = {n: sorted(set(common[n]) & set(FEASIBLE)) for n in N_GRID}

    # Per-fold Eq. 2 gate, mirrored from the Fold-1 script. Whether the mu/omega/eta
    # core survives every N is an empirical property of that fold's own LIME/SHAP
    # top-N (folds 3-4 do not even retain JobInvolvement in the model), so unlike in
    # the Fold-1 run it is reported per fold rather than asserted. Correctness gates
    # (Eq. 2 exact, non-empty, subset of the umbrella, monotone in N) stay hard.
    core_retained = sorted(set(FEASIBLE_CORE) & set(features))
    for n in N_GRID:
        assert feasible[n] == sorted(set(common[n]) & set(FEASIBLE)), (fold, n)
        assert feasible[n], f'empty feasible set at fold {fold}, N={n}'
        assert set(feasible[n]).issubset(set(FEASIBLE)), (fold, n)
    sizes = [len(feasible[n]) for n in sorted(N_GRID)]
    assert sizes == sorted(sizes), f'|F*_N| not monotone at fold {fold}: {sizes}'
    dropped_core = {n: sorted(set(core_retained) - set(feasible[n])) for n in N_GRID}
    if any(dropped_core.values()):
        print(f'  NOTE fold {fold}: core feature(s) absent from F*_N at '
              f'{[f"N={n}:{v}" for n, v in dropped_core.items() if v]}', flush=True)

    explainer = build_dice(X_train, y_train, model, features)
    fold_sets, skip_counts = {}, {}
    for n in N_GRID:
        generated, varied, skipped = run_counterfactuals(
            explainer, X_test, y_test, y_pred, indices, feasible[n])
        assert generated, f'no counterfactuals at fold {fold}, N={n}'
        outside = sorted({name.split('[')[0] for changes in generated
                          for name in changes} - set(feasible[n]))
        assert not outside, f'fold {fold}, N={n} recommended outside F*_N: {outside}'
        assert varied == set(features), (fold, n, sorted(varied ^ set(features)))
        fold_sets[n] = generated
        skip_counts[n] = skipped
        if skipped:
            print(f'  NOTE fold {fold} N={n}: {skipped} instance(s) found no '
                  f'counterfactual and were skipped.', flush=True)
    print(f'Fold {fold}: |C_12|={len(common[12])}, F*_12={feasible[12]}, '
          f'{len(fold_sets[12])} change-sets at N=12 '
          f'({len(fold_sets[32])} at N=32).', flush=True)
    return {'features': features, 'not_retained': not_retained,
            'core_retained': core_retained, 'dropped_core': dropped_core,
            'sizes': sizes, 'indices': indices,
            'common': common, 'feasible': feasible, 'sets': fold_sets,
            'skips': skip_counts}


def pool_stage(per_fold, folds):
    """Pool the per-fold change-sets, draw the twelve panels, write CSVs/manifest."""
    ALL_FOLDS_DIR.mkdir(exist_ok=True)
    pooled_sets = {n: [changes for fold in folds
                       for changes in per_fold[fold]['sets'][n]] for n in N_GRID}
    pooled_freq = {n: bootstrap_stats(pooled_sets[n]) for n in N_GRID}
    pooled_rec = {n: recommendation_counts(pooled_sets[n]) for n in N_GRID}
    for n in sorted(N_GRID, reverse=True):
        print(f'POOLED N={n}: {len(pooled_sets[n])} change-sets, '
              f'{len(pooled_rec[n])} distinct sets.', flush=True)

    # CSVs: pooled rows (Fold = 'pooled') plus the per-fold bootstrap detail.
    frequency_frames, recommendation_frames, summary_rows = [], [], []
    for n in N_GRID:
        frequency_frames.append(pooled_freq[n].assign(N=n, Fold='pooled'))
        recommendation_frames.append(pooled_rec[n].assign(N=n, Fold='pooled'))
        for fold in folds:
            detail = per_fold[fold]
            per_fold_sets = detail['sets'][n]
            frequency_frames.append(bootstrap_stats(per_fold_sets).assign(N=n, Fold=fold))
            detail_rec = recommendation_counts(per_fold_sets)
            recommendation_frames.append(detail_rec.assign(N=n, Fold=fold))
            summary_rows.append({
                'N': n, 'Fold': fold,
                'Retained_Features': len(detail['features']),
                'Common_Size': len(detail['common'][n]),
                'Common_Set': '; '.join(detail['common'][n]),
                'Feasible_Set': '; '.join(detail['feasible'][n]),
                'Feasible_Size': len(detail['feasible'][n]),
                'Change_Sets': len(per_fold_sets)})
    pd.concat(frequency_frames).to_csv(
        RESULTS / 'all_folds_cf_feature_frequencies.csv', index=False)
    pd.concat(recommendation_frames).to_csv(
        RESULTS / 'all_folds_cf_recommendation_sets.csv', index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RESULTS / 'all_folds_common_features_by_N.csv', index=False)

    for n in N_GRID:
        tag = fig_tag(n)
        bar_figure(pooled_rec[n].Recommendation_Set, pooled_rec[n].Frequency,
                   f'N = {n}: counterfactual recommendation-set frequency '
                   f'(pooled folds 1-5)',
                   'Frequency', ALL_FOLDS_DIR / f'top_{tag}_recomm.png')
        shown = pooled_freq[n][pooled_freq[n].Mean_Frequency >= 2]
        bar_figure(shown.Feature, shown.Mean_Frequency,
                   f'N = {n}: individual feature-change frequency '
                   f'(bootstrapped, pooled folds 1-5)',
                   f'Mean frequency ({int(CONFIDENCE * 100)}% CI)',
                   ALL_FOLDS_DIR / f'top_{tag}_freq.png',
                   errors=(shown.Mean_Frequency - shown.Lower_CI).tolist())

    manifest = {
        'folds': folds,
        'n_retained_features_per_fold': {str(f): len(per_fold[f]['features'])
                                         for f in folds},
        'umbrella_not_retained_per_fold': {str(f): per_fold[f]['not_retained']
                                           for f in folds},
        'core_retained_per_fold': {str(f): per_fold[f]['core_retained'] for f in folds},
        'core_missing_from_feasible_set_per_fold_per_n': {
            str(f): {str(n): per_fold[f]['dropped_core'][n] for n in N_GRID
                     if per_fold[f]['dropped_core'][n]} for f in folds},
        'feasible_set_sizes_per_fold': {str(f): per_fold[f]['sizes'] for f in folds},
        'n_true_positives_per_fold': {str(f): len(per_fold[f]['indices']) for f in folds},
        'n_change_sets_per_fold_per_n': {
            str(f): {str(n): len(per_fold[f]['sets'][n]) for n in N_GRID} for f in folds},
        'no_cf_skips_per_fold_per_n': {
            str(f): {str(n): per_fold[f]['skips'][n] for n in N_GRID
                     if per_fold[f]['skips'][n]} for f in folds},
        'feasible_sets_per_fold_per_n': {
            str(f): {str(n): per_fold[f]['feasible'][n] for n in N_GRID} for f in folds},
        'feasible_set_sizes_per_fold_per_n': {
            str(f): {str(n): len(per_fold[f]['feasible'][n]) for n in N_GRID}
            for f in folds},
        'pooled_change_sets_per_n': {str(n): len(pooled_sets[n]) for n in N_GRID},
        'pooled_recommendation_sets_per_n': {str(n): len(pooled_rec[n]) for n in N_GRID},
        'feasible_umbrella': sorted(FEASIBLE),
        'feasible_umbrella_union_over_folds': sorted({f for n in N_GRID
                                                     for detail in per_fold.values()
                                                     for f in detail['feasible'][n]}),
        'seeds': {'lime': SEED, 'shap': SEED, 'dice': SEED, 'bootstrap': SEED},
        'lime_num_features': N_LIME_PER_INSTANCE,
        'n_grid': N_GRID,
        'aggregation': 'pooled across folds 1-5 (per-fold detail in the CSVs)',
        'anchor': 'every fold verified against results/all_kfold_results.pkl (5e-7)',
    }
    (RESULTS / 'xai_run_manifest_all_folds.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'Wrote 12 pooled panels to {ALL_FOLDS_DIR.name}\\ and three results CSVs.')
    print(pooled_freq[12].to_string(index=False))
    print(pooled_rec[12].head(12).to_string(index=False))


def main():
    with open(ARTIFACT, 'rb') as handle:
        artifacts = pickle.load(handle)
    reference = pd.read_pickle(KFOLD)
    folds = sorted(artifacts)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if mode == 'pool':
        per_fold = {f: pickle.loads(
            (RESULTS / f'all_folds_checkpoint_fold{f}.pkl').read_bytes())
            for f in folds}
        pool_stage(per_fold, folds)
        return
    targets = [int(mode)] if mode.isdigit() else folds
    per_fold = {}
    for fold in targets:
        per_fold[fold] = fold_pipeline(fold, artifacts, reference)
        (RESULTS / f'all_folds_checkpoint_fold{fold}.pkl').write_bytes(
            pickle.dumps(per_fold[fold]))
        print(f'Fold {fold}: checkpoint saved.', flush=True)
    if targets == folds:
        pool_stage(per_fold, folds)


if __name__ == '__main__':
    main()
