"""Regenerate the LSC-XAI explainability stage (Sections 5.3-5.5) under fixed seeds.

No model is retrained. Fold-1 artifacts are loaded from
results/fold_artifacts_LR.pkl and verified against results/all_kfold_results.pkl
before a single explanation is computed.

Seeds (reported in the manuscript's software-implementation table)
------------------------------------------------------------------
LIME, main run        LimeTabularExplainer(random_state=42); a fresh explainer is
                      built per instance because LimeTabularExplainer stores a
                      stateful RandomState (lime_tabular.py:187) that advances on
                      every explain_instance call, so a shared object would make
                      results depend on call order.
LIME, stability study ten run seeds drawn from numpy.random.default_rng(42):
                      reproducible as a sequence while still measuring
                      seed-to-seed variability.
SHAP                  shap.Explainer(..., algorithm='permutation', seed=42); the
                      Permutation explainer calls numpy.random.seed(42) internally
                      (shap/explainers/_permutation.py:66).
DiCE                  generate_counterfactuals(random_seed=42, sample_size=1000)
                      with the numpy and stdlib global RNGs additionally pinned to
                      42, because dice_random.py:114 draws from the global numpy RNG
                      outside the random_seed guard.
Bootstrap resampling  numpy.random.default_rng(42).

Feasibility filtering (Eq. 2-3), and why the feasible set varies with N
-----------------------------------------------------------------------
The feasible sets are derived per configuration rather than pinned once:

    F*_N = C_N & F,   C_N = top-N(LIME) & top-N(SHAP)

where F is the auditable mu/omega/eta umbrella of Eq. 1 (identical to the umbrella used
by the notebook's counterfactual stage, notebook_raw.py:1650). Because C_N nests
monotonically in N, F*_N does too: 3 feasible features at N = 8 and N = 10, 4 at
N = 12, 14 and 16, and 6 at N = 32. The mu/omega/eta core
{JobInvolvement, Shift, OverTime} is feasible in every configuration, and larger N makes
*more* feasible features available, which is the intended behaviour. DiCE varies only the
features it is handed (dice_random.py:113-116), so no attribute outside F*_N can enter a
recommendation; "infeasible feature recommended" is asserted against below.

pandas-3 compatibility shim (dice-ml 0.12)
------------------------------------------
PublicData.__init__ -> _set_feature_dtypes stores categorical features as
string-typed CategoricalDtype, so DiceRandom.get_samples returns string categories
and PublicData.get_features_range reports them as strings. dice_random.py:110-111
builds the candidate frame as float64 from the (float) query instance, and
pandas 3 refuses to assign a string into a float64 column, whereas pandas 1.x
silently upcast the column to object. The query instance is therefore passed as
object dtype, and the returned counterfactual frame is coerced back to numeric
before the change-direction comparison. Both steps are dtype-only: the generated
feature values are unchanged.
"""
from pathlib import Path
import json
import pickle
import random
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
DEST = ROOT / 'manuscript'
ALL_N_DIR = RESULTS / 'xai_figures_all_N'
ARTIFACT = RESULTS / 'fold_artifacts_LR.pkl'
KFOLD = RESULTS / 'all_kfold_results.pkl'

# Fixed seeds
SEED = 42
N_LIME_STABILITY_RUNS = 10

# Design constants (must match the manuscript)
# Domain feasibility umbrella F (Eq. 1): every feature that passes all three binary
# criteria mu (mutability), omega (organizational controllability) and eta (ethical
# permissibility). Membership is the umbrella used by the notebook's counterfactual stage
# (notebook_raw.py:1650). Features failing a criterion (Age, tenure and career-history
# measures, ...) are deliberately absent. Two members (JobLevel, EnvironmentSatisfaction)
# are not retained by the Fold-1 feature selection and are therefore inert: they can never
# appear in C_N, so they cannot enter F*_N either.
FEASIBLE = ['OverTime', 'Shift', 'JobInvolvement', 'JobLevel',
            'EnvironmentSatisfaction', 'YearsSinceLastPromotion',
            'DailyRate', 'HourlyRate', 'MonthlyRate']
# The mu/omega/eta core: feasible and retained in every N configuration.
FEASIBLE_CORE = ['JobInvolvement', 'Shift', 'OverTime']
N_GRID = [32, 16, 14, 12, 10, 8]
N_APRIORI = 12
N_LIME_PER_INSTANCE = 12
TOTAL_CFS = 3
CF_SAMPLE_SIZE = 1000
N_BOOTSTRAP = 1000
CONFIDENCE = 0.95
MAX_CF_ATTEMPTS = 50
CF_OFFSET_SD = 1.5
# Fold-1 anchors from results/all_kfold_results.pkl
ANCHOR = {'Accuracy': 0.875000, 'Precision': 0.725643, 'Recall': 0.831757,
          'F1 Score': 0.761105, 'PR-AUC': 0.728821}


def load_fold1():
    """Load and verify the Fold-1 Logistic Regression artifact."""
    with open(ARTIFACT, 'rb') as handle:
        fold_artifacts = pickle.load(handle)
    fa = fold_artifacts[1]
    features = list(fa['retained_features'])
    X_train = fa['X_train_resampled'][features].reset_index(drop=True)
    y_train = np.asarray(fa['y_train_resampled']).reshape(-1)
    X_test = fa['X_test_scaled'][features].reset_index(drop=True)
    y_test = fa['y_test'].reset_index(drop=True).to_numpy()
    model = fa['trained_model']

    # Feasibility umbrella audit (Eq. 1). Two umbrella members (JobLevel,
    # EnvironmentSatisfaction) are not retained by the fold-wise feature selection, so
    # they can never enter C_N and are inert rather than errors. The mu/omega/eta core
    # must, however, be retained, otherwise Eq. 2 would lose a criterion-1 feature.
    not_retained = sorted(set(FEASIBLE) - set(features))
    missing_core = sorted(set(FEASIBLE_CORE) - set(features))
    assert not missing_core, f'feasible core absent from the Fold-1 model: {missing_core}'

    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score)
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    observed = {'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='macro'),
                'Recall': recall_score(y_test, y_pred, average='macro'),
                'F1 Score': f1_score(y_test, y_pred, average='macro'),
                'PR-AUC': average_precision_score(y_test, proba)}
    reference = pd.read_pickle(KFOLD)
    stored = reference[(reference.Model == 'Logistic Regression') & (reference.Fold == 1)].iloc[0]
    for metric, value in observed.items():
        assert abs(value - ANCHOR[metric]) < 5e-7, (metric, value, ANCHOR[metric])
        assert abs(value - stored[metric]) < 5e-7, (metric, value, stored[metric])
    print('PASS: Fold-1 reproduction gate (5 metrics match all_kfold_results.pkl exactly).')
    return features, X_train, y_train, X_test, y_test, model, y_pred, not_retained
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
    from collections import Counter
    counts = Counter()
    records = []
    indices = [i for i in range(len(X_test)) if y_test[i] == 1.0 and y_pred[i] == 1.0]
    for i in indices:
        explainer = build_lime_explainer(X_train, features, SEED)
        explanation = explainer.explain_instance(
            X_test.iloc[i].values, model.predict_proba, num_features=N_LIME_PER_INSTANCE)
        names, weights = top_features(explanation, N_LIME_PER_INSTANCE)
        for name, weight in zip(names, weights):
            counts[name] += 1
            records.append({'instance': i, 'lime_feature': name, 'weight': weight})
    print(f'LIME: explained {len(indices)} true-positive instances, '
          f'{len(counts)} distinct local feature strings.')
    return counts, pd.DataFrame(records), indices


def run_shap(X_test, model, features):
    """Permutation SHAP with a fixed seed; the API seeds numpy internally."""
    import shap
    explainer = shap.Explainer(model.predict, X_test.values,
                               feature_names=features,
                               algorithm='permutation', seed=SEED)
    values = explainer(X_test.values)
    print(f'SHAP: {type(explainer).__name__} produced {values.values.shape[0]} x '
          f'{values.values.shape[1]} attributions.')
    return values


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
    """Generate and annotate one instance's counterfactuals.

    query is passed as object dtype and the returned frame is coerced back to
    numeric purely for the pandas-3 dtype compatibility described in the module
    docstring; generated feature values are untouched.

    Returns (changes, columns): the annotated change-sets, plus the feature space of the
    returned counterfactual frame. The latter records the universe DiCE evaluated, so the
    audit in main() can confirm the generator never leaves the retained feature space.
    """
    np.random.seed(SEED)
    random.seed(SEED)
    cf = explainer.generate_counterfactuals(
        query.astype(object), total_CFs=TOTAL_CFS, desired_class='opposite',
        features_to_vary=features_to_vary, random_seed=SEED,
        sample_size=CF_SAMPLE_SIZE)
    example = cf.cf_examples_list[0]
    if example.final_cfs_df is None or example.final_cfs_df.empty:
        return [], set()
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
    return changes, set(frame.columns)


def run_counterfactuals(explainer, X_test, y_test, y_pred, indices, features_to_vary):
    """Batch counterfactual generation over the evaluated true positives.

    Returns (per_instance, varied): the pooled change-sets and the union of the
    columns that the generator was permitted to vary across the batch.
    """
    per_instance, varied = [], set()
    for i in indices:
        changes, requested = counterfactuals_for_instance(
            explainer, X_test.iloc[i:i + 1], features_to_vary, X_test.iloc[i])
        varied |= requested
        if changes:
            per_instance.extend(changes)
    return per_instance, varied
def bootstrap_stats(per_instance):
    """Mean frequency and percentile CI over bootstrapped resamples of instances."""
    from collections import Counter
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
    from collections import Counter
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


def lime_figure(X_test, X_train, model, features, indices, counts):
    """Local explanation for the first true-positive instance + aggregated view."""
    instance = indices[0]
    explainer = build_lime_explainer(X_train, features, SEED)
    explanation = explainer.explain_instance(
        X_test.iloc[instance].values, model.predict_proba, num_features=N_LIME_PER_INSTANCE)
    names, weights = top_features(explanation, N_LIME_PER_INSTANCE)
    order = np.argsort(np.abs(weights))
    labels = [names[i] for i in order]
    values = [weights[i] for i in order]
    colors = ['#c0392b' if v > 0 else '#2471a3' for v in values]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('LIME local weight (red = pushes toward attrition)')
    ax.set_title(f'Local LIME explanation: test instance {instance} (predicted attrition)')
    fig.tight_layout()
    fig.savefig(DEST / 'LIME_output_new.jpg', dpi=200)
    plt.close(fig)
    (RESULTS / 'lime_instance.html').write_text(explanation.as_html(), encoding='utf-8')

    ranked = counts.most_common()
    bar_figure([extract_base_feature_name(k) + f'  ({k})' for k, _ in ranked[:15]],
               [v for _, v in ranked[:15]],
               'Aggregated LIME top-feature frequency across true-positive instances',
               'Frequency', RESULTS / 'lime_aggregate.png', figsize=(14, 8))


def shap_figures(values):
    import shap
    plt.figure()
    shap.plots.beeswarm(values, show=False)
    plt.tight_layout()
    plt.savefig(DEST / 'SHAP_output_new.PNG', dpi=200, bbox_inches='tight')
    plt.close('all')
    plt.figure()
    shap.plots.bar(values, show=False)
    plt.tight_layout()
    plt.savefig(RESULTS / 'shap_bar.png', dpi=200, bbox_inches='tight')
    plt.close('all')
def main():
    ALL_N_DIR.mkdir(exist_ok=True)
    features, X_train, y_train, X_test, y_test, model, y_pred, not_retained = load_fold1()

    lime_counts, lime_records, indices = run_lime(
        X_train, X_test, y_test, y_pred, model, features)
    lime_records.to_csv(RESULTS / 'lime_top_features.csv', index=False)
    ranked_lime = lime_counts.most_common()

    shap_values = run_shap(X_test, model, features)
    shap_importance = pd.DataFrame({
        'Feature': features,
        'Mean_Abs_SHAP': np.abs(shap_values.values).mean(axis=0),
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    shap_importance.to_csv(RESULTS / 'shap_importances.csv', index=False)
    print('SHAP top 5:', shap_importance.Feature.head(5).tolist())

    # Cross-method agreement gate and feasibility filtering (Eq. 2-3).
    common = {n: intersect(ranked_lime, shap_values, features, n) for n in N_GRID}
    feasible = {n: sorted(set(common[n]) & set(FEASIBLE)) for n in N_GRID}
    pd.DataFrame([{'N': n, 'Common_Features': '; '.join(common[n]),
                   'Common_Size': len(common[n]),
                   'Feasible_Set': '; '.join(feasible[n]),
                   'Feasible_Size': len(feasible[n])} for n in N_GRID]
                 ).to_csv(RESULTS / 'common_features_by_N.csv', index=False)
    pd.DataFrame([{'N': n, 'Feasible_Set': '; '.join(feasible[n]),
                   'Size': len(feasible[n])} for n in N_GRID]
                 ).to_csv(RESULTS / 'feasible_sets_by_N.csv', index=False)

    # Eq. 2 must reproduce exactly, the mu/omega/eta core must survive every
    # configuration, and |F*_N| must be monotone non-decreasing in N because C_N is
    # nested. More available features at larger N is the intended behaviour, replacing
    # the single fixed three-feature set used previously.
    for n in N_GRID:
        assert feasible[n] == sorted(set(common[n]) & set(FEASIBLE)), n
        assert feasible[n], f'empty feasible set at N={n}'
        assert set(feasible[n]).issubset(set(FEASIBLE)), n
    dropped = {n: sorted(set(FEASIBLE_CORE) - set(feasible[n])) for n in N_GRID}
    assert not any(dropped.values()), f'core feasible feature dropped: {dropped}'
    sizes = [len(feasible[n]) for n in sorted(N_GRID)]
    assert sizes == sorted(sizes), f'|F*_N| grows as N shrinks: {sizes}'
    invariant = len({tuple(feasible[n]) for n in N_GRID}) == 1
    print(f'Feasible umbrella F (n={len(FEASIBLE)}): {sorted(FEASIBLE)}')
    print(f'  not retained by the Fold-1 selection (inert): {not_retained}')
    for n in sorted(N_GRID, reverse=True):
        print(f'  N={n}: |C_N|={len(common[n])}, F*_N={feasible[n]}')
    print(f'Feasible-set sizes for N={sorted(N_GRID)}: {sizes}')
    assert not invariant, f'feasible set does not vary with N (expected N-dependence)'

    explainer = build_dice(X_train, y_train, model, features)
    per_n, frames, sets, varied = {}, {}, {}, {}
    for n in N_GRID:
        generated, permitted = run_counterfactuals(
            explainer, X_test, y_test, y_pred, indices, feasible[n])
        per_n[n], varied[n] = generated, permitted
        frames[n] = bootstrap_stats(generated)
        sets[n] = recommendation_counts(generated)
        print(f'  N={n}: {len(generated)} counterfactual change-sets generated '
              f'from {len(feasible[n])} feasible features.')

    # The counterfactual output is *expected* to vary with N because F*_N does. The
    # invariants that must hold are that every configuration yields usable
    # counterfactuals and that only feasible attributes are ever recommended.
    empty = [n for n in N_GRID if not per_n[n]]
    assert not empty, f'no counterfactuals generated for N in {empty}'
    recommended = sorted({name.split('[')[0] for n in N_GRID
                          for changes in per_n[n] for name in changes})
    illegal = sorted(set(recommended) - set(FEASIBLE))
    assert not illegal, f'infeasible features recommended: {illegal}'
    for n in N_GRID:
        outside = sorted({name.split('[')[0] for changes in per_n[n]
                          for name in changes} - set(feasible[n]))
        assert not outside, f'N={n} recommended a feature outside F*_N: {outside}'
        assert varied[n] == set(features), (n, varied[n] ^ set(features))
    varying = len({tuple(sorted(map(tuple, per_n[n]))) for n in N_GRID}) > 1
    print(f'PASS: only feasible features recommended; recommendations vary with N = {varying}.')

    pd.concat([frames[n].assign(N=n) for n in N_GRID]).to_csv(
        RESULTS / 'cf_feature_frequencies_all_N.csv', index=False)
    pd.concat([sets[n].assign(N=n) for n in N_GRID]).to_csv(
        RESULTS / 'cf_recommendation_sets_all_N.csv', index=False)
    frames[N_APRIORI].to_csv(RESULTS / 'cf_feature_frequencies.csv', index=False)
    sets[N_APRIORI].to_csv(RESULTS / 'cf_recommendation_sets.csv', index=False)

    for n in N_GRID:
        tag = fig_tag(n)
        bar_figure(sets[n].Recommendation_Set, sets[n].Frequency,
                   f'N = {n}: counterfactual recommendation-set frequency',
                   'Frequency', ALL_N_DIR / f'top_{tag}_recomm.png')
        shown = frames[n][frames[n].Mean_Frequency >= 2]
        bar_figure(shown.Feature, shown.Mean_Frequency,
                   f'N = {n}: individual feature-change frequency (bootstrapped)',
                   f'Mean frequency ({int(CONFIDENCE * 100)}% CI)',
                   ALL_N_DIR / f'top_{tag}_freq.png',
                   errors=(shown.Mean_Frequency - shown.Lower_CI).tolist())

    # Manuscript figures: representative N = 12 pair plus LIME and SHAP.
    bar_figure(sets[N_APRIORI].Recommendation_Set, sets[N_APRIORI].Frequency,
               f'N = {N_APRIORI}: counterfactual recommendation-set frequency',
               'Frequency', DEST / f'top_{N_APRIORI}_recomm.png')
    shown = frames[N_APRIORI][frames[N_APRIORI].Mean_Frequency >= 2]
    bar_figure(shown.Feature, shown.Mean_Frequency,
               f'N = {N_APRIORI}: individual feature-change frequency (bootstrapped)',
               f'Mean frequency ({int(CONFIDENCE * 100)}% CI)',
               DEST / f'top_{N_APRIORI}_freq.png',
               errors=(shown.Mean_Frequency - shown.Lower_CI).tolist())
    lime_figure(X_test, X_train, model, features, indices, lime_counts)
    shap_figures(shap_values)

    manifest = {
        'representative_fold': 1,
        'n_retained_features': len(features),
        'n_test_instances': int(len(y_test)),
        'n_true_positives_explained': len(indices),
        'seeds': {'lime_main': SEED, 'lime_stability_runs': N_LIME_STABILITY_RUNS,
                  'shap': SEED, 'dice': SEED, 'bootstrap': SEED},
        'lime_num_features': N_LIME_PER_INSTANCE,
        'n_grid': N_GRID,
        'feasible_umbrella': sorted(FEASIBLE),
        'feasible_umbrella_size': len(FEASIBLE),
        'feasible_umbrella_not_retained': not_retained,
        'feasible_core': sorted(FEASIBLE_CORE),
        'feasible_sets_per_n': {str(n): feasible[n] for n in N_GRID},
        'feasible_set_sizes': {str(n): len(feasible[n]) for n in N_GRID},
        'common_set_sizes': {str(n): len(common[n]) for n in N_GRID},
        'feasible_set_invariant_across_n': bool(invariant),
        'counterfactual_sets_per_n': {str(n): len(per_n[n]) for n in N_GRID},
        'counterfactual_output_varies_across_n': bool(varying),
        'recommended_features': recommended,
        'anchor_metrics': ANCHOR,
    }
    (RESULTS / 'xai_run_manifest.json').write_text(
        json.dumps(manifest, indent=2), encoding='utf-8')
    print('Wrote CSVs, figures and results/xai_run_manifest.json.')
    print(frames[N_APRIORI].to_string(index=False))
    print(sets[N_APRIORI].head(12).to_string(index=False))


if __name__ == '__main__':
    main()

