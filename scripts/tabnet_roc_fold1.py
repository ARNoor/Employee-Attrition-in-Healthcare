# -*- coding: utf-8 -*-
"""Single illustrative Fold-1 TabNet ROC panel (tab_roc.png).

TabNet training is not deterministic in this environment (two protocol-identical
runs of neural_roc_replication.py diverged by 9.7e-1 in per-instance scores), so
no certified curve is attainable. This script trains TabNet ONCE under the
verbatim notebook cell-45 protocol, draws the Fold-1 ROC panel in the style of
the other appendix panels, and records the run's metrics with an explicit
non-certified status. No metric gate is applied and no manuscript claims change.
"""
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
OVERLEAF = ROOT / 'overleaf_figures'


def load_fold1():
    with open(RESULTS / 'fold_artifacts_LR.pkl', 'rb') as handle:
        fa = pickle.load(handle)[1]
    X_train = fa['X_train_resampled'].reset_index(drop=True)
    y_train = np.asarray(fa['y_train_resampled']).reshape(-1)
    X_test = fa['X_test_scaled'].reset_index(drop=True)
    y_test = fa['y_test'].reset_index(drop=True).to_numpy()
    return X_train, y_train, X_test, y_test


def train_tabnet_once(X_train, y_train, X_test, y_test):
    """Protocol-verbatim cell-45 run: TabNetClassifier(verbose=0), default seed 0."""
    import torch
    from pytorch_tabnet.tab_model import TabNetClassifier
    torch.manual_seed(0)
    np.random.seed(0)
    model = TabNetClassifier(verbose=0, seed=0)
    model.fit(np.asarray(X_train), np.asarray(y_train),
              eval_set=[(np.asarray(X_test), np.asarray(y_test))],
              patience=10, max_epochs=30)
    return model.predict_proba(np.asarray(X_test))[:, 1]


def observed_metrics(y_true, y_prob):
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score)
    y_pred = (y_prob >= 0.5).astype(int)
    ap_pos = average_precision_score(y_true, y_prob, pos_label=1)
    ap_neg = average_precision_score(1 - y_true, 1 - y_prob, pos_label=1)
    return {'Accuracy': float(accuracy_score(y_true, y_pred)),
            'Precision': float(precision_score(y_true, y_pred, average='macro')),
            'Recall': float(recall_score(y_true, y_pred, average='macro')),
            'F1 Score': float(f1_score(y_true, y_pred, average='macro')),
            'PR-AUC': float(ap_pos),
            'Macro PR-AUC': float((ap_pos + ap_neg) / 2)}


def draw_roc_panel(y_true, y_prob, path):
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    area = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f'TabNet (AUC = {area:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return float(area)


def main():
    X_train, y_train, X_test, y_test = load_fold1()
    print(f'Fold-1 TabNet: {X_train.shape[0]} resampled train rows, '
          f'{X_test.shape[0]} test rows, {int(y_test.sum())} positives.', flush=True)
    y_prob = train_tabnet_once(X_train, y_train, X_test, y_test)
    target = OVERLEAF / 'tab_roc.png'
    area = draw_roc_panel(y_test, y_prob, target)
    (RESULTS / 'tab_roc.png').write_bytes(target.read_bytes())
    stored = pd.read_pickle(RESULTS / 'all_kfold_results.pkl')
    stored_row = stored[(stored.Model == 'TabNet') & (stored.Fold == 1)].iloc[0]
    observed = observed_metrics(y_test, y_prob)
    payload = {
        'status': 'illustrative single run - NOT certified',
        'panel': str(target),
        'roc_auc_fold1': area,
        'stored_fold1_pr_auc': float(stored_row['PR-AUC']),
        'observed_metrics_this_run': observed,
        'protocol': 'notebook cell 45, Fold 1 only, single run: TabNetClassifier('
                    'verbose=0, seed=0) with torch/np seeds 0, eval_set on the test '
                    'fold, patience=10, max_epochs=30',
        'note': 'TabNet training is non-deterministic in this environment (two '
                'protocol-identical runs of neural_roc_replication.py diverged by '
                '9.7e-1 in per-instance scores), so this curve cannot be certified '
                'against the stored Fold-1 row and carries no bitwise-match claim. '
                'Drawn for illustration only.',
    }
    (RESULTS / 'tabnet_roc_fold1.json').write_text(
        json.dumps(payload, indent=2), encoding='utf-8')
    print(f'tab_roc.png: AUC = {area:.4f} -> {target}')
    print(f'observed metrics (this run): {json.dumps(observed)}')
    print(f"stored Fold-1 PR-AUC: {float(stored_row['PR-AUC']):.4f} "
          f'(certification not attempted; status: illustrative)')


if __name__ == '__main__':
    main()
