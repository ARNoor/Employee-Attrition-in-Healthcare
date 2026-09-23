"""Regenerate the missing MLP/CNN/TabNet ROC panels (appendix fig:roc_subfigures)
by replicating the seeded Fold-1 training protocol of notebook cell 45.

The seeded evaluation stored per-fold metrics (results/all_kfold_results.pkl) but
never persisted per-instance probability scores, so the curves are re-derived by
retraining the Fold-1 models under the documented protocol. Every metric is
verified against the stored Fold-1 row of the pickle (full precision) BEFORE any
figure is drawn; on any deviation >= 5e-7 the script aborts without drawing.

Protocol (verbatim from cell 45): data = the validated Fold-1 artifacts
(X_train_resampled is the standard-scaled SMOTE-NC training block, train std
1.0002; X_test_scaled the scaled test block); MLP torch.manual_seed(42) (the
notebook seeds random_state + fold index, so seed 42 for the first fold), hidden
20/20, BCELoss, Adam lr=1e-4, 30 epochs, batch 16 shuffle, threshold 0.5; CNN
TabularCNN(23, 32, 4, 1), Adam lr=1e-3, else identical; TabNet
TabNetClassifier(verbose=0), which self-seeds with its default seed 0, eval_set on
the test fold, patience 10, max_epochs 30.
Panels are Fold-1 curves in the style of the eleven existing appendix panels.
"""
from pathlib import Path
import json
import pickle
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / 'results'
OVERLEAF = ROOT / 'overleaf_figures'
TOLERANCE = 5e-7
SEED = 42


def load_fold1():
    with open(RESULTS / 'fold_artifacts_LR.pkl', 'rb') as handle:
        fa = pickle.load(handle)[1]
    X_train = fa['X_train_resampled'].reset_index(drop=True)
    y_train = np.asarray(fa['y_train_resampled']).reshape(-1)
    X_test = fa['X_test_scaled'].reset_index(drop=True)
    y_test = fa['y_test'].reset_index(drop=True).to_numpy()
    return X_train, y_train, X_test, y_test


class MLP(nn.Module):
    """Verbatim from notebook cell 45."""

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.batchNorm1 = nn.BatchNorm1d(hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.batchNorm2 = nn.BatchNorm1d(hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2)
        self.batchNorm3 = nn.BatchNorm1d(hidden_size2)
        self.fc4 = nn.Linear(hidden_size2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.batchNorm1(self.fc1(x)))
        x = F.relu(self.batchNorm2(self.fc2(x)))
        x = self.sigmoid(self.fc4(x))
        return x


class TabularCNN(nn.Module):
    """Verbatim from notebook cell 45."""

    def __init__(self, input_size, num_filters, kernel_size, output_size):
        super(TabularCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size)
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        conv_output_size = (input_size - kernel_size + 1) // 2
        if conv_output_size <= 0:
            raise ValueError('Convolutional output size is non-positive.')
        self.fc1 = nn.Linear(num_filters * conv_output_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.sigmoid(self.fc1(x))
        return x

class TabularDataset(Dataset):
    """Verbatim from notebook cell 45, accepting ndarray/DataFrame input."""

    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_and_score_network(model_name, X_train, y_train, X_test):
    """Retrain one Fold-1 PyTorch network under the cell-45 protocol."""
    train_loader = DataLoader(TabularDataset(X_train, y_train),
                              batch_size=16, shuffle=True)
    test_loader = DataLoader(TabularDataset(X_test, np.zeros(len(X_test))),
                             batch_size=32, shuffle=False)

    # Cell 45 seeds immediately before instantiating the network (seed 42 for the
    # first fold, since the notebook adds the 0-based fold index to random_state).
    torch.manual_seed(SEED)
    if model_name == 'MLP':
        model, learning_rate = MLP(X_train.shape[1], 20, 20, 1), 1e-4
    else:
        model, learning_rate = TabularCNN(X_train.shape[1], 32, 4, 1), 1e-3
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(30):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.unsqueeze(1))
            loss.backward()
            optimizer.step()

    model.eval()
    scores = []
    with torch.no_grad():
        for batch_X, _ in test_loader:
            scores.extend(model(batch_X).tolist())
    return np.array(scores).ravel()


def train_and_score_tabnet(X_train, y_train, X_test, y_test, single_thread=False):
    """TabNetClassifier(verbose=0) with its library-default seed (cell 45)."""
    from pytorch_tabnet.tab_model import TabNetClassifier
    if single_thread:
        torch.set_num_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    model = TabNetClassifier(verbose=0, seed=0)
    model.fit(np.asarray(X_train), np.asarray(y_train),
              eval_set=[(np.asarray(X_test), np.asarray(y_test))],
              patience=10, max_epochs=30)
    return model.predict_proba(np.asarray(X_test))[:, 1]

def fold1_reference():
    """The stored Fold-1 rows of the seeded evaluation (full precision)."""
    frame = pd.read_pickle(RESULTS / 'all_kfold_results.pkl')
    return frame[(frame.Model.isin(['MLP', 'CNN', 'TabNet']))
                 & (frame.Fold == 1)].set_index('Model')


def observed_metrics(name, y_true, y_prob):
    """Recompute cell-45/cell-47 metrics from the replicated Fold-1 scores."""
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 f1_score, precision_score, recall_score)
    y_pred = (y_prob >= 0.5).astype(int)
    ap_positive = average_precision_score(y_true, y_prob, pos_label=1)
    ap_negative = average_precision_score(1 - y_true, 1 - y_prob, pos_label=1)
    return {'Model': name, 'Fold': 1,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, average='macro'),
            'Recall': recall_score(y_true, y_pred, average='macro'),
            'F1 Score': f1_score(y_true, y_pred, average='macro'),
            'PR-AUC': ap_positive,
            'Macro PR-AUC': (ap_positive + ap_negative) / 2}


def draw_roc_panel(name, y_true, y_prob, path):
    """Match the eleven existing appendix panels (567x455 tight-cropped)."""
    from sklearn.metrics import auc, roc_curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    area = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(fpr, tpr, label=f'{name} (AUC = {area:.2f})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return area


def main():
    X_train, y_train, X_test, y_test = load_fold1()
    reference = fold1_reference()
    print(f'Fold-1 replication: {X_train.shape[0]} resampled train rows, '
          f'{X_test.shape[0]} test rows, {int(y_test.sum())} positives.', flush=True)

    gate_log, areas, panels, observed = {}, {}, {}, []

    def gate(name, y_prob):
        """Score one replicated model against its stored Fold-1 row."""
        row = observed_metrics(name, y_test, y_prob)
        stored = reference.loc[name]
        record = {'observed': {k: float(v) for k, v in row.items()
                               if k not in ('Model', 'Fold')},
                  'stored': {k: float(stored[k]) for k in row
                             if k not in ('Model', 'Fold')}}
        record['max_abs_deviation'] = max(
            abs(record['observed'][k] - record['stored'][k])
            for k in record['observed'])
        record['passed'] = bool(record['max_abs_deviation'] < TOLERANCE)
        return row, record

    def draw(name, y_prob):
        filename = {'MLP': 'mlp_roc.png', 'CNN': 'cnn_roc.png',
                    'TabNet': 'tab_roc.png'}[name]
        target = OVERLEAF / filename
        areas[name] = float(draw_roc_panel(name, y_test, y_prob, target))
        panels[name] = str(target)
        (RESULTS / filename).write_bytes(target.read_bytes())
        print(f'{filename}: AUC = {areas[name]:.4f} -> {target}', flush=True)

    def evaluate(name, y_prob, label=None):
        row, record = gate(name, y_prob)
        if label:
            row['Model'] = label
        observed.append(row)
        print(f'{name}: max |observed - stored| = '
              f'{record["max_abs_deviation"]:.3e} '
              f'-> {"PASS" if record["passed"] else "FAIL"}', flush=True)
        if record['passed']:
            draw(name, y_prob)
        return record

    # Cell-45 call order: TabNet first, then MLP, then CNN. Each network is
    # independently seeded by its own protocol, so order only matters for TabNet,
    # whose library self-seeds at construction (AbstractModel.__init__: seed 0).
    y_prob_tab1 = train_and_score_tabnet(X_train, y_train, X_test, y_test)
    print('TabNet trained (attempt 1: protocol-verbatim).', flush=True)
    gate_log['TabNet'] = {'attempt1': evaluate('TabNet', y_prob_tab1,
                                               'TabNet (attempt 1)')}
    gate_log['MLP'] = evaluate('MLP', train_and_score_network(
        'MLP', X_train, y_train, X_test))
    print('MLP trained.', flush=True)
    gate_log['CNN'] = evaluate('CNN', train_and_score_network(
        'CNN', X_train, y_train, X_test))
    print('CNN trained.', flush=True)

    if not gate_log['TabNet']['attempt1']['passed']:
        y_prob_tab2 = train_and_score_tabnet(X_train, y_train, X_test, y_test,
                                             single_thread=True)
        print('TabNet trained (attempt 2: single thread, explicit seed 0).',
              flush=True)
        gate_log['TabNet']['attempt2'] = evaluate('TabNet', y_prob_tab2,
                                                  'TabNet (attempt 2)')
        spread = float(np.abs(y_prob_tab1 - y_prob_tab2).max())
        gate_log['TabNet']['score_spread_between_attempts'] = spread
        if gate_log['TabNet']['attempt2']['passed']:
            draw('TabNet', y_prob_tab2)
        elif spread >= TOLERANCE:
            gate_log['TabNet']['verdict'] = (
                'not deterministic in this environment: two protocol-identical '
                f'runs disagree by {spread:.3e} in per-instance scores, so a '
                'bitwise match to the stored Fold-1 row is unattainable here')
        else:
            gate_log['TabNet']['verdict'] = (
                'deterministic in this environment, but the training trajectory '
                'differs from the stored run (max score spread between attempts '
                f'{spread:.3e}); consistent with a library-version difference '
                'rather than a seeding failure')

    observed_frame = pd.DataFrame(observed).set_index('Model')
    observed_frame.to_csv(RESULTS / 'neural_fold1_replication_metrics.csv')

    drawn = [n for n in ('MLP', 'CNN', 'TabNet') if n in panels]
    status = ('passed' if len(drawn) == 3 else
              'partial' if {'MLP', 'CNN'} <= set(drawn) else 'failed')
    (RESULTS / 'neural_roc_replication.json').write_text(
        json.dumps({'status': status,
                    'panels_drawn': drawn,
                    'protocol': 'notebook cell 45, Fold 1 only (torch seed 42 for '
                                'MLP/CNN; TabNetClassifier(verbose=0), library '
                                'default seed 0; attempt 2 adds single-thread '
                                'execution and an explicit seed)',
                    'tolerance': TOLERANCE,
                    'metrics': gate_log,
                    'roc_auc_fold1': areas,
                    'panels': panels,
                    'note': 'AUC drawn from the replicated Fold-1 probabilities; '
                            'every drawn panel passed the metric gate against '
                            'results/all_kfold_results.pkl at full precision.'},
                   indent=2), encoding='utf-8')
    print(f'Status: {status}; panels drawn: {drawn}.')
    if 'verdict' in gate_log.get('TabNet', {}):
        print('TabNet verdict: ' + gate_log['TabNet']['verdict'])


if __name__ == '__main__':
    main()
