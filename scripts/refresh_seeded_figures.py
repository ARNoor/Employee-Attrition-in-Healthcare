"""Refresh recoverable manuscript figures without training any models."""
from pathlib import Path
import base64
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / 'manuscript'
BACKUP = ROOT / 'results' / 'figures_before_seeded_update'
BACKUP.mkdir(exist_ok=True)

def backup(name):
    if not (BACKUP/name).exists():
        shutil.copy2(DEST/name, BACKUP/name)

notebook = json.loads((ROOT/'notebooks'/'Employee_Attrition_in_Healthcare.ipynb').read_text(encoding='utf-8'))
for name, index in [('tab_cm.png',2),('mlp_cm.png',52),('cnn_cm.png',54)]:
    image = notebook['cells'][45]['outputs'][index]['data']['image/png']
    backup(name)
    (DEST/name).write_bytes(base64.b64decode(image))
    assert plt.imread(DEST/name).size > 0

pairs = pd.read_csv(ROOT/'results/verified_pairwise_recall.csv')
df = pd.read_pickle(ROOT/'results/all_kfold_results.pkl')
models = sorted(df.Model.unique())
matrix = pd.DataFrame(np.nan, index=models, columns=models)
for _, row in pairs.iterrows():
    matrix.loc[row['Model 1'],row['Model 2']] = row.p_raw
    matrix.loc[row['Model 2'],row['Model 1']] = row.p_raw
fig, ax = plt.subplots(figsize=(12,10))
im = ax.imshow(matrix, vmin=0, vmax=1, cmap='viridis')
ax.set_xticks(range(14), models, rotation=65, ha='right')
ax.set_yticks(range(14), models)
for i in range(14):
    for j in range(14):
        if i != j:
            v=matrix.iloc[i,j]
            ax.text(j,i,f'{v:.4f}',ha='center',va='center',fontsize=6,color='white' if v<.5 else 'black')
ax.set_title('Exact pairwise Wilcoxon raw p-values (macro-recall)\nAll 91 Holm-adjusted p-values = 1.0000')
fig.colorbar(im,ax=ax,label='Raw two-sided p-value')
fig.tight_layout()
backup('wilcoxon_signed_rank_test.jpg')
fig.savefig(DEST/'wilcoxon_signed_rank_test.jpg',dpi=220)
plt.close(fig)

order=['Logistic Regression','KNN','Decision Tree','SVM','Ada Boost','Random Forest','Gradient Boosting','XGBoost','LightGBM','CatBoost','MLP','CNN','TabNet','Stacking']
means=df.groupby('Model').Recall.mean().reindex(order)
fig,ax=plt.subplots(figsize=(12,5))
ax.scatter(range(1,15),means)
for i,(name,value) in enumerate(means.items(),1):
    ax.annotate(name,(i,value),xytext=(0,8 if i%2 else -16),textcoords='offset points',ha='center',fontsize=8)
ax.set_xticks(range(1,15)); ax.set_ylim(.70,.855)
ax.set_xlabel('Assigned ordinal complexity score (1--14; heuristic, not measured runtime)')
ax.set_ylabel('Mean macro-recall (5 folds)'); ax.grid(alpha=.2)
fig.tight_layout()
backup('mean_recall_vs_complexity_graph.PNG')
fig.savefig(DEST/'mean_recall_vs_complexity_graph.PNG',dpi=220)
plt.close(fig)
print('Updated 3 saved Fold-1 confusion matrices and 2 regenerated statistical plots.')
print('Neural ROC panels were NOT regenerated; per-instance score arrays are not persisted in the metric pickle.')
