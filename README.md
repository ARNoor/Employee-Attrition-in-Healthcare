# LSC-XAI — Employee Attrition in Healthcare (Reproducibility Package)

Reproducibility materials for the **LSC-XAI** framework applied to healthcare
employee attrition: 14-class classifier comparison under seeded 5-fold CV,
LIME/SHAP/DiCE explainability with feasibility filtering, counterfactual
recommendation mining, statistical validation (Friedman/Wilcoxon/Holm), the
Saudi cross-dataset validation, and the LaTeX manuscript with validation tooling.

## Repository layout

```
.
├── README.md              this file
├── requirements.txt       pinned Python dependencies (tested on Python 3.12)
├── .gitignore             excludes environments, caches and runtime outputs
├── data/                  all input datasets
│   ├── watson_healthcare_modified.csv        raw IBM-Watson-derived dataset (notebook input)
│   ├── df_encoded.csv                        processed/encoded dataset (notebook input)
│   ├── df_encoded_outlhandled_new.csv        cross-dataset-validation train set
│   └── df_medical_sector_preprocessed.csv    Saudi medical-sector dataset
├── notebooks/
│   ├── Employee_Attrition_in_Healthcare.ipynb  main pipeline (training + XAI stages)
│   └── notebook_raw.py                         line-numbered export cited as provenance
│                                               by scripts (intentionally unmodified)
├── scripts/               all runnable pipeline stages (paths resolve relative to repo root)
├── manuscript/            complete Elsevier CAS SC LaTeX source + all figures
│                          (cas-sc-sample.tex, .cls/.sty/.bib/.bst, *.png|jpg)
├── results/               anchor artifacts, validator inputs and canonical outputs
└── overleaf_figures/      staging directory used by the ROC scripts (created/kept for runtime)
```

## Environment setup

```powershell
python -m venv .venv
.venv\Scripts\activate          # Windows (on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

All commands below assume the repository root as the current directory and an
activated environment. Seeded stages use fixed seeds (42 for LIME/SHAP/DiCE/
bootstrap, 111 for mutual information, 0 for TabNet) documented in the manuscript's
software-implementation table and in `results/xai_rerun_summary.md`.

## Reproduction workflow

| # | Command | What it does |
|---|---------|--------------|
| 0 | *(optional full retrain)* run `notebooks/Employee_Attrition_in_Healthcare.ipynb` top-to-bottom | Regenerates `data/df_encoded.csv`, `results/all_kfold_results.pkl` and `results/fold_artifacts_LR.pkl` (all three shipped, so this step is only needed to retrain from scratch) |
| 1 | `python scripts/audit_seeded_results.py` | Recomputes Friedman χ², all 91 exact Wilcoxon + Holm values; rewrites `results/verified_pairwise_recall.csv`; asserts against the manuscript |
| 2 | `python scripts/run_explainability.py` | Seeded Fold-1 LSC-XAI stage (LIME/SHAP/DiCE, feasible sets, panels); writes `results/` CSVs + figures into `manuscript/` |
| 3 | `python scripts/run_explainability_all_folds.py` | Five-fold replication + pooled panels (`all/1..5/pool` argument); writes `all_folds_figures/` + `results/all_folds_*.csv` |
| 4 | `python scripts/neural_roc_replication.py` | Certified MLP/CNN Fold-1 ROC replications gated at 5e-7 against the stored metrics |
| 5 | `python scripts/tabnet_roc_fold1.py` | Illustrative TabNet Fold-1 ROC panel (non-certified; see script docstring) |
| 6 | `python scripts/refresh_seeded_figures.py` | Extracts Fold-1 confusion matrices from the notebook outputs and redraws the statistical plots into `manuscript/` |
| 7 | `python scripts/validate_seeded_manuscript.py` | **End-to-end check**: manuscript syntax, all table values, seeds, manifests, figure paths → must end with `FINAL SOURCE/DATA VALIDATION PASSED` |
| 8 | `python scripts/make_compile_bundle.py` | Builds a self-contained Overleaf-ready zip of `manuscript/` |
| 9 | `python scripts/LSC_XAI_training_validation_ft_Saudi_dataset.py [--smote]` | Saudi cross-dataset validation (3 setups, bootstrap CIs); reports land in `outputs/` |
| 10 | `python scripts/recompute_stats.py` | Compatibility alias of step 1 |

Steps 1 and 7 are fast; steps 2–5 are compute-heavy (LIME/SHAP/DiCE per fold).

## Notes, assumptions and deviations from the original working directory

* **Layout rename:** the manuscript directory was `LSC_XAI_revised__Copy_/` and the
  notebook/data lived at the repo root in the original workspace; they are now in
  `manuscript/`, `notebooks/` and `data/`. Every script resolves paths from its own
  location (`scripts/..`), and notebook path strings were retargeted accordingly
  (sources only — cell outputs are untouched).
* **`results/fold_artifacts_LR.pkl`** (trained per-fold Logistic Regression
  artifacts, incl. SMOTE-NC blocks and scalers) is the input consumed by the
  explainability and ROC scripts; the notebook's save cell writes to this path.
* **`notebook_raw.py`** still contains original Colab paths on purpose: it is the
  line-numbered provenance export cited by `run_explainability*.py` docstrings.
* **TabNet is non-deterministic** in this software environment; its ROC panel is
  explicitly illustrative (status recorded in `results/tabnet_roc_fold1.json`).
* Runtime outputs (`outputs/`, `all_folds_figures/`, `results/xai_figures_all_N/`,
  checkpoints, logs, the compile zip) are git-ignored — rerun the scripts to get them.
* The full reproducibility changelog for the review rounds lives in
  `results/xai_rerun_summary.md`.
