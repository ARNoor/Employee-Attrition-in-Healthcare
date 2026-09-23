# Explainability re-run under fixed seeds (Sections 5.3-5.5)

Script: `run_explainability.py` (new). No model was retrained: Fold-1 artifacts are
loaded from `results/fold_artifacts_LR.pkl`.

## 1. Reproduction gate

Before any explanation is computed, the Fold-1 Logistic Regression is re-predicted and
the five metrics asserted against `results/all_kfold_results.pkl` (exact to 5e-7):

| Metric | Value |
|---|---|
| Accuracy | 0.875000 |
| Precision (macro) | 0.725643 |
| Recall (macro) | 0.831757 |
| F1 (macro) | 0.761105 |
| PR-AUC | 0.728821 |

Fold 1: 23 retained features, 336 test instances, 40 positives, **31 true positives**
explained (matches the `n = 31` reported in Section 5.6).

## 2. Seeds applied

| Stage | Mechanism | Value |
|---|---|---|
| Fold partition | `StratifiedKFold(random_state=42)` | 42 |
| Feature selection | `mutual_info_classif(random_state=111)`; `RFECV(RandomForest(...))` | 111 / 42 |
| SMOTENC | `SMOTENC(random_state=42)` | 42 |
| Logistic Regression | `LogisticRegression(random_state=42)` | 42 |
| LIME (main run) | `LimeTabularExplainer(random_state=42)`; `num_features=12` | 42 |
| LIME (stability, 10 runs) | seeds drawn from `np.random.default_rng(42)` | derived |
| SHAP | `shap.Explainer(..., algorithm='permutation', seed=42)`, applied via `numpy.random.seed` | 42 |
| DiCE | `generate_counterfactuals(random_seed=42, sample_size=1000)`, global numpy/stdlib RNGs also pinned | 42 |
| Bootstrap (1000x) | `np.random.default_rng(42)` | 42 |

Two implementation notes, both required for reproducibility and both documented in the
manuscript's software-implementation table:

1. **LIME is stateful.** `LimeTabularExplainer` stores a `RandomState` that advances on
   every `explain_instance` call, so repeated calls on one shared explainer differ. A
   fresh explainer is therefore constructed per explained instance, making each result
   independent of call order. Verified: fresh explainers with the same seed are identical.
2. **pandas-3 compatibility shim for DiCE 0.12.** `PublicData.__init__` ->
   `_set_feature_dtypes` stores categorical features as string-typed `CategoricalDtype`,
   so `DiceRandom.get_samples` returns string categories, and pandas 3 refuses to assign a
   string into the float64 candidate frame built at `dice_random.py:110-111` (pandas 1.x
   silently upcast). The query instance is passed as object dtype and the returned frame is
   coerced back to numeric for the change-direction comparison. Dtype-only: generated
   feature values are unchanged.

## 3. Results

**SHAP top features:** OverTime, Shift, JobInvolvement, Age, DistanceFromHome,
NumCompaniesWorked, YearsInCurrentRole, MonthlyIncome, TotalWorkingYears,
JobRole_Therapist, YearsSinceLastPromotion, DailyRate. The three mu/omega/eta-feasible
features are the top three.

**LIME-SHAP intersection narrows with N:** 18 features at N=32, 14 at N=16, 13 at N=14,
9 at N=12, 7 at N=10, 5 at N=8.

**Feasible umbrella (Eq. 1):** nine members — OverTime, Shift, JobInvolvement, JobLevel,
EnvironmentSatisfaction, YearsSinceLastPromotion, DailyRate, HourlyRate, MonthlyRate
(= the notebook's counterfactual stage, notebook_raw.py:1650). JobLevel and
EnvironmentSatisfaction are not retained by the Fold-1 selection (inert); HourlyRate
never reaches the top-N intersection.

**Feasible set F*_N = C_N ∩ F varies with N** (the change made to run_explainability.py):

| N | \|C_N\| | F*_N | \|F*_N\| |
|---|---|---|---|
| 32 | 18 | DailyRate, JobInvolvement, MonthlyRate, OverTime, Shift, YearsSinceLastPromotion | 6 |
| 16 | 14 | DailyRate, JobInvolvement, OverTime, Shift | 4 |
| 14 | 13 | DailyRate, JobInvolvement, OverTime, Shift | 4 |
| 12 | 9 | DailyRate, JobInvolvement, OverTime, Shift | 4 |
| 10 | 7 | JobInvolvement, OverTime, Shift | 3 |
| 8 | 5 | JobInvolvement, OverTime, Shift | 3 |

Larger N admits more candidate features, which exposes further umbrella members, so the
feasible set grows monotonically with N. Two properties are invariant: the core
{JobInvolvement, Shift, OverTime} is feasible in every configuration, and no infeasible
attribute can enter a recommendation because DiCE varies only F*_N. Every configuration
generates the same 93 counterfactual change-sets, so the enlargement is purely additive:
N <= 10 reproduces the previous three-feature frequencies exactly.

**Individual feature-change frequencies at the a-priori N = 12** (mean over 1000
bootstraps of 93 change-sets; per-N values in results/cf_feature_frequencies_all_N.csv):

| Feature changed | Mean | 95% CI |
|---|---|---|
| Shift[inc] | 61.02 | [52, 70] |
| DailyRate[inc] | 43.16 | [33, 53] |
| OverTime[dec] | 39.90 | [31, 49] |
| DailyRate[dec] | 38.08 | [29, 48] |
| JobInvolvement[inc] | 8.98 | [4, 15] |
| JobInvolvement[dec] | 0.97 | [0, 3] |

**Recommendation sets.** At N <= 10 (three-feature configuration, unchanged from the
previous run): Shift[inc] 40; OverTime[dec]+Shift[inc] 19; JobInvolvement[inc]+OverTime[dec] 9;
JobInvolvement[inc]+Shift[inc] 7; JobInvolvement[inc] 5; OverTime[dec] 5; all three 4;
JobInvolvement[dec]+OverTime[dec]+Shift[inc] 3; JobInvolvement[dec]+Shift[inc] 1
(9 distinct sets). At N = 12/14/16: 15 distinct sets; modal {DailyRate[dec], Shift[inc]}
and {DailyRate[inc], Shift[inc]}, 22 occurrences each. At N = 32: 10 modification types
over 27 distinct sets; the most frequent ({MonthlyRate[dec], Shift[inc]}) occurs 12 times.
Full per-N tables: results/cf_recommendation_sets_all_N.csv.

## 4. Figures regenerated

`LIME_output_new.jpg`, `SHAP_output_new.PNG`, and `top_{all,16,14,12,10,8}_{recomm,freq}.png`
(12 figures) written to `results/xai_figures_all_N/` and synced into `LSC_XAI_revised__Copy_`
(all 12 panels re-copied after the feasible-set change, so every threshold figure reflects
the per-N variation). Superseded figure versions (`figures_before_feasible_variation/`,
`figures_before_xai_rerun/`, `figures_before_seeded_update/`) were removed in the
cleanup and are preserved in `results/_cleanup_archive_2026-09-19.zip`. Supplementary artefacts:
`results/lime_aggregate.png`, `results/shap_bar.png`, `results/lime_instance.html`.

## 5. Manuscript edits

Seeding-related edits (earlier pass):

- Software-implementation table: LIME/SHAP row rewritten (was "no random seed fixed");
  DiCE row gained `random_seed=42` / `sample_size=1000`; new "XAI evaluation (Section 5.6)"
  row added; caption now points to the interpretability seeds.
- Section 5.6 determinism paragraph: "no random seed was fixed" and "likewise unseeded"
  replaced; SHAP ceiling now stated as deterministic for the fixed seed.

Feasible-set-variation edits (applied via the anchored script `apply_tex_edits.py`):

- Section 5.5: rewritten as a genuine sensitivity result — F*_N holds three features at
  N in {8,10}, four at N in {12,14,16} and six at N=32; the C_N core {JobInvolvement,
  Shift, OverTime} is invariant and the enlargement is purely additive (N <= 10 reproduces
  the three-feature frequencies exactly). All JobLevel[inc] claims removed (JobLevel is
  not a retained Fold-1 feature).
- Sensitivity figure caption: the twelve panels are now described as genuinely different
  (identical within the N <= 10 and N in {12,14,16} groups, as observed in the data).
- Table 9 (tab:common_feat): regenerated LIME/SHAP top-12 columns; nine starred common
  features at N=12 (the old table's JobLevel, EnvironmentSatisfaction and unstarred
  MonthlyRate/DailyRate rows are gone — none matches the seeded top-12s).
- tab:feasibility_complete: now assesses the nine C_12 features. New author-judgment rows:
  Age (0/0/0), JobRole_Therapist (0/0/1), MonthlyIncome (1/0/1, anchored to the grade
  structure and collinear with JobLevel) — all Psi=0; DailyRate added as a bold Psi=1 row
  with a pay-band rationale. DistanceFromHome and TotalWorkingYears rows removed (no
  longer in C_12; their exclusion rationales remain in the surrounding text).
- Companion paragraph after tab:feasibility_complete documents the umbrella members
  outside C_12 (HourlyRate, MonthlyRate, YearsSinceLastPromotion, JobLevel,
  EnvironmentSatisfaction) and why the last two can never enter F*_N.
- Section 5.4 bullets: DailyRate added with its organizational meaning and the per-N
  availability note (MonthlyRate, YearsSinceLastPromotion at N=32).
- Supplementary tables S1/S2: N=12 values (S1: Shift[inc] 61.02 [52,70], DailyRate[inc]
  43.16 [33,53], OverTime[dec] 39.90 [31,49], DailyRate[dec] 38.08 [29,48],
  JobInvolvement[inc] 8.98 [4,15], JobInvolvement[dec] 0.97 [0,3]; S2: 15 combinations
  summing to 93, verified against the rerun CSV).
- Section 5.6: LSC-XAI sparsity updated to 1 - 4/23 = 0.8261 with the per-N note; the
  counterfactual-quality paragraph now states the generator varies the N=12 feasible set.
- `validate_seeded_manuscript.py`: guards for the seed strings, the per-N feasible sizes
  (6,4,4,4,3,3), the new headline numbers, and against the retired invariance wording.

## 6. Open items requiring an author ruling

1. **RESOLVED.** Table 9 and `tab:feasibility_complete` now match the seeded model: the
   N=12 intersection is `{Age, DailyRate, JobInvolvement, JobRole_Therapist, MonthlyIncome,
   NumCompaniesWorked, OverTime, Shift, YearsInCurrentRole}` and the feasibility table
   assesses exactly those nine features. The mu/omega/eta assignments for the four
   newly-common features were made consistent with the implemented umbrella
   (notebook_raw.py:1650): DailyRate 1/1/1 (feasible; in F*_N from N=12 upward); Age
   0/0/0; JobRole_Therapist 0/0/1; MonthlyIncome 1/0/1. These are author judgments in the
   same sense as the original table (the manuscript's subjectivity note still applies)
   and should be confirmed by the authors.
2. **Section 5.6 metrics are still unseeded-derived.** `tab:xai_metrics` (LIME stability
   0.8034, fidelity 0.9375/0.9405, sparsity) and `tab:cf_metrics` (proximity/plausibility)
   were not recomputed: the repo does not contain that measurement code, and two
   specifications are missing - the resampling unit for the stability CI, and which 15 of
   the 31 true-positive instances the counterfactual survey used. Provide these and the
   stage can be regenerated the same way.
3. **PDF build** not performed (no TeX engine locally), by author decision.

## 7. Round-2 manuscript edits (reviewer comments 7-9)

Applied via the anchored script `apply_tex_edits_r2.py`; all additions in blue.

- Abstract: corrected driver list (was "including job level, shift structure and
  overtime") to job involvement, shift structure, overtime workload and daily pay rate.
- Section 6.2: the "most frequently recommended interventions increasing job level"
  sentence replaced with the corrected recommendations (shift-category reassignment,
  daily pay-rate adjustment, overtime reduction; job-involvement changes in the
  lower-frequency combination sets, Section 5.5 + Tables S1/S2).
- Section 6.2 claim-hierarchy example: the JobLevel counterfactual example (factually
  impossible - JobLevel is not a model input) replaced with an upward shift-category
  reassignment example.
- Section 3.2.1: exploratory Shift sentence corrected - the data show attrition 19.8%
  (category 0), 6.1% (1), 3.2% (2), 11.1% (3); the old "fewer shift assignments ->
  higher retention" wording was backwards. Categorical reading stated explicitly.
- Section 5.4: single operational definition added ([inc]/[dec] = movement up/down the
  encoded ordinal scale; Shift[inc] = reassignment between shift categories, not more
  shifts); Shift bullet rewritten accordingly; no-sign-reversal note added (EDA rates,
  fitted coefficients Shift -1.65 / JobInvolvement -1.14 / DailyRate -0.35 /
  OverTime +3.97 on the attrition logit, and counterfactual labels all agree).
- Section 5.5: JobInvolvement's N=12 frequencies named (8.98 [4,15] up; 0.97 [0,3] down).
- Validator: extended with negative guards (the five retired phrases) and positive
  guards (the corrected phrases); passing.
- Response files: reviewer_response_comment_7.md, _8.md, _9.md.

## 8. Round-3 manuscript edits (reviewer comments 13-14)

Applied via the anchored script `apply_tex_edits_r3.py`; all additions in blue.
Table 7 itself had already been reworked to five-fold mean +/- SD (Step 5 now equals
the Table 4 Logistic Regression row exactly); this round adds the remaining pieces.

- Section 5.3 lead + Table 7 caption: the ablation model is now identified in blue
  (Logistic Regression base model selected in Section 5.4); the caption also states
  every configuration uses the same fold-wise (leakage-safe) protocol as Table 4.
- Section 5.3: few-positive-cases caveat added (each held-out fold holds roughly 40
  attrition cases; descriptive fold-mean comparisons, no significance tests);
  'baseline point estimates' wording corrected to 'Step 1 means'.
- Section 5.6: sparsity paragraph opening value corrected 0.8696 -> 0.8261 (the last
  stale occurrence in active text; 0.4783 unchanged, design-consequence framing and
  per-N note already in place from earlier rounds).
- Validator: negative guard '0.8696 absent from active text'; positive guards for the
  corrected phrases, the Table 7 Step-5 row string, and the existing design-consequence
  sentences; passing.
- Response files: reviewer_response_comment_13.md, _14.md.

## 9. Directory cleanup (2026-09-19)

Removed as redundant; every removed item is preserved in `results/_cleanup_archive.zip`
(5.08 MB, 125 entries, verified with a full tar listing before deletion):

- One-shot anchored edit scripts `apply_tex_edits.py`, `apply_tex_edits_r2.py`,
  `apply_tex_edits_r3.py` (already applied; documented in Sections 5, 7 and 8).
- Superseded notebook backup `Employee_Attrition_in_Healthcare.ipynb.bak_pre_prauc`.
- One-shot extraction utilities `extract.ps1`, `extract_py.ps1`.
- Abandoned `employee-attrition-ml/` scaffold (stub modules plus its own unused
  `.venv`, ~220 MB; a zombie python process from that venv was stopped first).
  Not archived: the scaffold `.venv` (regenerable from the archived requirements.txt).
- `catboost_info/` training logs and the root `__pycache__/` bytecode (regenerable).
- Superseded figure backups `results/figures_before_seeded_update/`,
  `results/figures_before_xai_rerun/`, `results/figures_before_feasible_variation/`
  and the superseded tex backup `results/cas-sc-sample.before_xai_seeding.tex`.

Kept deliberately: `notebook_raw.py` (cited as provenance by run_explainability.py),
`results/cas-sc-sample.before_seeded_update.tex` (read by the validator PRE-EDIT
check), `seeded_manuscript.diff`, `feasible_rerun_log.txt`, all pipeline outputs,
data files, notebooks, active scripts and the reviewer responses.

## 10. All-folds LSC-XAI and the neural ROC replication (2026-09-20)

**All-folds run** (`run_explainability_all_folds.py`, per-fold checkpoints + pooled
aggregation; folds run in parallel with `--fold k`, pooled via `pool`):

- Every fold passed the 5e-7 reproduction gate against `all_kfold_results.pkl`.
- Per-fold feasible sizes |F*_N| (N = 32..8): f1 6/4/4/4/3/3, f2 5/5/4/3/3/3,
  f3 6/6/4/2/2/2, f4 4/3/3/2/2/2, f5 4/4/4/3/3/2. JobInvolvement is absent from the
  folds 3-4 models and therefore from their feasible sets at every N; HourlyRate is
  additionally dropped by fold 4; EnvironmentSatisfaction IS retained by folds 3-5
  and enters their feasible sets at large N (it does not appear in the Fold-1-only
  figures, where the model does not retain it).
- Pooled change-sets per N (32/16/14/12/10/8): 455/455/449/440/440/427 over
  47/46/29/21/12/12 distinct sets. Fold 5 skipped 2 instances per N (4 at N=8) for
  which DiCE found no counterfactual at all (recorded in the manifest; a sequential
  first run had aborted on exactly these because dice raises instead of returning
  an empty result - fixed by catching UserConfigValidationException per instance).
- A first sequential attempt also briefly ran three duplicate processes (two stale
  launches from earlier turns had survived); all were killed and the run was
  restarted cleanly, so no output was contaminated.
- One one-character bug was introduced and fixed during restructuring: the DiCE
  keyword `total_CFs` was mistyped `total_CFS`, silently landing in **kwargs and
  raising a misleading binding TypeError; the fix is character-exact.
- Outputs: `all_folds_figures/` (12 pooled panels),
  `results/all_folds_{common_features_by_N,cf_feature_frequencies,cf_recommendation_sets}.csv`,
  `results/xai_run_manifest_all_folds.json`, per-fold checkpoints
  `results/all_folds_checkpoint_fold{1..5}.pkl`, per-fold logs. NOT copied into
  `overleaf_figures/` or the manuscript: swapping the Fold-1 headline figures for
  pooled ones is a coordinated text decision (Section 5.4/5.5, Tables S1/S2 and the
  fold-scoping statement all describe the Fold-1 configuration).

**Neural ROC replication** (`neural_roc_replication.py`, Fold 1 only, cell-45
protocol, gate = bitwise match to `all_kfold_results.pkl`):

- MLP: PASS, max |observed-stored| = 0.0 (6 metrics) -> `mlp_roc.png` (AUC 0.9101).
- CNN: PASS, max |observed-stored| = 0.0 -> `cnn_roc.png` (AUC 0.8798).
- TabNet: FAIL and not reproducible - two protocol-identical runs disagree by 0.974
  in per-instance scores (best epoch 19 vs 26), i.e. TabNet training is
  non-deterministic in the current environment (torch 2.14 / current pytorch_tabnet;
  the original sklearn-1.6.1-era environment was removed in the 2026-09-19 cleanup
  and its exact versions are unrecoverable - the notebook holds no version
  fingerprints). No `tab_roc.png` was drawn: the stored Fold-1 row cannot be
  certified, and the stale pre-seeding panel must not be presented as seeded.
- Outputs: `overleaf_figures/{mlp,cnn}_roc.png` + copies in `results/`,
  `results/neural_roc_replication.json`, `results/neural_fold1_replication_metrics.csv`.

## 11. Manuscript integration (round 4, 2026-09-21)

`apply_tex_edits_r4a/b/c/d.py` integrated the five-fold LSC-XAI replication and the
certified neural ROC panels into `LSC_XAI_revised__Copy_/cas-sc-sample.tex`
(17 anchored edits; all new text in blue):

- Scope statement (Sec. 5.4) and threshold paragraph: pipeline "run on Fold 1 and then
  replicated across all five folds"; same a-priori threshold applied per fold.
- New blue paragraph in Sec. 5.5 ("Five-fold replication."): 154 TP instances;
  pooled change-sets 455/455/449/440/440/427 at N = 32/16/14/12/10/8; per-fold
  feasible-set sizes; OverTime+Shift feasible in all 30 fold-threshold cells,
  JobInvolvement in the 17 cells whose fold retains it; DailyRate fold-specific at
  N = 12 (Fold 1 only); pooled N = 12 means Shift[inc] 330.05 [312, 348],
  OverTime[dec] 198.90 [179, 219], JobInvolvement[inc] 44.90 [32, 59],
  DailyRate[inc] 43.19 [31, 56]; pooled N = 10 three-feature configuration
  (Shift[inc] 342.97 [326, 359]; JobInvolvement[inc] 61.00 [47, 75]).
- Sec. 5.6 clause and Sec. 6.2 clause: replication confirms the universal
  OverTime/Shift core and reorders the pooled ranking (job involvement third,
  DailyRate fold-specific); Tables 10-11 metrics remain single-fold protocol
  evaluations. Limitations paragraph extended accordingly.
- Supplementary Table S1: pooled five-fold block appended (8 rows incl. pooled-only
  Shift[dec] 3.02 [0, 7] and OverTime[inc] 1.08 [0, 3]).
- New appendix section "Pooled Five-Fold Recommendation Combinations at N = 12"
  (Supplementary Table, 21 rows, top: Shift[inc] 159; OverTime[dec]+Shift[inc] 92;
  OverTime[dec] 56) + pooled figure `af_top_12_{recomm,freq}.png`
  (label fig:all_folds_pooled).
- Appendix table `tab:all_folds_feasible` after the removed-features table:
  per-fold |F*_N| (change-sets) matrix, e.g. Fold 1: 6 (93) ... 3 (93),
  Fold 5: 4 (86) ... 2 (71).
- ROC appendix: certified `mlp_roc.png` / `cnn_roc.png` subfigures added with
  bitwise-reproduction note (MLP AUC 0.9101, CNN AUC 0.8798); TabNet panel omitted
  with non-determinism reason; certified copies copied over the stale 09-14 pngs in
  the manuscript folder.
- Software table: new row "LSC-XAI five-fold replication" (per-fold seeds,
  5e-7 reproduction gates before explanation).

Validation: `validate_seeded_manuscript.py` extended with a round-4 block that
cross-checks every quoted pooled number against
`results/all_folds_cf_recommendation_sets.csv` and
`results/all_folds_cf_feature_frequencies.csv` (pooled totals per N, 21 pooled
N = 12 sets summing to 440, top-set identities, pooled frequency means/CIs) and the
ROC appendix panel list (11 classical + certified MLP/CNN). Full validator run:
13/13 PASS; source syntax clean; no missing figure paths.



## 12. Comment 12: cross-fold stability of the recommendation set (2026-09-21)

Stability statistics derived for the reviewer response (all from
`results/all_folds_common_features_by_N.csv` + the pooled CSVs):

- Mean pairwise Jaccard of the per-fold feasible set F*_N across the 5 folds:
  0.5643 (N=32), 0.5579 (16), 0.6000 (14), 0.7167 (12), 0.8000 (10), 0.8000 (8) —
  stability rises as the threshold tightens.
- F*_N is an exactly nested chain across folds at N <= 12:
  {OverTime, Shift} (folds 3-4) subset {+JobInvolvement} (folds 2, 5)
  subset {+DailyRate} (fold 1) at N = 12.
- Recommendation stability at N = 12 per fold: Shift[inc] most frequent in every
  fold (fold means 61.02/63.11/68.91/70.04/67.12; 66-80% of each fold's
  change-sets); OverTime[dec] second in folds 2-4, third in folds 1 (DailyRate[inc]
  43.16 > 39.90) and 5 (JobInvolvement[inc] 25.64 > 23.06); no direction sign
  reversals.

Manuscript: one new blue passage added to the Section 5.5 replication paragraph
(Jaccard range, nested chain, rank-stability phrasing "second in folds 2--4 and
never below third in any fold"). Validator: stability strings pinned and the
Jaccard values re-derived from the CSVs. Response file:
`reviewer_response_comment_12.md`.

## 13. Illustrative TabNet Fold-1 ROC panel (2026-09-21)

`tabnet_roc_fold1.py`: single protocol-verbatim run (TabNetClassifier(verbose=0,
seed=0), torch/np seeds 0, early stopping best epoch 19) on the Fold-1 artifacts.
Panel `overleaf_figures/tab_roc.png` (+ copy in `results/`), ROC-AUC **0.7867**.
This run bitwise-matches attempt 1 of the replication (identical observed
metrics) but not the stored Fold-1 row (stored PR-AUC 0.5182 vs 0.4384) -
library-version difference, as recorded in Section 10. Status in
`results/tabnet_roc_fold1.json`: "illustrative single run - NOT certified".
No manuscript or validator changes; the manuscript's statement that TabNet has
no certified curve remains accurate.

Update (same day): the panel was inserted into the ROC appendix as requested.
`tab_roc.png` copied into `LSC_XAI_revised__Copy_/`; subfigure "TabNet
(illustrative)" added after the 1D-CNN panel; figure caption and body text now
state the panel is an illustrative single run (ROC-AUC 0.7867) that cannot be
certified against the stored Fold-1 row (non-deterministic training), read as
indicative only. Validator: `tab_roc.png` added to the expected panel list, new
stale/required strings pinned, and the quoted AUC cross-checked against
`results/tabnet_roc_fold1.json`. Full validator run passes.
