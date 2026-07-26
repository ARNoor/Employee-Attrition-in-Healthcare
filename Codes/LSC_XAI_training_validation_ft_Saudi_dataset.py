"""
=====================================================================================
ATTRITION MODEL — CROSS-DATASET VALIDATION (all 3 setups in one script)
=====================================================================================

Runs three validation setups on the same harmonized 17-feature set built from:
  - Train dataset : df_encoded_outlhandled_new.csv   (general HR, engineered/encoded)
  - Saudi dataset : df_medical_sector_preprocessed.csv (Saudi healthcare, manually
                    preprocessed — uses the same column names/encodings as train
                    for most fields, unlike the earlier saudi_processed_data.csv)

  Setup 1 — Cross-domain (cold transfer):
      Train on general HR data only. Test on 100% of Saudi data.
  Setup 2 — Partial Saudi in training:
      Train on general HR + a stratified 50% random half of Saudi data.
      Test on the remaining stratified 50% Saudi holdout half.
  Setup 3 — Saudi-only (in-domain):
      Train and test entirely on Saudi data, stratified 80/20 split.

By default, class imbalance is handled via class_weight='balanced' in Logistic
Regression (the version used in the manuscript's reported results — see notes
below). Pass --smote on the command line to instead run the StandardScaler +
SMOTE-NC variant (train-fold-only, no leakage) for comparison; this was tested
and found to perform no better (Setup 3 balanced accuracy dropped from 0.720 to
0.655), so it is NOT the default or recommended pipeline, but is included here
for full reproducibility of that comparison.

FEATURE ALIGNMENT — DOCUMENTED ASSUMPTIONS
--------------------------------------------
Only columns representing the same underlying construct in both datasets are
used (17 predictors total):

  DIRECTLY IDENTICAL ENCODING (used as-is, no rescaling) — 13 features:
    Gender, BusinessTravel, Education, EnvironmentSatisfaction, JobInvolvement,
    JobSatisfaction, WorkLifeBalance, OverTime, JobRole_Administrative,
    JobRole_Nurse, JobRole_Other, MaritalStatus_Married, MaritalStatus_Single
    (verified: identical value ranges on both sides)

  CONTINUOUS, DIFFERENT RAW SCALE — 4 features:
    - MonthlyIncome: train is confirmed log10-transformed (train range
      3.744-4.307 aligns tightly with log10 of Saudi's raw dollar values).
      Saudi's MonthlyIncome is log10-transformed to match.
    - Age, DistanceFromHome, YearsAtCompany: train's transform does not match
      any simple closed-form function of Saudi's raw values (checked and
      ruled out log10/log1p/sqrt — likely Yeo-Johnson or similar). These are
      min-max normalized to [0,1] INDEPENDENTLY per dataset — this preserves
      rank/relative position but not an exact shared scale (stated limitation).

  EXCLUDED (no defensible counterpart in the other dataset):
    Age-bin ambiguity in the OLDER saudi_processed_data.csv file is moot here
    since this Saudi file has raw Age. Train-only: DailyRate, HourlyRate,
    MonthlyRate, JobLevel, NumCompaniesWorked, PercentSalaryHike,
    PerformanceRating, RelationshipSatisfaction, Shift, TotalWorkingYears,
    TrainingTimesLastYear, YearsInCurrentRole, YearsSinceLastPromotion,
    YearsWithCurrManager, Department_Maternity, Department_Neurology,
    EducationField_*, JobRole_Therapist. Saudi-only: JobRole_Admin (distinct
    from JobRole_Administrative), MaritalStatus_Divorced (no way to verify
    this equals train's implicit zero-zero reference category).

EVALUATION
----------
Full confusion matrix, precision, recall/sensitivity, specificity, F1,
balanced accuracy, and PR-AUC (with its no-skill baseline = test set positive
rate), each with a 95% bootstrap CI (n_boot=2000, resampling the test set).

USAGE
-----
    python3 run_all_setups.py                 # default: class_weight='balanced'
    python3 run_all_setups.py --smote          # StandardScaler + SMOTE-NC variant

Expects the following files in /mnt/user-data/uploads/ (edit paths below if
running elsewhere):
    df_encoded_outlhandled_new.csv
    df_medical_sector_preprocessed.csv
=====================================================================================
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    balanced_accuracy_score, average_precision_score,
)

RANDOM_STATE = 42
N_BOOT = 2000
CI = 0.95

TRAIN_PATH = "/mnt/user-data/uploads/df_encoded_outlhandled_new.csv"
SAUDI_PATH = "/mnt/user-data/uploads/df_medical_sector_preprocessed.csv"

IDENTICAL_FEATURES = [
    "Gender", "BusinessTravel", "Education", "EnvironmentSatisfaction",
    "JobInvolvement", "JobSatisfaction", "WorkLifeBalance", "OverTime",
    "JobRole_Administrative", "JobRole_Nurse", "JobRole_Other",
    "MaritalStatus_Married", "MaritalStatus_Single",
]
RESCALED_FEATURES = ["Age", "DistanceFromHome", "YearsAtCompany"]  # per-dataset min-max
LOG_FEATURE = "MonthlyIncome"  # log10 applied to Saudi only; train already log10-scale
FEATURES = IDENTICAL_FEATURES + RESCALED_FEATURES + [LOG_FEATURE]

# column index ranges within FEATURES, used only for the --smote SMOTE-NC path
CATEGORICAL_IDX = list(range(0, len(IDENTICAL_FEATURES)))
CONTINUOUS_IDX = list(range(len(IDENTICAL_FEATURES), len(FEATURES)))

METRIC_LABELS = {
    "precision": "Precision",
    "recall_sensitivity": "Recall / Sensitivity",
    "specificity": "Specificity",
    "f1": "F1-score",
    "balanced_accuracy": "Balanced Accuracy",
    "pr_auc": "PR-AUC (Average Precision)",
}


# ---------------------------------------------------------------------------
# Feature harmonization
# ---------------------------------------------------------------------------
def minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return series * 0.0
    return (series - lo) / (hi - lo)


def build_train_common(train_raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=train_raw.index)
    for c in IDENTICAL_FEATURES:
        out[c] = train_raw[c].astype(float)
    for c in RESCALED_FEATURES:
        out[c] = minmax(train_raw[c].astype(float))
    out[LOG_FEATURE] = train_raw[LOG_FEATURE].astype(float)  # already log10-scale
    out["Attrition"] = train_raw["Attrition"].astype(int)
    return out


def build_saudi_common(saudi_raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=saudi_raw.index)
    for c in IDENTICAL_FEATURES:
        out[c] = saudi_raw[c].astype(float)
    for c in RESCALED_FEATURES:
        out[c] = minmax(saudi_raw[c].astype(float))
    out[LOG_FEATURE] = np.log10(saudi_raw[LOG_FEATURE].astype(float))
    out["Attrition"] = saudi_raw["Attrition"].astype(int)
    return out


def load_harmonized():
    train_raw = pd.read_csv(TRAIN_PATH)
    saudi_raw = pd.read_csv(SAUDI_PATH)
    return build_train_common(train_raw), build_saudi_common(saudi_raw)


# ---------------------------------------------------------------------------
# Metrics + bootstrap CI
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    return {
        "confusion_matrix": cm,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def bootstrap_ci(y_true, y_pred, y_proba, n_boot=N_BOOT, ci=CI, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scalar_keys = list(METRIC_LABELS.keys())
    boot = {k: [] for k in scalar_keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        m = compute_metrics(yt, yp, ypr)
        for k in scalar_keys:
            boot[k].append(m[k])
    alpha = (1 - ci) / 2
    out = {}
    for k in scalar_keys:
        arr = np.array(boot[k])
        arr = arr[~np.isnan(arr)]
        lo, hi = np.percentile(arr, [alpha * 100, (1 - alpha) * 100])
        out[k] = (lo, hi, len(arr))
    return out


def format_report(title, model_desc, y_train, y_test, point, ci_results, extra_notes=""):
    lines = []
    lines.append("=" * 78)
    lines.append(title)
    lines.append("=" * 78)
    lines.append(f"\nModel: {model_desc}")
    lines.append(f"Features ({len(FEATURES)}): {', '.join(FEATURES)}")
    lines.append(f"Train n={len(y_train)} (pos rate {y_train.mean():.3f}) | "
                  f"Test n={len(y_test)} (pos rate {y_test.mean():.3f})")

    cm = point["confusion_matrix"]
    tn, fp, fn, tp = cm.ravel()
    lines.append("\n--- Confusion Matrix (rows=actual, cols=predicted) ---")
    lines.append(f"                 Pred 0    Pred 1")
    lines.append(f"  Actual 0      {tn:6d}    {fp:6d}")
    lines.append(f"  Actual 1      {fn:6d}    {tp:6d}")

    lines.append("\n--- Metrics (point estimate, 95% bootstrap CI, n_valid_boot) ---")
    for key, label in METRIC_LABELS.items():
        val = point[key]
        lo, hi, n_valid = ci_results[key]
        lines.append(f"  {label:<28} {val:.4f}   95% CI [{lo:.4f}, {hi:.4f}]   (n_boot={n_valid})")

    no_skill = y_test.mean()
    lines.append(f"  {'PR-AUC no-skill baseline':<28} {no_skill:.4f}   "
                  f"(= test set positive rate; PR-AUC should exceed this to show real signal)")

    lines.append("\n--- Practical read on false negatives / false positives ---")
    lines.append(f"  False negatives (missed attrition risk): {fn} of {fn+tp} actual leavers "
                  f"({fn/(fn+tp)*100:.1f}% missed) -> these employees get no retention intervention.")
    lines.append(f"  False positives (flagged but stayed):    {fp} of {fp+tn} actual stayers "
                  f"({fp/(fp+tn)*100:.1f}% over-flagged) -> wasted retention spend / unnecessary manager "
                  f"attention, employee concern, stigmatization risk, added administrative burden.")
    if extra_notes:
        lines.append(f"\n{extra_notes}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model fitting: default (class_weight='balanced') or --smote (SMOTE-NC)
# ---------------------------------------------------------------------------
def fit_predict_balanced(X_train, y_train, X_test):
    model = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    return y_pred, y_proba, "Logistic Regression (class_weight='balanced')", None


def fit_predict_smotenc(X_train, y_train, X_test):
    from imblearn.over_sampling import SMOTENC  # imported lazily; only needed with --smote

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote_nc = SMOTENC(
        categorical_features=CATEGORICAL_IDX,
        random_state=RANDOM_STATE,
        k_neighbors=min(5, max(1, int((y_train == 1).sum()) - 1)),
    )
    X_train_res, y_train_res = smote_nc.fit_resample(X_train_scaled, y_train)

    model = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)  # no class_weight: SMOTE-NC already balances
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    desc = (f"Logistic Regression + StandardScaler + SMOTE-NC "
            f"(train resampled {len(y_train)}->{len(y_train_res)} rows, "
            f"pos rate ->{y_train_res.mean():.1%})")
    return y_pred, y_proba, desc, scaler


# ---------------------------------------------------------------------------
# Setup runners
# ---------------------------------------------------------------------------
def run_setup1(train, test, use_smote):
    X_train, y_train = train[FEATURES].values, train["Attrition"].values
    X_test, y_test = test[FEATURES].values, test["Attrition"].values

    fit_fn = fit_predict_smotenc if use_smote else fit_predict_balanced
    y_pred, y_proba, model_desc, _ = fit_fn(X_train, y_train, X_test)

    point = compute_metrics(y_test, y_pred, y_proba)
    ci_results = bootstrap_ci(y_test, y_pred, y_proba)
    notes = (f"  Attrition base rate: train {y_train.mean():.1%} vs Saudi test {y_test.mean():.1%} — "
             f"fully cross-domain (cold transfer) reference point.")
    report = format_report(
        "SETUP 1: CROSS-DOMAIN (train=general HR, test=100% Saudi)",
        model_desc, y_train, y_test, point, ci_results, notes,
    )
    return report, point, ci_results


def run_setup2(train, test, use_smote):
    saudi_train_half, saudi_test_half = train_test_split(
        test, test_size=0.5, stratify=test["Attrition"], random_state=RANDOM_STATE,
    )
    train_combined = pd.concat([train, saudi_train_half], ignore_index=True)
    test_final = saudi_test_half.reset_index(drop=True)

    X_train, y_train = train_combined[FEATURES].values, train_combined["Attrition"].values
    X_test, y_test = test_final[FEATURES].values, test_final["Attrition"].values

    fit_fn = fit_predict_smotenc if use_smote else fit_predict_balanced
    y_pred, y_proba, model_desc, _ = fit_fn(X_train, y_train, X_test)

    point = compute_metrics(y_test, y_pred, y_proba)
    ci_results = bootstrap_ci(y_test, y_pred, y_proba)
    notes = (f"  Saudi rows are {len(saudi_train_half)} of {len(train_combined)} pre-resampling "
             f"training rows ({len(saudi_train_half)/len(train_combined)*100:.1f}%).")
    report = format_report(
        "SETUP 2: general HR + 50% Saudi (train) -> 50% Saudi holdout (test)",
        model_desc, y_train, y_test, point, ci_results, notes,
    )
    return report, point, ci_results


def run_setup3(saudi, use_smote):
    train_df, test_df = train_test_split(
        saudi, test_size=0.2, stratify=saudi["Attrition"], random_state=RANDOM_STATE,
    )
    X_train, y_train = train_df[FEATURES].values, train_df["Attrition"].values
    X_test, y_test = test_df[FEATURES].values, test_df["Attrition"].values

    fit_fn = fit_predict_smotenc if use_smote else fit_predict_balanced
    y_pred, y_proba, model_desc, _ = fit_fn(X_train, y_train, X_test)

    point = compute_metrics(y_test, y_pred, y_proba)
    ci_results = bootstrap_ci(y_test, y_pred, y_proba)
    notes = (f"  Test set is small (n={len(y_test)}, only {int(y_test.sum())} positives) — "
             f"CIs will be wide accordingly. In-domain reference point.")
    report = format_report(
        "SETUP 3: SAUDI-ONLY, 80% train / 20% test (stratified)",
        model_desc, y_train, y_test, point, ci_results, notes,
    )
    return report, point, ci_results


# ---------------------------------------------------------------------------
# Summary table across all three setups
# ---------------------------------------------------------------------------
def print_summary_table(results):
    lines = ["\n" + "=" * 78, "SUMMARY ACROSS ALL THREE SETUPS", "=" * 78]
    header = f"{'Setup':<10}{'BalAcc':>18}{'Specificity':>20}{'Recall':>16}{'PR-AUC':>16}"
    lines.append(header)
    for name, (_, point, ci) in results.items():
        ba_lo, ba_hi, _ = ci["balanced_accuracy"]
        sp_lo, sp_hi, _ = ci["specificity"]
        re_lo, re_hi, _ = ci["recall_sensitivity"]
        pr_lo, pr_hi, _ = ci["pr_auc"]
        lines.append(
            f"{name:<10}"
            f"{point['balanced_accuracy']:.3f} [{ba_lo:.2f},{ba_hi:.2f}]".rjust(18) +
            f"{point['specificity']:.3f} [{sp_lo:.2f},{sp_hi:.2f}]".rjust(20) +
            f"{point['recall_sensitivity']:.3f} [{re_lo:.2f},{re_hi:.2f}]".rjust(16) +
            f"{point['pr_auc']:.3f} [{pr_lo:.2f},{pr_hi:.2f}]".rjust(16)
        )
    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run all 3 attrition validation setups.")
    parser.add_argument("--smote", action="store_true",
                         help="Use StandardScaler + SMOTE-NC (train-fold only) instead of "
                              "class_weight='balanced'. NOT the recommended default — tested "
                              "and found to underperform on Setup 3 (balanced accuracy "
                              "0.720 -> 0.655). Included for reproducibility of that comparison.")
    parser.add_argument("--outdir", default="/mnt/user-data/outputs",
                         help="Directory to write per-setup .txt reports to.")
    args = parser.parse_args()

    train, saudi = load_harmonized()

    results = {}
    r1, p1, c1 = run_setup1(train, saudi, args.smote)
    print(r1, "\n")
    results["Setup 1"] = (r1, p1, c1)

    r2, p2, c2 = run_setup2(train, saudi, args.smote)
    print(r2, "\n")
    results["Setup 2"] = (r2, p2, c2)

    r3, p3, c3 = run_setup3(saudi, args.smote)
    print(r3, "\n")
    results["Setup 3"] = (r3, p3, c3)

    print_summary_table(results)

    suffix = "_smote" if args.smote else "_balanced"
    import os
    os.makedirs(args.outdir, exist_ok=True)
    for name, (report, _, _) in results.items():
        fname = f"{name.lower().replace(' ', '')}{suffix}_report.txt"
        with open(os.path.join(args.outdir, fname), "w") as f:
            f.write(report)
    print(f"\nPer-setup reports written to {args.outdir}/ (suffix: {suffix})")


if __name__ == "__main__":
    main()
