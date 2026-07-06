import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, recall_score, precision_score, roc_auc_score,
    average_precision_score, classification_report, brier_score_loss,
)
from xgboost import XGBClassifier

from config import DOMAINS

warnings.filterwarnings("ignore")
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

# For a screening tool, missing a sick patient is worse than a false alarm, so
# each model's cutoff is tuned to catch at least this fraction of true cases.
TARGET_RECALL = 0.85


def tune_threshold(y_true, proba, target_recall: float = TARGET_RECALL) -> float:
    """
    Pick the HIGHEST probability cutoff that still catches >= target_recall of
    the disease cases. A higher cutoff means fewer false alarms, so among all
    thresholds that meet the recall target this gives the best precision.
    Falls back to 0.5 if the target cannot be met.
    """
    y_true = np.asarray(y_true)
    candidates = np.unique(np.concatenate([[0.0], np.round(proba, 4)]))
    best, found = 0.5, False
    for t in np.sort(candidates):                 # ascending: recall falls as t rises
        rec = recall_score(y_true, (proba >= t).astype(int), zero_division=0)
        if rec >= target_recall:
            best, found = float(t), True          # keep the largest t that still qualifies
    return best if found else 0.5


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _log1p_clip(a):
    """log1p with a floor at 0 (module-level so the pipeline stays picklable)."""
    return np.log1p(np.clip(a, 0, None))


def build_preprocessor(X: pd.DataFrame, skew_thresh: float = 4.0) -> ColumnTransformer:
    nums = X.select_dtypes(include=["number"]).columns.tolist()
    cats = X.select_dtypes(exclude=["number"]).columns.tolist()

    skew = [c for c in nums if abs(pd.to_numeric(X[c], errors="coerce").skew()) > skew_thresh]
    normal = [c for c in nums if c not in skew]

    log_tf = FunctionTransformer(_log1p_clip, feature_names_out="one-to-one")

    normal_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    skew_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", log_tf),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if normal:
        transformers.append(("num", normal_pipe, normal))
    if skew:
        transformers.append(("skew", skew_pipe, skew))
    if cats:
        transformers.append(("cat", cat_pipe, cats))
    if nums:
        transformers.append(("missing", MissingIndicator(features="all"), nums))

    return ColumnTransformer(transformers, remainder="drop")


# ---------------------------------------------------------------------------
# Candidate models (all imbalance-aware)
# ---------------------------------------------------------------------------

def candidate_models(y: pd.Series) -> dict:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = (neg / pos) if pos else 1.0
    return {
        "Logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample",
            random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", scale_pos_weight=spw,
            random_state=42, verbosity=0),
    }


# ---------------------------------------------------------------------------
# Train one domain
# ---------------------------------------------------------------------------

def train_domain(cfg, data_dir: str, verbose: bool = True) -> dict:
    raw = pd.read_csv(os.path.join(data_dir, cfg.filename), **cfg.read_kwargs)
    X, y = cfg.cleaner(raw)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)

    keep = y.map(y.value_counts()) >= 5
    X, y = X[keep].reset_index(drop=True), y[keep].reset_index(drop=True)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---- model selection by cross-validated macro-F1 (not accuracy) ----------
    best_name, best_cv_f1, best_model = None, -1.0, None
    cv_table = {}
    for name, clf in candidate_models(y_tr).items():
        pipe = Pipeline([("pre", build_preprocessor(X_tr)), ("clf", clf)])
        f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1_macro", n_jobs=-1).mean()
        cv_table[name] = round(float(f1), 4)
        if f1 > best_cv_f1:
            best_cv_f1, best_name, best_model = f1, name, clf

    method = "isotonic" if len(y_tr) >= 1000 else "sigmoid"  # isotonic needs data

    def _calibrated(X_fit, y_fit):
        base = Pipeline([("pre", build_preprocessor(X_fit)), ("clf", clone(best_model))])
        c = CalibratedClassifierCV(base, method=method, cv=5)
        c.fit(X_fit, y_fit)
        return c

    # ---- tune the decision threshold on a VALIDATION slice (not the test set)-
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.25, stratify=y_tr, random_state=42)
    val_proba = _calibrated(X_fit, y_fit).predict_proba(X_val)[:, 1]
    threshold = tune_threshold(y_val, val_proba, TARGET_RECALL)

    # ---- refit the calibrated model on ALL training data ---------------------
    calibrated = _calibrated(X_tr, y_tr)

    # ---- honest held-out evaluation AT THE TUNED THRESHOLD -------------------
    proba = calibrated.predict_proba(X_te)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "selected_model": best_name,
        "cv_macro_f1_by_model": cv_table,
        "decision_threshold": round(float(threshold), 3),
        "target_recall": TARGET_RECALL,
        "test_macro_f1": round(float(f1_score(y_te, pred, average="macro")), 4),
        "test_recall_disease": round(float(recall_score(y_te, pred, zero_division=0)), 4),
        "test_precision_disease": round(float(precision_score(y_te, pred, zero_division=0)), 4),
        "test_roc_auc": round(float(roc_auc_score(y_te, proba)), 4),
        "test_pr_auc": round(float(average_precision_score(y_te, proba)), 4),
        "test_brier": round(float(brier_score_loss(y_te, proba)), 4),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "class_balance": {int(k): int(v) for k, v in y.value_counts().items()},
        "calibration": method,
    }

    if verbose:
        print(f"\n=== {cfg.display} ===")
        print("  CV macro-F1 by model :", cv_table)
        print(f"  Selected             : {best_name}  (calibration={method})")
        print(f"  Decision threshold   : {metrics['decision_threshold']}  (target recall {TARGET_RECALL})")
        print(f"  Test macro-F1        : {metrics['test_macro_f1']}")
        print(f"  Test recall/precision: {metrics['test_recall_disease']} / {metrics['test_precision_disease']}")
        print(f"  Test ROC-AUC / PR-AUC: {metrics['test_roc_auc']} / {metrics['test_pr_auc']}")
        print(f"  Brier (lower=better) : {metrics['test_brier']}")
        print(classification_report(y_te, pred, zero_division=0))

    # ---- persist model + deployment metadata ---------------------------------
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(calibrated, os.path.join(MODEL_DIR, f"{cfg.key}_model.pkl"))

    num_cols = X.select_dtypes(include="number").columns
    ranges = {c: [float(np.nanpercentile(X[c], 1)), float(np.nanpercentile(X[c], 99))]
              for c in num_cols}
    meta = {
        "key": cfg.key,
        "display": cfg.display,
        "feature_names": list(X.columns),
        "required_features": cfg.required_features,
        "min_coverage": cfg.min_coverage,
        "base_weight": cfg.base_weight,
        "decision_threshold": metrics["decision_threshold"],
        "numeric_ranges": ranges,
        "metrics": metrics,
    }
    with open(os.path.join(MODEL_DIR, f"{cfg.key}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return metrics


def train_all(data_dir: str) -> dict:
    print("Training calibrated, imbalance-aware, threshold-tuned models...\n" + "=" * 62)
    summary = {}
    for cfg in DOMAINS.values():
        summary[cfg.key] = train_domain(cfg, data_dir)
    print("\n" + "=" * 62 + "\nDONE. Summary (thr / recall / precision / AUC / Brier):")
    for k, m in summary.items():
        print(f"  {k:9s}: {m['selected_model']:13s} "
              f"thr={m['decision_threshold']:.2f}  rec={m['test_recall_disease']:.3f}  "
              f"prec={m['test_precision_disease']:.3f}  "
              f"AUC={m['test_roc_auc']:.3f}  Brier={m['test_brier']:.3f}")
    return summary


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "."
    train_all(data)