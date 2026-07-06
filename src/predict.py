"""
Missing-data-AWARE prediction. This is the fix for the core real-world problem:
a real patient never has every feature for every disease, and the old code
either crashed (KeyError) or silently imputed the training mean and returned a
confident-looking-but-fabricated probability.

The rule here: never invent a diagnosis out of thin air.
  - If a domain's REQUIRED features are missing, or overall coverage is too low,
    that domain is SKIPPED (returned as available=False) instead of scored.
  - Only genuinely measured, partially-missing inputs are imputed, and every
    imputation / out-of-range value is reported as a warning.
  - Each prediction carries a coverage number so downstream code (and the user)
    knows how much of the score is real data versus filled-in.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
 
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
 
 
def load_domain(key: str):
    model = joblib.load(os.path.join(MODEL_DIR, f"{key}_model.pkl"))
    with open(os.path.join(MODEL_DIR, f"{key}_meta.json")) as f:
        meta = json.load(f)
    return model, meta
 
 
def _coverage(row: pd.Series, features) -> float:
    present = [f for f in features if f in row.index and pd.notna(row[f])]
    return len(present) / len(features) if features else 0.0
 
 
def _range_warnings(row: pd.Series, ranges: dict):
    warns = []
    for col, (lo, hi) in ranges.items():
        if col in row.index and pd.notna(row[col]):
            v = float(row[col])
            if v < lo or v > hi:
                warns.append(f"{col}={v:g} outside training range [{lo:g}, {hi:g}]")
    return warns
 
 
def predict_domain(key: str, patient: pd.Series) -> dict:
    """
    Predict disease probability for one domain for one patient (a pd.Series of
    raw features). Returns a dict describing availability, probability, and how
    trustworthy the inputs were.
    """
    model, meta = load_domain(key)
    feats = meta["feature_names"]
    required = meta["required_features"]
 
    # --- gate 1: required features must all be present -----------------------
    missing_required = [f for f in required if f not in patient.index or pd.isna(patient[f])]
    coverage = _coverage(patient, feats)
 
    if missing_required or coverage < meta["min_coverage"]:
        reason = (f"missing required {missing_required}" if missing_required
                  else f"coverage {coverage:.0%} < {meta['min_coverage']:.0%}")
        return {
            "domain": key, "display": meta["display"], "available": False,
            "reason": reason, "coverage": round(coverage, 3),
            "probability": None,
        }
 
    # --- build a one-row frame with exactly the trained columns --------------
    row = {f: (patient[f] if f in patient.index else np.nan) for f in feats}
    X = pd.DataFrame([row], columns=feats)
 
    warns = _range_warnings(patient, meta["numeric_ranges"])
    imputed = [f for f in feats if pd.isna(X.iloc[0][f])]
 
    proba = float(model.predict_proba(X)[0, 1])   # calibrated pipeline imputes the rest
    # Use the per-domain tuned cutoff (recall-oriented) for the yes/no flag.
    # The health SCORE still uses the raw calibrated probability, not the flag.
    threshold = meta.get("decision_threshold", 0.5)
    return {
        "domain": key, "display": meta["display"], "available": True,
        "probability": round(proba, 4),
        "threshold": threshold,
        "flag": int(proba >= threshold),          # 1 = screen positive
        "coverage": round(coverage, 3),
        "imputed_features": imputed,
        "range_warnings": warns,
    }
 
 
def predict_patient(patient: pd.Series, domains=None) -> dict:
    """Run every available domain for one patient. Returns {key: result}."""
    from config import DOMAINS
    keys = domains or list(DOMAINS.keys())
    return {k: predict_domain(k, patient) for k in keys}
 
 
def predict_batch(df: pd.DataFrame, domains=None) -> pd.DataFrame:
    """Row-wise prediction for a table of patients (missing columns are fine)."""
    from config import DOMAINS
    keys = domains or list(DOMAINS.keys())
    out = []
    for _, patient in df.iterrows():
        rec = {}
        for k in keys:
            r = predict_domain(k, patient)
            disp = r["display"]
            rec[f"{disp}_prob"] = r["probability"]
            rec[f"{disp}_flag"] = r.get("flag")
            rec[f"{disp}_available"] = r["available"]
            rec[f"{disp}_coverage"] = r["coverage"]
        out.append(rec)
    return pd.DataFrame(out, index=df.index)