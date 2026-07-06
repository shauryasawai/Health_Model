"""
Every dataset says:
  - where the file is and what the target column is
  - how to turn the raw CSV into (X, y) with a BINARY target (1 = disease)
  - which columns are LEAKY and must be dropped (features that encode the label)
  - which features are *required* to trust a prediction (used for gating)
  - a base clinical weight for the final aggregate score
"""

from dataclasses import dataclass, field
from typing import Callable, List
import numpy as np
import pandas as pd
 
 
def clean_diabetes(df: pd.DataFrame):
    df = df.drop_duplicates().copy()
    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome"])
    # FIX (silent-zero bug): a 0 in these columns is physiologically impossible
    # and actually means "not measured". Turn it into a real NaN.
    for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
        X[c] = X[c].replace(0, np.nan)
    return X, y
 
 
def clean_anaemia(df: pd.DataFrame):
    df = df.copy()
    df.columns = [c.strip().replace(" ", "") for c in df.columns]
    df = df.drop(columns=[c for c in ["S.No.", "S.No"] if c in df.columns])
    df = df.drop_duplicates()
    for c in ["RBC", "PCV", "MCV", "MCH", "MCHC", "RDW", "TLC", "PLT/mm3", "HGB", "Age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # WHO haemoglobin thresholds define the label (Sex 1 = male <13, 0 = female <12).
    df = df.dropna(subset=["HGB", "Sex"])
    y = np.where(df["Sex"] == 1, (df["HGB"] < 13.0).astype(int), (df["HGB"] < 12.0).astype(int))
    y = pd.Series(y, index=df.index).astype(int)
    df["Sex"] = df["Sex"].map({1: "Male", 0: "Female"}).fillna("Unknown")
    # FIX (leakage): the label is a threshold on HGB, so HGB and its proxy PCV
    # (~3x HGB) are dropped so the model must predict from independent indices.
    X = df.drop(columns=[c for c in ["HGB", "PCV"] if c in df.columns])
    return X, y
 
 
def clean_liver(df: pd.DataFrame):
    df = df.drop_duplicates().copy()
    # 1 = liver patient, 2 = healthy -> remap to 1 = disease, 0 = healthy.
    y = (df["Dataset"] == 1).astype(int)
    X = df.drop(columns=["Dataset"])
    return X, y
 
 
def clean_kidney(df: pd.DataFrame):
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "": np.nan})
    for c in ["pcv", "wc", "rc", "age", "bp", "sg", "al", "su",
              "bgr", "bu", "sc", "sod", "pot", "hemo"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["classification"] = df["classification"].str.replace(r"\s+", "", regex=True)
    df = df[df["classification"].isin(["ckd", "notckd"])]
    y = (df["classification"] == "ckd").astype(int)
    X = df.drop(columns=["classification"])
    return X, y
 
 
def clean_thyroid(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=[c for c in ["patient_id", "referral_source"] if c in df.columns])
    # target: '-' = no condition, any other code = a thyroid condition -> binary.
    y = (df["target"].astype(str).str.strip() != "-").astype(int)
    X = df.drop(columns=["target"])
    X = X.drop(columns=[c for c in ["TBG", "TBG_measured"] if c in X.columns])  # ~96% missing
    for c in ["TSH", "T3", "TT4", "T4U", "FTI", "age"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    return X, y
 
 
def clean_cardiac(df: pd.DataFrame):
    # UCI heart disease. target 'num': 0 = none, 1-4 = increasing severity.
    df = df.copy()
    df = df.drop(columns=[c for c in ["id", "dataset"] if c in df.columns])
    df = df.drop_duplicates()
    y = (pd.to_numeric(df["num"], errors="coerce").fillna(0) > 0).astype(int)  # -> binary
    X = df.drop(columns=["num"])
    for c in ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # 0 = "not measured" for these two in UCI.
    for c in ["trestbps", "chol"]:
        if c in X.columns:
            X[c] = X[c].replace(0, np.nan)
    return X, y
 
 
def clean_cardiovascular(df: pd.DataFrame):
    # Kaggle cardio_train (semicolon-delimited, 70k rows).
    df = df.copy()
    if "id" in df.columns:
        df = df.drop(columns=["id"])
    df = df.drop_duplicates()
    if "age" in df.columns:                       # age is stored in DAYS -> years
        df["age"] = (pd.to_numeric(df["age"], errors="coerce") / 365.25).round(1)
    y = df["cardio"].astype(int)
    X = df.drop(columns=["cardio"])
    # 70k rows is slow to train in a sandbox; a 10k stratified sample is plenty.
    if len(X) > 10000:
        idx = X.sample(10000, random_state=42).index
        X, y = X.loc[idx], y.loc[idx]
    return X.reset_index(drop=True), y.reset_index(drop=True)
 
 
@dataclass
class DomainConfig:
    key: str
    display: str
    filename: str
    cleaner: Callable
    required_features: List[str]
    base_weight: float
    read_kwargs: dict = field(default_factory=dict)
    min_coverage: float = 0.6
 
 
DOMAINS = {
    "cardiac": DomainConfig(
        key="cardiac", display="Heart", filename="heart_disease_uci.csv",
        cleaner=clean_cardiac,
        required_features=["age", "trestbps", "chol", "thalch"],
        base_weight=0.22,
    ),
    "cardiovascular": DomainConfig(
        key="cardiovascular", display="Cardiovascular", filename="cardio_train.csv",
        cleaner=clean_cardiovascular,
        required_features=["ap_hi", "ap_lo", "age", "cholesterol"],
        base_weight=0.20,
        read_kwargs={"delimiter": ";"},
    ),
    "diabetes": DomainConfig(
        key="diabetes", display="Diabetes", filename="diabetes.csv",
        cleaner=clean_diabetes,
        required_features=["Glucose", "BMI", "Age"],
        base_weight=0.16,
    ),
    "renal": DomainConfig(
        key="renal", display="Kidney", filename="kidney_disease.csv",
        cleaner=clean_kidney,
        required_features=["sc", "hemo", "al"],
        base_weight=0.15,
    ),
    "liver": DomainConfig(
        key="liver", display="Liver", filename="indian_liver_patient.csv",
        cleaner=clean_liver,
        required_features=["Total_Bilirubin", "Alamine_Aminotransferase", "Albumin"],
        base_weight=0.12,
    ),
    "thyroid": DomainConfig(
        key="thyroid", display="Thyroid", filename="thyroidDF.csv",
        cleaner=clean_thyroid,
        required_features=["TSH"],
        base_weight=0.08,
    ),
    "anaemia": DomainConfig(
        key="anaemia", display="Anaemia", filename="CBC data_for_meandeley_csv.csv",
        cleaner=clean_anaemia,
        required_features=["RBC", "MCV", "RDW", "Sex"],
        base_weight=0.07,
        read_kwargs={"skiprows": [1]},
    ),
}