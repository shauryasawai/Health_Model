"""
A gallery of demo patients to exercise the pipeline across scenarios:
  1. HEALTHY      - all 7 panels, everything normal   -> high score, no flags
  2. MULTI-SICK   - all 7 panels, several abnormal      -> low score, flags
  3. PARTIAL      - diabetes + CBC + heart + cardio      -> liver/kidney/thyroid SKIPPED
  4. OUT-OF-RANGE - extreme values                      -> range warnings
  5. SPARSE       - a panel present but missing a REQUIRED test -> that domain skipped
"""

import sys
import pandas as pd
from predict import predict_patient
from health_score import build_report, print_report
from config import DOMAINS


def healthy():
    return pd.Series({
        # diabetes
        "Pregnancies": 1, "Glucose": 88, "BloodPressure": 72, "SkinThickness": 22,
        "Insulin": 85, "BMI": 22.5, "DiabetesPedigreeFunction": 0.25, "Age": 34,
        # CBC / anaemia
        "Sex": "Male", "RBC": 5.1, "MCV": 89, "MCH": 30, "MCHC": 33, "RDW": 12.8,
        "TLC": 7.0, "PLT/mm3": 260,
        # liver
        "Gender": "Male", "Total_Bilirubin": 0.7, "Direct_Bilirubin": 0.2,
        "Alkaline_Phosphotase": 180, "Alamine_Aminotransferase": 22,
        "Aspartate_Aminotransferase": 24, "Total_Protiens": 7.2, "Albumin": 4.4,
        "Albumin_and_Globulin_Ratio": 1.5,
        # kidney
        "age": 34, "bp": 78, "sg": 1.02, "al": 0, "su": 0, "rbc": "normal", "pc": "normal",
        "pcc": "notpresent", "ba": "notpresent", "bgr": 95, "bu": 28, "sc": 0.9, "sod": 140,
        "pot": 4.2, "hemo": 15.2, "pcv": 46, "wc": 7200, "rc": 5.1, "htn": "no", "dm": "no",
        "cad": "no", "appet": "good", "pe": "no", "ane": "no",
        # thyroid
        "sex": "M", "on_thyroxine": "f", "query_on_thyroxine": "f", "on_antithyroid_meds": "f",
        "sick": "f", "pregnant": "f", "thyroid_surgery": "f", "I131_treatment": "f",
        "query_hypothyroid": "f", "query_hyperthyroid": "f", "lithium": "f", "goitre": "f",
        "tumor": "f", "hypopituitary": "f", "psych": "f", "TSH_measured": "t", "TSH": 1.8,
        "T3_measured": "t", "T3": 2.0, "TT4_measured": "t", "TT4": 108,
        "T4U_measured": "t", "T4U": 1.0, "FTI_measured": "t", "FTI": 110,
        # cardiac (UCI heart)
        "cp": "non-anginal", "trestbps": 120, "chol": 190, "fbs": "FALSE",
        "restecg": "normal", "thalch": 165, "exang": "FALSE", "oldpeak": 0.0,
        "slope": "upsloping", "ca": 0, "thal": "normal",
        # cardiovascular (cardio_train)
        "gender": 2, "height": 175, "weight": 70, "ap_hi": 118, "ap_lo": 78,
        "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
    })


def multi_sick():
    p = healthy().copy()
    p.update({
        "Glucose": 197, "BMI": 38.1, "Insulin": 330, "Age": 58, "DiabetesPedigreeFunction": 0.9,
        "RBC": 3.4, "MCV": 72, "MCH": 22, "RDW": 18.5, "Sex": "Female", "sex": "F", "Gender": "Female",
        "Total_Bilirubin": 4.8, "Direct_Bilirubin": 2.6, "Alamine_Aminotransferase": 145,
        "Aspartate_Aminotransferase": 190, "Albumin": 2.7, "Albumin_and_Globulin_Ratio": 0.7,
        "Alkaline_Phosphotase": 480,
        "sc": 3.8, "bu": 90, "hemo": 9.1, "al": 3, "sg": 1.01, "htn": "yes", "dm": "yes",
        "ane": "yes", "appet": "poor", "pcv": 27,
        "TSH": 14.5, "T3": 0.7, "TT4": 45, "FTI": 52,
        # cardiac abnormal
        "age": 58, "cp": "asymptomatic", "trestbps": 165, "chol": 305, "fbs": "TRUE",
        "restecg": "lv hypertrophy", "thalch": 105, "exang": "TRUE", "oldpeak": 3.2,
        "slope": "flat", "ca": 3, "thal": "reversable defect",
        # cardiovascular abnormal
        "gender": 1, "height": 170, "weight": 105, "ap_hi": 175, "ap_lo": 105,
        "cholesterol": 3, "gluc": 3, "smoke": 1, "alco": 1, "active": 0,
    })
    return p


def partial():
    # diabetes + CBC + HEART + CARDIO panels, but NO liver/kidney/thyroid tests.
    return pd.Series({
        "Pregnancies": 2, "Glucose": 168, "BloodPressure": 74, "SkinThickness": 31,
        "Insulin": 120, "BMI": 33.5, "DiabetesPedigreeFunction": 0.55, "Age": 51,
        "Sex": "Female", "RBC": 3.9, "PCV": 34, "MCV": 78, "MCH": 25, "MCHC": 31,
        "RDW": 16.5, "TLC": 8.2, "PLT/mm3": 250,
        # cardiac (UCI heart)
        "age": 51, "sex": "Female", "cp": "atypical angina", "trestbps": 138,
        "chol": 240, "fbs": "FALSE", "restecg": "normal", "thalch": 140,
        "exang": "FALSE", "oldpeak": 1.2, "slope": "flat", "ca": 0, "thal": "normal",
        # cardiovascular (cardio_train)
        "gender": 1, "height": 162, "weight": 82, "ap_hi": 138, "ap_lo": 88,
        "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
    })


def out_of_range():
    p = healthy().copy()
    p.update({"Glucose": 480, "Insulin": 900, "sc": 22.0, "TSH": 260,
              "Total_Bilirubin": 40, "ap_hi": 300, "trestbps": 260})
    return p


def sparse():
    return pd.Series({
        "Pregnancies": 3, "BloodPressure": 80, "SkinThickness": 28, "Insulin": 110,
        "BMI": 29.0, "DiabetesPedigreeFunction": 0.4, "Age": 45,
    })


SCENARIOS = {
    "1": ("HEALTHY (all panels)", healthy),
    "2": ("MULTI-SYSTEM SICK (all panels)", multi_sick),
    "3": ("PARTIAL (diabetes + CBC + heart + cardio)", partial),
    "4": ("OUT-OF-RANGE VALUES", out_of_range),
    "5": ("SPARSE (missing required Glucose)", sparse),
}


def run(which):
    label, fn = SCENARIOS[which]
    print("\n" + "#" * 62 + f"\n# SCENARIO {which}: {label}\n" + "#" * 62)
    rep = build_report(predict_patient(fn()), total_domains=len(DOMAINS))
    print_report(rep)


if __name__ == "__main__":
    keys = [sys.argv[1]] if len(sys.argv) > 1 else list(SCENARIOS.keys())
    for k in keys:
        run(k)