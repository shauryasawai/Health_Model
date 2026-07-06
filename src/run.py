"""
End-to-end demo:
  1. train all domains (calibrated, imbalance-aware, F1-selected)
  2. score a patient who only has SOME tests (missing-data-aware)
"""

import os
import sys
import pandas as pd

# Default data folder = <project_root>/data  (one level up from this src/ file)
DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

from train import train_all
from predict import predict_patient
from health_score import build_report, print_report
from config import DOMAINS


def demo_patient(data_dir):
    # A realistic partial patient: has diabetes + CBC + HEART + CARDIO panels,
    # but NO liver, kidney, or thyroid tests -> those three are SKIPPED.
    return pd.Series({
        # diabetes panel
        "Pregnancies": 2, "Glucose": 168, "BloodPressure": 74, "SkinThickness": 31,
        "Insulin": 120, "BMI": 33.5, "DiabetesPedigreeFunction": 0.55, "Age": 51,
        # CBC / anaemia panel
        "Sex": "Female", "RBC": 3.9, "PCV": 34, "MCV": 78, "MCH": 25,
        "MCHC": 31, "RDW": 16.5, "TLC": 8.2, "PLT/mm3": 250,
        # cardiac panel (UCI heart)
        "age": 51, "sex": "Female", "cp": "atypical angina", "trestbps": 138,
        "chol": 240, "fbs": "FALSE", "restecg": "normal", "thalch": 140,
        "exang": "FALSE", "oldpeak": 1.2, "slope": "flat", "ca": 0, "thal": "normal",
        # cardiovascular panel (cardio_train)
        "gender": 1, "height": 162, "weight": 82, "ap_hi": 138, "ap_lo": 88,
        "cholesterol": 2, "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
        # (no liver, kidney, or thyroid labs -> those domains should be SKIPPED)
    }) 


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATA_DIR

    train_all(data_dir)

    print("\n\n########## DEMO PATIENT (partial data) ##########")
    patient = demo_patient(data_dir)
    preds = predict_patient(patient)
    report = build_report(preds, total_domains=len(DOMAINS))
    print_report(report)