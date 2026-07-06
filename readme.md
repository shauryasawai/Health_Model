# 🩺 Healthigrity — Health Screening Score

Healthigrity looks at a person's medical test results and estimates how likely
they are to have each of seven common conditions — **heart disease,
cardiovascular risk, diabetes, kidney disease, liver disease, thyroid disorder,
and anaemia** — then combines those into one
easy-to-read **overall health score (0–100)**.

> ⚠️ **This is a screening estimate, not a diagnosis.** It is meant to point
> someone toward a doctor, not replace one. It has not been clinically validated.

---

## 📁 Folder layout

```
HEALTHIGRITY/
├── data/            ← the CSV datasets (input)
├── models/          ← the trained models (created automatically)
├── my_env/          ← your Python virtual environment
├── requirements.txt ← the libraries you need
└── src/             ← all the code lives here
    ├── config.py        ← cleans each dataset + settings
    ├── train.py         ← trains the models
    ├── predict.py       ← makes predictions for a patient
    ├── health_score.py  ← turns predictions into the score
    ├── run.py           ← trains everything + shows a demo
    └── demos.py         ← 5 example patients to test with
```

---

## 🚀 How to run it

**1. Create Virtual_ENV** (once):
```bash
python -m venv my_env
pip install virtualenv
my_env\Scripts\activate
```

**2. Install the libraries** (once):
```bash
pip install -r requirements.txt
```

**3. Train the models and see a demo** (run from the project root):
```bash
python src/run.py
```
This reads the CSVs from `data/`, trains all five models, saves them into
`models/`, and prints a sample patient report.

---

## 🧠 How it works (in plain words)

Think of it as a 3-step assembly line:

**Step 1 — Train (`train.py`)**
For each disease it learns a model from the data. It is careful to:
- pick the model that best *catches sick people* (not just overall accuracy),
- make its risk percentages *honest and meaningful* (calibration),
- choose a smart cut-off so it flags ~85% of real cases,
- avoid "cheating" by removing clues that give away the answer.

**Step 2 — Predict (`predict.py`)**
For a new person, it fills in a risk percentage for each disease.
If the person hasn't taken the tests a disease needs, it **skips that disease**
instead of making something up — and tells you how much data was missing.

**Step 3 — Score (`health_score.py`)**
Each risk percentage becomes a 0–100 health score (higher = healthier), and
those combine into one overall score. A single very unhealthy result pulls the
overall score down, just like in real life.

---

## ✅ Current model quality (on held-out test data)

| Disease         | Model         | Catches sick (recall) | Correct alarms (precision) |
|-----------------|---------------|-----------------------|----------------------------|
| Heart           | RandomForest  | 0.87                  | 0.85                       |
| Cardiovascular  | XGBoost       | 0.86                  | 0.64                       |
| Diabetes        | Logistic      | 0.89                  | 0.57                       |
| Kidney          | RandomForest  | 0.84                  | 1.00                       |
| Liver           | Logistic      | 0.83                  | 0.83                       |
| Thyroid         | XGBoost       | 0.82                  | 0.91                       |
| Anaemia         | Logistic      | 0.85                  | 1.00                       |

Heart, thyroid, kidney and anaemia are strong. Diabetes and cardiovascular have
more false alarms (genuinely hard datasets). Kidney/anaemia use small datasets,
so treat their near-perfect numbers with a little caution until tested on
outside data. The cardiovascular model only covers ages ~40–64, so younger
patients trigger a range warning.

---

## ⚙️ Things you can tweak

- **`TARGET_RECALL`** (top of `train.py`) — higher = catch more sick people but
  more false alarms; lower = fewer false alarms but miss more.
- **`required_features`** (in `config.py`) — which tests a disease *must* have
  before it will produce a score.