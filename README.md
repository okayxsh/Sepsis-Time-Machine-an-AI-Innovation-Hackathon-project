# The Sepsis Time Machine

**See sepsis coming hours before it arrives, and know why.**

An explainable early-warning system for sepsis in the ICU, built for the UOWD AI Innovation Hackathon (Medical AI track). A gradient-boosted model trained on real ICU data scores each patient's sepsis risk every hour, explains each alert with its top three reasons in plain English, and helps the team act on it: it writes an SBAR handover for the on-call doctor and tracks the Surviving Sepsis Campaign Hour-1 bundle.

The **Time Machine** replays a real patient the model has never seen, hour by hour, using only what the clinician could have known at each moment.

> Decision support, not a replacement for clinical judgement.

---

## Results

Measured on **300 held-out test patients** (21 septic) that the model never saw during training. The model was trained on a 1,500-patient subset of PhysioNet 2019.

| | **Our model** | SIRS (bedside rule) |
|---|---|---|
| AUROC (hourly) | **0.83** | 0.64 |
| Septic patients caught before onset | 17 / 21 (81%) | 17 / 21 (81%) |
| False alerts per 100 patient-days | **44** | 178 |
| Non-septic patients ever alerted | **35%** | 81% |
| Median lead time before onset | 24 h | 39 h* |

At the same catch rate as SIRS, the model raises **about 4× fewer false alarms**. That cut is aimed at alarm fatigue.

\*SIRS shows a longer lead time only because it fires on most patients early and stays on.

The alert threshold (0.12) was chosen on validation patients using patient-level Youden's J, and never on test. In the dashboard it is a live slider, so you can see the trade-off between sensitivity and false alarms.

---

## Features

| | |
|---|---|
| **Time Machine** | Hour-by-hour replay of vitals and risk, with normal-range bands, the alert marker and the onset reveal |
| **Explainable alerts** | SHAP top-3 reasons per alert, in plain English (e.g. *"Creatinine 2.8 mg/dL"*, *"Lactate just measured: 3"*) |
| **SBAR page** | One-click handover to the on-call doctor, written by Claude from computed values only. A deterministic template is used offline or on any error |
| **Hour-1 bundle** | Lactate, cultures, antibiotics, 30 mL/kg fluids (sized to weight), pressors, with a 60-minute countdown |
| **Alarm-fatigue dial** | Threshold slider showing catch rate, lead time and false alerts against the SIRS baseline |
| **Ward Board** | Risk overview of patients across the ward |
| **Model Performance** | Operating curve, trade-off chart, feature importance, threshold table |

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py              # dashboard on http://localhost:8501
```

The trained model and test predictions are in `artifacts/`, so the dashboard runs without retraining.

**Landing page** (optional):

```bash
cd landing && python -m http.server 8000   # http://localhost:8000
```

**SBAR with Claude** (optional): set `ANTHROPIC_API_KEY`. Without a key the app falls back to the template automatically. `SBAR_MODEL` overrides the model.

### Retrain from scratch

```bash
python download_physionet.py      # ~40k .psv files from the PhysioNet open-data mirror
python train.py                   # writes artifacts/
```

If a Kaggle export of the same dataset is at `data/Dataset.csv`, it is used instead of the download.

---

## How it works

**Data.** PhysioNet / Computing in Cardiology Challenge 2019: hourly vitals, labs and demographics from ICU stays in two US hospital systems. The challenge's `SepsisLabel` is already shifted 6 h before Sepsis-3 onset, so we define clinical onset as `first positive hour + 6`.

**Features (75, all causal).** Within-patient forward-fill and measurement flags. Hours since each key lab was last measured. 1 h and 6 h deltas. Trailing 6 h mean and std. Shock index. SIRS count. Static demographics. Each feature at hour *t* uses only data from hours ≤ *t*.

**Model.** LightGBM with class-imbalance weighting and early stopping on validation patients. Explanations come from SHAP `TreeExplainer`, deduplicated by underlying variable.

**Patient-level evaluation.**
- A septic patient counts as *caught* only if the first alert fires at or before clinical onset.
- Lead time is the time from that first alert to onset.
- False alerts are rising edges of the alert in non-septic patients, per 100 patient-days.

### Leakage safeguards
- 80 / 20 train/test split **by patient** (seed 42), so no patient appears in both.
- 15% of training patients are held out as validation for early stopping and threshold selection. The test set is used only for reporting.
- All features are causal, and `SepsisLabel` is never used as an input.

---

## Project structure

```
app.py                  Streamlit dashboard (Time Machine, Ward Board, Model Performance, About)
train.py                Patient-level split, LightGBM, SIRS baseline, metrics, demo patients
features.py             Data loading + causal feature engineering + plain-English explanations
sbar.py                 Claude SBAR call (10 s timeout) with template fallback
bundle.py               Surviving Sepsis Campaign Hour-1 bundle logic
download_physionet.py   PhysioNet 2019 downloader
artifacts/              Trained model, metrics, test predictions, threshold tables
landing/index.html      Pitch landing page
PITCH_SCRIPT.md         Demo script and judge Q&A
```

---

## Limitations

- Trained on a 1,500-patient subset; training on all ~40k patients is the next step.
- The test set is small (21 septic patients), so the metrics have wide confidence intervals.
- Retrospective data from US hospitals (2019). Before any clinical use it would need prospective validation, local (e.g. UAE) data and EHR integration.
- The model misses some patients. The dashboard shows one missed case on purpose.

---

## Acknowledgements

Data: Reyna MA *et al.*, *Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019*, Critical Care Medicine 48(2), 2020. Available on [PhysioNet](https://physionet.org/content/challenge-2019/1.0.0/) under the Open Data Commons Attribution License v1.0.

Clinical guidance: Surviving Sepsis Campaign Hour-1 bundle; Sepsis-3 definitions.
