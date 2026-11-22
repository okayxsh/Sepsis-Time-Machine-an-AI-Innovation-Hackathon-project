"""Causal feature engineering shared by train.py and app.py.

Every feature at hour h uses only rows with hour <= h of the same patient
(forward-fill, lagged deltas, trailing windows, cumulative flags). No backfill,
no centred windows, no whole-stay statistics, and never SepsisLabel.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")

VITALS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]
LABS = ["BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
        "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
        "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
        "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC",
        "Fibrinogen", "Platelets"]
STATIC = ["Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"]
KEY_LABS = ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "pH"]
DELTA_VARS = ["HR", "Resp", "MAP", "SBP", "Temp", "O2Sat"]
ROLL_VARS = ["HR", "Resp", "MAP", "Temp"]
ID, HOUR, LABEL = "Patient_ID", "Hour", "SepsisLabel"


# ----------------------------------------------------------------------------- loading
def load_raw(verbose=True):
    """Return one DataFrame with Patient_ID (str), Hour (0-based), all raw columns, SepsisLabel.

    Uses data/Dataset.csv (Kaggle) if present, else PhysioNet .psv files
    (cached to data/physionet_combined.parquet after the first read).
    """
    csv = os.path.join(DATA_DIR, "Dataset.csv")
    if os.path.exists(csv):
        df = pd.read_csv(csv)
        if verbose:
            print("Loaded Kaggle Dataset.csv")
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        if ID not in df.columns:
            raise ValueError(f"No {ID} column; columns = {list(df.columns)}")
        if HOUR not in df.columns:
            df[HOUR] = df.groupby(ID).cumcount()
    else:
        cache = os.path.join(DATA_DIR, "physionet_combined.parquet")
        if os.path.exists(cache):
            df = pd.read_parquet(cache)
            if verbose:
                print("Loaded cached PhysioNet parquet")
        else:
            files = sorted(glob.glob(os.path.join(DATA_DIR, "physionet", "training_set*", "*.psv")))
            if not files:
                raise FileNotFoundError("No data: put Dataset.csv in data/ or run download_physionet.py")
            if verbose:
                print(f"Reading {len(files)} .psv files ...")
            # fast path: concatenate raw text (all files share one header) and parse once
            import io
            header, chunks, ids, hrs = None, [], [], []
            for f in files:
                with open(f, "r") as fh:
                    lines = fh.read().strip().splitlines()
                if header is None:
                    header = lines[0]
                body = lines[1:]
                chunks.append("\n".join(body))
                ids.append(np.repeat(os.path.splitext(os.path.basename(f))[0], len(body)))
                hrs.append(np.arange(len(body)))
            df = pd.read_csv(io.StringIO(header + "\n" + "\n".join(chunks)), sep="|")
            df[ID] = np.concatenate(ids)
            df[HOUR] = np.concatenate(hrs)
            df.to_parquet(cache, index=False)
    df[ID] = df[ID].astype(str)
    df = df.sort_values([ID, HOUR], kind="stable").reset_index(drop=True)
    return df


# ----------------------------------------------------------------------------- features
def _grouped_window_stats(x, g, w=6):
    """Trailing w-row mean/std within group via cumulative sums (fast, causal)."""
    valid = x.notna()
    v = x.fillna(0.0)
    cs = v.groupby(g).cumsum()
    cq = (v * v).groupby(g).cumsum()
    cn = valid.astype(float).groupby(g).cumsum()
    cs_l = cs.groupby(g).shift(w).fillna(0.0)
    cq_l = cq.groupby(g).shift(w).fillna(0.0)
    cn_l = cn.groupby(g).shift(w).fillna(0.0)
    n = cn - cn_l
    s = cs - cs_l
    q = cq - cq_l
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s / n
        var = (q - n * mean * mean) / (n - 1)
    mean = mean.where(n > 0)
    std = np.sqrt(var.clip(lower=0)).where(n > 1)
    return mean, std


def sirs_count(df_ff):
    """SIRS criteria met (0-4) using forward-filled values. Missing -> criterion not met."""
    c = (df_ff["HR"] > 90).astype(int)
    c += (df_ff["Resp"] > 20).astype(int)
    c += ((df_ff["Temp"] > 38) | (df_ff["Temp"] < 36)).astype(int)
    c += ((df_ff["WBC"] > 12) | (df_ff["WBC"] < 4)).astype(int)
    return c


def build_features(raw):
    """raw: sorted by Patient_ID, Hour. Returns feature DataFrame aligned with raw's index."""
    g = raw[ID]
    F = pd.DataFrame(index=raw.index)
    measured = [c for c in VITALS + LABS if c in raw.columns]
    ff = raw[measured].groupby(g).ffill()
    for c in measured:
        F[c] = ff[c]
    # missingness / recency of key labs
    hour = raw[HOUR].astype(float)
    for c in KEY_LABS:
        if c not in raw.columns:
            continue
        m = raw[c].notna()
        F[f"{c}_measured"] = m.astype(int).groupby(g).cummax()
        last_h = hour.where(m).groupby(g).ffill()
        F[f"{c}_hrs_since"] = hour - last_h
    # deltas vs 1h and 6h ago (on forward-filled values)
    for c in DELTA_VARS:
        F[f"{c}_d1"] = ff[c] - ff[c].groupby(g).shift(1)
        F[f"{c}_d6"] = ff[c] - ff[c].groupby(g).shift(6)
    # trailing 6h window stats
    for c in ROLL_VARS:
        F[f"{c}_mean6"], F[f"{c}_std6"] = _grouped_window_stats(ff[c], g, 6)
    with np.errstate(divide="ignore", invalid="ignore"):
        F["ShockIndex"] = ff["HR"] / ff["SBP"].where(ff["SBP"] > 0)
    F["ShockIndex_d6"] = F["ShockIndex"] - F["ShockIndex"].groupby(g).shift(6)
    F["SIRS"] = sirs_count(ff)
    for c in STATIC:
        if c in raw.columns:
            F[c] = raw[c]
    return F.replace([np.inf, -np.inf], np.nan).astype("float32")


# ----------------------------------------------------------------------------- readable names
PRETTY = {
    "HR": ("Heart rate", "bpm"), "O2Sat": ("SpO₂", "%"), "Temp": ("Temperature", "°C"),
    "SBP": ("Systolic BP", "mmHg"), "MAP": ("Mean arterial pressure", "mmHg"),
    "DBP": ("Diastolic BP", "mmHg"), "Resp": ("Respiratory rate", "/min"),
    "EtCO2": ("End-tidal CO₂", "mmHg"), "BaseExcess": ("Base excess", "mmol/L"),
    "HCO3": ("Bicarbonate", "mmol/L"), "FiO2": ("FiO₂", ""), "pH": ("Blood pH", ""),
    "PaCO2": ("PaCO₂", "mmHg"), "SaO2": ("SaO₂", "%"), "AST": ("AST", "IU/L"),
    "BUN": ("Urea (BUN)", "mg/dL"), "Alkalinephos": ("Alkaline phosphatase", "IU/L"),
    "Calcium": ("Calcium", "mg/dL"), "Chloride": ("Chloride", "mmol/L"),
    "Creatinine": ("Creatinine", "mg/dL"), "Bilirubin_direct": ("Direct bilirubin", "mg/dL"),
    "Glucose": ("Glucose", "mg/dL"), "Lactate": ("Lactate", "mmol/L"),
    "Magnesium": ("Magnesium", "mg/dL"), "Phosphate": ("Phosphate", "mg/dL"),
    "Potassium": ("Potassium", "mmol/L"), "Bilirubin_total": ("Total bilirubin", "mg/dL"),
    "TroponinI": ("Troponin I", "ng/mL"), "Hct": ("Haematocrit", "%"), "Hgb": ("Haemoglobin", "g/dL"),
    "PTT": ("PTT", "s"), "WBC": ("White cell count", "×10⁹/L"), "Fibrinogen": ("Fibrinogen", "mg/dL"),
    "Platelets": ("Platelets", "×10⁹/L"), "Age": ("Age", "y"), "Gender": ("Sex (1=male)", ""),
    "Unit1": ("MICU unit", ""), "Unit2": ("SICU unit", ""),
    "HospAdmTime": ("Hours from hospital to ICU admission", "h"),
    "ICULOS": ("Hours in ICU", "h"), "ShockIndex": ("Shock index (HR/SBP)", ""),
    "SIRS": ("SIRS criteria met", "/4"),
}


def base_var(feat):
    for suf in ["_d1", "_d6", "_mean6", "_std6", "_measured", "_hrs_since"]:
        if feat.endswith(suf):
            return feat[: -len(suf)], suf
    return feat, ""


def _fmt(v, unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    if float(v).is_integer():
        return f"{v:.0f} {unit}".strip()
    s = f"{v:.0f}" if a >= 100 else f"{v:.1f}" if a >= 10 else f"{v:.2f}"
    return f"{s} {unit}".strip()


def explain_feature(feat, row):
    """Plain-English sentence for one feature given the patient's feature row (Series)."""
    base, suf = base_var(feat)
    name, unit = PRETTY.get(base, (base, ""))
    cur = row.get(base, np.nan)
    d6 = row.get(f"{base}_d6", np.nan)
    if suf == "_measured":
        return f"{name} {'has been measured' if row.get(feat, 0) >= 1 else 'not yet measured'}"
    if suf == "_hrs_since":
        if row.get(feat, 99) < 1:
            return f"{name} just measured: {_fmt(cur, unit)}"
        return f"{name} last measured {_fmt(row.get(feat))} h ago (value {_fmt(cur, unit)})"
    if suf == "_std6":
        return f"{name} unstable: 6 h variability ±{_fmt(row.get(feat), unit)} (now {_fmt(cur, unit)})"
    if suf == "_mean6":
        return f"{name} 6 h average {_fmt(row.get(feat), unit)} (now {_fmt(cur, unit)})"
    if suf == "_d1":
        return f"{name} {_fmt(cur, unit)}, changed {row.get(feat, np.nan):+.1f} in the last hour"
    if base in ("Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS"):
        return f"{name}: {_fmt(row.get(base), unit)}"
    if d6 is not None and pd.notna(d6) and pd.notna(cur):
        was = cur - d6
        return f"{name} {_fmt(cur, unit)} (was {_fmt(was, unit)} six hours ago)"
    return f"{name} {_fmt(cur, unit)}"
