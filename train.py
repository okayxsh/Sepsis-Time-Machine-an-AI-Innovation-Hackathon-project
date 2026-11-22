"""Sepsis Time Machine: training pipeline. Writes everything to artifacts/.

    python train.py

Patient-level split (no row leakage), causal features, LightGBM, SIRS baseline,
patient-level alerting metrics, demo-patient selection.
Threshold is chosen on VALIDATION patients (a subset of train), never on test.
"""
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from features import (HOUR, ID, LABEL, LABS, ROOT, STATIC, VITALS, build_features,
                      load_raw, sirs_count)

SEED = 42
ART = os.path.join(ROOT, "artifacts")
os.makedirs(ART, exist_ok=True)
THRESHOLDS = np.round(np.arange(0.05, 0.951, 0.05), 2)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:6.1f}s]", *a, flush=True)


# ----------------------------------------------------------------------------- patient metrics
def patient_table(df, score_col):
    """Per-patient info: septic?, clinical onset hour, length of stay."""
    g = df.groupby(ID, sort=False)
    pt = pd.DataFrame({
        "septic": g[LABEL].max().astype(bool),
        "n_hours": g[HOUR].count(),
        "last_hour": g[HOUR].max(),
    })
    fp = df[df[LABEL] == 1].groupby(ID)[HOUR].min()
    pt["first_pos_hour"] = fp
    pt["onset_hour"] = fp + 6
    return pt


def alert_metrics(df, score_col, t, pt=None):
    """Patient-level metrics for alert = score >= t.

    sensitivity: % septic patients whose first alert is at/before clinical onset
    lead time: onset - first alert (caught patients)
    false alerts / 100 patient-days: alert episodes (rising edges) in non-septic patients
    """
    if pt is None:
        pt = patient_table(df, score_col)
    a = df[score_col].values >= t
    ids = df[ID].values
    first_row = np.r_[True, ids[1:] != ids[:-1]]
    prev = np.r_[False, a[:-1]]
    prev[first_row] = False
    edges = a & ~prev
    al = df.loc[a, [ID, HOUR]].groupby(ID)[HOUR].min()
    first_alert = pt.index.map(al)
    first_alert = pd.Series(first_alert, index=pt.index, dtype=float)
    sep = pt[pt.septic]
    fa_sep = first_alert[sep.index]
    caught = fa_sep.notna() & (fa_sep <= sep.onset_hour)
    lead = (sep.onset_hour - fa_sep)[caught]
    non = ~pt.septic
    ep = pd.Series(edges, index=df.index).groupby(df[ID]).sum()
    ns_ids = pt.index[non]
    n_ep = ep.reindex(ns_ids).sum()
    hrs = pt.loc[non, "n_hours"].sum()
    return {
        "threshold": float(t),
        "sensitivity": float(caught.mean()) if len(sep) else float("nan"),
        "median_lead_h": float(lead.median()) if len(lead) else float("nan"),
        "false_alerts_per_100_pd": float(n_ep / hrs * 2400) if hrs else float("nan"),
        "nonseptic_pct_alerted": float(first_alert[ns_ids].notna().mean()),
        "n_septic": int(len(sep)), "n_caught": int(caught.sum()),
    }, first_alert


# ----------------------------------------------------------------------------- main
def main():
    raw = load_raw()
    log(f"rows={len(raw):,}  patients={raw[ID].nunique():,}  cols={raw.shape[1]}")
    print(raw.dtypes.to_string())
    print("septic patients %:", round(raw.groupby(ID)[LABEL].max().mean() * 100, 2),
          " positive rows %:", round(raw[LABEL].mean() * 100, 2))

    # --- patient-level split
    rng = np.random.RandomState(SEED)
    pids = np.array(sorted(raw[ID].unique()))
    rng.shuffle(pids)
    n_test = int(0.2 * len(pids))
    test_ids = set(pids[:n_test])
    trainval = pids[n_test:]
    n_val = int(0.15 * len(trainval))
    val_ids = set(trainval[:n_val])
    train_ids = set(trainval[n_val:])
    assert not (test_ids & (train_ids | val_ids))
    log(f"split patients: train={len(train_ids)} val={len(val_ids)} test={len(test_ids)}")

    # --- features
    X = build_features(raw)
    feat_names = list(X.columns)
    log(f"features: {len(feat_names)}")
    y = raw[LABEL].values.astype(int)
    is_tr = raw[ID].isin(train_ids).values
    is_va = raw[ID].isin(val_ids).values
    is_te = raw[ID].isin(test_ids).values

    # --- model
    pos = y[is_tr].sum()
    neg = is_tr.sum() - pos
    params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  scale_pos_weight=neg / pos, metric="auc", seed=SEED, verbose=-1, num_threads=0)
    dtr = lgb.Dataset(X[is_tr], y[is_tr], free_raw_data=False)
    dva = lgb.Dataset(X[is_va], y[is_va], reference=dtr)
    log(f"training LightGBM (scale_pos_weight={neg / pos:.1f}) ...")
    model = lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva], valid_names=["val"],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(50)])
    log(f"best iteration {model.best_iteration}, val AUROC {model.best_score['val']['auc']:.4f}")
    model.save_model(os.path.join(ART, "model.txt"), num_iteration=model.best_iteration)

    # --- choose threshold on validation patients (patient-level Youden: sens - % non-septic alerted)
    va = raw.loc[is_va, [ID, HOUR, LABEL]].copy()
    va["risk"] = model.predict(X[is_va], num_iteration=model.best_iteration)
    pt_va = patient_table(va, "risk")
    best_t, best_j, val_rows = 0.5, -1, []
    for t in np.round(np.arange(0.05, 0.951, 0.01), 2):
        m, _ = alert_metrics(va, "risk", t, pt_va)
        val_rows.append(m)
        j = m["sensitivity"] - m["nonseptic_pct_alerted"]
        if j > best_j:
            best_j, best_t = j, float(t)
    default_t = round(best_t, 2)
    log(f"default threshold (val Youden) = {default_t}")

    # --- test predictions
    te = raw.loc[is_te].copy()
    te["risk"] = model.predict(X[is_te], num_iteration=model.best_iteration)
    te["SIRS"] = X.loc[is_te, "SIRS"].values
    auroc = roc_auc_score(te[LABEL], te["risk"])
    auprc = average_precision_score(te[LABEL], te["risk"])
    sirs_auroc = roc_auc_score(te[LABEL], te["SIRS"])
    sirs_auprc = average_precision_score(te[LABEL], te["SIRS"])
    log(f"TEST AUROC={auroc:.4f} AUPRC={auprc:.4f} | SIRS AUROC={sirs_auroc:.4f}")
    if auroc > 0.95:
        print("WARNING: suspiciously high AUROC - check for leakage!")

    pt_te = patient_table(te, "risk")
    rows = [alert_metrics(te, "risk", t, pt_te)[0] for t in THRESHOLDS]
    pd.DataFrame(rows).to_csv(os.path.join(ART, "threshold_table.csv"), index=False)
    fine = pd.DataFrame([alert_metrics(te, "risk", t, pt_te)[0]
                         for t in np.round(np.arange(0.05, 0.951, 0.01), 2)])
    fine.to_csv(os.path.join(ART, "threshold_table_fine.csv"), index=False)
    te["SIRS_alert"] = (te["SIRS"] >= 2).astype(float)
    sirs_m, _ = alert_metrics(te, "SIRS_alert", 0.5, pt_te)
    at_default, first_alert = alert_metrics(te, "risk", default_t, pt_te)
    # our false-alert rate at the threshold that matches SIRS sensitivity
    match = fine[fine.sensitivity >= sirs_m["sensitivity"]]
    ours_at_sirs_sens = match.iloc[-1].to_dict() if len(match) else None

    metrics = {
        "data": {"rows": int(len(raw)), "patients": int(raw[ID].nunique()),
                 "train_patients": len(train_ids), "val_patients": len(val_ids),
                 "test_patients": len(test_ids), "test_rows": int(len(te)),
                 "septic_test_patients": int(pt_te.septic.sum()),
                 "pct_septic_patients": float(raw.groupby(ID)[LABEL].max().mean()),
                 "pct_positive_rows": float(raw[LABEL].mean())},
        "model": {"type": "LightGBM", "n_features": len(feat_names),
                  "best_iteration": int(model.best_iteration),
                  "val_auroc": float(model.best_score["val"]["auc"])},
        "test": {"auroc": auroc, "auprc": auprc, "prevalence_rows": float(te[LABEL].mean())},
        "sirs": {"auroc": sirs_auroc, "auprc": sirs_auprc, **sirs_m},
        "default_threshold": default_t,
        "threshold_rule": "patient-level Youden (sensitivity - % non-septic patients alerted) on validation patients",
        "at_default": at_default,
        "ours_at_sirs_sensitivity": ours_at_sirs_sens,
    }
    with open(os.path.join(ART, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # --- save test data: predictions + all raw columns (app recomputes features per patient)
    keep_raw = [c for c in VITALS + LABS + STATIC if c in te.columns]
    te[[ID, HOUR, "risk", LABEL] + keep_raw].reset_index(drop=True).to_parquet(
        os.path.join(ART, "test_preds.parquet"), index=False)

    # --- feature info
    imp = model.feature_importance("gain")
    with open(os.path.join(ART, "features.json"), "w") as f:
        json.dump({"features": feat_names,
                   "importance_gain": {n: float(v) for n, v in zip(feat_names, imp)}}, f, indent=2)

    # --- demo patients
    t = default_t
    pt_te["first_alert"] = first_alert
    pt_te["lead"] = pt_te.onset_hour - pt_te.first_alert
    risk_by = te.groupby(ID)["risk"]
    cand = pt_te[pt_te.septic & (pt_te.n_hours > 24) & (pt_te.first_pos_hour > 10)
                 & pt_te.lead.between(4, 8)].copy()

    def cleanliness(pid):
        d = te[te[ID] == pid]
        fa = pt_te.at[pid, "first_alert"]
        before = d[d[HOUR] < fa]["risk"]
        after = d[d[HOUR] >= fa]["risk"]
        # low & quiet before the alert, persistently high after it
        return (after >= t).mean() + (1 - before.max() / t if len(before) else 0) - before.std()

    cand["clean"] = [cleanliness(p) for p in cand.index]
    caught = cand.sort_values("clean", ascending=False).head(3)
    missed = pt_te[pt_te.septic & (pt_te.n_hours > 24) & (pt_te.first_pos_hour > 10)
                   & (pt_te.first_alert.isna() | (pt_te.first_alert > pt_te.onset_hour))]
    missed = missed.assign(mx=risk_by.max().reindex(missed.index)).sort_values("mx").head(1)
    nons = pt_te[~pt_te.septic & pt_te.first_alert.isna() & pt_te.n_hours.between(36, 72)]
    nons = nons.assign(mx=risk_by.max().reindex(nons.index)).sort_values("mx", ascending=False).head(1)

    demo = []
    for i, (pid, r) in enumerate(caught.iterrows(), 1):
        demo.append({"Patient_ID": pid, "kind": "caught", "label": f"★ Demo {i}: caught early",
                     "first_alert_hour": int(r.first_alert), "onset_hour": int(r.onset_hour),
                     "lead_h": int(r.lead)})
    for pid, r in missed.iterrows():
        demo.append({"Patient_ID": pid, "kind": "missed", "label": "Missed case",
                     "first_alert_hour": None if pd.isna(r.first_alert) else int(r.first_alert),
                     "onset_hour": int(r.onset_hour), "lead_h": None})
    for pid, r in nons.iterrows():
        demo.append({"Patient_ID": pid, "kind": "nonseptic", "label": "No sepsis",
                     "first_alert_hour": None, "onset_hour": None, "lead_h": None})
    with open(os.path.join(ART, "demo_patients.json"), "w") as f:
        json.dump({"threshold": t, "patients": demo}, f, indent=2)
    print(json.dumps(demo, indent=2))
    log("done")


if __name__ == "__main__":
    main()
