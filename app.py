"""Sepsis Time Machine - Streamlit app. Loads artifacts/ only (no training at runtime).

    streamlit run app.py
"""
import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from bundle import ITEMS, suggestions
from features import HOUR, ID, LABEL, PRETTY, base_var, build_features, explain_feature
from sbar import llm_sbar

warnings.filterwarnings("ignore", category=UserWarning, module="shap")
ROOT = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(ROOT, "artifacts")
RED, GREEN, GREY, AMBER, BLUE = "#ff4b4b", "#21c55d", "#9ca3af", "#f59e0b", "#38bdf8"
SS = st.session_state

st.set_page_config(page_title="Sepsis Time Machine", page_icon="🩺", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=JetBrains+Mono:wght@400;600&family=Mona+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, label {font-family: "Mona Sans", system-ui, -apple-system, "Segoe UI", sans-serif;}
h1, h2, h3, h4 {font-family: "Instrument Serif", Georgia, serif !important; font-weight: 400 !important; letter-spacing: -.01em;}
.block-container {padding-top: 2.4rem; padding-bottom: 1rem; max-width: 1500px;}
.topbar {display:flex; justify-content:space-between; align-items:center; gap:16px; flex-wrap:wrap;
         border:1px solid #101c2f; background:#091120; border-radius:14px; padding:12px 18px; margin-bottom:14px;}
.topbar .brand {font-family:"Instrument Serif",Georgia,serif; font-size:1.9rem; line-height:1;}
.topbar .brand em {color:#3fd0c9;}
.topbar .meta {font-family:"JetBrains Mono",Consolas,monospace; font-size:.85rem; color:#8fa3bd;}
.topbar .live {font-family:"JetBrains Mono",Consolas,monospace; font-size:.8rem; color:#f0626a; border:1px solid #f0626a;
               border-radius:999px; padding:3px 10px; margin-right:8px;}
.topbar a {color:#3fd0c9; text-decoration:none; font-size:.9rem;}
.stTabs [data-baseweb="tab"] {font-size:1.05rem; font-weight:600;}
.banner {border-radius: 10px; padding: 14px 18px; font-size: 1.45rem; font-weight: 800; margin: 4px 0 10px 0;}
.alert  {background: #7f1d1d; border: 2px solid #ff4b4b; color: #fff;}
.ok     {background: #0f2e1d; border: 1px solid #21c55d; color: #bbf7d0;}
.onset  {background: #1f2937; border: 1px solid #9ca3af; color: #e5e7eb; font-size: 1.15rem;}
.caught {background: #14532d; border: 2px solid #21c55d; color: #fff;}
.missed {background: #451a03; border: 2px solid #f59e0b; color: #fff;}
.bignum {font-size: 2.6rem; font-weight: 800; line-height: 1.0;}
.biglbl {font-size: 0.85rem; color: #9ca3af; text-transform: uppercase; letter-spacing: .05em;}
.reason {background: #1e293b; border-left: 4px solid #ff4b4b; padding: 8px 12px; margin: 6px 0;
         border-radius: 6px; font-size: 1.05rem;}
.badge {display:inline-block; padding: 3px 10px; border-radius: 999px; font-weight: 700; color: #fff;}
table.ward {width:100%; border-collapse: collapse; font-size: 1.0rem;}
table.ward th {text-align:left; color:#9ca3af; border-bottom:1px solid #374151; padding:6px;}
table.ward td {border-bottom:1px solid #1f2937; padding:7px 6px;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ loading
@st.cache_resource
def load_model():
    booster = lgb.Booster(model_file=os.path.join(ART, "model.txt"))
    with open(os.path.join(ART, "features.json")) as f:
        fj = json.load(f)
    return booster, fj["features"], fj["importance_gain"]


@st.cache_resource
def load_explainer():
    import shap
    return shap.TreeExplainer(load_model()[0])


@st.cache_resource
def load_preds():
    p = pd.read_parquet(os.path.join(ART, "test_preds.parquet"))
    p[ID] = p[ID].astype(str)
    idx = p.groupby(ID, sort=False).indices
    return p, idx


@st.cache_resource
def load_meta():
    with open(os.path.join(ART, "metrics.json")) as f:
        metrics = json.load(f)
    with open(os.path.join(ART, "demo_patients.json")) as f:
        demo = json.load(f)
    thr = pd.read_csv(os.path.join(ART, "threshold_table.csv"))
    fine_path = os.path.join(ART, "threshold_table_fine.csv")
    fine = pd.read_csv(fine_path) if os.path.exists(fine_path) else thr
    return metrics, demo, thr, fine


@st.cache_resource
def patient(pid):
    preds, idx = load_preds()
    _, feats, _ = load_model()
    d = preds.iloc[idx[pid]].sort_values(HOUR).reset_index(drop=True)
    F = build_features(d)[feats]
    fp = d.loc[d[LABEL] == 1, HOUR]
    info = {"septic": len(fp) > 0, "onset": int(fp.min()) + 6 if len(fp) else None,
            "last": int(d[HOUR].max()), "first": int(d[HOUR].min())}
    return d, F, info


@st.cache_resource
def patient_lists():
    preds, idx = load_preds()
    g = preds.groupby(ID, sort=False).agg(septic=(LABEL, "max"), n=(HOUR, "count"))
    return g


def shap_row(F, i):
    sv = load_explainer().shap_values(F.iloc[[i]])
    if isinstance(sv, list):
        sv = sv[1]
    return np.asarray(sv)[0]


@st.cache_data(show_spinner=False)
def top_reasons(pid, i, k=3):
    _, F, _ = patient(pid)
    _, feats, _ = load_model()
    sv = shap_row(F, i)
    row = F.iloc[i]
    out, used = [], set()
    for j in np.argsort(-sv):
        if sv[j] <= 0:
            break
        b, _ = base_var(feats[j])
        # show dynamic, clinically actionable drivers (static demographics/admin fields skipped)
        if b in used or b in ("ICULOS", "HospAdmTime", "Unit1", "Unit2", "Age", "Gender"):
            continue
        used.add(b)
        out.append((feats[j], float(sv[j]), explain_feature(feats[j], row)))
        if len(out) == k:
            break
    return out


def pretty_feat(f):
    b, suf = base_var(f)
    name = PRETTY.get(b, (b, ""))[0]
    return name + {"_d1": " (1 h change)", "_d6": " (6 h change)", "_mean6": " (6 h mean)",
                   "_std6": " (6 h variability)", "_measured": " measured yet",
                   "_hrs_since": " – hrs since measured", "": ""}[suf]


def fmt(x, nd=0):
    return "–" if x is None or pd.isna(x) else f"{x:.{nd}f}"


def risk_color(r, t):
    return RED if r >= t else AMBER if r >= 0.6 * t else GREEN


# ============================================================================ sidebar
if not os.path.exists(os.path.join(ART, "model.txt")):
    st.error("No artifacts found. Run `python train.py` first.")
    st.stop()

metrics, demo, thr_tab, thr_fine = load_meta()
preds, pidx = load_preds()
plist = patient_lists()

st.sidebar.markdown("## 🩺 Sepsis Time Machine")
st.sidebar.caption("Explainable early warning for ICU sepsis")
options, labels = [], {}
for p in demo["patients"]:
    options.append(p["Patient_ID"])
    labels[p["Patient_ID"]] = f"{p['label']} · {p['Patient_ID']}"
others_sep = [p for p in plist[plist.septic == 1].index if p not in labels][:150]
others_non = [p for p in plist[(plist.septic == 0) & (plist.n > 24)].index if p not in labels][:150]
for p in others_sep:
    labels[p] = f"Septic · {p}"
for p in others_non:
    labels[p] = f"Non-septic · {p}"
options += others_sep + others_non
# open on the strongest showcase (caught demo with the longest lead) so the presenter starts ready
caught = [p for p in demo["patients"] if p["kind"] == "caught"]
best = max(caught, key=lambda p: p["lead_h"] or 0)["Patient_ID"] if caught else options[0]
pid = st.sidebar.selectbox("Patient (unseen test set)", options, index=options.index(best),
                           format_func=lambda p: labels[p])
t = st.sidebar.slider("Alert threshold", 0.05, 0.95, float(metrics["default_threshold"]), 0.01,
                      help="Higher = fewer false alarms, fewer/later catches. See Model Performance.")
weight = st.sidebar.number_input("Patient weight (kg)", 30, 250, 70, 1,
                                 help="Dataset has no weight; needed for the 30 mL/kg fluid bolus.")
speed = st.sidebar.select_slider("Replay speed (s / ICU hour)", [0.1, 0.2, 0.35, 0.5, 1.0], 0.2)
st.sidebar.markdown("---")
st.sidebar.caption(f"Model: LightGBM · {metrics['model']['n_features']} causal features · "
                   f"trained on {metrics['data']['train_patients']:,} patients · "
                   f"test AUROC {metrics['test']['auroc']:.3f}")
st.sidebar.caption("Decision-support prototype; clinicians make the final decision.")

d, F, info = patient(pid)
risk = d["risk"].values
hours = d[HOUR].values
last = info["last"]
above = np.where(risk >= t)[0]
alert_i = int(above[0]) if len(above) else None
alert_h = int(hours[alert_i]) if alert_i is not None else None

# ----- replay state (must be set before the slider is created)
hkey = f"hour_{pid}"
if hkey not in SS:
    SS[hkey] = info["first"]
if SS.get("playing") and SS.get("play_pid") == pid:
    if SS[hkey] < last:
        SS[hkey] = min(last, SS[hkey] + 1)
    else:
        SS["playing"] = False

_pt_kind = next((p["label"] for p in demo["patients"] if p["Patient_ID"] == pid), labels[pid].split(" · ")[0])
st.markdown(f"""
<div class="topbar">
  <div class="brand">Sepsis <em>Time Machine</em></div>
  <div class="meta"><span class="live">● REPLAY</span>{_pt_kind} · patient {pid} · {last - info['first'] + 1} h ICU stay ·
     threshold {t:.2f} · test AUROC {metrics['test']['auroc']:.2f} vs SIRS {metrics['sirs']['auroc']:.2f}</div>
  <a href="http://localhost:8000" target="_blank">← Project overview</a>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["⏱ Time Machine", "🏥 Ward Board", "📈 Model Performance",
                                  "ℹ About / Limitations"])

# ============================================================================ TAB 1
with tab1:
    def _set(v):
        SS[hkey] = v
        SS["playing"] = False

    def _play():
        SS["playing"] = not SS.get("playing", False)
        SS["play_pid"] = pid
        if SS["playing"] and SS[hkey] >= last:
            SS[hkey] = info["first"]

    c = st.columns([1, 1, 1, 1, 6])
    c[0].button("⏸ Pause" if SS.get("playing") else "▶ Play", on_click=_play, use_container_width=True)
    c[1].button("⏮ Reset", on_click=_set, args=(info["first"],), use_container_width=True)
    c[2].button("−1 h", on_click=_set, args=(max(info["first"], SS[hkey] - 1),), use_container_width=True)
    c[3].button("+1 h", on_click=_set, args=(min(last, SS[hkey] + 1),), use_container_width=True)
    h = c[4].slider("ICU hour", info["first"], last, key=hkey, label_visibility="collapsed")

    m = hours <= h
    cur_i = int(np.where(m)[0][-1])
    cur = F.iloc[cur_i]
    fired = alert_h is not None and alert_h <= h
    onset = info["onset"]
    show_onset = info["septic"] and onset is not None and h >= min(onset, last)

    left, right = st.columns([3, 2], gap="large")
    with left:
        # ---- vitals
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=("Heart rate (bpm)", "Resp. rate (/min)",
                                            "Mean arterial pressure (mmHg)", "Temperature (°C)"))
        for r, (col, color) in enumerate([("HR", "#f472b6"), ("Resp", BLUE), ("MAP", "#a78bfa"),
                                          ("Temp", AMBER)], 1):
            fig.add_trace(go.Scatter(x=hours[m], y=F[col].values[m], mode="lines", line=dict(color=color, width=3),
                                     name=col, connectgaps=True), row=r, col=1)
        # shaded normal ranges make the drift obvious from the back of the room
        for r, (lo, hi_) in enumerate([(60, 100), (12, 20), (65, 100), (36, 38)], 1):
            fig.add_hrect(y0=lo, y1=hi_, fillcolor="rgba(63,208,201,0.08)", line_width=0, row=r, col=1)
        fig.add_hline(y=65, line=dict(color=RED, dash="dot", width=1), row=3, col=1)
        fig.update_annotations(font_size=15)
        fig.update_layout(font=dict(size=14))
        for r in range(1, 5):
            if fired:
                fig.add_vline(x=alert_h, line=dict(color=RED, width=2), row=r, col=1)
            if show_onset:
                fig.add_vline(x=onset, line=dict(color=GREY, width=2, dash="dash"), row=r, col=1)
        fig.update_xaxes(range=[info["first"] - 0.5, last + 0.5])
        fig.update_layout(height=470, template="plotly_dark", showlegend=False,
                          margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

        # ---- risk
        fr = go.Figure()
        fr.add_trace(go.Scatter(x=hours[m], y=risk[m], mode="lines", fill="tozeroy",
                                line=dict(color=BLUE, width=3), fillcolor="rgba(56,189,248,0.15)", name="risk"))
        hi = m & (risk >= t)
        fr.add_trace(go.Scatter(x=hours[hi], y=risk[hi], mode="markers", marker=dict(color=RED, size=7),
                                name="≥ threshold"))
        fr.add_hline(y=t, line=dict(color=RED, dash="dash", width=2),
                     annotation_text=f"threshold {t:.2f}", annotation_position="top left")
        if fired:
            fr.add_vline(x=alert_h, line=dict(color=RED, width=3),
                         annotation_text=f"⚠ Alert h{alert_h}", annotation_position="top right")
        if show_onset:
            fr.add_vline(x=onset, line=dict(color=GREY, width=3, dash="dash"),
                         annotation_text=f"Clinical onset h{onset}", annotation_position="bottom right")
        fr.update_xaxes(range=[info["first"] - 0.5, last + 0.5], title="ICU hour")
        fr.update_yaxes(range=[0, 1.02], title="Sepsis risk")
        fr.update_layout(height=260, template="plotly_dark", showlegend=False, font=dict(size=14),
                         margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fr, use_container_width=True)

    with right:
        # ---- headline numbers
        k = st.columns(4)
        rc = risk_color(risk[cur_i], t)
        k[0].markdown(f"<div class='biglbl'>Risk · h{h}</div><div class='bignum' style='color:{rc}'>"
                      f"{risk[cur_i]:.2f}</div>", unsafe_allow_html=True)
        k[1].markdown(f"<div class='biglbl'>HR</div><div class='bignum'>{fmt(cur['HR'])}</div>",
                      unsafe_allow_html=True)
        mc = RED if pd.notna(cur["MAP"]) and cur["MAP"] < 65 else "#fff"
        k[2].markdown(f"<div class='biglbl'>MAP</div><div class='bignum' style='color:{mc}'>{fmt(cur['MAP'])}</div>",
                      unsafe_allow_html=True)
        k[3].markdown(f"<div class='biglbl'>Lactate</div><div class='bignum'>{fmt(cur['Lactate'], 1)}</div>",
                      unsafe_allow_html=True)
        sex = {1: "M", 0: "F"}.get(int(cur["Gender"]) if pd.notna(cur["Gender"]) else -1, "?")
        st.caption(f"Patient {pid} · {fmt(cur['Age'])} y · {sex} · ICU hour {h} of {last} · "
                   f"RR {fmt(cur['Resp'])} · Temp {fmt(cur['Temp'], 1)} °C · SpO₂ {fmt(cur['O2Sat'])} % · "
                   f"WBC {fmt(cur['WBC'], 1)}")

        # ---- banners
        if fired:
            st.markdown(f"<div class='banner alert'>⚠ SEPSIS RISK – Hour {alert_h}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='banner ok'>✓ No alert so far (risk below {t:.2f})</div>",
                        unsafe_allow_html=True)
        if show_onset:
            st.markdown(f"<div class='banner onset'>Clinical onset (Sepsis-3) – Hour {onset}</div>",
                        unsafe_allow_html=True)
            if alert_h is not None and alert_h <= onset:
                st.markdown(f"<div class='banner caught'>✅ Caught {onset - alert_h} hours early</div>",
                            unsafe_allow_html=True)
            else:
                st.markdown("<div class='banner missed'>✗ Missed – no alert before onset at this threshold</div>",
                            unsafe_allow_html=True)

        # ---- why
        why_i = alert_i if fired else cur_i
        reasons = top_reasons(pid, why_i)
        st.markdown(f"#### {'Why this alert' if fired else 'Current top risk drivers'} "
                    f"<span style='color:#9ca3af;font-size:0.9rem'>(SHAP, hour {int(hours[why_i])})</span>",
                    unsafe_allow_html=True)
        if reasons:
            for f_, v_, txt in reasons:
                st.markdown(f"<div class='reason'>{txt} <span style='color:#9ca3af;float:right'>"
                            f"+{v_:.2f}</span></div>", unsafe_allow_html=True)
        else:
            st.caption("No features are currently pushing risk up.")

        # ---- bundle + SBAR (after alert)
        if fired:
            sug = suggestions(cur["MAP"], cur["Lactate"], weight)
            bstart = f"bundle_start_{pid}_{alert_h}"
            bff = f"bundle_ff_{pid}_{alert_h}"
            if bstart not in SS:
                SS[bstart] = time.time()
                SS[bff] = 0.0
            elapsed = (time.time() - SS[bstart]) / 60 + SS[bff]
            pending = [lbl.split(" (")[0] for key, lbl in ITEMS
                       if sug[key][0] in ("required", "consider") and not SS.get(f"chk_{pid}_{key}")]

            with st.container(border=True):
                hc = st.columns([3, 2])
                hc[0].markdown("#### ⏱ Hour-1 Bundle")
                remaining = max(0, int((60 - elapsed) * 60))
                color = RED if remaining == 0 else AMBER if remaining < 900 else GREEN
                with hc[1]:
                    components.html(f"""
<div id="cd" style="font:800 30px system-ui;color:{color};text-align:right"></div>
<script>
let s={remaining};const el=document.getElementById('cd');
function tick(){{const m=Math.floor(s/60),x=s%60;
el.textContent=s>0?(m+':'+String(x).padStart(2,'0')+' left'):'OVERDUE';
el.style.color=s<=0?'{RED}':(s<900?'{AMBER}':'{GREEN}');if(s>0)s--;}}
tick();setInterval(tick,1000);
</script>""", height=48)
                for key, lbl in ITEMS:
                    status, note = sug[key]
                    tag = {"required": ":red[**REQUIRED**]", "consider": ":orange[**CONSIDER**]",
                           "standard": ":green[standard]", "n/a": ":gray[not indicated]"}[status]
                    done = st.checkbox(f"{lbl}  {tag}", key=f"chk_{pid}_{key}")
                    overdue = (not done) and elapsed > 60 and status in ("required", "consider")
                    st.caption((":red[**OVERDUE** – ] " if overdue else "") + note)
                fc = st.columns(2)
                if fc[0].button("Simulate +15 min", use_container_width=True):
                    SS[bff] += 15
                    st.rerun()
                if fc[1].button("Restart timer", use_container_width=True):
                    SS[bstart] = time.time()
                    SS[bff] = 0.0
                    st.rerun()

            # ---- SBAR
            skey = f"sbar_{pid}_{alert_h}"
            if st.button("📟 Page the doctor (SBAR)", type="primary", use_container_width=True):
                a_row = F.iloc[alert_i]
                payload = {
                    "age": None if pd.isna(cur["Age"]) else int(cur["Age"]),
                    "gender": None if pd.isna(cur["Gender"]) else int(cur["Gender"]),
                    "icu_hour": int(h), "alert_hour": alert_h,
                    "vitals": {c_: (None if pd.isna(cur[c_]) else round(float(cur[c_]), 1))
                               for c_ in ["HR", "Resp", "MAP", "SBP", "Temp", "O2Sat"]},
                    "labs": {c_: (None if pd.isna(cur[c_]) else round(float(cur[c_]), 2))
                             for c_ in ["Lactate", "WBC", "Creatinine", "Platelets", "Bilirubin_total", "pH"]},
                    "change_6h": {c_: (None if pd.isna(cur[f"{c_}_d6"]) else round(float(cur[f"{c_}_d6"]), 1))
                                  for c_ in ["HR", "Resp", "MAP", "Temp"]},
                    "shock_index": None if pd.isna(cur["ShockIndex"]) else round(float(cur["ShockIndex"]), 2),
                    "risk": round(float(risk[cur_i]), 3), "risk_at_alert": round(float(risk[alert_i]), 3),
                    "threshold": round(t, 2),
                    "top_reasons": [r_[2] for r_ in top_reasons(pid, alert_i)],
                    "bundle_pending": pending,
                    "weight_kg": weight,
                }
                with st.spinner("Writing SBAR ..."):
                    SS[skey] = llm_sbar(payload)
            if skey in SS:
                text, src = SS[skey]
                with st.container(border=True):
                    st.markdown(f"**SBAR page** · <span style='color:#9ca3af'>source: {src}</span>",
                                unsafe_allow_html=True)
                    st.text(text)

# ============================================================================ TAB 2
@st.cache_resource
def ward_patients():
    rng = np.random.RandomState(7)
    long = plist[plist.n >= 48]
    sep = list(long[long.septic == 1].index)
    non = list(long[long.septic == 0].index)
    chosen = [p["Patient_ID"] for p in demo["patients"]]
    chosen += list(rng.choice(sep, min(4, len(sep)), replace=False))
    chosen += list(rng.choice(non, min(15 - len(chosen), len(non)), replace=False))
    return list(dict.fromkeys(chosen))


with tab2:
    wh = st.slider("Ward clock – ICU hour for every bed", 0, 72, 24, key="ward_h")
    rows = []
    for bed, p in enumerate(ward_patients(), 1):
        dd, FF, ii = patient(p)
        hh = dd[HOUR].values
        i = int(np.searchsorted(hh, min(wh, ii["last"]), side="right") - 1)
        if i < 0:
            continue
        rr = dd["risk"].values
        ab = np.where(rr[: i + 1] >= t)[0]
        r6 = rr[i] - rr[max(0, i - 6)]
        reason = top_reasons(p, i, 1)
        rows.append({"bed": bed, "pid": p, "age": FF["Age"].iloc[i], "sex": FF["Gender"].iloc[i],
                     "hour": int(hh[i]), "risk": rr[i], "trend": r6,
                     "reason": reason[0][2] if reason else "–",
                     "since": (int(hh[i] - hh[ab[0]]) if len(ab) else None),
                     "discharged": wh > ii["last"]})
    rows.sort(key=lambda r: -r["risk"])
    n_alert = sum(r["risk"] >= t for r in rows)
    st.markdown(f"**{len(rows)} beds · {n_alert} currently above threshold {t:.2f}**")
    html = ["<table class='ward'><tr><th>Bed</th><th>Patient</th><th>Age/Sex</th><th>ICU h</th>"
            "<th>Risk</th><th>6 h trend</th><th>Top reason</th><th>Alert</th></tr>"]
    for r in rows:
        col = risk_color(r["risk"], t)
        arrow = "▲" if r["trend"] > 0.05 else "▼" if r["trend"] < -0.05 else "▶"
        since = f"<span style='color:{RED};font-weight:700'>{r['since']} h ago</span>" if r["since"] is not None else "–"
        sx = {1: "M", 0: "F"}.get(int(r["sex"]) if pd.notna(r["sex"]) else -1, "?")
        hr_txt = f"{r['hour']}" + (" (end of stay)" if r["discharged"] else "")
        html.append(f"<tr><td>{r['bed']}</td><td>{r['pid']}</td><td>{fmt(r['age'])}/{sx}</td><td>{hr_txt}</td>"
                    f"<td><span class='badge' style='background:{col}'>{r['risk']:.2f}</span></td>"
                    f"<td>{arrow} {r['trend']:+.2f}</td><td>{r['reason']}</td><td>{since}</td></tr>")
    html.append("</table>")
    st.markdown("".join(html), unsafe_allow_html=True)
    st.caption("Unseen test-set patients replayed side by side. Risk badge: red ≥ threshold, amber ≥ 60 % of "
               "threshold, green below. Top reason = largest positive SHAP contribution at that hour.")

# ============================================================================ TAB 3
with tab3:
    row = thr_fine.iloc[(thr_fine.threshold - t).abs().argsort().iloc[0]]
    s = metrics["sirs"]
    c = st.columns(5)
    c[0].metric("AUROC – our model", f"{metrics['test']['auroc']:.3f}",
                f"{metrics['test']['auroc'] - s['auroc']:+.3f} vs SIRS")
    c[1].metric("AUROC – SIRS rule", f"{s['auroc']:.3f}")
    c[2].metric(f"Septic patients caught early (t={row.threshold:.2f})", f"{row.sensitivity * 100:.0f}%",
                f"{(row.sensitivity - s['sensitivity']) * 100:+.0f} pts vs SIRS")
    c[3].metric("Median lead time", f"{row.median_lead_h:.0f} h")
    c[4].metric("False alerts / 100 patient-days", f"{row.false_alerts_per_100_pd:.0f}",
                f"{row.false_alerts_per_100_pd - s['false_alerts_per_100_pd']:+.0f} vs SIRS", delta_color="inverse")
    st.caption(f"Test set: {metrics['data']['test_patients']:,} unseen patients "
               f"({metrics['data']['septic_test_patients']} septic), {metrics['data']['test_rows']:,} patient-hours. "
               f"AUPRC {metrics['test']['auprc']:.3f} (prevalence {metrics['test']['prevalence_rows'] * 100:.1f} %). "
               f"SIRS (≥2 of HR>90, RR>20, T>38/<36, WBC>12/<4): catches {s['sensitivity'] * 100:.0f} % early with "
               f"{s['false_alerts_per_100_pd']:.0f} false alerts/100 patient-days; "
               f"{s['nonseptic_pct_alerted'] * 100:.0f} % of non-septic patients alerted at least once.")
    o = metrics.get("ours_at_sirs_sensitivity")
    if o:
        st.info(f"**At the same sensitivity as SIRS** ({s['sensitivity'] * 100:.0f} %), our model needs "
                f"**{o['false_alerts_per_100_pd']:.0f}** false alerts per 100 patient-days vs SIRS "
                f"**{s['false_alerts_per_100_pd']:.0f}** (threshold {o['threshold']:.2f}).")

    g1, g2 = st.columns(2)
    with g1:
        ft = go.Figure()
        ft.add_trace(go.Scatter(x=thr_fine.threshold, y=thr_fine.sensitivity * 100, name="% septic caught early",
                                line=dict(color=GREEN, width=3)))
        ft.add_trace(go.Scatter(x=thr_fine.threshold, y=thr_fine.false_alerts_per_100_pd,
                                name="False alerts / 100 pt-days", line=dict(color=RED, width=3), yaxis="y2"))
        ft.add_trace(go.Scatter(x=thr_fine.threshold, y=thr_fine.median_lead_h, name="Median lead (h)",
                                line=dict(color=BLUE, width=2, dash="dot")))
        ft.add_vline(x=t, line=dict(color="#fff", dash="dash"), annotation_text=f"t = {t:.2f}")
        ft.update_layout(title="Alarm-fatigue trade-off", template="plotly_dark", height=420,
                         xaxis_title="Threshold", yaxis=dict(title="% caught / lead h"),
                         yaxis2=dict(title="False alerts / 100 pt-days", overlaying="y", side="right"),
                         legend=dict(orientation="h", y=-0.2), paper_bgcolor="rgba(0,0,0,0)",
                         margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(ft, use_container_width=True)
    with g2:
        fo = go.Figure()
        fo.add_trace(go.Scatter(x=thr_fine.false_alerts_per_100_pd, y=thr_fine.sensitivity * 100, mode="lines",
                                name="Our model (all thresholds)", line=dict(color=BLUE, width=3)))
        fo.add_trace(go.Scatter(x=[row.false_alerts_per_100_pd], y=[row.sensitivity * 100], mode="markers",
                                name=f"Ours @ t={row.threshold:.2f}", marker=dict(color="#fff", size=14)))
        fo.add_trace(go.Scatter(x=[s["false_alerts_per_100_pd"]], y=[s["sensitivity"] * 100], mode="markers",
                                name="SIRS ≥ 2", marker=dict(color=RED, size=16, symbol="x")))
        fo.update_layout(title="Operating curve: catches vs false alarms", template="plotly_dark", height=420,
                         xaxis_title="False alerts / 100 patient-days (lower is better)",
                         yaxis_title="% septic caught before onset", legend=dict(orientation="h", y=-0.2),
                         paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fo, use_container_width=True)

    _, feats, imp = load_model()
    top = sorted(imp.items(), key=lambda kv: -kv[1])[:15][::-1]
    fi = go.Figure(go.Bar(x=[v for _, v in top], y=[pretty_feat(k_) for k_, _ in top], orientation="h",
                          marker_color=BLUE))
    fi.update_layout(title="Top 15 features (LightGBM gain)", template="plotly_dark", height=480,
                     paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fi, use_container_width=True)
    with st.expander("Threshold table (test set, patient level)"):
        st.dataframe(thr_tab.style.format({"sensitivity": "{:.1%}", "nonseptic_pct_alerted": "{:.1%}",
                                           "median_lead_h": "{:.1f}", "false_alerts_per_100_pd": "{:.1f}"}),
                     use_container_width=True, hide_index=True)

# ============================================================================ TAB 4
with tab4:
    md = metrics["data"]
    st.markdown(f"""
### Data
**PhysioNet / Computing in Cardiology Challenge 2019** (also on Kaggle as *Prediction of Sepsis*):
{md['patients']:,} ICU patients, {md['rows']:,} patient-hours from two US hospital systems.
Hourly vital signs, sparse lab results and demographics. {md['pct_septic_patients'] * 100:.1f} % of patients
develop sepsis (Sepsis-3). The challenge label is already shifted 6 h early, so
**clinical onset = first positive hour + 6**.

### Method
- **Patient-level split** 80/20 (seed 42): {md['train_patients']:,} train, {md['val_patients']:,} validation
  (early stopping + threshold choice), **{md['test_patients']:,} test patients never seen in training**.
- **Causal features only** ({metrics['model']['n_features']}): forward-fill within patient, 1 h / 6 h deltas,
  trailing 6 h mean & variability, shock index (HR/SBP), lab-measurement recency flags, SIRS count,
  demographics. No backfill, no centred windows, no SepsisLabel-derived features.
- **LightGBM** gradient boosted trees with class-imbalance weighting; native missing-value handling.
- **Alert threshold** chosen on validation patients ({metrics['threshold_rule']}): **{metrics['default_threshold']:.2f}**.
- **Explanations:** SHAP TreeExplainer on the alert-hour row → top 3 positive contributors in plain English.
- **SBAR page:** an LLM (Claude) rephrases **only** the computed values; a deterministic template is used
  whenever the API is unavailable.
- **Baseline:** SIRS ≥ 2 criteria (HR > 90, RR > 20, T > 38 or < 36 °C, WBC > 12 or < 4).

### Metric definitions (patient level)
- *Caught early*: the first alert (risk ≥ threshold) happens at or before clinical onset.
- *Lead time*: onset hour − first alert hour.
- *False alerts / 100 patient-days*: alert episodes (rising edges) in non-septic patients ÷ their ICU hours × 2400.

### Limitations
- Retrospective data from 2 US hospitals; **not prospectively validated**; not tested on UAE populations.
- Labels follow Sepsis-3 as defined by the challenge; the definition itself is imperfect.
- Many lab results are missing; forward-fill assumes values persist.
- Demo patients were picked as illustrative examples (including a missed case); headline metrics are over the
  whole test set.
- **Decision support only, not a diagnosis.**

> ⚕️ *Decision-support prototype; clinicians make the final decision.*
""")

# ============================================================================ replay loop
if SS.get("playing") and SS.get("play_pid") == pid:
    time.sleep(speed)
    st.rerun()
