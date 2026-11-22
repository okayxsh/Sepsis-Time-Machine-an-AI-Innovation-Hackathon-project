# Sepsis Time Machine – 3-person pitch (~3.5 min)

Setup: landing page (`landing/index.html`) full-screen on the projector, and the dashboard
(`streamlit run app.py`, http://localhost:8501) already open in a second tab on **★ Demo 2**,
hour slider at 0. Press → / Space on the landing page to step through the sections.

---

## Speaker 1 – The hook and the problem (landing page: Hero → Problem) · ~45 s

*[Hero section on screen; the risk line animates]*

> "Quick show of hands: who's heard of sepsis?
> It's what happens when your body's response to an infection starts attacking your own organs.
> It kills around **11 million people a year**, roughly **one in every five deaths worldwide**."

*[Press → to go to Problem]*

> "The cruel part is that it's very treatable if you catch it early, and **every hour of delay raises the risk of death**.
> But ICUs still rely on simple rules like SIRS. They miss cases, they fire so often that staff tune them out,
> and they never tell the doctor *why*.
> So we asked: what if you could see sepsis coming hours before it arrives, and know exactly why?"

*[Hand-off]* "My teammate will show you how."

---

## Speaker 2 – The idea, how it works, results and trust (landing page: Idea → Time Machine → Results → Trust) · ~75 s

*[Press → to go to Idea]*

> "Sepsis Time Machine does three things: **predict, explain, act.**
> **Predict:** every hour, a LightGBM model trained on **real ICU data** from the PhysioNet 2019 challenge
> reads the patient's vital signs, lab results and trends and scores their sepsis risk.
> **Explain:** every alert comes with its top three reasons in plain English, computed with SHAP,
> for example 'creatinine 2.8, lactate rising'. No black box.
> **Act:** one click writes an SBAR page to the on-call doctor and starts the Surviving Sepsis
> Hour-1 bundle checklist."

*[Press → to go to Time Machine]*

> "To prove it, we built a Time Machine. It replays a real patient the model has **never seen**, hour by hour,
> showing only what the clinician could have known at that moment."

*[Press → to go to Results]*

> "On unseen patients our model scores an **AUROC of 0.83 against 0.64 for SIRS**.
> At the same catch rate it produces about **four times fewer false alarms**, and that is what fixes alarm fatigue.
> These are early numbers from a 1,500-patient subset; the full 40,000-patient run is in progress."

*[Press → to go to Trust]*

> "And we built it to be trusted. We split by patient so there's no leakage. The AI never invents values.
> If the internet goes down, a template takes over. And we'll show you a case we **missed**.
> Over to our live demo."

---

## Speaker 3 – Live dashboard (switch to the http://localhost:8501 tab) · ~90 s

*[Time Machine tab, ★ Demo 2, press **▶ Play**]*

> "This is a real ICU patient from our held-out test set. Watch the vital signs on the left and the risk line below.
> For the first hours everything looks normal… now the breathing rate is creeping up and kidney function is worsening…"

*[Alert fires at hour 18; press ⏸ Pause]*

> "**Hour 18: sepsis risk alert.** And here's *why*: creatinine 2.8, a fresh lactate of 3, acidotic blood pH.
> Those are real values a doctor can verify in seconds."

*[Click **📟 Page the doctor (SBAR)**]*

> "One click writes a structured SBAR page to the on-call doctor using only the numbers you see on screen."

*[Point to the Hour-1 Bundle; tick 'lactate' and 'cultures'; click **Simulate +15 min**]*

> "The Hour-1 bundle starts a countdown: lactate, cultures before antibiotics, antibiotics, fluids sized to
> the patient's weight. Anything not done in 60 minutes turns red."

*[Press ▶ Play again until the onset appears]*

> "And here's the reveal: **clinical onset at hour 25. We caught it 7 hours early.**"

*[Switch to the **Model Performance** tab; drag the sidebar threshold slider]*

> "This slider is the alarm-fatigue dial. Lower it and you catch more patients but get more false alarms;
> raise it and the opposite happens. The hospital chooses where to sit, and SIRS is the red X, well behind our curve."

*[Optional, if time: pick **Missed case** in the sidebar]*

> "And to be honest: here's a patient we missed. The alert came an hour after onset. No model is perfect,
> which is why this is decision support."

*[Close]*

> "Sepsis Time Machine: **hours earlier, and it tells you why.** Real ICU data, explainable,
> built to reduce alarm fatigue. Decision support, not a replacement for clinicians. Thank you."

---

## Likely judge questions (any speaker)
- **Data leakage?** The split is by patient (80/20, fixed seed), with features built only from past and current hours, and the threshold is chosen on validation patients, never on test.
- **Why not just an LLM?** The prediction is a trained model on real data with measured AUROC. The LLM only rephrases computed values for the SBAR page, and a template takes over offline.
- **How early is "early"?** The challenge labels sepsis 6 h before Sepsis-3 onset. We count a patient as caught only if the first alert comes at or before clinical onset.
- **Deployment?** We'd need prospective validation, local (UAE) data and EHR integration. Today it's a decision-support prototype.
