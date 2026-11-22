"""SBAR page to the on-call physician: Claude API call with a deterministic offline fallback.

The LLM only rephrases values we computed (risk, vitals, SHAP reasons). If the key is
missing, the network is down, the call is slow (>10 s) or anything else fails, the
template below is used, so the demo never depends on wifi.
"""
import json
import os

SYSTEM_PROMPT = (
    "You are a clinical communication assistant. Write a concise SBAR page to the on-call "
    "physician using ONLY the values provided. Do not invent history, diagnoses, medications "
    "or values. Under 120 words. Use four labelled lines: S:, B:, A:, R:. "
    "End with: 'AI-generated decision support. Verify clinically.'"
)
MODEL = os.environ.get("SBAR_MODEL", "claude-opus-5-5")
DISCLAIMER = "AI-generated decision support. Verify clinically."


def _v(x, unit="", nd=0):
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.{nd}f}{(' ' + unit) if unit else ''}"
    except (TypeError, ValueError):
        return str(x)


def template_sbar(p):
    """Deterministic SBAR from the same JSON payload (offline fallback)."""
    v, labs, d6 = p.get("vitals", {}), p.get("labs", {}), p.get("change_6h", {})
    sex = {1: "male", 0: "female"}.get(p.get("gender"), "patient")
    reasons = "; ".join(p.get("top_reasons", [])[:3]) or "n/a"
    pending = [b for b in p.get("bundle_pending", [])]
    s = (f"S: Sepsis early-warning alert. {_v(p.get('age'))}-year-old {sex}, ICU hour {p.get('icu_hour')}. "
         f"Risk {_v(p.get('risk'), nd=2)} (alert threshold {_v(p.get('threshold'), nd=2)}).")
    b = (f"B: HR {_v(v.get('HR'))}, RR {_v(v.get('Resp'))}, MAP {_v(v.get('MAP'))}, "
         f"SBP {_v(v.get('SBP'))}, Temp {_v(v.get('Temp'), nd=1)}, SpO2 {_v(v.get('O2Sat'))}. "
         f"Lactate {_v(labs.get('Lactate'), nd=1)}, WBC {_v(labs.get('WBC'), nd=1)}, "
         f"Creatinine {_v(labs.get('Creatinine'), nd=2)}. "
         f"6 h change: HR {_v(d6.get('HR'))}, MAP {_v(d6.get('MAP'))}, RR {_v(d6.get('Resp'))}.")
    a = f"A: Model flags rising sepsis risk. Main drivers: {reasons}."
    r = ("R: Please assess at bedside. Consider Hour-1 sepsis bundle"
         + (f" (pending: {', '.join(pending)})." if pending else ".")
         )
    return "\n".join([s, b, a, r, DISCLAIMER])


def llm_sbar(payload, timeout=10.0):
    """Return (text, source). source is 'claude' or 'template (<reason>)'."""
    try:
        import anthropic
    except ImportError:
        return template_sbar(payload), "template (anthropic package not installed)"
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return template_sbar(payload), "template (no API key set)"
    try:
        client = anthropic.Anthropic(timeout=timeout, max_retries=0)
        resp = client.beta.messages.create(
            model=MODEL,
            max_tokens=1024,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                       "Patient data (JSON):\n" + json.dumps(payload, indent=1, default=str)}],
        )
        if resp.stop_reason == "refusal":
            return template_sbar(payload), "template (model declined)"
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not text:
            return template_sbar(payload), "template (empty response)"
        if DISCLAIMER not in text:
            text += "\n" + DISCLAIMER
        return text, f"Claude ({resp.model})"
    except Exception as e:  # network, timeout, auth, rate-limit, SDK mismatch: never break the demo
        return template_sbar(payload), f"template ({type(e).__name__})"
