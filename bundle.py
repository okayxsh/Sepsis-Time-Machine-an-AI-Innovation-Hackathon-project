"""Surviving Sepsis Campaign Hour-1 bundle (2018) - item list + auto-suggestions."""
import math

ITEMS = [
    ("lactate", "Measure lactate (re-measure within 2–4 h if initial > 2 mmol/L)"),
    ("cultures", "Obtain blood cultures BEFORE antibiotics"),
    ("antibiotics", "Give broad-spectrum antibiotics"),
    ("fluids", "Rapid 30 mL/kg crystalloid for hypotension (MAP < 65) or lactate ≥ 4"),
    ("pressors", "Vasopressors if hypotensive during/after fluids, to keep MAP ≥ 65 mmHg"),
]


def _num(x):
    try:
        x = float(x)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def suggestions(map_mmhg, lactate, weight_kg):
    """Return {item_key: (status, note)} where status is 'required'|'consider'|'standard'|'n/a'."""
    m, lac = _num(map_mmhg), _num(lactate)
    vol = int(round(weight_kg * 30))
    hypo = m is not None and m < 65
    high_lac = lac is not None and lac >= 4
    out = {}
    if lac is None:
        out["lactate"] = ("required", "No lactate on record – measure now")
    elif lac > 2:
        out["lactate"] = ("required", f"Lactate {lac:.1f} mmol/L > 2 – re-measure in 2–4 h")
    else:
        out["lactate"] = ("standard", f"Last lactate {lac:.1f} mmol/L")
    out["cultures"] = ("required", "Before the first antibiotic dose")
    out["antibiotics"] = ("required", "Within 1 hour of recognition")
    if hypo or high_lac:
        why = " and ".join(w for w, c in [(f"MAP {m:.0f} < 65" if m is not None else "", hypo),
                                          (f"lactate {lac:.1f} ≥ 4" if lac is not None else "", high_lac)] if c)
        out["fluids"] = ("required", f"{why} → give {vol} mL crystalloid ({weight_kg:.0f} kg × 30 mL)")
    else:
        out["fluids"] = ("n/a", f"Not indicated now (MAP {m:.0f} mmHg)" if m is not None else
                         "Not indicated now")
    if hypo:
        out["pressors"] = ("consider", f"MAP {m:.0f} < 65 – start if still low during/after fluids")
    else:
        out["pressors"] = ("n/a", "Only if MAP stays < 65 after fluids")
    return out
