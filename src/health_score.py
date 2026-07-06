import numpy as np
 
BASE_WEIGHTS = {
    "cardiac": 0.22, "diabetes": 0.16, "renal": 0.15,
    "liver": 0.12, "thyroid": 0.08, "anaemia": 0.07,
}
# zone lower-bound -> weight multiplier (worse health = more influence)
ZONE_MULT = [(85, 0.8), (70, 1.0), (50, 1.5), (30, 2.5), (0, 4.0)]
ZONE_LABELS = [(85, "EXCELLENT"), (70, "GOOD"), (50, "FAIR"), (30, "POOR"), (0, "CRITICAL")]
 
NEUTRAL_SCORE = 70.0   # score assigned exactly at a model's decision threshold
 
 
def individual_score(p: float, threshold: float = 0.5,
                     neutral: float = NEUTRAL_SCORE) -> float:
    """
    Threshold-anchored health score in [0, 100].
 
    A probability equal to the model's tuned threshold maps to `neutral` (70,
    the GOOD/FAIR boundary). Lower probabilities rise toward 100; higher ones
    fall toward 0. This keeps the score consistent with the screen flag.
    """
    p = float(np.clip(p, 0.0, 1.0))
    t = float(np.clip(threshold, 1e-6, 1 - 1e-6))
    if p <= t:                                   # model says healthy
        score = 100.0 - (100.0 - neutral) * (p / t)
    else:                                        # model says disease
        score = neutral * (1.0 - (p - t) / (1.0 - t))
    return float(np.clip(score, 0.0, 100.0))
 
 
def _zone_label(score: float) -> str:
    return next(lbl for lo, lbl in ZONE_LABELS if score >= lo)
 
 
def _zone_mult(score: float) -> float:
    return next(m for lo, m in ZONE_MULT if score >= lo)
 
 
def overall_score(component_scores: dict, base_weights: dict = None) -> dict:
    """
    component_scores : {domain: health_score} for AVAILABLE domains only.
    Returns overall score + the weights actually used + zone.
    """
    base_weights = base_weights or BASE_WEIGHTS
    if not component_scores:
        return {"overall_score": None, "zone": None, "weights_used": {}}
 
    adj = {c: base_weights.get(c, 0.10) * _zone_mult(s) for c, s in component_scores.items()}
    total = sum(adj.values())
    w = {c: v / total for c, v in adj.items()}      # renormalised over AVAILABLE domains
 
    log_sum = sum(w[c] * np.log(max(component_scores[c], 0.5) / 100.0) for c in component_scores)
    overall = float(np.clip(100.0 * np.exp(log_sum), 0, 100))
    return {
        "overall_score": round(overall, 2),
        "zone": _zone_label(overall),
        "weights_used": {c: round(v, 3) for c, v in w.items()},
    }
 
 
def build_report(domain_predictions: dict, total_domains: int = None) -> dict:
    """
    domain_predictions : output of predict_patient() -> {key: result dict}.
    Produces the full patient report, honest about coverage/completeness.
    """
    available = {k: r for k, r in domain_predictions.items() if r.get("available")}
    skipped = {k: r["reason"] for k, r in domain_predictions.items() if not r.get("available")}
    total = total_domains or len(domain_predictions)
 
    comp_scores, comp_detail = {}, {}
    for k, r in available.items():
        thr = r.get("threshold", 0.5)
        s = round(individual_score(r["probability"], thr), 2)
        comp_scores[k] = s
        comp_detail[k] = {
            "display": r["display"],
            "disease_probability": r["probability"],
            "flag": r.get("flag"),               # 1 = screen positive (tuned cutoff)
            "threshold": thr,
            "health_score": s,
            "zone": _zone_label(s),
            "coverage": r["coverage"],
            "range_warnings": r.get("range_warnings", []),
            "imputed_features": r.get("imputed_features", []),
        }
 
    agg = overall_score(comp_scores)
    n_used = len(comp_scores)
 
    if n_used == 0:
        note = "No score - not enough test data for any system."
    elif n_used < total:
        note = f"Based on {n_used} of {total} systems.  Interpret with caution - several systems lacked data."
    else:
        note = f"Based on {n_used} of {total} systems."
 
    return {
        "overall_score": agg["overall_score"],
        "overall_zone": agg["zone"],
        "domains_used": n_used,
        "domains_total": total,
        "completeness": round(n_used / total, 2) if total else 0.0,
        "confidence_note": note,
        "weights_used": agg["weights_used"],
        "components": comp_detail,
        "skipped_domains": skipped,
        "risk_areas": dict(sorted(
            {k: v for k, v in comp_scores.items() if v < 70}.items(),
            key=lambda x: x[1])),
        "disclaimer": "Screening estimate only. Not clinically validated; not a diagnosis.",
    }
 
 
def print_report(rep: dict):
    print("\n" + "=" * 60)
    if rep["overall_score"] is None:
        print("  OVERALL HEALTH SCORE : NO SCORE")
    else:
        print(f"  OVERALL HEALTH SCORE : {rep['overall_score']}  ({rep['overall_zone']})")
    print(f"  {rep['confidence_note']}")
    print("=" * 60)
    for k, d in rep["components"].items():
        w = rep["weights_used"].get(k, 0)
        flag = "  <== SCREEN POSITIVE" if d.get("flag") else ""
        print(f"  {d['display']:9s}  score={d['health_score']:6.2f}  {d['zone']:9s}"
              f"  p(disease)={d['disease_probability']:.3f}  thr={d['threshold']:.2f}"
              f"  weight={w:.2f}  cov={d['coverage']:.0%}{flag}")
        for warn in d["range_warnings"]:
            print(f"             ! {warn}")
    for k, reason in rep["skipped_domains"].items():
        print(f"  {k:9s}  SKIPPED ({reason})")
    print("-" * 60)
    print(f"  {rep['disclaimer']}")