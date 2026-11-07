import numpy as np
from drift import feature_psi, score_psi, ks_stats


def label_free_report(X, scores, ref, thr, ref_sample, cut):
    """Everything we can measure the moment a week ends (no fraud labels needed)."""
    fp = feature_psi(X, ref)
    top = ref["top_features"]
    num_cols = [c for c, s in ref["features"].items() if s["type"] == "num"]
    ks = ks_stats(X, ref_sample, num_cols)
    rep = {
        "rows": len(X),
        "alert_rate": float((np.asarray(scores) >= thr).mean()),
        "score_psi": score_psi(scores, ref),
        "n_top_major": int(sum(fp[c] > cut for c in top)),
        "max_feature_psi": float(max(fp.values())),
        "max_ks": float(np.nanmax(list(ks.values()))),
    }
    return rep, fp, ks


def check_rules(rep, known_pr_auc, rules, baseline_pr_auc):
    """Returns a list of reasons. An empty list means no alert."""
    reasons = []
    if rep["score_psi"] > rules["score_psi"]:
        reasons.append("score drift")
    if rep["n_top_major"] >= rules["n_top_major"]:
        reasons.append("feature drift")
    if known_pr_auc is not None and known_pr_auc < (1 - rules["pr_auc_drop"]) * baseline_pr_auc:
        reasons.append("performance drop")
    return reasons