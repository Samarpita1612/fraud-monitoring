import numpy as np


def best_threshold(y_true, amount, scores, review_cost):
    """Tries every possible line and returns the one with the highest net benefit.
    Net benefit = fraud $ caught - review_cost * number of alerts."""
    y = np.asarray(y_true)
    amt = np.asarray(amount, dtype="float64")
    s = np.asarray(scores, dtype="float64")

    order = np.argsort(-s)                      # most suspicious first
    s_sorted = s[order]
    caught = np.cumsum((amt * y)[order])        # fraud $ caught if we flag the top k
    alerts = np.arange(1, len(s) + 1)
    net = caught - review_cost * alerts

    k = int(np.argmax(net))
    if net[k] <= 0:                             # flagging anything loses money
        return 1.01, 0.0
    return float(s_sorted[k]), float(net[k])


def summarize(y_true, amount, scores, threshold, review_cost):
    """Plain-language numbers for a chosen threshold."""
    y = np.asarray(y_true)
    amt = np.asarray(amount, dtype="float64")
    flag = np.asarray(scores) >= threshold

    caught_usd = float((amt * y)[flag].sum())
    total_usd = float((amt * y).sum())
    n_alerts = int(flag.sum())
    n_fraud = int(y.sum())
    return {
        "threshold": round(float(threshold), 4),
        "alerts": n_alerts,
        "alert_rate": round(n_alerts / len(y), 4),
        "precision": round(float(y[flag].mean()), 4) if n_alerts else 0.0,
        "recall": round(float(y[flag].sum() / n_fraud), 4) if n_fraud else 0.0,
        "fraud_usd_caught": round(caught_usd, 2),
        "fraud_usd_total": round(total_usd, 2),
        "net_benefit": round(caught_usd - review_cost * n_alerts, 2),
    }