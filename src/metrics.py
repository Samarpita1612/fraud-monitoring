import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def evaluate(y_true, scores, precision_target=0.8):
    """Returns the numbers we track for every model."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    prec, rec, _ = precision_recall_curve(y_true, scores)
    ok = prec >= precision_target
    recall_at_p = float(rec[ok].max()) if ok.any() else 0.0

    return {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        f"recall@{int(precision_target * 100)}%prec": round(recall_at_p, 4),
    }