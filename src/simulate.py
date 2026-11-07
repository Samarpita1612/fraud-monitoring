import numpy as np
import pandas as pd

from features import RAW_FEATURES
from metrics import evaluate
from threshold import summarize
from drift import feature_psi, score_psi
from retrain import compare


def run_policy(prod, models, chall, trigger, v1_info, review_cost, min_gap=2, n_weeks=12):
    state = {"key": "v1", "thr": v1_info["thr"], "ref": v1_info["ref"],
             "baseline": v1_info["baseline"], "deployed": -1, "last_try": -99}
    weekly, events = [], []

    for w in range(n_weeks):
        b = prod[prod.batch == w]
        s = b["s_" + state["key"]].to_numpy()
        summ = summarize(b["isFraud"], b["TransactionAmt"], s, state["thr"], review_cost)
        weekly.append({"week": w, "model": state["key"],
                       "pr_auc": evaluate(b["isFraud"], s)["pr_auc"],
                       "net_benefit": summ["net_benefit"], "alert_rate": summ["alert_rate"],
                       "precision": summ["precision"], "recall": summ["recall"]})

        if w == n_weeks - 1:                     # nothing left to serve after the last week
            break

        # what the monitor knows at the end of week w
        fp = feature_psi(b[RAW_FEATURES], state["ref"])
        sc_psi = score_psi(s, state["ref"])

        pooled = None
        weeks_ok = list(range(state["deployed"] + 1, w - 1))     # weeks this model served, labels matured
        if len(weeks_ok) >= 2:
            d = prod[prod.batch.isin(weeks_ok[-2:])]
            pooled = evaluate(d["isFraud"], d["s_" + state["key"]])["pr_auc"]

        reasons = trigger(w, state, fp, sc_psi, pooled)
        if not reasons or (w - state["last_try"]) < min_gap:
            continue

        state["last_try"] = w
        c = chall[w]
        ok, cmp_ = compare(models[state["key"]], c["model"], c["holdout"], review_cost)
        events.append({"week": w, "reasons": ", ".join(reasons), "champion": state["key"],
                       "challenger": c["key"], "pooled_pr_auc": pooled,
                       "champ_holdout_pr_auc": cmp_["champion"]["pr_auc"],
                       "chall_holdout_pr_auc": cmp_["challenger"]["pr_auc"],
                       "chall_threshold": cmp_["challenger"]["thr"], "promoted": ok})
        if ok:
            state.update(key=c["key"], thr=cmp_["challenger"]["thr"], ref=c["ref"],
                         baseline=cmp_["challenger"]["pr_auc"], deployed=w)

    return pd.DataFrame(weekly), pd.DataFrame(events)