import numpy as np
import pandas as pd
import lightgbm as lgb

from features import RAW_FEATURES
from metrics import evaluate
from threshold import best_threshold

PARAMS = dict(n_estimators=247, learning_rate=0.05, num_leaves=63,
              min_child_samples=100, subsample=0.8, subsample_freq=1,
              colsample_bytree=0.8, random_state=42, verbose=-1)

TRAIN_DAYS = 60        # recent history used by a retrain
HOLDOUT_DAYS = 14      # newest matured days, kept aside for the practice match
LABEL_DELAY = 14       # days until a label is known


def matured_end(week):
    """At the END of production week `week`, rows with day < this value have known labels."""
    return 90 + 7 * (week + 1) - LABEL_DELAY


def window(feat, start_day, end_day):
    return feat[(feat.day >= start_day) & (feat.day < end_day)]


def train_challenger(feat, week):
    me = matured_end(week)
    tr = window(feat, me - HOLDOUT_DAYS - TRAIN_DAYS, me - HOLDOUT_DAYS)
    ho = window(feat, me - HOLDOUT_DAYS, me)
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(tr[RAW_FEATURES], tr["isFraud"])
    return model, tr, ho


def compare(champ, chall, ho, review_cost):
    """Practice match on the holdout. Each model gets its own best threshold,
    so neither has an unfair edge."""
    X, y = ho[RAW_FEATURES], ho["isFraud"]
    amt = ho["TransactionAmt"].to_numpy()
    out = {}
    for name, m in [("champion", champ), ("challenger", chall)]:
        s = m.predict_proba(X)[:, 1]
        thr, net = best_threshold(y, amt, s, review_cost)
        out[name] = {"pr_auc": evaluate(y, s)["pr_auc"], "net": net, "thr": thr}
    ok = (out["challenger"]["pr_auc"] >= 1.02 * out["champion"]["pr_auc"]
          and out["challenger"]["net"] >= out["champion"]["net"])
    return ok, out