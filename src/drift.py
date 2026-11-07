import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

EPS = 1e-4          # stops division by zero when a bucket is empty


def psi_from_props(ref, new):
    ref = np.clip(np.asarray(ref, dtype="float64"), EPS, None)
    new = np.clip(np.asarray(new, dtype="float64"), EPS, None)
    return float(np.sum((new - ref) * np.log(new / ref)))


def props_for(series, spec):
    """Share of rows in each bucket, using the SAME buckets saved on Day 4."""
    n = len(series)
    if spec["type"] == "num":
        nn = series.dropna().to_numpy(dtype="float64")
        idx = np.searchsorted(spec["edges"], nn, side="left")
        counts = np.bincount(idx, minlength=len(spec["edges"]) + 1)
        return np.append(counts, series.isna().sum()) / n
    vc = series.value_counts()
    vc.index = vc.index.astype(str)
    return np.array([vc.get(c, 0) for c in spec["cats"]] + [series.isna().sum()]) / n


def feature_psi(X, ref):
    return {c: psi_from_props(spec["props"], props_for(X[c], spec))
            for c, spec in ref["features"].items()}


def score_psi(scores, ref):
    spec = {"type": "num", **ref["score"]}
    new = props_for(pd.Series(scores), spec)[:-1]    # scores are never blank, so drop the blank bucket
    return psi_from_props(ref["score"]["props"], new)


def ks_stats(X_new, X_ref_sample, numeric_cols):
    out = {}
    for c in numeric_cols:
        a = X_ref_sample[c].dropna().to_numpy(dtype="float64")
        b = X_new[c].dropna().to_numpy(dtype="float64")
        out[c] = float(ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan
    return out