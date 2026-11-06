import numpy as np
import pandas as pd


def _num_ref(s):
    nn = s.dropna().to_numpy(dtype="float64")
    edges = np.unique(np.quantile(nn, np.linspace(0.1, 0.9, 9))).tolist()
    # a value v goes in bin i where edges[i] is the first edge >= v
    idx = np.searchsorted(edges, nn, side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1)
    props = (np.append(counts, s.isna().sum()) / len(s)).tolist()   # last = blanks
    return {"type": "num", "edges": edges, "props": props}


def _cat_ref(s):
    cats = list(s.cat.categories)
    vc = s.value_counts()
    props = [float(vc.get(c, 0) / len(s)) for c in cats] + [float(s.isna().mean())]
    return {"type": "cat", "cats": [str(c) for c in cats], "props": props}


def build_reference(X_train, features, val_scores, importance):
    """Snapshot of what 'normal' looked like when the model was trained."""
    feats = {}
    for c in features:
        feats[c] = _cat_ref(X_train[c]) if isinstance(X_train[c].dtype, pd.CategoricalDtype) else _num_ref(X_train[c])

    # model scores: measured on validation rows (the model has not memorised those)
    sc = np.asarray(val_scores, dtype="float64")
    edges = np.unique(np.quantile(sc, np.linspace(0.1, 0.9, 9))).tolist()
    idx = np.searchsorted(edges, sc, side="left")
    props = (np.bincount(idx, minlength=len(edges) + 1) / len(sc)).tolist()

    return {
        "features": feats,
        "score": {"edges": edges, "props": props},
        "top_features": list(importance.sort_values(ascending=False).head(10).index),
    }