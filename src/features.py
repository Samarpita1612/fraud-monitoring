import numpy as np
import pandas as pd

META = ["TransactionID", "isFraud", "TransactionDT", "day", "week", "split", "batch"]

NUM_RAW = (["TransactionAmt", "card1", "card2", "card3", "card5",
            "addr1", "addr2", "dist1", "dist2"]
           + [f"C{i}" for i in range(1, 15)]
           + [f"D{i}" for i in range(1, 16)])

CAT_COLS = (["ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"]
            + [f"M{i}" for i in range(1, 10)])

ENG = ["hour", "dayofweek", "log_amt", "amt_cents", "email_match", "n_missing",
       "n_prior_txn", "n_prior_24h", "secs_since_last", "amt_vs_uid_mean"]

FEATURES = NUM_RAW + CAT_COLS + ENG


def build_features(df):
    """Adds engineered columns. Uses ONLY the current row and rows before it.
    Input must be sorted by TransactionDT."""
    df = df.copy()

    # how many boxes were left blank (measured before adding anything new)
    df["n_missing"] = df[NUM_RAW + CAT_COLS].isna().sum(axis=1).astype("int8")

    # when
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype("int8")
    df["dayofweek"] = ((df["TransactionDT"] // 86400) % 7).astype("int8")

    # amount
    df["log_amt"] = np.log1p(df["TransactionAmt"]).astype("float32")
    df["amt_cents"] = (df["TransactionAmt"] % 1).astype("float32")

    # 1 only if both emails exist and are equal
    df["email_match"] = (df["P_emaildomain"] == df["R_emaildomain"]).astype("int8")

    # pseudo-customer: a number for each unique (card1, card2, addr1) combination
    key = pd.DataFrame({"a": df["card1"],
                        "b": df["card2"].fillna(-1),
                        "c": df["addr1"].fillna(-1)})
    df["uid"] = key.groupby(["a", "b", "c"], sort=False).ngroup()

    g = df.groupby("uid", sort=False)

    # how many earlier payments by this pseudo-customer (current row not counted)
    df["n_prior_txn"] = g.cumcount().astype("int32")

    # seconds since their previous payment (empty for the first one)
    df["secs_since_last"] = g["TransactionDT"].diff().astype("float32")

    # this amount vs their average earlier amount
    amt = df["TransactionAmt"].astype("float64")
    prior_sum = amt.groupby(df["uid"], sort=False).cumsum() - amt
    prior_mean = prior_sum / df["n_prior_txn"].replace(0, np.nan)
    df["amt_vs_uid_mean"] = (amt / prior_mean).astype("float32")

    # how many earlier payments in the last 24 hours
    dt = df["TransactionDT"].to_numpy()
    out = np.zeros(len(df), dtype="int32")
    for idx in g.indices.values():
        t = dt[idx]
        out[idx] = np.arange(len(t)) - np.searchsorted(t, t - 86400, side="left")
    df["n_prior_24h"] = out

    return df.drop(columns=["uid"])


def to_categorical(df, train_mask):
    """Turns text columns into categories. The list of categories comes from
    TRAIN rows only, like a model that has only seen the past."""
    df = df.copy()
    for c in CAT_COLS:
        cats = df.loc[train_mask, c].dropna().unique()
        df[c] = pd.Categorical(df[c], categories=cats)
    return df
