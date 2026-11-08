import altair as alt
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
MODELS = ROOT / "models"
POLICIES = ["never retrain", "every 4 weeks", "on alert"]
POLICY_COLORS = alt.Scale(domain=POLICIES, range=["#7f7f7f", "#1f77b4", "#e4572e"])
GAIN_COLORS = alt.Scale(domain=["every 4 weeks", "on alert"], range=["#1f77b4", "#e4572e"])

st.set_page_config(page_title="Fraud model monitoring", layout="wide")


@st.cache_data
def load():
    weekly = pd.read_csv(DOCS / "day6_weekly_by_policy.csv")
    events = pd.read_csv(DOCS / "day6_events.csv")
    gpsi = pd.read_csv(DOCS / "day6_group_psi_on_alert.csv")
    sp = DOCS / "day7_stress_test.csv"
    stress = pd.read_csv(sp) if sp.exists() else None
    registry = json.load(open(MODELS / "registry.json"))
    return weekly, events, gpsi, stress, registry


weekly, events, gpsi, stress, registry = load()

st.title("Fraud detection: monitoring and retraining")
st.caption("IEEE-CIS Fraud Detection data. 12-week production replay with a 14-day label delay. "
           "Review cost assumed at $5 per alert; amounts treated as dollars.")

policy = st.sidebar.selectbox("Policy to inspect", POLICIES, index=2)
st.sidebar.write("Charts compare all three policies. The numbers at the top and the retrain "
                 "log show the selected policy.")

w = weekly[weekly.policy == policy].sort_values("week")
ev = events[events.policy == policy]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Model serving in week 11", w.iloc[-1]["model"])
c2.metric("Mean weekly PR-AUC", f"{w.pr_auc.mean():.3f}")
c3.metric("Total net benefit", f"${w.net_benefit.sum():,.0f}")
c4.metric("Retrains promoted", int(ev.promoted.sum()))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Policies", "Drift", "Retrains and models", "Stress test", "About"])

with tab1:
    st.subheader("PR-AUC by week")
    pr_chart = (alt.Chart(weekly).mark_line(point=True)
                .encode(x=alt.X("week:O", title="Production week"),
                        y=alt.Y("pr_auc:Q", title="PR-AUC",
                                scale=alt.Scale(domain=[0.35, 0.70])),
                        color=alt.Color("policy:N", scale=POLICY_COLORS, title="Policy"))
                .properties(height=320))
    st.altair_chart(pr_chart, use_container_width=True)
    st.caption("The y-axis starts at 0.35 to make differences visible. Week-to-week swings of "
               "0.05 to 0.1 are normal here, so small gaps between policies are not conclusive.")

    cum = weekly.sort_values(["policy", "week"]).copy()
    cum["cumulative"] = cum.groupby("policy")["net_benefit"].cumsum()

    st.subheader("Cumulative net benefit ($)")
    cum_chart = (alt.Chart(cum).mark_line(point=True)
                 .encode(x=alt.X("week:O", title="Production week"),
                         y=alt.Y("cumulative:Q", title="Cumulative net benefit ($)"),
                         color=alt.Color("policy:N", sort=POLICIES, title="Policy"))
                 .properties(height=320))
    st.altair_chart(cum_chart, use_container_width=True)

    st.subheader("Extra net benefit vs never retrain ($)")
    piv = cum.pivot(index="week", columns="policy", values="cumulative")
    gain = (piv[["every 4 weeks", "on alert"]].sub(piv["never retrain"], axis=0)
            .reset_index().melt("week", var_name="policy", value_name="extra"))
    gain_chart = (alt.Chart(gain).mark_line(point=True)
                  .encode(x=alt.X("week:O", title="Production week"),
                          y=alt.Y("extra:Q", title="Extra net benefit ($)"),
                          color=alt.Color("policy:N", scale=GAIN_COLORS, title="Policy"))
                  .properties(height=260))
    st.altair_chart(gain_chart, use_container_width=True)

    totals = weekly.groupby("policy").agg(mean_pr_auc=("pr_auc", "mean"),
                                          total_net_benefit=("net_benefit", "sum")).loc[POLICIES]
    st.dataframe(totals.style.format({"mean_pr_auc": "{:.3f}", "total_net_benefit": "${:,.0f}"}))


with tab2:
    st.subheader("Feature-group drift (PSI) for the on-alert policy")
    st.caption("PSI is measured against the reference of whichever model was serving that week. "
               "Above 0.25 is major drift. The reference resets when a model is promoted.")
    g = gpsi.set_index("week")
    st.dataframe(g.drop(columns="model").style
                 .background_gradient(cmap="Reds", vmin=0, vmax=0.5).format("{:.2f}"), height=460)
    st.write("Serving model by week: " + ", ".join(f"{r.week}: {r.model}" for r in gpsi.itertuples()))

with tab3:
    st.subheader(f"Retrain attempts: {policy}")
    if len(ev):
        st.dataframe(ev.reset_index(drop=True))
    else:
        st.write("No retrain attempts for this policy.")
    st.subheader("Model registry")
    st.caption("The registry records the on-alert simulation. In the never-retrain policy, v1 stayed the champion.")
    st.dataframe(pd.json_normalize(registry))

with tab4:
    st.subheader("Alert-rule stress test")
    if stress is None:
        st.info("Run notebook 07 to create docs/day7_stress_test.csv")
    else:
        st.caption("Problems injected into weeks 8-11 while model c7 was serving. "
                   "Rules were frozen before the test.")
        st.dataframe(stress, height=620, use_container_width=True)

with tab5:
    st.subheader("Assumptions and limitations")
    st.markdown("""
- Review cost of $5 per alert is an assumption; amounts are treated as dollars; catching a fraud is assumed to save its full amount.
- One run, one random seed, 12 production weeks. Weekly PR-AUC varies by 0.05 to 0.1, so small differences are not conclusive.
- Alert rules were designed after seeing the production weeks, so those results are a development result, not a clean test.
- Most of the retraining gain comes from one model (c7, weeks 8-11). Earlier retrains were close to break-even, and c0 lost money in its two live weeks.
- Every challenger passed the promotion guard by a wide margin, so the guard was never shown to reject a bad model.
- The performance rule needs two matured weeks of the current model, so it rarely had a chance to fire, and in the stress test it fired on untouched data (0.505 vs a 0.508 line).
- PSI has no upper limit: a feed that goes fully blank produced values around 14.
- The pseudo customer ID (card1, card2, addr1) is an approximation.
""")