"""Scrollable narrative: turning a demand forecast into a purchase decision.

A top-to-bottom story. Each beat pairs simplified math, a chart, and the outcome:

  Act 1  One number, four lies     — per-week WMAPE looks broken everywhere
  Act 2  Buy to the lead time      — reframe to lead-time demand; error collapses
  Act 3  A playbook per pattern    — smooth→min/max ... lumpy→policy (a gradient)
  Act 4  Does it work? KPIs        — service / fill / overstock per quadrant

This is the interactive *review* page's companion: same cached run, told as a
narrative for working through the recommended process.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app_common import (QUAD_COLORS, QUAD_ORDER, PLOTLY_TEMPLATE,
                        sidebar_controls, load)
from forecastlab import inventory

st.set_page_config(page_title="Forecast → Purchase", layout="wide")

n_skus, n_weeks, horizon, seed = sidebar_controls()
ctx = load(n_skus, n_weeks, horizon, seed)
wide, classes, results = ctx["wide"], ctx["classes"], ctx["results"]
overall = ctx["overall"]

MODEL = "global_tweedie"
rate = results["forecasts"][MODEL]
ltf = inventory.lead_time_frame(wide, rate, classes, horizon)

# Per-quadrant per-week WMAPE for the same model.
bq = results["by_quadrant"]
week_wmape = (bq[bq["model"] == MODEL].set_index("quadrant")["wmape"]).to_dict()
lt_kpi = inventory.by_quadrant(ltf, service=0.95)


def rep_sku(quad: str) -> str | None:
    """Representative SKU for a quadrant: the one with the most demand events."""
    pool = classes[classes["quadrant"] == quad]
    return pool["n_events"].idxmax() if len(pool) else None


def pooled_block_ratios(quad: str) -> np.ndarray:
    """Pooled distribution of (L-week demand / that SKU's mean L-week demand).

    Standardising by each SKU's own mean lets us pool heterogeneous SKUs onto one
    axis: 1.0 = an average interval. The *spread* of this cloud is exactly the
    buffer a buyer must hold above the mean forecast.
    """
    tw = wide.shape[1] - horizon
    out = []
    for sku in classes.index[classes["quadrant"] == quad]:
        train = wide.loc[sku].values[:tw]
        blocks = np.array([train[i:i + horizon].sum()
                           for i in range(0, len(train) - horizon + 1, horizon)])
        m = blocks.mean()
        if m > 0:
            out.extend((blocks / m).tolist())
    return np.array(out)


def badge(text: str, color: str) -> str:
    return (f"<span style='background:{color};color:white;padding:3px 10px;"
            f"border-radius:12px;font-size:0.85em;font-weight:600'>{text}</span>")


# =========================================================================== #
# HERO
# =========================================================================== #
st.title("From forecast to purchase order")
st.markdown(
    "A forecast is not a buy. This page follows one global Tweedie model from raw "
    "accuracy down to the **min/max levels and policies** a buyer actually uses — "
    "and shows why each demand pattern needs a *different* answer. Scroll. ⬇️"
)
st.divider()

# =========================================================================== #
# ACT 1 — one number, four lies
# =========================================================================== #
st.header("1 · One number, four lies")
st.markdown(
    "The first instinct is to score the forecast week by week. Do that and it "
    "looks broken almost everywhere:"
)
st.latex(r"\mathrm{WMAPE_{week}}=\frac{\sum_t \lvert a_t-\hat f_t\rvert}{\sum_t a_t}")

cols = st.columns(4)
for col, q in zip(cols, QUAD_ORDER):
    sku = rep_sku(q)
    with col:
        st.markdown(f"**{q.title()}**")
        if sku is not None:
            s = wide.loc[sku].values
            spark = go.Figure(go.Bar(y=s[-52:], marker_color=QUAD_COLORS[q]))
            spark.update_layout(height=90, margin=dict(l=0, r=0, t=0, b=0),
                                template=PLOTLY_TEMPLATE,
                                xaxis=dict(visible=False), yaxis=dict(visible=False))
            st.plotly_chart(spark, use_container_width=True,
                            key=f"spark_{q}", config={"displayModeBar": False})
        ww = week_wmape.get(q, float("nan"))
        st.metric("per-week WMAPE", f"{ww:.2f}")
        verdict = ("looks fine" if ww < 0.6 else "looks bad" if ww < 1.0 else "looks awful")
        st.caption(verdict)

st.error(
    "**This is the wrong number.** Weekly WMAPE mostly measures *timing* — which "
    "week a sporadic spike lands — and no model can predict that for lumpy parts. "
    "It says nothing about whether you'd buy the right quantity."
)
st.divider()

# =========================================================================== #
# ACT 2 — buy to the lead time
# =========================================================================== #
st.header("2 · Buy to the lead time")
c_math, c_chart = st.columns([1, 1.4])
with c_math:
    st.markdown(
        "Purchasing never buys a single week. It buys enough to cover a **lead "
        "time** of *L* weeks. So score the quantity you actually order — the "
        "**lead-time demand** — not the weekly wiggle:"
    )
    st.latex(r"D_L=\sum_{t=1}^{L} d_t \qquad \hat D_L=\text{rate}\times L")
    st.latex(r"\mathrm{WMAPE_{LT}}=\frac{\sum_i \lvert D_{L,i}-\hat D_{L,i}\rvert}"
             r"{\sum_i D_{L,i}}")
    allk = lt_kpi.loc["ALL"]
    st.metric(f"Lead-time WMAPE (L={horizon}w)", f"{allk['lt_wmape']:.2f}",
              help="Error on the actual purchase quantity")
    st.metric("Bias", f"{allk['lt_bias']:+.2f}",
              help="+ over-buy, − under-buy. Near zero = no systematic dead stock.")
with c_chart:
    rows = []
    for q in QUAD_ORDER:
        rows.append({"quadrant": q, "view": "per-week", "wmape": week_wmape.get(q, np.nan)})
        rows.append({"quadrant": q, "view": "lead-time",
                     "wmape": lt_kpi.loc[q, "lt_wmape"] if q in lt_kpi.index else np.nan})
    fig = px.bar(pd.DataFrame(rows), x="quadrant", y="wmape", color="view",
                 barmode="group", template=PLOTLY_TEMPLATE,
                 color_discrete_map={"per-week": "#bbbbbb", "lead-time": "#1f3b57"},
                 title="The same forecast, scored two ways")
    fig.update_layout(height=380, yaxis_title="WMAPE")
    st.plotly_chart(fig, use_container_width=True, key="reframe_bar")
st.success(
    "Aggregate to the buy and the error roughly halves, with near-zero bias. The "
    "forecast was usable all along — we were measuring it wrong."
)
st.divider()

# =========================================================================== #
# ACT 3 — a playbook per pattern
# =========================================================================== #
st.header("3 · A playbook per pattern")
st.markdown(
    "The two Syntetos-Boylan cut points don't just pick a *model* — they pick a "
    "**management style**, on a gradient from pure forecasting to pure policy. "
    "Each chart below is the pooled distribution of *actual ÷ expected* lead-time "
    "demand for that quadrant: **1.0 is an average interval, and the spread to "
    "the right is the buffer you must carry above the forecast.**"
)

PLAYBOOK = {
    "smooth": dict(
        title="Smooth → textbook min/max",
        approach=("Forecast-driven", "#2ca02c"),
        text=("Demand every week, steady size. The lead-time distribution is tight "
              "around 1.0, so the point forecast *is* essentially the buy and "
              "safety stock is a thin sliver."),
        math=r"S=\hat D_L+\underbrace{z\,\sigma_L}_{\text{small}}\;\approx\;\hat D_L"),
    "erratic": dict(
        title="Erratic → forecast + a real buffer",
        approach=("Forecast + safety stock", "#ff7f0e"),
        text=("Frequent demand, but spike sizes swing. The mean is reliable, the "
              "variance is not — so the forecast still anchors the buy, you just "
              "widen the safety stock to absorb the size swings."),
        math=r"S=\hat D_L+z\,\sigma_L\quad(\sigma_L\text{ large})"),
    "intermittent": dict(
        title="Intermittent → order to a quantile, not a mean",
        approach=("Distribution-driven", "#1f77b4"),
        text=("Long zero runs, consistent spikes. The point forecast is a small "
              "average that is *never* the actual weekly demand. Don't buy the "
              "mean — buy a service **quantile** of the lead-time distribution."),
        math=r"\mathrm{ROP}=F_{D_L}^{-1}(0.95)"),
    "lumpy": dict(
        title="Lumpy → manage by policy, not by SKU forecast",
        approach=("Policy-first", "#d62728"),
        text=("Sporadic **and** wildly variable. The distribution is so dispersed "
              "and zero-heavy that any single-SKU number is low-confidence. Stop "
              "chasing accuracy: **pool** demand across branches (risk pooling "
              "shrinks the relative spread by ~√n), hold deliberate **buffer / "
              "consignment** stock, or move slow movers to **make-to-order**. The "
              "forecast becomes a guardrail, not the plan."),
        math=r"\sigma_{\text{pooled}}\approx\frac{\sigma}{\sqrt{n}}\ll\textstyle\sum_i \sigma_i"),
}

for q in QUAD_ORDER:
    pb = PLAYBOOK[q]
    st.subheader(pb["title"])
    st.markdown(badge(*pb["approach"]), unsafe_allow_html=True)
    c_txt, c_fig = st.columns([1, 1.3])
    with c_txt:
        st.markdown(pb["text"])
        st.latex(pb["math"])
        if q in lt_kpi.index:
            k = lt_kpi.loc[q]
            m1, m2 = st.columns(2)
            m1.metric("within ±25% of actual buy", f"{100*k['within_25pct']:.0f}%")
            m2.metric("overstock for 95% service", f"{100*k['overstock_ss']:.0f}%")
    with c_fig:
        ratios = pooled_block_ratios(q)
        if ratios.size:
            p95 = float(np.quantile(ratios, 0.95))
            fig = px.histogram(x=ratios, nbins=40, template=PLOTLY_TEMPLATE,
                               color_discrete_sequence=[QUAD_COLORS[q]],
                               histnorm="probability")
            fig.add_vline(x=1.0, line_dash="dot", line_color="black",
                          annotation_text="forecast (mean)")
            fig.add_vline(x=p95, line_color="#1f3b57",
                          annotation_text=f"95% buy ≈ {p95:.1f}×")
            fig.update_layout(height=300, bargap=0.03, showlegend=False,
                              xaxis_title="actual ÷ expected lead-time demand",
                              yaxis_title="", margin=dict(t=30, b=10))
            st.plotly_chart(fig, use_container_width=True, key=f"ratio_{q}")
    st.divider()

# =========================================================================== #
# ACT 4 — does it work? KPIs
# =========================================================================== #
st.header("4 · Does it work? Service KPIs per pattern")
service = st.select_slider("Target service level", options=[0.90, 0.95, 0.99],
                           value=0.95, format_func=lambda v: f"{int(v*100)}%")
kpi = inventory.by_quadrant(ltf, service=service)

c1, c2 = st.columns([1.3, 1])
with c1:
    seg = [q for q in QUAD_ORDER if q in kpi.index]
    bars = pd.DataFrame({
        "quadrant": seg * 2,
        "policy": ["point forecast only"] * len(seg) + ["+ safety stock"] * len(seg),
        "fill": [kpi.loc[q, "fill_point"] for q in seg] + [kpi.loc[q, "fill_ss"] for q in seg],
    })
    fig = px.bar(bars, x="quadrant", y="fill", color="policy", barmode="group",
                 template=PLOTLY_TEMPLATE, title="Fill rate: point forecast vs + safety stock",
                 color_discrete_map={"point forecast only": "#cccccc", "+ safety stock": "#2ca02c"})
    fig.add_hline(y=service, line_dash="dot", line_color="#d62728",
                  annotation_text=f"target {int(service*100)}%")
    fig.update_layout(height=380, yaxis_tickformat=".0%", yaxis_title="fill rate")
    st.plotly_chart(fig, use_container_width=True, key="fill_bar")
with c2:
    st.markdown("**Cost of that service** — extra units carried above demand:")
    show = kpi[["lt_wmape", "fill_point", "fill_ss", "stockout_ss", "overstock_ss"]].copy()
    show.columns = ["LT WMAPE", "fill (pt)", "fill (+SS)", "stockout (+SS)", "overstock"]
    st.dataframe(
        show.style.format({"LT WMAPE": "{:.2f}", "fill (pt)": "{:.0%}",
                           "fill (+SS)": "{:.0%}", "stockout (+SS)": "{:.0%}",
                           "overstock": "{:+.0%}"}),
        use_container_width=True)

# Model value: same inventory, better model -> more service.
naive_ltf = inventory.lead_time_frame(wide, results["forecasts"]["naive"], classes, horizon)
tw_fill = inventory.service_metrics(ltf, service)["fill_ss"]
nv_fill = inventory.service_metrics(naive_ltf, service)["fill_ss"]
st.info(
    f"**Why the model still matters.** At the same {int(service*100)}% policy, the "
    f"global Tweedie forecast fills **{tw_fill:.0%}** of demand vs **{nv_fill:.0%}** "
    "for a naive baseline — tighter, less-biased forecasts buy the same service "
    "with less safety stock."
)

st.subheader("The one-line verdict per pattern")
st.markdown(
    "- **Smooth** — buy straight off the forecast; min/max is enough.\n"
    "- **Erratic** — forecast anchors the buy; widen safety stock for size swings.\n"
    "- **Intermittent** — order to a service *quantile* of the lead-time "
    "distribution, not the mean.\n"
    "- **Lumpy** — don't trust a per-SKU number; pool, buffer, or make-to-order, "
    "and use the forecast only as a guardrail."
)
st.caption("All figures recompute live from the sidebar controls. Synthetic data; "
           "the structural story — not the absolute numbers — is the takeaway.")
