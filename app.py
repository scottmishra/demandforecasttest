"""Streamlit app: the intermittent-demand forecasting playbook, made visible.

Run with:  streamlit run app.py

The app walks through the recommended process end to end on synthetic data for a
fictional industrial PVF / MRO distributor:

  1. Look at the demand panel and its intermittency.
  2. Classify every SKU on the Syntetos-Boylan plane (ADI vs CV^2).
  3. For a chosen SKU, show the routing decision and every candidate forecast.
  4. Validate the hypothesis: a single global Tweedie-objective gradient-boosted
     model beats per-series local methods on intermittent / lumpy demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from forecastlab import synth, backtest, classification, routing

st.set_page_config(page_title="Intermittent Demand Lab", layout="wide")

QUAD_COLORS = {
    "smooth": "#2ca02c",
    "erratic": "#ff7f0e",
    "intermittent": "#1f77b4",
    "lumpy": "#d62728",
    "no_demand": "#7f7f7f",
}


# --------------------------------------------------------------------------- #
# Cached compute
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Generating panel and fitting models…")
def compute(n_skus: int, n_weeks: int, horizon: int, seed: int):
    cfg = synth.SynthConfig(n_skus=n_skus, n_weeks=n_weeks, seed=seed)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    train_weeks = n_weeks - horizon
    results = backtest.run_backtest(wide, skus, train_weeks, horizon)
    return demand, skus, wide, train_weeks, results


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("⚙️ Experiment")
n_skus = st.sidebar.slider("SKUs", 100, 600, 320, step=20)
n_weeks = st.sidebar.slider("Weeks of history", 104, 208, 156, step=4)
horizon = st.sidebar.slider("Forecast horizon (weeks)", 4, 26, 13, step=1)
seed = st.sidebar.number_input("Random seed", value=7, step=1)
st.sidebar.caption(
    "All data is synthetic. The generator reproduces MRO structure — zeros, "
    "spikes, obsolescence, cold-start, and cross-SKU attribute signal — but "
    "does **not** use a Tweedie likelihood, so the Tweedie objective has to "
    "earn its win."
)

demand, skus, wide, train_weeks, results = compute(n_skus, n_weeks, horizon, int(seed))
classes = results["classes"]
overall = results["overall"]

st.title("Intermittent-Demand Forecasting — does global Tweedie win?")
st.markdown(
    "**Hypothesis under test:** for a parts distributor's sparse, lumpy demand, "
    "a *single global* gradient-boosted model with a **Tweedie** objective beats "
    "running one *local* model per SKU (Croston / SBA / TSB / exponential "
    "smoothing). The grain is the SKU itself — there is no BOM to roll up."
)

tab_data, tab_dist, tab_class, tab_route, tab_verdict = st.tabs(
    ["1 · Demand panel", "2 · Distributions", "3 · Classify",
     "4 · Route a SKU", "5 · Verdict"]
)

# --------------------------------------------------------------------------- #
# Tab 1 — data
# --------------------------------------------------------------------------- #
with tab_data:
    c1, c2, c3, c4 = st.columns(4)
    nz_frac = (demand["demand"] > 0).mean()
    c1.metric("SKUs", f"{skus.shape[0]:,}")
    c2.metric("Weeks", n_weeks)
    c3.metric("Zero-demand weeks", f"{100*(1-nz_frac):.0f}%")
    c4.metric("Obsolescent / cold-start",
              f"{skus['is_obsolete'].mean()*100:.0f}% / {skus['is_coldstart'].mean()*100:.0f}%")

    st.subheader("Quadrant mix (classified on training history)")
    mix = classes["quadrant"].value_counts().rename_axis("quadrant").reset_index(name="skus")
    fig = px.bar(mix, x="quadrant", y="skus", color="quadrant",
                 color_discrete_map=QUAD_COLORS)
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Example series, one per quadrant")
    for quad in ["smooth", "erratic", "intermittent", "lumpy"]:
        members = classes.index[classes["quadrant"] == quad]
        if len(members) == 0:
            continue
        sku = members[0]
        row = wide.loc[sku].values
        f = go.Figure()
        f.add_bar(x=np.arange(len(row)), y=row, marker_color=QUAD_COLORS[quad])
        f.add_vline(x=train_weeks, line_dash="dash", line_color="black",
                    annotation_text="holdout")
        f.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10),
                        title=f"{quad} — {sku} "
                              f"(ADI {classes.loc[sku,'adi']:.2f}, "
                              f"CV² {classes.loc[sku,'cv2']:.2f})")
        st.plotly_chart(f, use_container_width=True)

# --------------------------------------------------------------------------- #
# Tab 2 — per-SKU demand distributions
# --------------------------------------------------------------------------- #
with tab_dist:
    st.subheader("Demand distribution of each generated SKU")
    st.markdown(
        "Every SKU is a draw from its own latent process: a **Bernoulli arrival** "
        "(controls how often demand occurs) and a **gamma spike size** (controls "
        "how big and how variable). These are the distributions the forecasters "
        "have to recover — note the spike at zero that defines intermittent demand."
    )

    mode = st.radio("View", ["Single SKU", "Grid by quadrant"], horizontal=True)

    if mode == "Single SKU":
        qf = st.selectbox("Filter by quadrant", ["(any)"] + list(QUAD_COLORS),
                          key="dist_qf")
        pool = classes if qf == "(any)" else classes[classes["quadrant"] == qf]
        if len(pool) == 0:
            st.info("No SKUs in that quadrant for this configuration.")
        else:
            sku = st.selectbox("SKU", list(pool.index), key="dist_sku")
            series = wide.loc[sku].values
            nz = series[series > 0]
            attrs = skus.set_index("sku").loc[sku]

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Quadrant", classes.loc[sku, "quadrant"])
            c2.metric("Zero weeks", f"{100*(series == 0).mean():.0f}%")
            c3.metric("Mean spike", f"{nz.mean():.1f}" if nz.size else "—")
            c4.metric("Max", f"{series.max():.0f}")
            c5.metric("Events", int((series > 0).sum()))

            d1, d2 = st.columns(2)
            with d1:
                fig = px.histogram(x=series, nbins=30,
                                   color_discrete_sequence=[QUAD_COLORS[classes.loc[sku, "quadrant"]]])
                fig.update_layout(height=320, bargap=0.05,
                                  title="All weekly demand (incl. zeros)",
                                  xaxis_title="units / week", yaxis_title="weeks")
                st.plotly_chart(fig, use_container_width=True)
            with d2:
                if nz.size:
                    fig = px.histogram(x=nz, nbins=20,
                                       color_discrete_sequence=["#555"])
                    fig.update_layout(height=320, bargap=0.05,
                                      title="Nonzero spike sizes only",
                                      xaxis_title="units / event", yaxis_title="events")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("This SKU has no demand events in the window.")
            st.caption(
                f"Attributes — equipment: `{attrs['equipment_type']}`, family: "
                f"`{attrs['product_family']}`, branch: `{attrs['branch']}`, "
                f"price: ${attrs['unit_price']:.2f}"
                + ("  ·  ⚠️ obsolescent" if attrs["is_obsolete"] else "")
                + ("  ·  🆕 cold-start" if attrs["is_coldstart"] else "")
            )
    else:
        quad = st.selectbox("Quadrant", ["smooth", "erratic", "intermittent", "lumpy"],
                            index=2, key="dist_grid_q")
        members = list(classes.index[classes["quadrant"] == quad])
        n_show = st.slider("How many SKUs", 6, min(48, max(6, len(members))),
                           min(24, len(members)), step=6) if len(members) > 6 else len(members)
        members = members[:n_show]
        st.caption(f"Showing {len(members)} of "
                   f"{(classes['quadrant'] == quad).sum()} {quad} SKUs — "
                   "each panel is one SKU's distribution of nonzero spike sizes.")
        ncols = 4
        long_rows = []
        for sku in members:
            nz = wide.loc[sku].values
            nz = nz[nz > 0]
            for v in nz:
                long_rows.append({"sku": sku, "size": v})
        if long_rows:
            grid_df = pd.DataFrame(long_rows)
            fig = px.histogram(grid_df, x="size", facet_col="sku", facet_col_wrap=ncols,
                               nbins=15, color_discrete_sequence=[QUAD_COLORS[quad]],
                               height=140 * int(np.ceil(len(members) / ncols)))
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1],
                                                       font_size=9))
            fig.update_layout(bargap=0.05, showlegend=False,
                              margin=dict(l=10, r=10, t=24, b=10))
            fig.update_xaxes(matches=None, showticklabels=True, title_text="")
            fig.update_yaxes(matches=None, showticklabels=False, title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No demand events to show for this quadrant.")

# --------------------------------------------------------------------------- #
# Tab 3 — classification plane
# --------------------------------------------------------------------------- #
with tab_class:
    st.subheader("The Syntetos-Boylan plane")
    st.markdown(
        "Each part is placed by **ADI** (how sporadic) and **CV²** (how variable "
        "the spike sizes). The dashed cut points — ADI = 1.32, CV² = 0.49 — "
        "define the four routing quadrants."
    )
    plot_df = classes.replace({"adi": {np.inf: np.nan}}).dropna(subset=["adi"]).copy()
    plot_df["cv2_clip"] = plot_df["cv2"].clip(upper=4)
    plot_df["adi_clip"] = plot_df["adi"].clip(upper=12)
    fig = px.scatter(
        plot_df, x="adi_clip", y="cv2_clip", color="quadrant",
        color_discrete_map=QUAD_COLORS, hover_name=plot_df.index,
        hover_data={"n_events": True, "adi_clip": False, "cv2_clip": False},
        labels={"adi_clip": "ADI (avg inter-demand interval)", "cv2_clip": "CV² of demand size"},
    )
    fig.add_vline(x=classification.ADI_CUT, line_dash="dash", line_color="gray")
    fig.add_hline(y=classification.CV2_CUT, line_dash="dash", line_color="gray")
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Routing playbook")
    st.table(pd.DataFrame(
        [(q, t) for q, t in classification.RECOMMENDED.items()],
        columns=["quadrant", "recommended approach"],
    ))

# --------------------------------------------------------------------------- #
# Tab 4 — route a single SKU
# --------------------------------------------------------------------------- #
with tab_route:
    st.subheader("Pick a SKU and watch the playbook decide")
    quad_filter = st.selectbox("Filter by quadrant", ["(any)"] + list(QUAD_COLORS))
    pool = classes if quad_filter == "(any)" else classes[classes["quadrant"] == quad_filter]
    if len(pool) == 0:
        st.info("No SKUs in that quadrant for this configuration.")
    else:
        sku = st.selectbox("SKU", list(pool.index))
        hist = wide.loc[sku].values
        gm = results["global_models"].get("global_tweedie")
        decision = routing.route_one(
            hist, horizon, train_weeks=train_weeks,
            global_model=gm, skus=skus, wide=wide, sku=sku,
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Quadrant", decision["quadrant"])
        c2.metric("ADI / CV²", f"{decision['adi']:.2f} / {decision['cv2']:.2f}")
        c3.metric("Demand events (train)", decision["n_events"])
        st.info(f"**Playbook recommendation:** {decision['recommended_text']}  \n"
                f"Conventional local choice for this quadrant: "
                f"`{decision['recommended_local_model']}`")

        actual = hist[train_weeks:train_weeks + horizon]
        cand = decision["candidate_rates"]
        f = go.Figure()
        f.add_bar(x=np.arange(len(hist)), y=hist, name="demand",
                  marker_color=QUAD_COLORS[decision["quadrant"]], opacity=0.6)
        f.add_vline(x=train_weeks, line_dash="dash", line_color="black")
        hx = np.arange(train_weeks, train_weeks + horizon)
        highlight = {"croston", "sba", "tsb", "ses", "global_tweedie"}
        for name, rate in cand.items():
            if name in highlight:
                f.add_scatter(x=hx, y=np.full(horizon, rate), mode="lines",
                              name=name,
                              line=dict(width=3 if name == "global_tweedie" else 1.5,
                                        dash="solid" if name == "global_tweedie" else "dot"))
        f.update_layout(height=420, title=f"{sku} — history, holdout, and candidate rates",
                        legend=dict(orientation="h"))
        st.plotly_chart(f, use_container_width=True)

        cand_df = pd.DataFrame({"model": list(cand), "forecast_rate": list(cand.values())})
        cand_df["holdout_actual_rate"] = actual.mean()
        cand_df["abs_rate_error"] = (cand_df["forecast_rate"] - actual.mean()).abs()
        st.dataframe(cand_df.sort_values("abs_rate_error").round(3),
                     use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Tab 5 — verdict
# --------------------------------------------------------------------------- #
with tab_verdict:
    st.subheader("Overall leaderboard")
    st.caption("Lower is better. WMAPE = Σ|actual−forecast| / Σ actual over the "
               "held-out weeks; Bias% > 0 means over-forecasting.")
    show = overall.copy().sort_values("wmape")

    def _wmape_shade(col: pd.Series) -> list[str]:
        # Matplotlib-free green->red gradient: best WMAPE green, worst red.
        lo, hi = col.min(), col.max()
        rng = (hi - lo) or 1.0
        out = []
        for v in col:
            t = (v - lo) / rng  # 0 best .. 1 worst
            r = int(120 + 135 * t)
            g = int(200 - 120 * t)
            out.append(f"background-color: rgba({r}, {g}, 90, 0.45)")
        return out

    st.dataframe(
        show.style.format({"wmape": "{:.3f}", "mase": "{:.3f}", "rmsse": "{:.3f}",
                           "bias_pct": "{:+.3f}"})
        .apply(_wmape_shade, subset=["wmape"]),
        use_container_width=True, hide_index=True,
    )

    ov = overall.set_index("model")["wmape"]
    locals_only = ov.drop([m for m in ov.index if m.startswith("global_")])
    best_local = locals_only.idxmin()
    tw, sq = ov.get("global_tweedie", np.nan), ov.get("global_squared", np.nan)
    lift = 100 * (locals_only[best_local] - tw) / locals_only[best_local]
    abl = 100 * (sq - tw) / sq

    c1, c2, c3 = st.columns(3)
    c1.metric("Global Tweedie WMAPE", f"{tw:.3f}")
    c2.metric(f"Lift vs best local ({best_local})", f"{lift:+.1f}%")
    c3.metric("Tweedie vs squared-error (same features)", f"{abl:+.1f}%")
    if tw <= locals_only[best_local] and tw <= sq:
        st.success("Hypothesis supported: the global Tweedie model is the best "
                   "single default, and the Tweedie objective beats squared error "
                   "on the same global features.")
    else:
        st.warning("On this configuration the global Tweedie model is not the "
                   "outright winner — try a different seed/horizon to explore.")

    st.subheader("WMAPE by quadrant")
    pivot = results["by_quadrant"].pivot(index="model", columns="quadrant", values="wmape")
    order = [q for q in ["smooth", "erratic", "intermittent", "lumpy"] if q in pivot.columns]
    fig = px.imshow(pivot[order], text_auto=".2f", color_continuous_scale="RdYlGn_r",
                    aspect="auto", labels=dict(color="WMAPE"))
    fig.update_layout(height=380)
    st.plotly_chart(fig, use_container_width=True)

    if gm := results["global_models"].get("global_tweedie"):
        st.subheader("What the global model leans on")
        imp = gm.feature_importance().head(15).iloc[::-1]
        fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                     labels={"x": "importance", "y": ""})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("`weeks_since_sale` and the rolling demand-rate features "
                   "typically dominate — temporal recency and intermittency, not "
                   "calendar seasonality, drive intermittent demand.")
