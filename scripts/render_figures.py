"""Render the app's views to PNG without a browser.

The remote container has no working headless browser (the Playwright CDN is
blocked and the system chromium is a snap stub), so we cannot screenshot the
live Streamlit page. Instead this reproduces the *same figures the app builds*,
from the same synthetic data, and writes them as images via Plotly + kaleido.

    python -m scripts.render_figures            # writes PNGs to shots/
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from forecastlab import synth, backtest, classification

OUT = os.path.join(os.path.dirname(__file__), os.pardir, "shots")
QUAD_COLORS = {
    "smooth": "#2ca02c", "erratic": "#ff7f0e",
    "intermittent": "#1f77b4", "lumpy": "#d62728", "no_demand": "#7f7f7f",
}
TEMPLATE = "plotly_white"


def _save(fig, name, scale=2):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.update_layout(template=TEMPLATE)
    fig.write_image(path, scale=scale)
    print("wrote", os.path.relpath(path))
    return path


def single_sku_distribution(wide, classes, skus, sku, fname):
    series = wide.loc[sku].values
    nz = series[series > 0]
    quad = classes.loc[sku, "quadrant"]
    attrs = skus.set_index("sku").loc[sku]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("All weekly demand (incl. zeros)", "Nonzero spike sizes only"),
    )
    fig.add_histogram(x=series, nbinsx=30, marker_color=QUAD_COLORS[quad],
                      row=1, col=1, showlegend=False)
    fig.add_histogram(x=nz, nbinsx=20, marker_color="#555",
                      row=1, col=2, showlegend=False)
    fig.update_xaxes(title_text="units / week", row=1, col=1)
    fig.update_yaxes(title_text="weeks", row=1, col=1)
    fig.update_xaxes(title_text="units / event", row=1, col=2)
    fig.update_yaxes(title_text="events", row=1, col=2)
    fig.update_layout(
        width=1100, height=420, bargap=0.05,
        title_text=(f"{sku} — {quad}  ·  "
                    f"ADI {classes.loc[sku,'adi']:.2f}, CV² {classes.loc[sku,'cv2']:.2f}  ·  "
                    f"{100*(series==0).mean():.0f}% zero weeks  ·  "
                    f"{attrs['equipment_type']} / {attrs['branch']}"),
    )
    _save(fig, fname)


def grid_by_quadrant(wide, classes, quad, fname, n=16):
    members = list(classes.index[classes["quadrant"] == quad])[:n]
    rows = []
    for sku in members:
        v = wide.loc[sku].values
        for x in v[v > 0]:
            rows.append({"sku": sku, "size": x})
    df = pd.DataFrame(rows)
    fig = px.histogram(df, x="size", facet_col="sku", facet_col_wrap=4, nbins=15,
                       color_discrete_sequence=[QUAD_COLORS[quad]])
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1], font_size=10))
    fig.update_xaxes(matches=None, title_text="")
    fig.update_yaxes(matches=None, showticklabels=False, title_text="")
    fig.update_layout(width=1100, height=700, bargap=0.05, showlegend=False,
                      title_text=f"Per-SKU spike-size distributions — {quad} quadrant "
                                 f"(16 of {(classes['quadrant']==quad).sum()} SKUs)")
    _save(fig, fname)


def example_series(wide, classes, train_weeks, fname):
    quads = ["smooth", "erratic", "intermittent", "lumpy"]
    fig = make_subplots(rows=4, cols=1, subplot_titles=[
        (lambda s: f"{q} — {s} (ADI {classes.loc[s,'adi']:.2f}, CV² {classes.loc[s,'cv2']:.2f})")
        (classes.index[classes["quadrant"] == q][0]) for q in quads])
    for i, q in enumerate(quads, start=1):
        sku = classes.index[classes["quadrant"] == q][0]
        row = wide.loc[sku].values
        fig.add_bar(x=np.arange(len(row)), y=row, marker_color=QUAD_COLORS[q],
                    showlegend=False, row=i, col=1)
        fig.add_vline(x=train_weeks, line_dash="dash", line_color="black", row=i, col=1)
    fig.update_layout(width=1100, height=760,
                      title_text="Example demand series, one per quadrant "
                                 "(dashed line = holdout boundary)")
    _save(fig, fname)


def sb_plane(classes, fname):
    df = classes.replace({"adi": {np.inf: np.nan}}).dropna(subset=["adi"]).copy()
    df["cv2_clip"] = df["cv2"].clip(upper=4)
    df["adi_clip"] = df["adi"].clip(upper=12)
    fig = px.scatter(df, x="adi_clip", y="cv2_clip", color="quadrant",
                     color_discrete_map=QUAD_COLORS, hover_name=df.index,
                     labels={"adi_clip": "ADI (avg inter-demand interval)",
                             "cv2_clip": "CV² of demand size"})
    fig.add_vline(x=classification.ADI_CUT, line_dash="dash", line_color="gray")
    fig.add_hline(y=classification.CV2_CUT, line_dash="dash", line_color="gray")
    fig.update_layout(width=1000, height=620,
                      title_text="Syntetos-Boylan classification plane "
                                 "(cuts at ADI=1.32, CV²=0.49)")
    _save(fig, fname)


def verdict(results, fname_heat, fname_board):
    pivot = results["by_quadrant"].pivot(index="model", columns="quadrant", values="wmape")
    order = [q for q in ["smooth", "erratic", "intermittent", "lumpy"] if q in pivot.columns]
    pivot = pivot.loc[results["overall"].sort_values("wmape")["model"], order]
    fig = px.imshow(pivot, text_auto=".2f", color_continuous_scale="RdYlGn_r",
                    aspect="auto", labels=dict(color="WMAPE"))
    fig.update_layout(width=900, height=430,
                      title_text="WMAPE by quadrant (lower = better; global_tweedie wins the hard ones)")
    _save(fig, fname_heat)

    ov = results["overall"].sort_values("wmape")
    fig = go.Figure(data=[go.Table(
        header=dict(values=["model", "WMAPE", "MASE", "RMSSE", "Bias%", "n SKUs"],
                    fill_color="#1f3b57", font=dict(color="white", size=13), align="left"),
        cells=dict(
            values=[ov["model"],
                    ov["wmape"].round(3), ov["mase"].round(3), ov["rmsse"].round(3),
                    ov["bias_pct"].round(3), ov["n_skus"]],
            fill_color=[["#d6f5d6" if m == "global_tweedie" else "white" for m in ov["model"]]],
            align="left", height=26))])
    fig.update_layout(width=820, height=360,
                      title_text="Overall leaderboard (global_tweedie highlighted)")
    _save(fig, fname_board)


def main():
    cfg = synth.SynthConfig(n_skus=320, n_weeks=156, seed=7)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    train_weeks = 156 - 13
    classes = classification.classify_panel(wide, train_weeks=train_weeks)
    print("fitting models for the verdict figures…")
    results = backtest.run_backtest(wide, skus, train_weeks, horizon=13)
    classes = results["classes"]

    inter = classes.index[classes["quadrant"] == "intermittent"][0]
    lumpy = classes.index[classes["quadrant"] == "lumpy"][0]

    single_sku_distribution(wide, classes, skus, inter, "01_dist_single_intermittent.png")
    single_sku_distribution(wide, classes, skus, lumpy, "02_dist_single_lumpy.png")
    grid_by_quadrant(wide, classes, "lumpy", "03_dist_grid_lumpy.png")
    grid_by_quadrant(wide, classes, "intermittent", "04_dist_grid_intermittent.png")
    example_series(wide, classes, train_weeks, "05_panel_examples.png")
    sb_plane(classes, "06_classification_plane.png")
    verdict(results, "07_verdict_heatmap.png", "08_verdict_leaderboard.png")


if __name__ == "__main__":
    main()
