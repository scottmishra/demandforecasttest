"""Shared helpers for the Streamlit pages (Explore + Story).

Both pages need the same cached experiment and the same sidebar knobs, so they
live here and are imported by ``app.py`` and ``pages/`` alike. Keeping them in
one place means the two pages always show numbers from the *same* run.
"""

from __future__ import annotations

import streamlit as st

from forecastlab import synth, backtest

QUAD_COLORS = {
    "smooth": "#2ca02c",
    "erratic": "#ff7f0e",
    "intermittent": "#1f77b4",
    "lumpy": "#d62728",
    "no_demand": "#7f7f7f",
}
QUAD_ORDER = ["smooth", "erratic", "intermittent", "lumpy"]
PLOTLY_TEMPLATE = "plotly_white"


@st.cache_resource(show_spinner="Generating panel and fitting models…")
def compute(n_skus: int, n_weeks: int, horizon: int, seed: int):
    """Generate the panel and run the full backtest. Cached across pages."""
    cfg = synth.SynthConfig(n_skus=n_skus, n_weeks=n_weeks, seed=seed)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    train_weeks = n_weeks - horizon
    results = backtest.run_backtest(wide, skus, train_weeks, horizon)
    return demand, skus, wide, train_weeks, results


def sidebar_controls():
    """Render the shared experiment controls. Keyed so values persist across pages."""
    st.sidebar.title("⚙️ Experiment")
    n_skus = st.sidebar.slider("SKUs", 100, 600, 320, step=20, key="n_skus")
    n_weeks = st.sidebar.slider("Weeks of history", 104, 208, 156, step=4, key="n_weeks")
    horizon = st.sidebar.slider("Forecast / lead-time (weeks)", 4, 26, 13, step=1,
                                key="horizon")
    seed = st.sidebar.number_input("Random seed", value=7, step=1, key="seed")
    st.sidebar.caption(
        "All data is synthetic. The generator reproduces MRO structure — zeros, "
        "spikes, obsolescence, cold-start, and cross-SKU attribute signal — but "
        "does **not** use a Tweedie likelihood, so the objective has to earn its win."
    )
    return n_skus, n_weeks, horizon, int(seed)


def load(n_skus, n_weeks, horizon, seed):
    """Convenience: run/fetch the cached experiment and unpack the common bits."""
    demand, skus, wide, train_weeks, results = compute(n_skus, n_weeks, horizon, seed)
    return {
        "demand": demand, "skus": skus, "wide": wide,
        "train_weeks": train_weeks, "results": results,
        "classes": results["classes"], "overall": results["overall"],
        "horizon": horizon,
    }
