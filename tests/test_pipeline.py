"""Smoke + sanity tests for the forecasting lab.

These are intentionally lightweight (small panel, few estimators implicitly via
the default config) so they run in a few seconds, but they exercise every code
path the Streamlit app touches: synth -> classify -> route -> backtest.
"""

from __future__ import annotations

import numpy as np

from forecastlab import synth, classification, backtest, routing
from forecastlab.models.local import LOCAL_MODELS, croston, sba, tsb


def _small_panel():
    cfg = synth.SynthConfig(n_skus=60, n_weeks=104, seed=1)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    return demand, skus, wide


def test_synth_shapes_and_zeros():
    demand, skus, wide = _small_panel()
    assert wide.shape == (60, 104)
    # MRO demand must be mostly zeros, else the whole premise is wrong.
    assert (demand["demand"] == 0).mean() > 0.3
    # No negative demand.
    assert (demand["demand"] >= 0).all()


def test_classification_covers_quadrants():
    _, _, wide = _small_panel()
    classes = classification.classify_panel(wide, train_weeks=91)
    quads = set(classes["quadrant"].unique())
    # We engineered all four; at least the two hard ones must appear.
    assert {"intermittent", "lumpy"} & quads


def test_classifier_cutpoints():
    # A perfectly regular series -> smooth.
    reg = np.array([10.0] * 50)
    assert classification.classify_one(reg)["quadrant"] == "smooth"
    # Sporadic, consistent size -> intermittent.
    s = np.zeros(50)
    s[::5] = 4.0
    assert classification.classify_one(s)["quadrant"] == "intermittent"


def test_croston_family_nonnegative_and_finite():
    h = np.zeros(60)
    h[[3, 10, 11, 30, 55]] = [5, 2, 9, 1, 7]
    for fn in (croston, sba, tsb):
        out = fn(h, 8)
        assert out.shape == (8,)
        assert np.all(out >= 0) and np.all(np.isfinite(out))
    # SBA must be <= Croston (it de-biases downward).
    assert sba(h, 1)[0] <= croston(h, 1)[0] + 1e-9


def test_tsb_decays_on_obsolescence():
    # Demand then a long tail of zeros -> TSB rate should fall toward zero.
    h = np.zeros(80)
    h[:20:4] = 6.0  # demand only in the first 20 weeks
    rate_early = tsb(h[:24], 1)[0]
    rate_late = tsb(h, 1)[0]
    assert rate_late < rate_early


def test_backtest_runs_and_global_competitive():
    demand, skus, wide = _small_panel()
    train_weeks = 91
    res = backtest.run_backtest(wide, skus, train_weeks, horizon=13)
    overall = res["overall"].set_index("model")["wmape"]
    # Every model beats a degenerate WMAPE and is finite.
    assert overall.notna().all()
    assert (overall < 3.0).all()
    # The global tweedie model should at least beat the naive baseline.
    assert overall["global_tweedie"] < overall["naive"]


def test_route_one_includes_global():
    demand, skus, wide = _small_panel()
    train_weeks = 91
    res = backtest.run_backtest(wide, skus, train_weeks, horizon=13)
    gm = res["global_models"]["global_tweedie"]
    sku = wide.index[0]
    decision = routing.route_one(
        wide.loc[sku].values, 13, train_weeks=train_weeks,
        global_model=gm, skus=skus, wide=wide, sku=sku,
    )
    assert "global_tweedie" in decision["candidate_rates"]
    assert decision["quadrant"] in classification.RECOMMENDED
    assert set(LOCAL_MODELS) <= set(decision["candidate_rates"])
