"""Forecasting models used in the lab.

Two flavours:

* ``local`` — one model fit per series (naive, SES, Croston, SBA, TSB). These
  are the conventional recommendations for the intermittent quadrants.
* ``global_xgb`` — a single model fit across *all* series with shared features
  and a Tweedie objective. This is the hypothesis under test.

Every forecaster exposes ``forecast(history, horizon) -> np.ndarray`` for the
local case, or a panel-level ``fit`` / ``predict`` for the global case.
"""
