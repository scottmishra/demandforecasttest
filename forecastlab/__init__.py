"""forecastlab — a small lab for validating intermittent-demand forecasting choices.

The package synthesises a realistic SKU-level demand panel for a fictional
industrial PVF / MRO distributor, classifies each part by demand pattern
(Syntetos-Boylan), routes it to an appropriate model, and backtests a set of
forecasting methods so the central hypothesis can be checked empirically:

    A *single global* gradient-boosted model trained across the whole SKU
    portfolio with a *Tweedie* objective beats per-series local methods on
    intermittent / lumpy demand.

Nothing here is tied to a real customer; all data is generated.
"""

__all__ = [
    "synth",
    "classification",
    "backtest",
    "routing",
    "inventory",
]
