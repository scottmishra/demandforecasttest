"""Is the forecast good enough to *buy* against? (headless)

Per-period WMAPE is the wrong question for an intermittent SKU: you cannot
predict which week a spike lands, so weekly point error is irreducibly large.
Purchasing cares about lead-time demand, bias, and the service level it can hit.

This prints those, overall and by quadrant, reusing forecastlab.inventory so the
numbers match the Streamlit "Inventory story" page exactly.
"""

from __future__ import annotations

import pandas as pd

from forecastlab import synth, backtest, inventory

H = 13          # holdout / protection interval (weeks)
SERVICE = 0.95


def main():
    cfg = synth.SynthConfig(n_skus=320, n_weeks=156, seed=7)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    results = backtest.run_backtest(wide, skus, 156 - H, H)
    classes = results["classes"]

    pd.set_option("display.width", 170)
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")

    week = results["overall"].set_index("model").loc["global_tweedie", "wmape"]
    print("=" * 80)
    print(f"per-week WMAPE (global_tweedie) : {week:.2f}   <- the wrong, scary number")
    print("=" * 80)

    for model in ["global_tweedie", "tsb", "naive"]:
        ltf = inventory.lead_time_frame(wide, results["forecasts"][model], classes, H)
        kpi = inventory.by_quadrant(ltf, service=SERVICE)
        print(f"\nLEAD-TIME ({H}-week) PURCHASING VIEW — model = {model}")
        print(kpi.to_string())

    print("\nReading guide:")
    print("  lt_wmape       error on the quantity you actually buy (L-week total)")
    print("  lt_bias        + over-buy, - under-buy")
    print("  within_25pct   share of live SKUs whose LT forecast is within +/-25%")
    print("  fill_point     fill rate stocking only the point forecast")
    print(f"  fill_ss        fill rate after {int(SERVICE*100)}% safety stock from the spread")
    print("  overstock_ss   excess units carried to reach that service")


if __name__ == "__main__":
    main()
