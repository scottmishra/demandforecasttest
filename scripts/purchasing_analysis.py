"""Is the forecast good enough to *buy* against?

Per-period WMAPE is the wrong question for an intermittent SKU: you cannot
predict which week a spike lands, so weekly point error is irreducibly large
even for a perfect model. Purchasing does not care about that. It cares about:

  1. Lead-time demand   — total demand over the reorder protection interval
                          (here, the 13-week holdout). This is what a min/max or
                          order-up-to level is sized against.
  2. Bias               — systematic over/under buying compounds into dead stock
                          or chronic stockouts.
  3. The distribution   — safety stock needs a spread, not just a mean. We
                          simulate the service level you actually achieve.

This script reuses the same backtest and reports all three, overall and by
Syntetos-Boylan quadrant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecastlab import synth, backtest

H = 13          # holdout / protection interval (weeks)
Z95 = 1.645     # normal quantile for a 95% cycle-service target


def lead_time_table(wide, results, model):
    """Per-SKU lead-time actual vs forecast totals for one model."""
    rate = results["forecasts"][model]
    tw = wide.shape[1] - H
    rows = []
    for sku, hist in zip(wide.index, wide.values):
        actual_lt = float(hist[tw:tw + H].sum())
        fc_lt = float(rate[sku] * H)
        # Lead-time demand sigma estimated from non-overlapping H-week sums in
        # the training history — the spread a planner would use for safety stock.
        train = hist[:tw]
        blocks = [train[i:i + H].sum() for i in range(0, len(train) - H + 1, H)]
        sigma_lt = float(np.std(blocks)) if len(blocks) > 1 else float(train.std() * np.sqrt(H))
        rows.append({"sku": sku, "actual_lt": actual_lt, "fc_lt": fc_lt,
                     "sigma_lt": sigma_lt,
                     "quadrant": results["classes"].loc[sku, "quadrant"]})
    return pd.DataFrame(rows)


def summarise(df, label):
    a, f = df["actual_lt"], df["fc_lt"]
    tot = a.sum()
    wmape = (a - f).abs().sum() / tot
    bias = (f - a).sum() / tot
    # Per-SKU accuracy band (only SKUs with real demand).
    live = df[a > 0]
    within25 = (np.abs(live["fc_lt"] - live["actual_lt"]) <= 0.25 * live["actual_lt"]).mean()
    within50 = (np.abs(live["fc_lt"] - live["actual_lt"]) <= 0.50 * live["actual_lt"]).mean()

    # Service-level simulation over the protection interval.
    # Policy A: order-up-to the point forecast (no safety stock).
    S0 = np.maximum(np.round(f.values), 0)
    fill0 = np.minimum(a.values, S0).sum() / tot
    stockout0 = (a.values > S0).mean()
    # Policy B: forecast + 95% safety stock from the lead-time spread.
    S1 = np.maximum(np.round(f.values + Z95 * df["sigma_lt"].values), 0)
    fill1 = np.minimum(a.values, S1).sum() / tot
    stockout1 = (a.values > S1).mean()
    overstock1 = (S1.sum() - a.sum()) / tot  # excess units carried vs demand

    return {
        "segment": label, "n": len(df),
        "LT_WMAPE": wmape, "LT_bias": bias,
        "within_25pct": within25, "within_50pct": within50,
        "fill_pointfc": fill0, "stockout_pointfc": stockout0,
        "fill_with_ss": fill1, "stockout_with_ss": stockout1,
        "overstock_with_ss": overstock1,
    }


def main():
    cfg = synth.SynthConfig(n_skus=320, n_weeks=156, seed=7)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    tw = 156 - H
    results = backtest.run_backtest(wide, skus, tw, H)

    overall = results["overall"].set_index("model")
    print("=" * 78)
    print("PER-PERIOD vs LEAD-TIME accuracy  (model = global_tweedie)")
    print("=" * 78)
    print(f"  per-week WMAPE (headline) : {overall.loc['global_tweedie','wmape']:.2f}"
          "   <- looks terrible, but it's the wrong metric")

    for model in ["global_tweedie", "tsb", "naive"]:
        df = lead_time_table(wide, results, model)
        rows = [summarise(df, "ALL")]
        for q in ["smooth", "erratic", "intermittent", "lumpy"]:
            sub = df[df["quadrant"] == q]
            if len(sub):
                rows.append(summarise(sub, q))
        out = pd.DataFrame(rows).set_index("segment")
        pd.set_option("display.width", 160)
        pd.set_option("display.float_format", lambda v: f"{v:.2f}")
        print("\n" + "=" * 78)
        print(f"LEAD-TIME ({H}-week) PURCHASING VIEW — model = {model}")
        print("=" * 78)
        print(out.to_string())

    print("\nReading guide:")
    print("  LT_WMAPE        error on the quantity you actually buy (13-wk total)")
    print("  LT_bias         + = systematic over-buy, - = under-buy")
    print("  within_25pct    share of live SKUs whose LT forecast is within +/-25%")
    print("  fill_pointfc    fill rate if you stock only the point forecast")
    print("  fill_with_ss    fill rate after adding 95% safety stock from the spread")
    print("  overstock_with_ss  excess units carried (inventory cost of that service)")


if __name__ == "__main__":
    main()
