"""Headless experiment runner — validates the Tweedie-global hypothesis.

Generates the synthetic panel, runs the backtest, and prints the leaderboard
overall and by quadrant. Also writes artefacts to ``artifacts/`` so the
Streamlit app can load precomputed results instead of recomputing on launch.

Usage:
    python -m scripts.run_experiment            # default config
    python -m scripts.run_experiment --weeks 156 --horizon 13 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from forecastlab import synth, backtest

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "artifacts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-skus", type=int, default=320)
    ap.add_argument("--weeks", type=int, default=156)
    ap.add_argument("--horizon", type=int, default=13)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--save", action="store_true", help="write artefacts for the app")
    args = ap.parse_args()

    cfg = synth.SynthConfig(n_skus=args.n_skus, n_weeks=args.weeks, seed=args.seed)
    demand, skus = synth.generate_panel(cfg)
    wide = synth.to_wide(demand)
    train_weeks = args.weeks - args.horizon

    print(f"panel: {skus.shape[0]} skus x {args.weeks} weeks  "
          f"(train {train_weeks}, holdout {args.horizon})")
    print("archetype mix:")
    print(skus["true_archetype"].value_counts().to_string())
    print()

    results = backtest.run_backtest(wide, skus, train_weeks, args.horizon)

    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")

    print("=" * 64)
    print("OVERALL LEADERBOARD (lower WMAPE / MASE / RMSSE is better)")
    print("=" * 64)
    print(results["overall"].to_string(index=False))
    print()

    print("=" * 64)
    print("BY QUADRANT — WMAPE (the hypothesis lives in intermittent/lumpy)")
    print("=" * 64)
    pivot = results["by_quadrant"].pivot(index="model", columns="quadrant", values="wmape")
    print(pivot.to_string())
    print()

    # Headline verdict.
    overall = results["overall"].set_index("model")["wmape"]
    best = overall.idxmin()
    tw = overall.get("global_tweedie", np.nan)
    sq = overall.get("global_squared", np.nan)
    best_local = overall.drop([c for c in overall.index if c.startswith("global_")]).idxmin()
    print("VERDICT")
    print(f"  best overall model      : {best}  (WMAPE {overall[best]:.3f})")
    print(f"  global tweedie WMAPE    : {tw:.3f}")
    print(f"  global squared WMAPE    : {sq:.3f}   <- Tweedie vs squared ablation")
    print(f"  best local model        : {best_local}  (WMAPE {overall[best_local]:.3f})")
    if np.isfinite(tw) and np.isfinite(overall[best_local]):
        lift = 100 * (overall[best_local] - tw) / overall[best_local]
        print(f"  global-tweedie lift vs best local: {lift:.1f}% WMAPE")

    if args.save:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        demand.to_parquet(os.path.join(ARTIFACT_DIR, "demand.parquet"))
        skus.to_parquet(os.path.join(ARTIFACT_DIR, "skus.parquet"))
        results["overall"].to_csv(os.path.join(ARTIFACT_DIR, "overall.csv"), index=False)
        results["by_quadrant"].to_csv(os.path.join(ARTIFACT_DIR, "by_quadrant.csv"), index=False)
        results["per_sku"].to_csv(os.path.join(ARTIFACT_DIR, "per_sku.csv"), index=False)
        results["classes"].to_csv(os.path.join(ARTIFACT_DIR, "classes.csv"))
        with open(os.path.join(ARTIFACT_DIR, "meta.json"), "w") as f:
            json.dump(vars(args), f, indent=2)
        print(f"\nartefacts written to {os.path.abspath(ARTIFACT_DIR)}")


if __name__ == "__main__":
    main()
