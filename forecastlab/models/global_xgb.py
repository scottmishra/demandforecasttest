"""Global gradient-boosted forecaster — the hypothesis under test.

Design choices that matter:

* **One model, all SKUs.** Features are shared across the portfolio so the model
  borrows strength: a cold-start valve inherits the demand shape of other valves
  in its branch and price band through the static attribute features.

* **Direct lead-time-rate target.** For each (sku, origin) we predict the *mean
  weekly demand over the next H weeks*. This is the same quantity the local
  Croston-family methods output (a flat rate feeding min/max), so the comparison
  is apples-to-apples, and it sidesteps the brittle recursive feedback of
  multi-step lag models on series full of zeros.

* **Tweedie objective.** ``reg:tweedie`` with ``1 < power < 2`` models a
  compound-Poisson-Gamma response — a point mass at zero plus a skewed positive
  part — which is the shape of intermittent demand. We also expose a
  squared-error variant so the experiment can isolate the contribution of the
  objective from the contribution of going global.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

# Static attribute columns fed to the model. Note we deliberately exclude the
# hidden generative labels (true_archetype, is_obsolete, is_coldstart): a real
# distributor would not have them. "age" (weeks of history) is a fair proxy that
# captures cold-start without leaking the synthetic flag.
STATIC_CATEGORICAL = ["equipment_type", "product_family", "branch"]
STATIC_NUMERIC = ["unit_price"]


def series_features(hist: np.ndarray, origin: int) -> dict:
    """Features describing one series at a forecast origin.

    ``hist`` is the full demand vector; only ``hist[:origin]`` is used so no
    future information leaks in.
    """
    h = np.asarray(hist[:origin], dtype=float)
    n = len(h)
    feat: dict[str, float] = {}

    # Recent lags.
    for lag in (1, 2, 3, 4):
        feat[f"lag_{lag}"] = float(h[-lag]) if n >= lag else 0.0

    # Rolling means / std over several windows (demand-rate at multiple scales).
    for w in (4, 8, 13, 26, 52):
        win = h[-w:] if n >= 1 else np.array([0.0])
        feat[f"roll_mean_{w}"] = float(win.mean()) if win.size else 0.0
    for w in (8, 26):
        win = h[-w:] if n >= 1 else np.array([0.0])
        feat[f"roll_std_{w}"] = float(win.std()) if win.size else 0.0

    # Intermittency descriptors.
    for w in (13, 26, 52):
        win = h[-w:] if n >= 1 else np.array([0.0])
        feat[f"nz_frac_{w}"] = float((win > 0).mean()) if win.size else 0.0
    nz = h[h > 0]
    feat["mean_nz_size"] = float(nz.mean()) if nz.size else 0.0
    feat["max_recent"] = float(h[-26:].max()) if n else 0.0

    # Weeks since last sale — repeatedly cited as the single most predictive
    # feature for intermittent demand.
    if nz.size:
        last_sale = np.nonzero(h > 0)[0][-1]
        feat["weeks_since_sale"] = float(n - 1 - last_sale)
    else:
        feat["weeks_since_sale"] = float(n)

    # Maturity / cold-start proxy.
    feat["age"] = float(n)
    feat["total_events"] = float(nz.size)

    # Seasonal position of the origin.
    feat["sin_woy"] = float(np.sin(2 * np.pi * (origin % 52) / 52.0))
    feat["cos_woy"] = float(np.cos(2 * np.pi * (origin % 52) / 52.0))
    return feat


@dataclass
class GlobalXGB:
    horizon: int
    objective: str = "tweedie"  # "tweedie" | "squared"
    tweedie_power: float = 1.3
    n_estimators: int = 400
    learning_rate: float = 0.05
    max_depth: int = 6
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_origin_gap: int = 4  # spacing between training origins (weeks)
    seed: int = 0

    def _xgb_params(self) -> dict:
        if self.objective == "tweedie":
            obj = {"objective": "reg:tweedie", "tweedie_variance_power": self.tweedie_power}
        elif self.objective == "squared":
            obj = {"objective": "reg:squarederror"}
        elif self.objective == "poisson":
            obj = {"objective": "count:poisson"}
        else:  # pragma: no cover
            raise ValueError(self.objective)
        return {
            **obj,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "tree_method": "hist",
            "enable_categorical": True,
            "random_state": self.seed,
            "n_jobs": 0,
        }

    def _build_rows(self, wide: pd.DataFrame, skus: pd.DataFrame, train_weeks: int,
                    origins: list[int]) -> pd.DataFrame:
        """Assemble a training/inference frame at the given origins."""
        attr = skus.set_index("sku")
        rows = []
        for sku, hist in zip(wide.index, wide.values):
            a = attr.loc[sku]
            for origin in origins:
                feat = series_features(hist, origin)
                for c in STATIC_CATEGORICAL:
                    feat[c] = a[c]
                for c in STATIC_NUMERIC:
                    feat[c] = float(a[c])
                feat["sku"] = sku
                feat["origin"] = origin
                # Target only defined when the next H weeks are inside train data.
                end = origin + self.horizon
                if end <= train_weeks:
                    feat["target"] = float(hist[origin:end].mean())
                else:
                    feat["target"] = np.nan
                rows.append(feat)
        df = pd.DataFrame(rows)
        for c in STATIC_CATEGORICAL:
            df[c] = df[c].astype("category")
        return df

    def fit(self, wide: pd.DataFrame, skus: pd.DataFrame, train_weeks: int) -> "GlobalXGB":
        # Training origins: a rolling set ending one horizon before the cutoff,
        # spaced to decorrelate consecutive windows. More origins => more rows.
        first = 26  # require some history before the earliest origin
        last = train_weeks - self.horizon
        origins = list(range(first, last + 1, self.min_origin_gap))
        if not origins:
            origins = [max(first, last)]
        df = self._build_rows(wide, skus, train_weeks, origins)
        df = df.dropna(subset=["target"])
        self._feature_cols = [
            c for c in df.columns if c not in ("sku", "origin", "target")
        ]
        X = df[self._feature_cols]
        y = df["target"].values
        self.model_ = xgb.XGBRegressor(**self._xgb_params())
        self.model_.fit(X, y)
        return self

    def predict(self, wide: pd.DataFrame, skus: pd.DataFrame, origin: int) -> pd.Series:
        """Predict the per-period demand rate at a single origin, per SKU."""
        df = self._build_rows(wide, skus, train_weeks=origin, origins=[origin])
        preds = self.model_.predict(df[self._feature_cols])
        preds = np.clip(preds, 0.0, None)
        return pd.Series(preds, index=df["sku"].values, name="forecast")

    def feature_importance(self) -> pd.Series:
        imp = self.model_.feature_importances_
        return pd.Series(imp, index=self._feature_cols).sort_values(ascending=False)
