"""Classify -> route -> forecast for a single SKU.

This is the decision process the app makes visible: given one part's history,
which quadrant is it, which model does the playbook recommend, and what does
each candidate model say. The global Tweedie model is always offered as the
"borrow strength" option alongside the quadrant's conventional recommendation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import classification
from .models.local import LOCAL_MODELS

# The conventional local model the playbook routes each quadrant to.
QUADRANT_LOCAL_MODEL = {
    "smooth": "ses",
    "erratic": "ses",
    "intermittent": "croston",
    "lumpy": "sba",
    "no_demand": "naive",
}


def route_one(history: np.ndarray, horizon: int, train_weeks: int | None = None,
              global_model=None, skus: pd.DataFrame | None = None,
              wide: pd.DataFrame | None = None, sku: str | None = None) -> dict:
    """Produce the routing decision and candidate forecasts for one SKU.

    If a fitted ``global_model`` plus the panel context are supplied, the global
    Tweedie forecast is included as a candidate.
    """
    if train_weeks is None:
        train_weeks = len(history)
    train = np.asarray(history[:train_weeks], dtype=float)

    cls = classification.classify_one(train)
    quad = cls["quadrant"]
    local_choice = QUADRANT_LOCAL_MODEL[quad]

    candidates = {
        name: float(fn(train, horizon)[0]) for name, fn in LOCAL_MODELS.items()
    }
    if global_model is not None and wide is not None and skus is not None and sku is not None:
        gpred = global_model.predict(wide.loc[[sku]], skus[skus["sku"] == sku],
                                     origin=train_weeks)
        candidates["global_tweedie"] = float(gpred.iloc[0])

    return {
        "adi": cls["adi"],
        "cv2": cls["cv2"],
        "n_events": cls["n_events"],
        "quadrant": quad,
        "recommended_text": classification.RECOMMENDED[quad],
        "recommended_local_model": local_choice,
        "candidate_rates": candidates,
    }
