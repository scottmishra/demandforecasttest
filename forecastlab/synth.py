"""Synthetic demand panel for a fictional industrial PVF / MRO distributor.

Why synthetic: the real pilot data is confidential, and we only need data that
*reproduces the structural features* that make MRO demand hard — long runs of
zeros, occasional spikes, obsolescence, cold-start parts, and (crucially)
cross-SKU structure tied to part attributes. That last property is what lets a
global model "borrow strength"; if attributes carried no signal, a global model
could not beat a well-tuned local one and the hypothesis would be untestable.

The generator deliberately does NOT bake in a Tweedie likelihood. Demand
arrivals are Bernoulli; sizes are a mixture of geometric / negative-binomial /
lognormal-rounded draws depending on the part. The Tweedie objective therefore
has to *earn* its win on data it was not handed.

Grain: one row per (sku, week). Default horizon ~3 years of weekly buckets.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Syntetos-Boylan archetypes we want represented in the catalogue. The cut
# points used by the classifier are ADI = 1.32 and CV^2 = 0.49; we aim each
# archetype to land in its quadrant on average (with natural spread).
ARCHETYPES = ("smooth", "erratic", "intermittent", "lumpy")

EQUIPMENT_TYPES = (
    "valve",
    "flange",
    "fitting",
    "pipe",
    "pump_part",
    "instrument",
    "gasket",
    "fastener",
)

# Branches stand in for regional distribution centres. Branch carries a demand
# multiplier so the global model can learn location effects.
BRANCHES = ("gulf_coast", "permian", "bakken", "midcon", "northeast")


@dataclass
class SynthConfig:
    n_skus: int = 320
    n_weeks: int = 156  # ~3 years of weekly buckets
    seed: int = 7
    # Fraction of parts that go obsolete (demand probability decays to ~0).
    obsolete_frac: float = 0.12
    # Fraction of parts introduced partway through the window (cold-start).
    coldstart_frac: float = 0.10
    # Annual seasonal amplitude applied to smooth/erratic occurrence rates.
    seasonal_amp: float = 0.35


def _archetype_params(rng: np.random.Generator, archetype: str) -> dict:
    """Latent generative parameters for one part, given its archetype.

    Returns base weekly occurrence probability ``p`` (controls ADI) and a size
    distribution spec (controls CV^2 of nonzero demand).
    """
    if archetype == "smooth":
        # Frequent demand, low size dispersion -> low ADI, low CV^2.
        p = rng.uniform(0.75, 0.98)
        size_mean = rng.uniform(8, 40)
        size_cv = rng.uniform(0.15, 0.45)
    elif archetype == "erratic":
        # Frequent demand, high size dispersion -> low ADI, high CV^2.
        p = rng.uniform(0.70, 0.95)
        size_mean = rng.uniform(6, 30)
        size_cv = rng.uniform(0.9, 1.8)
    elif archetype == "intermittent":
        # Sporadic demand, consistent size -> high ADI, low CV^2.
        p = rng.uniform(0.12, 0.35)
        size_mean = rng.uniform(2, 12)
        size_cv = rng.uniform(0.2, 0.55)
    elif archetype == "lumpy":
        # Sporadic AND variable -> high ADI, high CV^2. The hard case.
        p = rng.uniform(0.08, 0.30)
        size_mean = rng.uniform(3, 20)
        size_cv = rng.uniform(1.1, 2.2)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(archetype)
    return {"p": p, "size_mean": size_mean, "size_cv": size_cv}


def _draw_size(rng: np.random.Generator, mean: float, cv: float) -> float:
    """Draw one nonzero demand size with target mean and coefficient of var.

    Uses a gamma shape/scale (continuous, skewed) rounded up to >=1 unit. Gamma
    gives us independent control of mean and CV without being Tweedie.
    """
    cv = max(cv, 1e-3)
    shape = 1.0 / (cv * cv)
    scale = mean / shape
    val = rng.gamma(shape, scale)
    return float(max(1.0, round(val)))


def _seasonal_factor(week_idx: np.ndarray, amp: float, phase: float) -> np.ndarray:
    """Multiplicative seasonal factor in roughly [1-amp, 1+amp]."""
    return 1.0 + amp * np.sin(2 * np.pi * (week_idx / 52.0) + phase)


def generate_panel(config: SynthConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate the demand panel and the SKU attribute table.

    Returns
    -------
    demand : DataFrame with columns [sku, week, date, demand]
    skus   : DataFrame with one row per sku and its static attributes plus the
             hidden generative archetype (kept for analysis / colouring only —
             models never receive ``true_archetype``).
    """
    config = config or SynthConfig()
    rng = np.random.default_rng(config.seed)

    weeks = np.arange(config.n_weeks)
    start = pd.Timestamp("2023-01-02")  # a Monday
    dates = start + pd.to_timedelta(weeks * 7, unit="D")

    sku_rows = []
    demand_rows = []

    # Branch and equipment-type demand multipliers — latent structure the
    # global model can exploit through the attribute features.
    branch_mult = {b: rng.uniform(0.6, 1.6) for b in BRANCHES}
    equip_mult = {e: rng.uniform(0.7, 1.4) for e in EQUIPMENT_TYPES}

    # Roughly balanced archetype mix, slightly weighted to the hard quadrants
    # because that is where MRO portfolios actually live.
    archetype_weights = np.array([0.22, 0.20, 0.30, 0.28])

    for i in range(config.n_skus):
        archetype = rng.choice(ARCHETYPES, p=archetype_weights)
        params = _archetype_params(rng, archetype)
        equip = rng.choice(EQUIPMENT_TYPES)
        branch = rng.choice(BRANCHES)
        family = f"{equip}_fam_{rng.integers(0, 4)}"
        # Price loosely ties to size/equipment; the global model can use it as a
        # cold-start prior signal.
        unit_price = float(
            round(rng.lognormal(mean=np.log(50) + 0.4 * (params["size_mean"] / 20), sigma=0.6), 2)
        )
        sku = f"SKU{i:04d}"

        # Effective weekly occurrence probability vector across the window.
        phase = rng.uniform(0, 2 * np.pi)
        seas = _seasonal_factor(weeks, config.seasonal_amp, phase)
        if archetype in ("intermittent", "lumpy"):
            seas = 1.0 + 0.4 * (seas - 1.0)  # muted seasonality on sporadic parts
        p_vec = np.clip(params["p"] * seas, 0.001, 0.999)

        # Obsolescence: probability decays after a random onset week.
        is_obsolete = rng.random() < config.obsolete_frac
        if is_obsolete:
            onset = rng.integers(config.n_weeks // 3, int(config.n_weeks * 0.8))
            decay = np.ones(config.n_weeks)
            tail = np.arange(config.n_weeks - onset)
            decay[onset:] = np.exp(-tail / rng.uniform(8, 25))
            p_vec = p_vec * decay

        # Cold-start: no demand possible before an introduction week.
        is_coldstart = rng.random() < config.coldstart_frac
        intro_week = 0
        if is_coldstart:
            intro_week = int(rng.integers(config.n_weeks // 2, int(config.n_weeks * 0.85)))
            p_vec = p_vec.copy()
            p_vec[:intro_week] = 0.0

        mult = branch_mult[branch] * equip_mult[equip]

        occur = rng.random(config.n_weeks) < p_vec
        demand = np.zeros(config.n_weeks)
        for w in np.nonzero(occur)[0]:
            demand[w] = _draw_size(rng, params["size_mean"] * mult, params["size_cv"])

        sku_rows.append(
            {
                "sku": sku,
                "equipment_type": equip,
                "product_family": family,
                "branch": branch,
                "unit_price": unit_price,
                "is_obsolete": is_obsolete,
                "is_coldstart": is_coldstart,
                "intro_week": intro_week,
                "true_archetype": archetype,  # analysis only — never a feature
            }
        )
        for w in range(config.n_weeks):
            demand_rows.append((sku, w, dates[w], demand[w]))

    demand_df = pd.DataFrame(demand_rows, columns=["sku", "week", "date", "demand"])
    skus_df = pd.DataFrame(sku_rows)
    return demand_df, skus_df


def to_wide(demand: pd.DataFrame) -> pd.DataFrame:
    """Pivot the long panel to a wide sku x week matrix of demand."""
    wide = demand.pivot(index="sku", columns="week", values="demand").fillna(0.0)
    wide = wide.sort_index(axis=1)
    return wide


if __name__ == "__main__":  # pragma: no cover - smoke test
    d, s = generate_panel()
    print(f"panel: {d.shape[0]:,} rows  |  {s.shape[0]} skus  |  {d['week'].nunique()} weeks")
    print(s["true_archetype"].value_counts())
    nz = d.groupby("sku")["demand"].apply(lambda x: (x > 0).mean())
    print(f"mean nonzero-week fraction: {nz.mean():.3f}")
