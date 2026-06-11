# Intermittent-Demand Forecasting Lab

A small, self-contained lab for **validating one hypothesis**:

> For an industrial PVF / MRO parts distributor, a **single global**
> gradient-boosted model trained across the whole SKU portfolio with a
> **Tweedie** objective beats the conventional approach of fitting **one local
> model per SKU** (Croston / SBA / TSB / exponential smoothing) — especially on
> intermittent and lumpy demand.

The SKU *is* the natural grain here (a distributor sells parts directly — there
is no finished product to roll a BOM up to), so the interesting question is not
aggregation, it is **intermittency**: long runs of zeros punctuated by spikes,
plus obsolescence and cold-start parts. All data in this repo is **synthetic**;
nothing is tied to a real customer or dataset.

## The result

Running the default experiment (320 SKUs × 156 weekly buckets, 13-week holdout):

```
OVERALL LEADERBOARD (lower WMAPE is better)
         model    wmape   mase  rmsse  bias%
global_tweedie    0.738  1.094  0.708  -0.05   <- best single default
           tsb    0.795  1.109  0.724  +0.00   <- best local method
global_squared    0.797  1.231  0.734  +0.05   <- same features, no Tweedie
           ses    0.797  1.114  0.736  -0.03
    moving_avg    0.801  1.112  0.740  -0.02
           sba    0.837  1.117  0.752  +0.00
       croston    0.846  1.128  0.753  +0.06
         naive    0.946  1.229  0.874  -0.05
```

Two things matter:

1. **Global Tweedie is the best single default**, ~7% lower WMAPE than the best
   local method — consistent with the "3–8 WMAPE points" rule of thumb for
   borrowing strength across series.
2. The **ablation** (`global_squared`: identical global features, squared-error
   objective) is materially worse than `global_tweedie`. So the win is not just
   "go global" — the **Tweedie objective itself** is pulling weight, because it
   models the point-mass-at-zero-plus-skewed-positive shape of intermittent
   demand directly.

Numbers move with seed / horizon / panel size, but the ordering is stable.

## The process the app makes visible

1. **Classify** every SKU on the Syntetos-Boylan plane — ADI (how sporadic) vs
   CV² (how variable the spike sizes) — with the standard 1.32 / 0.49 cut points.
2. **Route**: smooth/erratic → conventional smoothing is fine; intermittent/lumpy
   → Croston-family *or* the global model. TSB for obsolescence-prone parts
   because it decays toward zero during zero-demand runs (Croston/SBA do not).
3. **Forecast & compare**: backtest all methods on a held-out window, scored
   overall and by quadrant.

## Layout

```
forecastlab/
  synth.py            synthetic MRO demand panel (zeros, spikes, obsolescence,
                      cold-start, cross-SKU attribute signal — but NOT a Tweedie
                      likelihood, so the objective has to earn its win)
  classification.py   ADI / CV² → Syntetos-Boylan quadrant + routing playbook
  models/
    local.py          naive, moving avg, SES, Croston, SBA, TSB (from scratch)
    global_xgb.py     global XGBoost; reg:tweedie vs reg:squarederror ablation
  backtest.py         holdout backtest + WMAPE / MASE / RMSSE / Bias, by quadrant
  routing.py          classify → recommend → forecast for a single SKU
app.py                Streamlit dashboard (the 4-step process, interactive)
scripts/run_experiment.py   headless runner + artefact dump
tests/                end-to-end smoke + sanity tests
```

## Run it

```bash
pip install -r requirements.txt

# headless validation (prints the leaderboard + verdict)
python -m scripts.run_experiment

# interactive app
streamlit run app.py

# tests
python -m pytest -q
```

## Modelling notes

- **Direct lead-time-rate target.** The global model predicts the *mean weekly
  demand over the next H weeks* — the same flat rate the Croston-family methods
  output for a min/max calculation — so the comparison is apples-to-apples and
  avoids the brittle recursive feedback of multi-step lag models on zero-heavy
  series.
- **Features that carry the signal.** Lags and multi-window rolling demand rates,
  `weeks_since_sale` (repeatedly the most predictive single feature for
  intermittent demand), nonzero-fraction and mean-nonzero-size (demand
  probability vs size), an `age` proxy for cold-start, and static attributes
  (equipment type, family, branch, price) that let cold-start SKUs inherit a
  prior from similar parts. The hidden generative labels are never fed to the
  model.
- **Next step for a real pilot.** The downstream consumer is min/max and safety
  stock, so production output should be the *lead-time demand distribution* per
  SKU, not a point forecast — e.g. XGBoost quantile regression or sampling from
  the fitted Tweedie. This lab validates the point-accuracy claim first; the
  probabilistic layer sits on top of the same global model.
