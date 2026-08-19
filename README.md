# CueCharge

**Smart Battery Dispatch for Small-Scale Solar Operators**

CueCharge is a research MVP testing one question:

> Can forecast-informed battery dispatch create meaningful incremental savings for a small commercial customer that already owns a battery?

## Current MVP

CueCharge compares three cases on a representative San Francisco commercial building profile:

1. **Unmanaged** — solar + building load with no battery dispatch.
2. **Fixed Rule** — a simple battery controller that charges off-peak and discharges during the 16:00–21:00 peak window.
3. **CueCharge** — a linear-programming dispatch optimizer using a load forecast, tariff prices, battery constraints, and demand-peak carryover.

The important metric is **incremental value**: Fixed Rule bill minus CueCharge bill.

## Data & assumptions

- Building demand source: compact hourly commercial building profiles committed to this repository.
- Sub-hourly demand: four synthetic 15-minute intervals are reconstructed from each hourly value. These are **not real 15-minute meter readings**.
- Solar: NREL PVWatts TMY3 simulation for San Francisco. TMY3 represents a typical meteorological year; it is not measured site weather.
- Forecasting: Random Forest using recent daily/weekly load history. Solar uses a simple yesterday-profile persistence forecast.
- Battery: configurable power and energy capacity with 93% round-trip-style efficiency assumptions and SOC constraints.
- Tariff: simplified commercial tariff model for backtesting. It is **not an official utility bill calculation**.

## What this does not claim yet

CueCharge is not a live battery controller and does not currently integrate with battery hardware. VPP/aggregation revenue is excluded from the headline result. Results are historical simulations and should not be interpreted as realized customer savings.

## Run locally

```bash
pip install streamlit pandas numpy scipy scikit-learn plotly requests
streamlit run app.py
```

If using PVWatts, provide an `NREL_API_KEY` through Streamlit secrets or an environment variable. Never commit the key to Git.
