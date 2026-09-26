# Retail Forecasting Metrics — Interpretation Notes

These notes exist so the Insight Agent can explain whether a number is
good or bad, not just repeat it. Retrieved when a question needs
interpretive context (e.g. "is this a good MAPE?") rather than a fresh
number pulled from the database.

## MAPE (Mean Absolute Percentage Error)
Measures average forecast error as a percentage of actual sales. Lower is
better. As a rough field benchmark for retail daily sales forecasting:
- Under 10% is considered strong.
- 10-20% is acceptable/typical for noisy retail demand.
- Above 20% suggests the model is missing significant structure (promos,
  holidays, store-specific effects) in the data.

The best result in this project — Prophet + walk-forward + promo,
MAPE 0.078 (7.8%) — falls in the "strong" range.

## Why Promo Matters as a Feature
Promotional periods (promo = 1) tend to significantly shift daily sales
volume relative to non-promo days at the same store. A model that ignores
promo as an input will systematically under- or over-predict on
promotional days, since it has no signal to separate "high demand because
of the day of week / season" from "high demand because of an active
promotion." This is why promo was included as a regressor in the
best-performing model.

## Store Type and Assortment
Rossmann stores are categorized by store_type (a, c, d in this cleaned
dataset — type 'b' was excluded) and assortment (product range offered).
These are structural differences between stores, not something that
changes day to day, so they are most useful for segmenting a question
("how does type a compare to type d") rather than as a time-varying
signal.

## Rolling Average as a Feature vs. as a Naive Baseline
A 7-day rolling average of past sales (excluding the current day) is a
reasonable naive forecasting baseline on its own, and is also useful as
an input feature to a more sophisticated model. The distinction matters:
using it correctly (excluding the current day) as a baseline tells you
how much lift a real model provides over "just assume tomorrow looks like
the last week." Using it incorrectly (including the current day, as in
the leakage bug documented above) makes any baseline look artificially
strong and invalidates the comparison.