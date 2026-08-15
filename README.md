# Bayesian Stochastic Volatility Modeling of S&P 500 Returns

A comparison of classical (GARCH) and Bayesian (stochastic volatility) approaches to estimating
market volatility, built around the 2008 financial crisis, with the sampler failures and fixes
documented along the way.
## Credits 
Extension Credits to LinkedIn Commenters:

* Ashraf SAID : https://www.linkedin.com/in/saidashraf/
* Peter Cotton : https://www.linkedin.com/in/petercotton/
* Steve Jewson : https://www.linkedin.com/in/steve-jewson-phd-052bb417/

## Overview

Volatility is never directly observed — only the return on a given day is. This project builds and
compares two fundamentally different ways of inferring it from S&P 500 returns:

* **The classical approach (GARCH(1,1)-t):** volatility follows a fixed mathematical rule, fit by
  maximum likelihood. It outputs a single number for each day's volatility, with a standard error
  attached — no broader sense of what else might be plausible.
* **The Bayesian approach (stochastic volatility via MCMC):** log-volatility is treated as an
  unknown, randomly-evolving hidden process. Instead of one number, sampling recovers a full
  posterior distribution over what volatility could plausibly have been on any given day.

The notebook documents the full process, including a sampler failure (a "funnel" geometry problem)
and how a non-centered reparameterization fixes it, a formal test of whether the model's fat-tailed
return assumption is actually earning its keep, and several practical questions the Bayesian
posterior can answer that a single point estimate can't (e.g. "how confident are we that volatility
was higher in October 2008 than in the smaller 2010 spike?").

Key findings:

* GARCH(1,1)-t on the full 2008–2019 sample gives α + β ≈ 1.0, indicating volatility shocks decay
  very slowly — a spike in turbulence keeps influencing the forecast for months, matching the
  observed slow recovery after the September–October 2008 crash.
* A first, "obvious" Bayesian stochastic-volatility model (log-volatility as a direct Gaussian
  random walk) samples badly: effective sample size on the step-size parameter was 41 out of 1,000
  draws, a textbook posterior-funnel problem.
* A non-centered reparameterization — sampling independent, unconstrained innovations and building
  log-volatility from them deterministically — fixes this completely: ~25x improvement in effective
  sample size, r_hat landing exactly at 1.00 across every parameter, same underlying model.
* The recovered Bayesian volatility path closely tracks GARCH's on the same 700-day crisis window,
  peaking near 4.5–4.8% daily in September–October 2008 — but adds a genuine credible interval on
  top of the point estimate, not just a line.
* Comparing t-distribution vs. Normal return innovations via PSIS-LOO cross-validation and posterior
  predictive checks: the t-distribution comes out ahead, but the difference (elpd_diff = 0.87) is
  smaller than its own uncertainty (dse = 1.97) — fat tails are a reasonable assumption at this
  sample size, not a decisively proven one.
* The posterior enables direct probability statements a point estimate can't, e.g. the probability
  that volatility on one specific day exceeded another, or a full day-by-day curve of "probability
  the market was in a high-volatility regime."

## Data

Daily S&P 500 log returns, 2008–2019 (2,906 trading days), read from
[`par-sp500-daily.csv`](par-sp500-daily.csv), included in this repo (columns: `Date`, `Close`, and
`change` — the daily log return used throughout the notebook). The window is chosen
deliberately to include the 2008 financial crisis, giving genuine volatility clustering — long calm
stretches punctuated by violent spikes — rather than the comparatively mild behavior of a shorter,
more recent sample. The computationally-heavy Bayesian models (Parts 3–6) use a 700-day sub-window
(May 2008 – February 2011) covering the crisis and its immediate aftermath; the GARCH baseline is
fit on both the full sample and the same 700-day window, so the two approaches are compared on
identical data.

## Setup

```bash
pip install pymc arviz arch pandas numpy scipy matplotlib
```

Place `par-sp500-daily.csv` alongside the notebook, then run top to bottom. Parts 3–6 (MCMC
sampling) are the computationally heavy sections — expect noticeably longer runtimes there than in
the GARCH baseline.

## Structure

* **Part 0** — motivation: why volatility estimation matters and what this project sets out to compare
* **Part 1** — data: loading full-sample returns and carving out the 700-day crisis window used later
* **Part 2** — the industry baseline: GARCH(1,1)-t fit by maximum likelihood, on both the full sample
  and the crisis window, with an explanation of what the fitted parameters mean
* **Part 3** — a first, direct attempt at Bayesian stochastic volatility (log-volatility as a
  centered Gaussian random walk); intentionally left in to show the resulting sampler failure
  (a posterior funnel) and how to diagnose it via r_hat and effective sample size
* **Part 4** — fixing the funnel with a non-centered reparameterization of the identical model, a
  before/after comparison of sampling diagnostics, and the recovered volatility path plotted against
  GARCH's for a direct comparison
* **Part 5** — testing the fat-tailed-returns assumption: refitting with Normal instead of
  Student-t innovations and comparing the two via PSIS-LOO cross-validation and posterior
  predictive checks
* **Part 6** — practical questions the posterior can answer beyond a single volatility path: pairwise
  day-vs-day probability comparisons, a full time series of "probability of a high-volatility
  regime," a Value-at-Risk calculation from the posterior, and a clarification of what the model is
  and isn't doing (in-sample smoothing vs. genuine out-of-sample forecasting)

## Limitations

* Both Bayesian models are fit on a single 700-day window; conclusions about the fat-tails
  comparison and the funnel diagnosis haven't been re-checked against a separate historical period.
* The stochastic-volatility model assumes a zero mean return (all signal is attributed to
  volatility, none to drift), which is standard for short-horizon volatility work but not
  universally appropriate.
* The credible intervals shown are largely **in-sample smoothing**, not out-of-sample forecasts —
  the model is anchored by the actual observed return on each day when estimating that day's
  volatility. The notebook is explicit about this distinction, since it's easy to mistake a tight
  smoothed interval for forecasting accuracy.
* MCMC sampling is considerably more expensive than the GARCH maximum-likelihood fit, which limits
  how easily this approach scales to longer histories or larger universes of assets without further
  optimization.
