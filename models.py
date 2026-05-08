"""
models.py — risk scoring, trading signals, counterfactual simulation.

All logic is explainable rule-based + simple stats (no black-box ML).
This makes it easy to explain to a JPM mentor line by line.
"""

import pandas as pd
import numpy as np


# ─── Spike Probability Score ──────────────────────────────────────────────────

def spike_probability(temp_f, wind_mph, gas_price_usd, hour, lmp_zscore):
    """
    Estimate probability of a price spike in the next 12 hours.
    Returns float in [0, 1] and a dict of factor contributions.

    Logic is transparent and explainable — each factor has a clear
    real-world justification rooted in electricity market mechanics.

    Sources of price spikes:
      1. Extreme heat → AC demand surge
      2. Low wind    → less cheap renewable generation, more gas dispatch
      3. High gas    → higher marginal cost for gas peakers (sets LMP)
      4. Peak hours  → morning/evening demand peaks
      5. Already elevated → momentum / mean-reversion lag
    """
    factors = {}

    # 1. Temperature: extreme heat (>95°F) or cold (<35°F) drives demand
    if temp_f > 95:
        factors["heat_stress"] = min((temp_f - 95) / 20, 1.0) * 0.30
    elif temp_f < 35:
        factors["cold_stress"] = min((35 - temp_f) / 20, 1.0) * 0.20
    else:
        factors["heat_stress"] = 0.0

    # 2. Low wind → less renewable displacement of gas peakers
    factors["low_wind"] = max(0, (15 - wind_mph) / 15) * 0.20

    # 3. Gas price: Henry Hub > $3.50 means peaker costs are high
    factors["gas_cost"] = min(max(gas_price_usd - 2.0, 0) / 4.0, 1.0) * 0.25

    # 4. Peak demand hours (7-10am, 5-9pm)
    is_peak = (7 <= hour <= 10) or (17 <= hour <= 21)
    factors["peak_hour"] = 0.15 if is_peak else 0.0

    # 5. Already elevated (z-score momentum)
    factors["price_momentum"] = min(max(lmp_zscore / 3.0, 0), 1.0) * 0.10

    prob = sum(factors.values())
    prob = float(np.clip(prob, 0.0, 0.97))
    return prob, factors


def trading_signal(prob, current_lmp, roll24_mean):
    """
    Return a trading/operational signal based on spike probability.

    Signal levels:
      BUY EARLY  — lock in supply now before price rises
      HEDGE      — use futures/options to cap price exposure
      HOLD       — monitor, no action needed
    """
    if prob >= 0.60:
        signal = "🔴  HEDGE NOW"
        rationale = f"Spike probability {prob:.0%} is high. Consider forward contracts or demand reduction."
    elif prob >= 0.35:
        signal = "🟡  BUY EARLY"
        rationale = f"Spike probability {prob:.0%} is elevated. Procuring early is likely cheaper than spot."
    else:
        signal = "🟢  HOLD"
        rationale = f"Spike probability {prob:.0%} is low. Current LMP (${current_lmp:.2f}/MWh) near 24hr avg (${roll24_mean:.2f}/MWh)."
    return signal, rationale


# ─── Counterfactual Simulation ────────────────────────────────────────────────

def simulate_temp_shock(base_temp_f, delta_f, wind_mph, gas_price, hour, lmp_zscore):
    """
    'What if temperature rises by delta_f degrees?'
    Returns dict comparing base vs shocked spike probability.
    """
    base_prob, _ = spike_probability(base_temp_f, wind_mph, gas_price, hour, lmp_zscore)
    shock_prob, _ = spike_probability(base_temp_f + delta_f, wind_mph, gas_price, hour, lmp_zscore)
    return {
        "base_temp": base_temp_f,
        "shocked_temp": base_temp_f + delta_f,
        "base_prob": base_prob,
        "shock_prob": shock_prob,
        "delta_prob": shock_prob - base_prob,
    }


def simulate_gas_shock(temp_f, wind_mph, base_gas, delta_gas, hour, lmp_zscore):
    """
    'What if gas prices rise by delta_gas $/MMBtu?'
    """
    base_prob, _ = spike_probability(temp_f, wind_mph, base_gas, hour, lmp_zscore)
    shock_prob, _ = spike_probability(temp_f, wind_mph, base_gas + delta_gas, hour, lmp_zscore)
    return {
        "base_gas": base_gas,
        "shocked_gas": base_gas + delta_gas,
        "base_prob": base_prob,
        "shock_prob": shock_prob,
        "delta_prob": shock_prob - base_prob,
    }


# ─── Expected Value of Hedging ────────────────────────────────────────────────

def hedge_expected_value(prob_spike, current_lmp, spike_magnitude_mwh, hedge_cost_mwh, volume_mw):
    """
    Simple expected value calculation for hedging decision.

    EV(hedge)   = -hedge_cost * volume  (certain cost)
    EV(no hedge) = -prob_spike * spike_magnitude * volume  (expected loss)
    Net benefit of hedging = EV(no hedge) - EV(hedge)

    Returns dict with all components, positive net = hedge is worth it.
    """
    ev_no_hedge = -prob_spike * spike_magnitude_mwh * volume_mw
    ev_hedge = -hedge_cost_mwh * volume_mw
    net_benefit = ev_no_hedge - ev_hedge  # how much worse no-hedge is
    return {
        "prob_spike": prob_spike,
        "ev_no_hedge_usd": ev_no_hedge,
        "ev_hedge_usd": ev_hedge,
        "net_benefit_of_hedging_usd": -net_benefit,  # positive = hedge recommended
        "recommendation": "HEDGE" if net_benefit < 0 else "PASS",
    }


# ─── Volatility Regime ────────────────────────────────────────────────────────

def classify_vol_regime(volatility_ratio):
    """
    Classify current price volatility regime.
    volatility_ratio = rolling std / rolling mean (coefficient of variation)
    """
    if volatility_ratio > 0.5:
        return "HIGH VOL", "🔴"
    elif volatility_ratio > 0.25:
        return "ELEVATED", "🟡"
    else:
        return "NORMAL", "🟢"


# ─── Day-Ahead vs Real-Time Spread ───────────────────────────────────────────

def da_rt_spread_signal(merged_df):
    """
    Compute rolling day-ahead vs real-time spread proxy.
    In real markets, DA price = forecast price, RT = actual clearing price.
    Here we approximate: DA proxy = 24h forward rolling mean.
    Large positive spread (RT >> DA) = unexpected supply tightness.
    """
    df = merged_df.copy()
    df["da_proxy"] = df["lmp"].shift(24)  # yesterday same hour as DA proxy
    df["da_rt_spread"] = df["lmp"] - df["da_proxy"]
    df["spread_zscore"] = (
        (df["da_rt_spread"] - df["da_rt_spread"].rolling(168).mean())
        / (df["da_rt_spread"].rolling(168).std() + 1e-9)
    )
    return df[["datetime", "lmp", "da_proxy", "da_rt_spread", "spread_zscore"]].dropna()
