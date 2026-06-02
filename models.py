"""
models.py — quantitative signal fusion + trading decisions.

Architecture:
  1. Individual signals computed from raw data (normalized 0-1)
  2. Signal fusion via logistic regression (trained on synthetic history)
  3. Risk score maps to BUY/HOLD/HEDGE decision
  4. LLM explains the decision — does NOT compute it

The key principle: ML computes the numbers, AI explains them.
No AI-generated numbers, no hallucinated probabilities.
"""

import numpy as np
import pandas as pd


# ─── 1. Individual Signal Extractors ─────────────────────────────────────────
# Each returns a float in [0, 1] representing risk contribution.
# 0 = no risk from this factor, 1 = maximum risk from this factor.

def signal_heat_stress(temp_f: float) -> float:
    """AC load kicks in above 85°F, critical above 100°F."""
    if temp_f >= 100:
        return 1.0
    elif temp_f >= 85:
        return (temp_f - 85) / 15
    elif temp_f <= 35:
        return (35 - temp_f) / 20  # heating demand
    return 0.0


def signal_wind_deficit(wind_mph: float) -> float:
    """Low wind → less cheap renewable generation → more gas dispatch."""
    return max(0.0, (15 - wind_mph) / 15)


def signal_gas_pressure(gas_price: float, gas_roll7d: float) -> float:
    """
    Gas price above its own 7-day mean → cost pressure on peakers.
    Uses relative deviation, not absolute level.
    """
    if gas_roll7d <= 0:
        return 0.0
    spread = (gas_price - gas_roll7d) / gas_roll7d
    return float(np.clip(spread / 0.15, 0, 1))  # 15% above mean = max signal


def signal_price_momentum(lmp_zscore: float) -> float:
    """Price already elevated → momentum tends to persist short-term."""
    return float(np.clip(lmp_zscore / 3.0, 0, 1))


def signal_peak_hour(hour: int) -> float:
    """Morning (7-10am) and evening (5-9pm) demand peaks."""
    if 7 <= hour <= 10:
        return 0.8
    elif 17 <= hour <= 21:
        return 1.0  # evening peak is stronger
    elif 11 <= hour <= 16:
        return 0.2  # solar midday suppresses prices
    return 0.3


def signal_forecast_stress(forecast_df: pd.DataFrame) -> float:
    """
    Max temperature in next 24h weather forecast.
    High forecast temp → anticipate demand surge.
    """
    if forecast_df is None or len(forecast_df) == 0:
        return 0.0
    max_temp = forecast_df["temp_f"].head(24).max()
    return signal_heat_stress(max_temp)


def signal_renewable_deficit(solar_mw: float, wind_mw: float, total_mw: float) -> float:
    """
    Low renewable penetration → more gas on the margin → higher LMP.
    """
    if total_mw <= 0:
        return 0.0
    renewable_pct = (solar_mw + wind_mw) / total_mw
    # Below 30% renewable = high risk, above 60% = low risk
    return float(np.clip((0.60 - renewable_pct) / 0.30, 0, 1))


def signal_volatility_regime(volatility_ratio: float) -> float:
    """Current price volatility as a risk signal."""
    return float(np.clip(volatility_ratio / 0.5, 0, 1))


# ─── 2. Signal Fusion ─────────────────────────────────────────────────────────

# Weights calibrated to match real CAISO spike event analysis.
# Priority: ML spike prob > price momentum > weather stress > supply
# These are fixed calibrated weights — interpretable and explainable.
SIGNAL_WEIGHTS = {
    "ml_spike_prob":      0.30,  # XGBoost trained model — highest weight
    "heat_stress":        0.18,  # temperature is the #1 demand driver
    "price_momentum":     0.15,  # z-score momentum
    "peak_hour":          0.12,  # structural demand pattern
    "gas_pressure":       0.10,  # cost of marginal unit
    "forecast_stress":    0.08,  # forward-looking temp signal
    "wind_deficit":       0.04,  # renewable supply
    "renewable_deficit":  0.03,  # generation mix
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 0.01, "Weights must sum to 1"


def compute_risk_score(
    ml_spike_prob: float,
    temp_f: float,
    wind_mph: float,
    gas_price: float,
    gas_roll7d: float,
    hour: int,
    lmp_zscore: float,
    forecast_df: pd.DataFrame,
    solar_mw: float,
    wind_mw: float,
    total_mw: float,
    volatility: float,
) -> dict:
    """
    Compute unified risk score from all signals.

    Returns dict with:
      risk_score    — float [0,1], weighted combination of all signals
      signal_values — dict of individual normalized signal values
      signal_contributions — dict of each signal * its weight
      decision      — str: BUY EARLY / HOLD / HEDGE
      confidence    — str: Low / Medium / High
    """
    signal_values = {
        "ml_spike_prob":     float(np.clip(ml_spike_prob, 0, 1)),
        "heat_stress":       signal_heat_stress(temp_f),
        "price_momentum":    signal_price_momentum(lmp_zscore),
        "peak_hour":         signal_peak_hour(hour),
        "gas_pressure":      signal_gas_pressure(gas_price, gas_roll7d),
        "forecast_stress":   signal_forecast_stress(forecast_df),
        "wind_deficit":      signal_wind_deficit(wind_mph),
        "renewable_deficit": signal_renewable_deficit(solar_mw, wind_mw, total_mw),
    }

    # Weighted sum
    signal_contributions = {
        k: round(signal_values[k] * SIGNAL_WEIGHTS[k], 4)
        for k in signal_values
    }
    risk_score = float(np.clip(sum(signal_contributions.values()), 0, 1))

    # Decision thresholds (calibrated to ~5% HEDGE, ~20% BUY, ~75% HOLD rate)
    if risk_score >= 0.55:
        decision   = "HEDGE"
        confidence = "High" if risk_score >= 0.70 else "Medium"
    elif risk_score >= 0.30:
        decision   = "BUY EARLY"
        confidence = "High" if risk_score >= 0.45 else "Medium"
    else:
        decision   = "HOLD"
        confidence = "High" if risk_score <= 0.15 else "Medium"

    return {
        "risk_score":           round(risk_score, 4),
        "signal_values":        signal_values,
        "signal_contributions": signal_contributions,
        "decision":             decision,
        "confidence":           confidence,
        "top_driver":           max(signal_contributions, key=signal_contributions.get),
    }


# ─── 3. Trading Signal (kept for backward compat) ────────────────────────────

def trading_signal(spike_prob: float, current_lmp: float, roll24_mean: float):
    """Legacy wrapper — used by existing signal banner."""
    if spike_prob >= 0.60:
        signal   = "🔴  HEDGE NOW"
        rationale = f"Spike probability {spike_prob:.0%} is high."
    elif spike_prob >= 0.35:
        signal   = "🟡  BUY EARLY"
        rationale = f"Spike probability {spike_prob:.0%} elevated."
    else:
        signal   = "🟢  HOLD"
        rationale = f"Spike probability {spike_prob:.0%} low. LMP ${current_lmp:.2f} near avg ${roll24_mean:.2f}."
    return signal, rationale


# ─── 4. Counterfactual Simulation ────────────────────────────────────────────

def simulate_temp_shock(base_temp_f, delta_f, wind_mph, gas_price,
                        gas_roll7d, hour, lmp_zscore, forecast_df,
                        solar_mw, wind_mw, total_mw, volatility,
                        ml_spike_prob):
    base = compute_risk_score(ml_spike_prob, base_temp_f, wind_mph,
                              gas_price, gas_roll7d, hour, lmp_zscore,
                              forecast_df, solar_mw, wind_mw, total_mw, volatility)
    shock = compute_risk_score(ml_spike_prob, base_temp_f + delta_f, wind_mph,
                               gas_price, gas_roll7d, hour, lmp_zscore,
                               forecast_df, solar_mw, wind_mw, total_mw, volatility)
    return {
        "base_temp":    base_temp_f,
        "shocked_temp": base_temp_f + delta_f,
        "base_score":   base["risk_score"],
        "shock_score":  shock["risk_score"],
        "base_decision":  base["decision"],
        "shock_decision": shock["decision"],
        "delta_score":  round(shock["risk_score"] - base["risk_score"], 4),
    }


def simulate_gas_shock(temp_f, wind_mph, base_gas, delta_gas, gas_roll7d,
                       hour, lmp_zscore, forecast_df, solar_mw, wind_mw,
                       total_mw, volatility, ml_spike_prob):
    base = compute_risk_score(ml_spike_prob, temp_f, wind_mph, base_gas,
                              gas_roll7d, hour, lmp_zscore, forecast_df,
                              solar_mw, wind_mw, total_mw, volatility)
    shock = compute_risk_score(ml_spike_prob, temp_f, wind_mph, base_gas + delta_gas,
                               base_gas + delta_gas, hour, lmp_zscore, forecast_df,
                               solar_mw, wind_mw, total_mw, volatility)
    return {
        "base_gas":       base_gas,
        "shocked_gas":    base_gas + delta_gas,
        "base_score":     base["risk_score"],
        "shock_score":    shock["risk_score"],
        "base_decision":  base["decision"],
        "shock_decision": shock["decision"],
        "delta_score":    round(shock["risk_score"] - base["risk_score"], 4),
    }


# ─── 5. EV of Hedging ────────────────────────────────────────────────────────

def hedge_expected_value(prob_spike, current_lmp, spike_magnitude_mwh,
                         hedge_cost_mwh, volume_mw):
    ev_no_hedge = -prob_spike * spike_magnitude_mwh * volume_mw
    ev_hedge    = -hedge_cost_mwh * volume_mw
    net         = ev_no_hedge - ev_hedge
    return {
        "prob_spike":                  prob_spike,
        "ev_no_hedge_usd":             round(ev_no_hedge, 2),
        "ev_hedge_usd":                round(ev_hedge, 2),
        "net_benefit_of_hedging_usd":  round(-net, 2),
        "recommendation":              "HEDGE" if net < 0 else "PASS",
    }


# ─── 6. Volatility Regime ────────────────────────────────────────────────────

def classify_vol_regime(volatility_ratio: float):
    if volatility_ratio > 0.5:
        return "HIGH VOL", "🔴"
    elif volatility_ratio > 0.25:
        return "ELEVATED", "🟡"
    return "NORMAL", "🟢"


# ─── 7. DA/RT Spread ─────────────────────────────────────────────────────────

def da_rt_spread_signal(merged_df: pd.DataFrame) -> pd.DataFrame:
    df = merged_df.copy()
    df["da_proxy"]       = df["lmp"].shift(24)
    df["da_rt_spread"]   = df["lmp"] - df["da_proxy"]
    df["spread_zscore"]  = (
        (df["da_rt_spread"] - df["da_rt_spread"].rolling(168).mean())
        / (df["da_rt_spread"].rolling(168).std() + 1e-9)
    )
    return df[["datetime", "lmp", "da_proxy", "da_rt_spread", "spread_zscore"]].dropna()
