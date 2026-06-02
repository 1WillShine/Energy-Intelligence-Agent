"""
app.py — Energy Intelligence Platform
Run: streamlit run app.py

Requires:
  pip install streamlit plotly pandas numpy requests anthropic
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import anthropic

from pipeline import fetch_all, _synthetic_prices, _synthetic_gas, _placeholder_news, fetch_weather_forecast
from ingestion.realtime import DataManager
from forecasting.model import (
    generate_training_data, train, predict, build_features, FEATURES,
    compute_normal_regime_intervals, price_scenario_analysis,
)
from models import (
    compute_risk_score, trading_signal, simulate_temp_shock,
    simulate_gas_shock, hedge_expected_value, classify_vol_regime, da_rt_spread_signal,
    spike_probability, SIGNAL_WEIGHTS,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="GridEdge Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* Dark professional theme */
  .stApp { background-color: #0d1117; color: #e6edf3; }
  section[data-testid="stSidebar"] { background-color: #161b22; }
  .metric-card {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 10px;
  }
  .metric-value { font-size: 2rem; font-weight: 700; color: #58a6ff; }
  .metric-label { font-size: 0.8rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
  .signal-box {
    border-radius: 8px;
    padding: 18px 24px;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
    margin: 10px 0;
  }
  .news-card {
    background: #161b22;
    border-left: 3px solid #58a6ff;
    padding: 12px 16px;
    margin-bottom: 8px;
    border-radius: 0 6px 6px 0;
  }
  .supply-node {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
    font-size: 0.85rem;
  }
  h1, h2, h3 { color: #e6edf3 !important; }
  .stTabs [data-baseweb="tab"] { color: #8b949e; }
  .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom-color: #58a6ff !important; }

  /* Pulse animation for active decision state */
  @keyframes pulse-green {
    0%, 100% { box-shadow: 0 0 0 0 rgba(68,187,68,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(68,187,68,0); }
  }
  @keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 0 0 rgba(255,68,68,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(255,68,68,0); }
  }
  @keyframes pulse-blue {
    0%, 100% { box-shadow: 0 0 0 0 rgba(88,166,255,0.4); }
    50%       { box-shadow: 0 0 0 8px rgba(88,166,255,0); }
  }
  @keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  /* Decision state active gets pulse */
  div[style*="background:#44bb44"] { animation: pulse-green 2s infinite; }
  div[style*="background:#ff4444"] { animation: pulse-red   2s infinite; }
  div[style*="background:#58a6ff"] { animation: pulse-blue  2s infinite; }

  /* Synthesis panel fade-in */
  .synthesis-panel { animation: fadeIn 0.5s ease; }

  /* Metric cards */
  div[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 12px;
    transition: border-color 0.2s;
  }
  div[data-testid="metric-container"]:hover {
    border-color: #58a6ff;
  }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

# ─── Load secrets (Streamlit Cloud) or fall back to empty ────────────────────
def _get_secret(key: str, default: str = "") -> str:
    """Read from st.secrets (Streamlit Cloud) or return default."""
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

_default_provider = _get_secret("DEFAULT_AI_PROVIDER", "Groq (Free)")
_default_groq_key = _get_secret("GROQ_API_KEY", "")
_default_gemini_key = _get_secret("GEMINI_API_KEY", "")
_default_anthropic_key = _get_secret("ANTHROPIC_API_KEY", "")

with st.sidebar:
    st.markdown("## ⚡ GridEdge Intelligence")
    st.markdown("*Energy Market Intelligence Platform*")
    st.markdown("---")

    st.markdown("### 🤖 AI Provider")
    ai_provider = st.selectbox(
        "Provider",
        ["Groq (Free)", "Google Gemini (Free)", "Anthropic Claude (Paid)"],
        index=["Groq (Free)", "Google Gemini (Free)", "Anthropic Claude (Paid)"].index(_default_provider),
    )
    if ai_provider == "Groq (Free)":
        ai_key = st.text_input("Groq API Key", value=_default_groq_key, type="password",
                               help="Free at console.groq.com")
        if _default_groq_key:
            st.caption("✅ Pre-configured · Groq Llama 3.3 70B")
        else:
            st.caption("Free · console.groq.com · no credit card")
    elif ai_provider == "Google Gemini (Free)":
        ai_key = st.text_input("Gemini API Key", value=_default_gemini_key, type="password",
                               help="Free at aistudio.google.com")
        if _default_gemini_key:
            st.caption("✅ Pre-configured · Gemini 2.5 Flash")
        else:
            st.caption("Free · aistudio.google.com · no credit card")
    else:
        ai_key = st.text_input("Anthropic API Key", value=_default_anthropic_key, type="password",
                               help="Paid at console.anthropic.com")
        if _default_anthropic_key:
            st.caption("✅ Pre-configured · Claude Sonnet")
        else:
            st.caption("Paid · ~$3/M tokens")
    anthropic_key = ai_key if ai_provider == "Anthropic Claude (Paid)" else ""

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    price_days = st.slider("Historical window (days)", 30, 90, 60)
    auto_refresh = st.toggle("Auto-refresh (5 min)", value=False)

    st.markdown("---")
    st.markdown("### 📍 Market")
    st.markdown("**Zone:** NP-15 (N. California)")
    st.markdown("**Node:** TH_NP15_GEN-APND")
    st.markdown("**ISO:** CAISO")

    st.markdown("---")
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

# ─── Data Loading ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data(days):
    return fetch_all(price_days=days, mix_days=7)


with st.spinner("Loading energy market data..."):
    data = load_data(price_days)

# Train ML models
@st.cache_resource(show_spinner="Training ML models on 2 years of data...")
def load_models():
    training_df = generate_training_data(years=2, seed=42)
    models      = train(training_df)
    # Compute tight normal-regime intervals on same training data
    models["normal_intervals"] = compute_normal_regime_intervals(training_df)
    return models

ml = load_models()

# Use tight normal-regime intervals instead of wide quantile intervals
_ni = ml["normal_intervals"]
ml_forecast_normal_low  = _ni["ci_low"]
ml_forecast_normal_high = _ni["ci_high"]
ml_forecast_resid_std   = _ni["resid_std"]

merged = data["merged"]
genmix = data["genmix"]
gas = data["gas"]
news = data["news"]
forecast = data["forecast"]

# Debug: show merge diagnostics if something is empty
if merged.empty or merged.dropna().empty:
    st.error("⚠️ Data merge failed — showing diagnostics below.")
    st.write("**prices shape:**", data["prices"].shape)
    st.write("**prices sample:**", data["prices"].head(3))
    st.write("**weather shape:**", data["weather"].shape)
    st.write("**weather sample:**", data["weather"].head(3))
    st.write("**merged shape:**", merged.shape)
    st.write("**merged after dropna:**", merged.dropna().shape)
    st.stop()

# Current snapshot — use last row that has all key columns filled
latest = merged.dropna(subset=["lmp", "temp_f", "wind_mph"]).iloc[-1]
current_lmp = latest["lmp"]
current_temp = latest["temp_f"]
current_wind = latest["wind_mph"]
current_zscore = latest.get("lmp_zscore", 0.0)
current_vol = latest.get("volatility", 0.1)
current_gas = float(gas["gas_price"].iloc[-1])
current_hour = int(latest["hour"])
roll24_mean = latest["lmp_roll24"] if pd.notna(latest.get("lmp_roll24")) else current_lmp

# ML prediction
try:
    merged_with_gas = merged.copy()
    merged_with_gas["gas_price"] = current_gas
    feat_live = build_features(merged_with_gas)
    ml_pred = predict(ml, feat_live.iloc[-1])
    spike_prob = ml_pred["spike_prob"]
    ml_forecast_point = ml_pred["point"]
    ml_forecast_q10   = ml_pred["q10"]
    ml_forecast_q90   = ml_pred["q90"]
except Exception:
    spike_prob_tuple = spike_probability(current_temp, current_wind, current_gas, current_hour, current_zscore)
    spike_prob = spike_prob_tuple[0]
    ml_forecast_point = roll24_mean
    ml_forecast_q10   = roll24_mean * 0.8
    ml_forecast_q90   = roll24_mean * 1.3

# ── Signal fusion (all quantitative, no AI involvement) ──
_gm_latest    = genmix.iloc[-1]
_gas_roll7d   = float(gas["gas_price"].rolling(7).mean().iloc[-1]) if len(gas) >= 7 else current_gas
_gas_roll7d   = _gas_roll7d if not np.isnan(_gas_roll7d) else current_gas

risk = compute_risk_score(
    ml_spike_prob    = spike_prob,
    temp_f           = current_temp,
    wind_mph         = current_wind,
    gas_price        = current_gas,
    gas_roll7d       = _gas_roll7d,
    hour             = current_hour,
    lmp_zscore       = current_zscore,
    forecast_df      = forecast,
    solar_mw         = float(_gm_latest["solar_mw"]),
    wind_mw          = float(_gm_latest["wind_mw"]),
    total_mw         = float(_gm_latest["total_mw"]),
    volatility       = current_vol,
)

decision      = risk["decision"]          # BUY EARLY / HOLD / HEDGE
risk_score    = risk["risk_score"]        # 0-1 float
confidence    = risk["confidence"]        # Low/Medium/High
top_driver    = risk["top_driver"]        # name of highest-weight signal
signal_vals   = risk["signal_values"]     # dict of normalized signals
signal_contribs = risk["signal_contributions"]

# Legacy signal for backward compat
signal, rationale = trading_signal(spike_prob, current_lmp, roll24_mean)
vol_label, vol_color = classify_vol_regime(current_vol)

# Factor breakdown for risk tab (rule-based, kept for explainability chart)
_, factors = spike_probability(current_temp, current_wind, current_gas, current_hour, current_zscore)

# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh:
    import time as _time
    _time.sleep(300)   # 5 minutes
    st.cache_data.clear()
    st.rerun()

# ─── Multi-provider AI call ───────────────────────────────────────────────────

def call_ai(messages: list, system: str, provider: str, key: str) -> str:
    """
    Unified AI call supporting Groq, Gemini, and Anthropic.
    Returns response text or a clear error message (never silently fails).
    """
    if not key:
        return "⚠️ Enter your API key in the sidebar to activate the AI."

    try:
        if provider == "Groq (Free)":
            try:
                from openai import OpenAI
            except ImportError:
                return "⚠️ Run: pip install openai"
            client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=1000,
            )
            return resp.choices[0].message.content

        elif provider == "Google Gemini (Free)":
            try:
                from openai import OpenAI
            except ImportError:
                return "⚠️ Run: pip install openai"
            client = OpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=key,
            )
            resp = client.chat.completions.create(
                model="gemini-2.5-flash-preview-04-17",
                messages=[{"role": "system", "content": system}] + messages,
                max_tokens=1000,
            )
            return resp.choices[0].message.content

        else:  # Anthropic
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=key)
            resp = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system,
                messages=messages,
            )
            return resp.content[0].text

    except Exception as e:
        return f"AI error: {e}"

# ─── Header ───────────────────────────────────────────────────────────────────

# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:4px;">
  <div>
    <span style="font-size:1.8rem; font-weight:800; color:#e6edf3;">⚡ GridEdge Intelligence</span>
    <span style="font-size:0.85rem; color:#8b949e; margin-left:12px;">
      CAISO NP-15 · {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC
      {"· 🟢 Live" if auto_refresh else "· ⚪ Manual refresh"}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Three-state decision indicator ──────────────────────────────────────────
# This is the primary output of the entire system

_buy_active   = decision == "BUY EARLY"
_hedge_active = decision == "HEDGE"
_hold_active  = decision == "HOLD"

def _state_style(active, color):
    if active:
        return f"background:{color}; color:#000; font-weight:800; border:2px solid {color}; border-radius:8px; padding:10px 18px; font-size:1.05rem; text-align:center;"
    return f"background:#1c2128; color:#555; font-weight:500; border:1px solid #30363d; border-radius:8px; padding:10px 18px; font-size:1.05rem; text-align:center;"

dec_col1, dec_col2, dec_col3, dec_spacer = st.columns([1, 1, 1, 3])
with dec_col1:
    st.markdown(f'<div style="{_state_style(_buy_active, "#44bb44")}">📥 BUY EARLY</div>', unsafe_allow_html=True)
with dec_col2:
    st.markdown(f'<div style="{_state_style(_hold_active, "#58a6ff")}">⏸ HOLD</div>', unsafe_allow_html=True)
with dec_col3:
    st.markdown(f'<div style="{_state_style(_hedge_active, "#ff4444")}">🛡 HEDGE</div>', unsafe_allow_html=True)
with dec_spacer:
    _rationale_txt = f"Risk score: {risk_score:.2f}/1.0 · Confidence: {confidence} · Top driver: {top_driver.replace('_',' ')}"
    st.markdown(f'<div style="color:#8b949e; font-size:0.85rem; padding:12px 0;">{_rationale_txt}</div>', unsafe_allow_html=True)

st.markdown("---")

# ─── KPI Row ─────────────────────────────────────────────────────────────────

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.metric("⚡ LMP", f"${current_lmp:.2f}/MWh",
              delta=f"{current_lmp - roll24_mean:+.1f} vs 24h avg")
with k2:
    st.metric("🔮 ML Forecast", f"${ml_forecast_point:.0f}/MWh",
              delta=f"80%: ${ml_forecast_q10:.0f}–${ml_forecast_q90:.0f}")
with k3:
    st.metric("⚠️ Spike Prob", f"{spike_prob:.0%}",
              delta="ML model" if spike_prob > 0.3 else None,
              delta_color="inverse")
with k4:
    st.metric("🌡️ Temperature", f"{current_temp:.0f}°F")
with k5:
    st.metric("🔥 Gas", f"${current_gas:.2f}/MMBtu")
with k6:
    st.metric("📊 Volatility", vol_label)

st.markdown("---")

# ─── AI Synthesis Panel (proactive, runs automatically) ───────────────────────

with st.expander("🧠 AI Market Intelligence — Live Synthesis", expanded=True):
    if not ai_key:
        st.info("Add an API key in the sidebar (Groq is free) to enable live AI synthesis.")
    else:
        # Build full market context
        _gm = genmix.iloc[-1]
        _news_text = "\n".join([f"- {n['title']}" for n in news[:4]])
        _fc_24h = forecast.head(24)
        _max_fc_temp = _fc_24h["temp_f"].max() if len(_fc_24h) > 0 else current_temp

        _synthesis_system = f"""You are GridEdge, an AI energy market analyst for CAISO NP-15.

CRITICAL RULES:
- You ONLY interpret the numbers provided below. You do NOT generate new numbers.
- Do NOT invent prices, probabilities, or statistics not listed here.
- Do NOT make claims about news events beyond the headlines provided.
- Every statement must trace directly to a number in the LIVE SIGNALS section.

LIVE SIGNALS (computed by quantitative models, not by you):
Decision engine: {decision} | Risk score: {risk_score:.2f}/1.0 | Confidence: {confidence}
Top risk driver: {top_driver.replace('_',' ')}

Signal breakdown (normalized 0-1):
  ML spike probability:  {signal_vals['ml_spike_prob']:.2f}  (XGBoost model output)
  Heat stress:           {signal_vals['heat_stress']:.2f}  (temp {current_temp:.0f}°F)
  Price momentum:        {signal_vals['price_momentum']:.2f}  (z-score {current_zscore:.2f})
  Peak hour:             {signal_vals['peak_hour']:.2f}  (hour {current_hour}:00)
  Gas pressure:          {signal_vals['gas_pressure']:.2f}  (${current_gas:.2f}/MMBtu)
  Forecast stress:       {signal_vals['forecast_stress']:.2f}  (max {_max_fc_temp:.0f}°F next 24h)
  Wind deficit:          {signal_vals['wind_deficit']:.2f}  ({current_wind:.0f}mph)
  Renewable deficit:     {signal_vals['renewable_deficit']:.2f}

Market context:
  LMP now:      ${current_lmp:.2f}/MWh (24h avg ${roll24_mean:.2f}, z={current_zscore:.2f})
  ML forecast:  ${ml_forecast_point:.0f}/MWh next hour (80% CI: ${ml_forecast_q10:.0f}-${ml_forecast_q90:.0f})
  Volatility:   {vol_label}
  Generation:   Solar {_gm['solar_mw']:,.0f}MW | Wind {_gm['wind_mw']:,.0f}MW | Gas {_gm['gas_mw']:,.0f}MW

Recent headlines (do not embellish):
{_news_text}

Output format (STRICT — use exactly these labels):
DECISION: {decision}
CONFIDENCE: {confidence}
REASONING: [2-3 sentences. Cite specific numbers from above. No invented statistics.]
WATCH: [One specific threshold or event that would change the decision. Must be quantitative.]"""

        _cache_key = f"synthesis_{datetime.utcnow().strftime('%Y%m%d%H')}{(datetime.utcnow().minute // 5) * 5}"

        if "synthesis_cache" not in st.session_state or st.session_state.get("synthesis_key") != _cache_key:
            with st.spinner("Synthesizing market signals..."):
                _synthesis = call_ai(
                    messages=[{"role": "user", "content": "Synthesize current market conditions and give procurement recommendation."}],
                    system=_synthesis_system,
                    provider=ai_provider,
                    key=ai_key,
                )
            st.session_state["synthesis_cache"] = _synthesis
            st.session_state["synthesis_key"] = _cache_key
        else:
            _synthesis = st.session_state["synthesis_cache"]

        # Parse and display synthesis
        _lines = _synthesis.strip().splitlines()
        for _line in _lines:
            if _line.startswith("DECISION:"):
                _dec = _line.replace("DECISION:", "").strip()
                _dc = "#44bb44" if "BUY" in _dec else "#ff4444" if "HEDGE" in _dec else "#58a6ff"
                st.markdown(f'<div style="font-size:1.3rem; font-weight:800; color:{_dc}; margin-bottom:6px;">{_dec}</div>', unsafe_allow_html=True)
            elif _line.startswith("CONFIDENCE:"):
                st.markdown(f'<span style="color:#8b949e; font-size:0.9rem;">{_line}</span>', unsafe_allow_html=True)
            elif _line.startswith("REASONING:"):
                st.markdown(_line.replace("REASONING:", "**Why:**"))
            elif _line.startswith("WATCH:"):
                st.markdown(_line.replace("WATCH:", "👁 **Watch:**"))
            elif _line.strip():
                st.markdown(_line)

        st.caption(f"Synthesized from: LMP data · ML forecast · {len(news)} news signals · weather forecast · generation mix")

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Market", "🏭 Supply Chain", "☁️ Weather & Forecast",
    "⚠️ Risk & Signals", "🔮 ML Forecast", "🤖 AI Agent", "📰 News Feed"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("### Electricity Price & Market Dynamics")

    # Price chart with volatility bands + spike flags
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.20],
                        subplot_titles=("LMP Price ($/MWh)", "24h Rolling Volatility", "Z-Score"),
                        vertical_spacing=0.06)

    df_plot = merged.dropna().tail(24 * 30)  # last 30 days

    # Price line + bands
    fig.add_trace(go.Scatter(x=df_plot["datetime"], y=df_plot["lmp"],
                             name="LMP", line=dict(color="#58a6ff", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df_plot["datetime"], y=df_plot["lmp_roll24"],
                             name="24h Mean", line=dict(color="#f0a500", width=1, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pd.concat([df_plot["datetime"], df_plot["datetime"][::-1]]),
        y=pd.concat([df_plot["lmp_roll24"] + 2 * df_plot["lmp_std24"],
                     (df_plot["lmp_roll24"] - 2 * df_plot["lmp_std24"])[::-1]]),
        fill="toself", fillcolor="rgba(88,166,255,0.08)",
        line=dict(color="rgba(0,0,0,0)"), name="±2σ Band", showlegend=True
    ), row=1, col=1)

    # Spike markers
    spikes = df_plot[df_plot["spike"] == 1]
    fig.add_trace(go.Scatter(x=spikes["datetime"], y=spikes["lmp"],
                             mode="markers", marker=dict(color="#ff4444", size=7, symbol="triangle-up"),
                             name="Spike (>2σ)"), row=1, col=1)

    # Volatility
    fig.add_trace(go.Scatter(x=df_plot["datetime"], y=df_plot["volatility"],
                             fill="tozeroy", fillcolor="rgba(240,165,0,0.15)",
                             line=dict(color="#f0a500"), name="Volatility", showlegend=False), row=2, col=1)

    # Z-score
    fig.add_trace(go.Scatter(x=df_plot["datetime"], y=df_plot["lmp_zscore"],
                             line=dict(color="#bc8cff"), name="Z-Score", showlegend=False), row=3, col=1)
    fig.add_hline(y=2.0, line_dash="dot", line_color="#ff4444", row=3, col=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color="#44bb44", row=3, col=1)

    fig.update_layout(height=600, template="plotly_dark",
                      paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
                      legend=dict(orientation="h", y=1.02))
    fig.update_yaxes(gridcolor="#21262d")
    fig.update_xaxes(gridcolor="#21262d")
    st.plotly_chart(fig, use_container_width=True)

    # DA/RT Spread
    st.markdown("### Day-Ahead vs Real-Time Spread")
    spread_df = da_rt_spread_signal(merged).tail(24 * 14)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=spread_df["datetime"], y=spread_df["da_rt_spread"],
                          marker_color=np.where(spread_df["da_rt_spread"] > 0, "#ff4444", "#44bb44"),
                          name="DA/RT Spread"))
    fig2.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                       plot_bgcolor="#0d1117", height=250,
                       title="RT LMP − DA Proxy ($/MWh) — positive = real-time tighter than forecast")
    st.plotly_chart(fig2, use_container_width=True)

    # Gas price correlation
    st.markdown("### Gas Price vs LMP (Supply Cost Transmission)")
    daily_lmp = merged.groupby(merged["datetime"].dt.date)["lmp"].mean().reset_index()
    daily_lmp.columns = ["date", "avg_lmp"]
    daily_lmp["date"] = pd.to_datetime(daily_lmp["date"])
    gas_merged = pd.merge(daily_lmp, gas, on="date", how="inner")
    if len(gas_merged) > 5:
        fig3 = px.scatter(gas_merged, x="gas_price", y="avg_lmp",
                          trendline="ols",
                          labels={"gas_price": "Henry Hub Gas ($/MMBtu)", "avg_lmp": "Avg Daily LMP ($/MWh)"},
                          template="plotly_dark", color_discrete_sequence=["#58a6ff"])
        fig3.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=300)
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SUPPLY CHAIN
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("### Electricity Supply Chain")

    # Flow diagram using Plotly Sankey
    st.markdown("#### Supply Chain Flow: Fuel → Generation → Grid → Consumer")

    latest_mix = genmix.iloc[-1]
    total = latest_mix["total_mw"]

    fig_sankey = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=20, thickness=25,
            label=["☀️ Solar", "💨 Wind", "⚛️ Nuclear", "💧 Hydro", "🔥 Natural Gas",
                   "🏭 Generation Pool", "🔌 Transmission Grid", "🏢 Commercial",
                   "🏠 Residential", "🏭 Industrial"],
            color=["#f0a500", "#58a6ff", "#bc8cff", "#1f9cf0", "#ff6b35",
                   "#30363d", "#21262d", "#44bb44", "#44bb44", "#44bb44"],
            x=[0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.65, 1.0, 1.0, 1.0],
            y=[0.05, 0.25, 0.45, 0.65, 0.85, 0.45, 0.45, 0.15, 0.50, 0.85],
        ),
        link=dict(
            source=[0, 1, 2, 3, 4, 5, 6, 6, 6],
            target=[5, 5, 5, 5, 5, 6, 7, 8, 9],
            value=[
                latest_mix["solar_mw"], latest_mix["wind_mw"],
                latest_mix["nuclear_mw"], latest_mix["hydro_mw"], latest_mix["gas_mw"],
                total, total * 0.35, total * 0.38, total * 0.27,
            ],
            color=["rgba(240,165,0,0.4)", "rgba(88,166,255,0.4)", "rgba(188,140,255,0.4)",
                   "rgba(31,156,240,0.4)", "rgba(255,107,53,0.4)",
                   "rgba(48,54,61,0.5)", "rgba(68,187,68,0.3)", "rgba(68,187,68,0.3)", "rgba(68,187,68,0.3)"],
        )
    ))
    fig_sankey.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", height=420,
        font=dict(color="#e6edf3", size=13)
    )
    st.plotly_chart(fig_sankey, use_container_width=True)

    # Generation mix over time
    st.markdown("#### Generation Mix Over Time (MW)")
    gm = genmix.tail(7 * 24)
    fig_mix = go.Figure()
    colors = {"solar_mw": "#f0a500", "wind_mw": "#58a6ff",
              "nuclear_mw": "#bc8cff", "hydro_mw": "#1f9cf0", "gas_mw": "#ff6b35"}
    labels = {"solar_mw": "Solar", "wind_mw": "Wind",
              "nuclear_mw": "Nuclear", "hydro_mw": "Hydro", "gas_mw": "Gas"}
    for col, color in colors.items():
        fig_mix.add_trace(go.Scatter(
            x=gm["datetime"], y=gm[col],
            stackgroup="one", name=labels[col],
            line=dict(width=0), fillcolor=color.replace("#", "rgba(") + ",0.7)"
                      if False else color,
            mode="lines"
        ))
    fig_mix.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                          plot_bgcolor="#0d1117", height=350,
                          yaxis_title="MW", legend=dict(orientation="h"))
    st.plotly_chart(fig_mix, use_container_width=True)

    # Current mix donut
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.markdown("#### Current Generation Mix")
        vals = [latest_mix["solar_mw"], latest_mix["wind_mw"], latest_mix["nuclear_mw"],
                latest_mix["hydro_mw"], latest_mix["gas_mw"]]
        lbls = ["Solar", "Wind", "Nuclear", "Hydro", "Gas"]
        fig_donut = go.Figure(go.Pie(
            labels=lbls, values=vals, hole=0.55,
            marker_colors=["#f0a500", "#58a6ff", "#bc8cff", "#1f9cf0", "#ff6b35"]
        ))
        fig_donut.update_layout(template="plotly_dark", paper_bgcolor="#0d1117", height=300,
                                showlegend=True, legend=dict(orientation="h"))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_b:
        st.markdown("#### Renewable Penetration")
        genmix["renewable_pct"] = (genmix["solar_mw"] + genmix["wind_mw"] + genmix["hydro_mw"]) / genmix["total_mw"] * 100
        fig_ren = go.Figure(go.Scatter(
            x=genmix["datetime"].tail(168), y=genmix["renewable_pct"].tail(168),
            fill="tozeroy", fillcolor="rgba(68,187,68,0.2)",
            line=dict(color="#44bb44"), name="Renewable %"
        ))
        fig_ren.add_hline(y=50, line_dash="dash", line_color="#f0a500",
                          annotation_text="50% threshold")
        fig_ren.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                              plot_bgcolor="#0d1117", height=300,
                              yaxis_title="% Renewable")
        st.plotly_chart(fig_ren, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — WEATHER & FORECAST
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Weather Intelligence")
    st.caption("Weather is the #1 driver of electricity demand spikes. Temperature extremes → AC/heating load → LMP price spikes.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Temperature vs LMP (Last 30 Days)")
        df_tw = merged.dropna().tail(24 * 30)
        fig_tw = go.Figure()
        fig_tw.add_trace(go.Scatter(x=df_tw["datetime"], y=df_tw["temp_f"],
                                    name="Temperature (°F)", line=dict(color="#f0a500"), yaxis="y1"))
        fig_tw.add_trace(go.Scatter(x=df_tw["datetime"], y=df_tw["lmp"],
                                    name="LMP ($/MWh)", line=dict(color="#58a6ff"), yaxis="y2"))
        fig_tw.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=350,
            yaxis=dict(title="Temperature (°F)", gridcolor="#21262d"),
            yaxis2=dict(title="LMP ($/MWh)", overlaying="y", side="right"),
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig_tw, use_container_width=True)

    with col2:
        st.markdown("#### 7-Day Forecast")
        fig_fc = make_subplots(rows=2, cols=1, shared_xaxes=True,
                               subplot_titles=("Temp Forecast (°F)", "Wind Forecast (mph)"),
                               vertical_spacing=0.1)
        fig_fc.add_trace(go.Scatter(x=forecast["datetime"], y=forecast["temp_f"],
                                    fill="tozeroy", fillcolor="rgba(240,165,0,0.15)",
                                    line=dict(color="#f0a500"), name="Temp"), row=1, col=1)
        fig_fc.add_hline(y=95, line_dash="dot", line_color="#ff4444",
                         annotation_text="Heat stress (95°F)", row=1, col=1)
        fig_fc.add_trace(go.Scatter(x=forecast["datetime"], y=forecast["wind_mph"],
                                    fill="tozeroy", fillcolor="rgba(88,166,255,0.15)",
                                    line=dict(color="#58a6ff"), name="Wind"), row=2, col=1)
        fig_fc.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                             plot_bgcolor="#0d1117", height=350)
        st.plotly_chart(fig_fc, use_container_width=True)

    # Solar radiation vs solar generation
    st.markdown("#### Solar Radiation → Solar Generation Relationship")
    gm_weather = pd.merge(
        genmix[["datetime", "solar_mw"]],
        merged[["datetime", "solar_rad"]],
        on="datetime", how="inner"
    ).dropna()
    if len(gm_weather) > 10:
        fig_sol = px.scatter(gm_weather, x="solar_rad", y="solar_mw",
                             labels={"solar_rad": "Solar Radiation (W/m²)", "solar_mw": "Solar Generation (MW)"},
                             trendline="ols", template="plotly_dark",
                             color_discrete_sequence=["#f0a500"])
        fig_sol.update_layout(paper_bgcolor="#0d1117", plot_bgcolor="#0d1117", height=280)
        st.plotly_chart(fig_sol, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RISK & SIGNALS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### Risk Intelligence & Trading Signals")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Spike Probability Breakdown")
        st.markdown(f"**Current probability: {spike_prob:.1%}**")

        factor_names = list(factors.keys())
        factor_vals = [v * 100 for v in factors.values()]
        colors_bar = ["#ff4444" if v > 10 else "#f0a500" if v > 5 else "#44bb44" for v in factor_vals]
        fig_factors = go.Figure(go.Bar(
            x=factor_vals, y=factor_names, orientation="h",
            marker_color=colors_bar,
            text=[f"{v:.1f}%" for v in factor_vals], textposition="outside"
        ))
        fig_factors.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                                  plot_bgcolor="#0d1117", height=280,
                                  xaxis_title="Contribution to spike probability (%)",
                                  xaxis=dict(range=[0, 35]))
        st.plotly_chart(fig_factors, use_container_width=True)

    with col2:
        st.markdown("#### Expected Value of Hedging")
        vol_mw = st.slider("Volume (MW)", 10, 500, 100)
        spike_mag = st.slider("Spike magnitude ($/MWh)", 50, 500, 150)
        hedge_cost = st.slider("Hedge cost ($/MWh)", 2, 30, 8)

        ev = hedge_expected_value(spike_prob, current_lmp, spike_mag, hedge_cost, vol_mw)
        net = ev["net_benefit_of_hedging_usd"]
        st.markdown(f"""
        | | Value |
        |---|---|
        | Spike probability | {ev['prob_spike']:.1%} |
        | EV (no hedge) | **${ev['ev_no_hedge_usd']:,.0f}** |
        | EV (hedge) | **${ev['ev_hedge_usd']:,.0f}** |
        | Net benefit of hedging | **${net:,.0f}** |
        | Recommendation | **{ev['recommendation']}** |
        """)

    # Counterfactual simulations
    st.markdown("---")
    st.markdown("#### 🔬 Counterfactual Simulation")
    st.caption("*'What if' analysis — how sensitive is spike risk to key inputs?*")

    sim_col1, sim_col2 = st.columns(2)

    with sim_col1:
        st.markdown("**Temperature shock**")
        delta_temp = st.slider("Temperature change (°F)", -20, +30, +5)
        sim_t = simulate_temp_shock(current_temp, delta_temp, current_wind,
                                    current_gas, current_hour, current_zscore)
        fig_sim_t = go.Figure(go.Bar(
            x=["Base", f"Base + {delta_temp}°F"],
            y=[sim_t["base_prob"] * 100, sim_t["shock_prob"] * 100],
            marker_color=["#58a6ff", "#ff4444" if sim_t["shock_prob"] > 0.4 else "#f0a500"],
            text=[f"{sim_t['base_prob']:.1%}", f"{sim_t['shock_prob']:.1%}"],
            textposition="outside"
        ))
        fig_sim_t.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                                plot_bgcolor="#0d1117", height=220,
                                yaxis=dict(range=[0, 110], title="Spike Probability (%)"))
        st.plotly_chart(fig_sim_t, use_container_width=True)
        st.caption(f"Δ Probability: **{sim_t['delta_prob']:+.1%}**")

    with sim_col2:
        st.markdown("**Gas price shock**")
        delta_gas = st.slider("Gas price change ($/MMBtu)", -2.0, +4.0, +1.0, step=0.25)
        sim_g = simulate_gas_shock(current_temp, current_wind, current_gas,
                                   delta_gas, current_hour, current_zscore)
        fig_sim_g = go.Figure(go.Bar(
            x=["Base", f"Base + ${delta_gas:.2f}"],
            y=[sim_g["base_prob"] * 100, sim_g["shock_prob"] * 100],
            marker_color=["#58a6ff", "#ff4444" if sim_g["shock_prob"] > 0.4 else "#f0a500"],
            text=[f"{sim_g['base_prob']:.1%}", f"{sim_g['shock_prob']:.1%}"],
            textposition="outside"
        ))
        fig_sim_g.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                                plot_bgcolor="#0d1117", height=220,
                                yaxis=dict(range=[0, 110], title="Spike Probability (%)"))
        st.plotly_chart(fig_sim_g, use_container_width=True)
        st.caption(f"Δ Probability: **{sim_g['delta_prob']:+.1%}**")

    # Rolling spike probability over time
    st.markdown("#### Rolling Spike Probability (Last 14 Days)")
    df_roll = merged.dropna().tail(14 * 24).copy()
    df_roll["spike_prob_est"] = (
        (np.maximum(df_roll["temp_f"] - 95, 0) / 20 * 0.30).clip(0, 0.30) +
        (np.maximum(15 - df_roll["wind_mph"], 0) / 15 * 0.20).clip(0, 0.20) +
        (df_roll["lmp_zscore"].clip(0, 3) / 3 * 0.10)
    ).clip(0, 0.97)

    fig_roll = go.Figure()
    fig_roll.add_trace(go.Scatter(x=df_roll["datetime"], y=df_roll["spike_prob_est"],
                                  fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
                                  line=dict(color="#ff4444"), name="Est. Spike Prob"))
    fig_roll.add_hline(y=0.60, line_dash="dash", line_color="#ff4444",
                       annotation_text="HEDGE threshold")
    fig_roll.add_hline(y=0.35, line_dash="dash", line_color="#f0a500",
                       annotation_text="BUY EARLY threshold")
    fig_roll.update_layout(template="plotly_dark", paper_bgcolor="#0d1117",
                           plot_bgcolor="#0d1117", height=280,
                           yaxis=dict(title="Spike Probability", tickformat=".0%"))
    st.plotly_chart(fig_roll, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — AI AGENT
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### 🔮 ML Price Forecast")
    st.caption("XGBoost · 2-year training · TimeSeriesSplit CV · Normal-regime intervals (separate from spike risk)")

    cv = ml["cv_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Mean MAE", f"${cv['mae'].mean():.2f}/MWh")
    with m2:
        st.metric("Normal CI width", f"${ml_forecast_normal_high - ml_forecast_normal_low:.0f}/MWh",
                  delta="tighter than quantile regression")
    with m3:
        st.metric("Residual Std", f"${ml_forecast_resid_std:.2f}/MWh")
    with m4:
        st.metric("Spike rate (train)", f"{ml['spike_rate']:.2%}")

    st.markdown("---")

    fa, fb = st.columns(2)
    with fa:
        st.markdown("#### Next Hour Price Forecast")
        st.caption("Normal-regime interval (excludes spike scenario — shown separately below)")

        # Gauge-style visualization
        fig_fc = go.Figure()
        # Confidence band
        fig_fc.add_trace(go.Scatter(
            x=["Low (10%)", "Point", "High (90%)"],
            y=[ml_forecast_normal_low, ml_forecast_point, ml_forecast_normal_high],
            mode="markers+lines",
            marker=dict(size=[12, 18, 12],
                        color=["#44bb44", "#58a6ff", "#f0a500"],
                        symbol=["circle", "diamond", "circle"]),
            line=dict(color="#30363d", width=2),
            name="Normal regime CI",
        ))
        fig_fc.add_hline(y=current_lmp, line_dash="dash", line_color="#ff4444",
                         annotation_text=f"Current ${current_lmp:.0f}")
        fig_fc.add_hline(y=ml_forecast_point, line_dash="dot", line_color="#58a6ff",
                         annotation_text=f"Forecast ${ml_forecast_point:.0f}")
        fig_fc.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=280, yaxis_title="LMP ($/MWh)",
            title=f"80% CI: ${ml_forecast_normal_low:.0f}–${ml_forecast_normal_high:.0f}/MWh (normal conditions)",
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        st.markdown(f"""
        | | Value |
        |---|---|
        | Point forecast | **${ml_forecast_point:.2f}/MWh** |
        | Normal 80% CI | **${ml_forecast_normal_low:.0f}–${ml_forecast_normal_high:.0f}/MWh** |
        | Spike probability | **{spike_prob:.1%}** (separate risk) |
        | Current LMP | **${current_lmp:.2f}/MWh** |
        | Model accuracy | **±${ml_forecast_resid_std:.2f}/MWh** avg error |
        """)

    with fb:
        st.markdown("#### Feature Importance")
        imp = ml["feature_importance"].head(12)
        fig_imp = go.Figure(go.Bar(
            x=imp["importance"],
            y=imp["feature"].str.replace("_", " "),
            orientation="h",
            marker_color=[
                "#ff4444" if v > 0.06 else "#f0a500" if v > 0.03 else "#58a6ff"
                for v in imp["importance"]
            ],
        ))
        fig_imp.update_layout(
            template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            height=380, xaxis_title="Importance score",
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    # ── Scenario Analysis ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 📊 Price Scenario Analysis")
    st.caption("What each scenario means for a commercial electricity buyer")

    scenarios = price_scenario_analysis(
        point       = ml_forecast_point,
        ci_low      = ml_forecast_normal_low,
        ci_high     = ml_forecast_normal_high,
        spike_prob  = spike_prob,
        roll24_mean = roll24_mean,
        current_lmp = current_lmp,
    )

    sc_cols = st.columns(len(scenarios))
    for col, sc in zip(sc_cols, scenarios):
        with col:
            st.markdown(f"""
<div style="background:#161b22; border-left:3px solid {sc['color']};
     border-radius:0 8px 8px 0; padding:14px; height:100%;">
  <div style="font-size:0.75rem; color:#8b949e; text-transform:uppercase;
       letter-spacing:1px; margin-bottom:6px;">{sc['scenario']}</div>
  <div style="font-size:1.3rem; font-weight:700; color:{sc['color']};
       margin-bottom:6px;">{sc['price']}</div>
  <div style="font-size:0.78rem; color:#8b949e; margin-bottom:4px;">
    Probability: {sc['probability']}</div>
  <div style="font-size:0.82rem; color:#e6edf3; margin-bottom:8px;">
    {sc['implication']}</div>
  <div style="font-size:0.78rem; color:{sc['color']}; font-weight:600;">
    → {sc['action']}</div>
</div>""", unsafe_allow_html=True)

    # CV performance chart
    st.markdown("---")
    st.markdown("#### Model Validation (TimeSeriesSplit)")
    st.caption("Fold N always trains on past data, validates on future — prevents leakage")
    fig_cv = go.Figure()
    fig_cv.add_trace(go.Bar(
        x=[f"Fold {i}" for i in cv["fold"]],
        y=cv["mae"],
        marker_color=["#ff4444" if v > cv["mae"].mean()*1.2 else "#58a6ff" for v in cv["mae"]],
        text=[f"${v:.1f}" for v in cv["mae"]],
        textposition="outside",
    ))
    fig_cv.add_hline(y=cv["mae"].mean(), line_dash="dash", line_color="#f0a500",
                     annotation_text=f"Mean MAE: ${cv['mae'].mean():.2f}")
    fig_cv.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
        height=260, yaxis_title="MAE ($/MWh)",
    )
    st.plotly_chart(fig_cv, use_container_width=True)

    st.markdown("#### Why These Features Matter")
    st.markdown("""
    | Feature | Economic reason |
    |---|---|
    | `lmp_roll24_std` | High recent volatility → more volatility expected |
    | `is_peak` | 7-10am / 5-9pm demand peaks structurally drive prices |
    | `lmp_lag24` | Same hour yesterday: strongest single predictor |
    | `heat_stress` | AC load above 85°F creates nonlinear demand surge |
    | `gas_price` | Gas peakers set marginal cost → LMP in high-demand hours |
    | `solar_roll6` | 6-hour solar average suppresses midday prices |
    | `lmp_zscore` | Price above rolling mean tends to revert or escalate |
    """)

with tab6:
    st.markdown("### 🤖 AI Energy Intelligence Agent")

    _provider_display = ai_provider.split(" (")[0]
    st.caption(f"Powered by {_provider_display} · Live market context injected · Ask anything about current conditions")

    # Connection status indicator
    if ai_key:
        st.success(f"✅ {_provider_display} connected — agent ready")
    else:
        st.warning(f"⚠️ No API key. Get a free Groq key at console.groq.com (takes 2 min, no credit card)")

    if not ai_key:
        st.markdown("""
        **Once connected, the agent can:**
        - Explain the current BUY/HOLD/HEDGE recommendation in plain English
        - Interpret how today's news affects electricity prices
        - Answer what-if questions: "what if temp rises 10°F?"
        - Explain LMP, DA/RT spread, and market mechanics
        - Compare current conditions to historical regimes
        """)
    else:
        # Build live market context for the agent
        news_text = "\n".join([f"- {n['title']}: {n['description']}" for n in news[:5]])
        gm = genmix.iloc[-1]

        system_prompt = f"""You are an expert energy market analyst and quantitative trader specializing in 
US electricity markets, specifically CAISO (California ISO). You have deep expertise in:
- LMP (Locational Marginal Pricing) mechanics and drivers
- Day-ahead vs real-time market dynamics and arbitrage
- Electricity supply chain: fuel → generation → transmission → consumers
- Risk management: hedging with futures, forward contracts, options
- How weather, gas prices, and renewables drive electricity prices
- How to think probabilistically about price spikes

You are currently monitoring the CAISO NP-15 node. Here is the live market snapshot:

CURRENT MARKET DATA:
- LMP Price: ${current_lmp:.2f}/MWh (24h avg: ${roll24_mean:.2f}/MWh)
- Temperature: {current_temp:.0f}°F in San Francisco
- Wind Speed: {current_wind:.0f} mph
- Natural Gas (Henry Hub): ${current_gas:.2f}/MMBtu
- Price Z-Score: {current_zscore:.2f} (>2.0 = spike territory)
- Volatility Regime: {vol_label}
- Spike Probability (next 12h): {spike_prob:.1%}
- Current Trading Signal: {signal}

GENERATION MIX RIGHT NOW:
- Solar: {gm['solar_mw']:,.0f} MW ({gm['solar_mw']/gm['total_mw']*100:.0f}%)
- Wind: {gm['wind_mw']:,.0f} MW ({gm['wind_mw']/gm['total_mw']*100:.0f}%)
- Gas: {gm['gas_mw']:,.0f} MW ({gm['gas_mw']/gm['total_mw']*100:.0f}%)
- Nuclear: {gm['nuclear_mw']:,.0f} MW
- Hydro: {gm['hydro_mw']:,.0f} MW
- Total: {gm['total_mw']:,.0f} MW

RECENT NEWS HEADLINES:
{news_text}

When answering:
- Be precise and quantitative where possible
- Connect market mechanics to the numbers above
- If news is relevant, cite specific headlines
- Be concise but substantive — this is for a sophisticated trader
- Do NOT use generic disclaimers like "consult a financial advisor"
"""

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Suggested prompts
        st.markdown("**Quick questions:**")
        q_cols = st.columns(3)
        quick_questions = [
            "What's driving the current spike signal?",
            "How does today's news affect my position?",
            "Explain the DA/RT spread I'm seeing",
            "Should I hedge given current gas prices?",
            "What would a 10°F temp spike do to prices?",
            "Explain LMP mechanics in simple terms",
        ]
        for i, q in enumerate(quick_questions):
            with q_cols[i % 3]:
                if st.button(q, key=f"quick_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})

        st.markdown("---")

        # Display chat history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        if prompt := st.chat_input("Ask about current market conditions, news, trading signals..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing..."):
                    reply = call_ai(
                        messages=[{"role": m["role"], "content": m["content"]}
                                  for m in st.session_state.messages],
                        system=system_prompt,
                        provider=ai_provider,
                        key=ai_key,
                    )
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})

        if st.session_state.messages:
            if st.button("Clear conversation"):
                st.session_state.messages = []
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — NEWS FEED
# ══════════════════════════════════════════════════════════════════════════════

with tab7:
    st.markdown("### 📰 Energy Market News Feed")
    st.caption("Recent headlines relevant to CAISO electricity prices and supply chain.")

    for article in news:
        st.markdown(f"""
<div class="news-card">
  <strong>{article['title']}</strong><br>
  <span style="color:#8b949e; font-size:0.85rem;">{article.get('description', '')}</span><br>
  <span style="color:#58a6ff; font-size:0.78rem;">🕐 {article['published'][:16].replace('T', ' ')} UTC</span>
  {"&nbsp;&nbsp;<a href='" + article['url'] + "' target='_blank' style='color:#58a6ff;'>Read →</a>" if article['url'] != '#' else ''}
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 💡 Market Interpretation")
    st.caption("How the above headlines connect to electricity prices:")

    for article in news[:3]:
        title = article["title"].lower()
        if "heat" in title or "temperature" in title:
            st.info(f"🌡️ **Heat events** → demand surge → LMP spike risk HIGH")
        elif "gas" in title or "lng" in title:
            st.warning(f"🔥 **Gas price movement** → shifts marginal cost of gas peakers → direct LMP impact")
        elif "solar" in title or "wind" in title or "renewable" in title:
            st.success(f"☀️ **High renewable output** → displaces gas → can push LMPs lower or even negative")
        elif "transmission" in title or "outage" in title or "constraint" in title:
            st.error(f"⚡ **Grid constraints** → congestion costs → localized LMP spikes")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
    "GridEdge Intelligence · CAISO NP-15 · Data: CAISO OASIS, Open-Meteo, EIA · "
    "For educational and research purposes only · Not financial advice"
    "</div>",
    unsafe_allow_html=True
)
