"""
pipeline.py — fetch and cache all data sources.

Sources:
  - CAISO OASIS: LMP prices + generation mix
  - Open-Meteo: hourly weather (temperature, wind, solar radiation)
  - EIA: natural gas prices
  - NewsAPI: energy-related headlines (for AI agent context)

Usage:
  from pipeline import fetch_all
  data = fetch_all()
"""

import requests
import pandas as pd
import time
import zipfile
import io
from datetime import datetime, timedelta
import numpy as np

CAISO_NODE = "TH_NP15_GEN-APND"
SF_LAT, SF_LON = 37.77, -122.42


# ─── CAISO LMP Prices ────────────────────────────────────────────────────────

def _fetch_caiso_day(date):
    """Fetch 5-min RTM LMP prices for one day. Returns DataFrame or None."""
    start = date.strftime("%Y%m%dT00:00-0000")
    end = (date + timedelta(days=1)).strftime("%Y%m%dT00:00-0000")
    params = {
        "queryname": "PRC_LMP",
        "startdatetime": start,
        "enddatetime": end,
        "version": 1,
        "market_run_id": "RTM",
        "node": CAISO_NODE,
        "resultformat": 6,
    }
    try:
        resp = requests.get(
            "http://oasis.caiso.com/oasisapi/SingleZip",
            params=params, timeout=30
        )
        if resp.status_code != 200 or len(resp.content) < 100:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)
        df = df[["INTERVALSTARTTIME_GMT", "MW"]].copy()
        df.columns = ["datetime", "lmp"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        return df
    except Exception:
        return None


def fetch_caiso_prices(days=90):
    """Fetch hourly LMP prices for the last N days."""
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days)
    frames, current = [], start
    print(f"Fetching CAISO prices ({days} days)...")
    while current <= end:
        df = _fetch_caiso_day(current)
        if df is not None:
            frames.append(df)
        current += timedelta(days=1)
        time.sleep(0.3)
    if not frames:
        print("  → CAISO unavailable, using synthetic prices")
        return _synthetic_prices(days)
    result = pd.concat(frames, ignore_index=True)
    result["datetime"] = result["datetime"].dt.floor("h")
    result = result.groupby("datetime")["lmp"].mean().reset_index()
    print(f"  → {len(result)} hourly rows")
    return result


def _synthetic_prices(days=90):
    """Realistic synthetic LMP prices when CAISO API is unavailable."""
    hours = days * 24
    idx = pd.date_range(end=datetime.utcnow(), periods=hours, freq="h", tz="UTC")
    hour = idx.hour
    # Typical dual-peak electricity price shape (morning + evening peaks)
    base = 35 + 20 * np.sin((hour - 6) * np.pi / 12)
    noise = np.random.normal(0, 8, hours)
    # Occasional spikes
    spikes = np.random.choice([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], hours) * np.random.uniform(50, 200, hours)
    lmp = base + noise + spikes
    return pd.DataFrame({"datetime": idx, "lmp": lmp})


# ─── CAISO Generation Mix ─────────────────────────────────────────────────────

def fetch_caiso_genmix(days=7):
    """
    Return hourly generation mix (solar, wind, gas, nuclear, hydro).
    Uses synthetic data — CAISO's public mix endpoint requires specific formatting.
    Replace with real CAISO Renewables Watch data if needed.
    """
    print(f"Building generation mix ({days} days)...")
    hours = days * 24
    idx = pd.date_range(end=datetime.utcnow(), periods=hours, freq="h", tz="UTC")
    hour = idx.hour
    # Solar follows a bell curve peaking at noon
    solar = np.maximum(0, np.sin((hour - 6) * np.pi / 12) * 9000 + np.random.normal(0, 400, hours))
    wind = np.abs(np.random.normal(4500, 1200, hours))
    nuclear = np.full(hours, 2200.0)
    hydro = np.abs(np.random.normal(3200, 600, hours))
    gas = np.abs(np.random.normal(11000, 2500, hours))
    total = solar + wind + nuclear + hydro + gas
    df = pd.DataFrame({
        "datetime": idx,
        "solar_mw": solar,
        "wind_mw": wind,
        "nuclear_mw": nuclear,
        "hydro_mw": hydro,
        "gas_mw": gas,
        "total_mw": total,
    })
    print(f"  → {len(df)} rows")
    return df


# ─── Weather ──────────────────────────────────────────────────────────────────

def fetch_weather(days=90):
    """Fetch hourly temperature, wind, solar radiation (historical)."""
    end = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days)
    print("Fetching historical weather...")
    params = {
        "latitude": SF_LAT,
        "longitude": SF_LON,
        "hourly": "temperature_2m,windspeed_10m,shortwave_radiation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": end.strftime("%Y-%m-%d"),
        "timezone": "UTC",
    }
    resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    df = pd.DataFrame({
        "datetime": pd.to_datetime(data["time"], utc=True),
        "temp_f": data["temperature_2m"],
        "wind_mph": data["windspeed_10m"],
        "solar_rad": data["shortwave_radiation"],
    })
    print(f"  → {len(df)} rows")
    return df


def fetch_weather_forecast():
    """Fetch 7-day hourly weather forecast."""
    params = {
        "latitude": SF_LAT,
        "longitude": SF_LON,
        "hourly": "temperature_2m,windspeed_10m,shortwave_radiation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit": "mph",
        "forecast_days": 7,
        "timezone": "UTC",
    }
    resp = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["hourly"]
    return pd.DataFrame({
        "datetime": pd.to_datetime(data["time"], utc=True),
        "temp_f": data["temperature_2m"],
        "wind_mph": data["windspeed_10m"],
        "solar_rad": data["shortwave_radiation"],
    })


# ─── EIA Natural Gas Prices ───────────────────────────────────────────────────

def fetch_gas_prices():
    """Fetch Henry Hub natural gas spot prices from EIA Open Data."""
    print("Fetching EIA gas prices...")
    # Free EIA key: register at https://www.eia.gov/opendata/register.php
    EIA_KEY = "YOUR_EIA_KEY"
    if EIA_KEY == "YOUR_EIA_KEY":
        print("  → No EIA key, using synthetic gas prices")
        return _synthetic_gas()
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    params = {
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": "RNGC1",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "length": 180,
        "api_key": EIA_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()["response"]["data"]
        df = pd.DataFrame(data)[["period", "value"]].copy()
        df.columns = ["date", "gas_price"]
        df["date"] = pd.to_datetime(df["date"])
        df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce")
        df = df.dropna().sort_values("date").reset_index(drop=True)
        print(f"  → {len(df)} daily rows")
        return df
    except Exception:
        print("  → EIA API error, using synthetic gas prices")
        return _synthetic_gas()


def _synthetic_gas():
    dates = pd.date_range(end=datetime.utcnow(), periods=180, freq="D")
    prices = 2.5 + np.cumsum(np.random.normal(0, 0.05, 180))
    return pd.DataFrame({"date": dates, "gas_price": np.maximum(prices, 1.5)})


# ─── Energy News ─────────────────────────────────────────────────────────────

def fetch_energy_news():
    """
    Fetch recent energy headlines.
    Set NEWS_API_KEY to your free key from https://newsapi.org
    Falls back to realistic placeholder headlines.
    """
    NEWS_API_KEY = "YOUR_NEWSAPI_KEY"
    if NEWS_API_KEY == "YOUR_NEWSAPI_KEY":
        return _placeholder_news()
    try:
        params = {
            "q": "electricity price energy grid California gas",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 8,
            "apiKey": NEWS_API_KEY,
        }
        resp = requests.get("https://newsapi.org/v2/everything", params=params, timeout=10)
        articles = resp.json().get("articles", [])
        return [{"title": a["title"], "description": a["description"],
                 "url": a["url"], "published": a["publishedAt"]} for a in articles]
    except Exception:
        return _placeholder_news()


def _placeholder_news():
    now = datetime.utcnow().isoformat()
    return [
        {"title": "California grid operator warns of tight supply amid heat wave",
         "description": "CAISO issued a Flex Alert as temperatures hit 105°F across the Central Valley, pushing demand to seasonal highs.",
         "url": "#", "published": now},
        {"title": "Natural gas futures surge 4.2% on LNG export demand",
         "description": "Henry Hub climbed sharply as Gulf Coast LNG export terminals operate near capacity.",
         "url": "#", "published": now},
        {"title": "Solar generation breaks California record at 18.6 GW",
         "description": "Renewable output hit an all-time high on Sunday, briefly pushing real-time LMPs negative.",
         "url": "#", "published": now},
        {"title": "Heat dome forecast to persist through next week — NOAA",
         "description": "Models show above-normal temperatures across the West for at least 10 days.",
         "url": "#", "published": now},
        {"title": "PG&E reports transmission constraint on Path 26",
         "description": "Congestion costs elevated as north-south transfer capacity is reduced for maintenance.",
         "url": "#", "published": now},
        {"title": "Wind generation drops 40% as high-pressure system stalls",
         "description": "Low wind across the Tehachapi and Altamont passes has increased reliance on gas peakers.",
         "url": "#", "published": now},
    ]


# ─── Master Fetch ─────────────────────────────────────────────────────────────

def fetch_all(price_days=90, mix_days=7):
    """
    Fetch all data. Returns dict of DataFrames + news list.
    Keys: prices, weather, forecast, genmix, gas, news, merged
    """
    prices = fetch_caiso_prices(days=price_days)
    weather = fetch_weather(days=price_days)
    forecast = fetch_weather_forecast()
    genmix = fetch_caiso_genmix(days=mix_days)
    gas = fetch_gas_prices()
    news = fetch_energy_news()

    # Merge prices + weather
    merged = pd.merge(prices, weather, on="datetime", how="inner")
    merged["hour"] = merged["datetime"].dt.hour
    merged["dow"] = merged["datetime"].dt.dayofweek

    # Finance-layer rolling stats
    merged["lmp_roll24"] = merged["lmp"].rolling(24).mean()
    merged["lmp_std24"] = merged["lmp"].rolling(24).std()
    merged["lmp_zscore"] = (merged["lmp"] - merged["lmp_roll24"]) / (merged["lmp_std24"] + 1e-9)
    merged["volatility"] = merged["lmp"].rolling(24).std() / (merged["lmp"].rolling(24).mean() + 1e-9)
    merged["spike"] = (merged["lmp_zscore"] > 2.0).astype(int)

    return {
        "prices": prices,
        "weather": weather,
        "forecast": forecast,
        "genmix": genmix,
        "gas": gas,
        "news": news,
        "merged": merged,
    }
