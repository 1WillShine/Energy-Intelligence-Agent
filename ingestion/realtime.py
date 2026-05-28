"""
ingestion/realtime.py — Real-time data manager.

Architecture:
  - CAISO RTM LMP: polled every 5 minutes, appended to local parquet
  - Weather: fetched every 60 minutes, merged on hourly boundary
  - Historical bootstrap: pulls 90 days on first run automatically
  - All data stored in data/ directory as parquet (fast, compressed)

Usage:
  from ingestion.realtime import DataManager
  dm = DataManager()
  dm.bootstrap()          # one-time: pulls 90 days of history
  latest = dm.update()    # call every 5 min: returns merged DataFrame
"""

import requests
import pandas as pd
import numpy as np
import zipfile
import io
import time
import os
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

CAISO_NODE   = "TH_NP15_GEN-APND"
SF_LAT       = 37.77
SF_LON       = -122.42
DATA_DIR     = Path("data")
PRICE_FILE   = DATA_DIR / "lmp_history.parquet"
WEATHER_FILE = DATA_DIR / "weather_history.parquet"
MERGED_FILE  = DATA_DIR / "merged.parquet"
BOOTSTRAP_DAYS = 90


# ─── CAISO fetcher ────────────────────────────────────────────────────────────

def _fetch_caiso_interval(start: datetime, end: datetime) -> pd.DataFrame | None:
    """
    Fetch RTM LMP prices for a time interval from CAISO OASIS.
    Returns DataFrame with [datetime, lmp] or None on failure.
    Max interval: 1 day (CAISO API limit).
    """
    params = {
        "queryname":     "PRC_LMP",
        "startdatetime": start.strftime("%Y%m%dT%H:%M-0000"),
        "enddatetime":   end.strftime("%Y%m%dT%H:%M-0000"),
        "version":       1,
        "market_run_id": "RTM",
        "node":          CAISO_NODE,
        "resultformat":  6,
    }
    try:
        resp = requests.get(
            "http://oasis.caiso.com/oasisapi/SingleZip",
            params=params, timeout=30,
        )
        if resp.status_code != 200 or len(resp.content) < 200:
            return None
        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f)

        # CAISO column names vary slightly by query — handle both
        time_col = next((c for c in df.columns if "INTERVALSTART" in c), None)
        val_col  = next((c for c in df.columns if c in ["MW", "LMP_TYPE"]), None)

        if time_col is None:
            return None

        # If MW column exists use it; otherwise look for VALUE
        if "MW" in df.columns:
            df = df[[time_col, "MW"]].copy()
            df.columns = ["datetime", "lmp"]
        elif "VALUE" in df.columns:
            df = df[[time_col, "VALUE"]].copy()
            df.columns = ["datetime", "lmp"]
        else:
            return None

        df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.floor("h")
        df["lmp"] = pd.to_numeric(df["lmp"], errors="coerce")
        df = df.dropna()
        df = df.groupby("datetime")["lmp"].mean().reset_index()
        return df

    except Exception as e:
        log.warning(f"CAISO fetch failed: {e}")
        return None


def fetch_caiso_history(days: int = 90) -> pd.DataFrame:
    """
    Bootstrap: pull N days of historical LMP data.
    Fetches day by day, skips failures, falls back to synthetic.
    """
    end   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    frames  = []
    current = start
    total   = 0

    log.info(f"Bootstrapping {days} days of CAISO history...")

    while current < end:
        next_day = current + timedelta(days=1)
        df = _fetch_caiso_interval(current, min(next_day, end))
        if df is not None and len(df) > 0:
            frames.append(df)
            total += len(df)
        current = next_day
        time.sleep(0.5)  # polite rate limiting

    if frames:
        result = pd.concat(frames, ignore_index=True)
        result = result.drop_duplicates("datetime").sort_values("datetime")
        log.info(f"Fetched {len(result)} real hourly LMP rows from CAISO")
        return result
    else:
        log.warning("CAISO API unavailable — using synthetic history")
        return _synthetic_prices(days)


def fetch_caiso_latest(lookback_hours: int = 2) -> pd.DataFrame | None:
    """
    Fetch the most recent N hours of RTM LMP.
    Called every 5 minutes for real-time updates.
    """
    end   = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(hours=lookback_hours)
    return _fetch_caiso_interval(start, end)


# ─── Weather fetcher ──────────────────────────────────────────────────────────

def fetch_weather_history(days: int = 90) -> pd.DataFrame:
    """Fetch N days of historical hourly weather from Open-Meteo archive."""
    end   = datetime.utcnow() - timedelta(days=1)
    start = end - timedelta(days=days)
    params = {
        "latitude":         SF_LAT,
        "longitude":        SF_LON,
        "hourly":           "temperature_2m,windspeed_10m,shortwave_radiation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "start_date":       start.strftime("%Y-%m-%d"),
        "end_date":         end.strftime("%Y-%m-%d"),
        "timezone":         "UTC",
    }
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params, timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()["hourly"]
        return pd.DataFrame({
            "datetime":  pd.to_datetime(data["time"], utc=True),
            "temp_f":    data["temperature_2m"],
            "wind_mph":  data["windspeed_10m"],
            "solar_rad": data["shortwave_radiation"],
        })
    except Exception as e:
        log.warning(f"Weather history fetch failed: {e}")
        return _synthetic_weather(days)


def fetch_weather_current() -> pd.DataFrame:
    """
    Fetch current + 7-day forecast weather.
    Called every 60 minutes.
    """
    params = {
        "latitude":         SF_LAT,
        "longitude":        SF_LON,
        "hourly":           "temperature_2m,windspeed_10m,shortwave_radiation",
        "temperature_unit": "fahrenheit",
        "windspeed_unit":   "mph",
        "forecast_days":    7,
        "timezone":         "UTC",
    }
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params=params, timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()["hourly"]
        return pd.DataFrame({
            "datetime":  pd.to_datetime(data["time"], utc=True),
            "temp_f":    data["temperature_2m"],
            "wind_mph":  data["windspeed_10m"],
            "solar_rad": data["shortwave_radiation"],
        })
    except Exception as e:
        log.warning(f"Weather forecast fetch failed: {e}")
        return None


# ─── Gas prices ───────────────────────────────────────────────────────────────

def fetch_gas_prices(eia_key: str = "") -> pd.DataFrame:
    """
    Fetch Henry Hub gas prices from EIA.
    Falls back to synthetic if no key provided.
    Get free key at: https://www.eia.gov/opendata/register.php
    """
    if not eia_key:
        return _synthetic_gas()
    url = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"
    params = {
        "frequency":        "daily",
        "data[0]":          "value",
        "facets[series][]": "RNGC1",
        "sort[0][column]":  "period",
        "sort[0][direction]":"desc",
        "length":           180,
        "api_key":          eia_key,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        data = resp.json()["response"]["data"]
        df = pd.DataFrame(data)[["period", "value"]].copy()
        df.columns = ["date", "gas_price"]
        df["date"]      = pd.to_datetime(df["date"])
        df["gas_price"] = pd.to_numeric(df["gas_price"], errors="coerce")
        return df.dropna().sort_values("date").reset_index(drop=True)
    except Exception as e:
        log.warning(f"EIA fetch failed: {e}")
        return _synthetic_gas()


# ─── Synthetic fallbacks ──────────────────────────────────────────────────────

def _synthetic_prices(days: int = 90) -> pd.DataFrame:
    """Calibrated synthetic LMP when CAISO API is unavailable."""
    np.random.seed(42)
    hours = days * 24
    idx   = pd.date_range(
        end=datetime.utcnow().replace(minute=0, second=0, microsecond=0),
        periods=hours, freq="h", tz="UTC",
    )
    hour  = idx.hour
    month = idx.month
    dow   = idx.dayofweek

    base     = 20 + np.exp(-0.5*((hour-8)/1.5)**2)*25 + np.exp(-0.5*((hour-18)/2)**2)*35
    seasonal = 1.0 + 0.35 * np.sin((month - 3) * np.pi / 6)
    wkend    = np.where(dow >= 5, 0.82, 1.0)

    gas_shock = np.zeros(hours)
    for i in range(1, hours):
        gas_shock[i] = 0.995 * gas_shock[i-1] + np.random.normal(0, 0.03)

    lmp = (base * seasonal * wkend
           + (gas_shock) * 8
           + np.random.normal(0, 6, hours))
    spikes = np.random.choice([0]*9+[1], hours) * np.random.lognormal(4.5, 0.8, hours)
    lmp = lmp + spikes

    return pd.DataFrame({"datetime": idx, "lmp": lmp.round(4)})


def _synthetic_weather(days: int = 90) -> pd.DataFrame:
    """Synthetic weather fallback."""
    np.random.seed(43)
    hours = days * 24
    idx   = pd.date_range(
        end=datetime.utcnow().replace(minute=0, second=0, microsecond=0),
        periods=hours, freq="h", tz="UTC",
    )
    month = idx.month
    hour  = idx.hour
    temp  = (58 + 12 * np.sin((month-3)*np.pi/6)
             + 8 * np.sin((hour-14)*np.pi/12)
             + np.random.normal(0, 4, hours))
    solar = np.maximum(0, np.sin((hour-6)*np.pi/12)) * 400 + np.random.normal(0, 30, hours)
    wind  = np.abs(np.random.normal(8, 4, hours))
    return pd.DataFrame({
        "datetime":  idx,
        "temp_f":    temp.round(1),
        "wind_mph":  wind.round(1),
        "solar_rad": np.maximum(solar, 0).round(1),
    })


def _synthetic_gas(days: int = 180) -> pd.DataFrame:
    np.random.seed(44)
    dates  = pd.date_range(end=datetime.utcnow(), periods=days, freq="D")
    prices = 3.5 + np.cumsum(np.random.normal(0, 0.05, days))
    return pd.DataFrame({"date": dates, "gas_price": np.maximum(prices, 1.5).round(4)})


# ─── Merge ────────────────────────────────────────────────────────────────────

def _merge_and_enrich(prices: pd.DataFrame, weather: pd.DataFrame,
                      gas: pd.DataFrame) -> pd.DataFrame:
    """
    Merge price + weather on hourly datetime.
    Add gas price (daily → forward-filled to hourly).
    Add rolling finance features.
    """
    # Ensure clean hourly timestamps on both sides
    prices  = prices.copy()
    weather = weather.copy()
    prices["datetime"]  = pd.to_datetime(prices["datetime"],  utc=True).dt.floor("h")
    weather["datetime"] = pd.to_datetime(weather["datetime"], utc=True).dt.floor("h")

    merged = pd.merge(prices, weather, on="datetime", how="inner")

    # Forward-fill daily gas price to hourly
    gas = gas.copy()
    gas["datetime"] = pd.to_datetime(gas["date"], utc=True)
    gas_hourly = (
        pd.DataFrame({"datetime": merged["datetime"]})
        .merge(gas[["datetime", "gas_price"]], on="datetime", how="left")
    )
    gas_hourly["gas_price"] = gas_hourly["gas_price"].ffill().bfill()
    merged["gas_price"] = gas_hourly["gas_price"].values

    # Time features
    merged["hour"] = merged["datetime"].dt.hour
    merged["dow"]  = merged["datetime"].dt.dayofweek

    # Finance features
    merged["lmp_roll24"]  = merged["lmp"].rolling(24).mean()
    merged["lmp_std24"]   = merged["lmp"].rolling(24).std()
    merged["lmp_zscore"]  = ((merged["lmp"] - merged["lmp_roll24"])
                             / (merged["lmp_std24"] + 1e-9))
    merged["volatility"]  = (merged["lmp"].rolling(24).std()
                             / (merged["lmp"].rolling(24).mean() + 1e-9))
    merged["spike"]       = (merged["lmp_zscore"] > 2.0).astype(int)

    return merged.sort_values("datetime").reset_index(drop=True)


# ─── DataManager ─────────────────────────────────────────────────────────────

class DataManager:
    """
    Manages all data fetching, caching, and merging.

    Usage in app.py:
        dm = DataManager(eia_key="your_key")
        dm.bootstrap()           # run once at startup
        merged = dm.update()     # run every 5 minutes
        forecast = dm.get_forecast()  # current weather forecast
    """

    def __init__(self, eia_key: str = ""):
        self.eia_key = eia_key
        self._last_weather_fetch: datetime | None = None
        self._weather_cache: pd.DataFrame | None  = None
        self._gas_cache: pd.DataFrame | None       = None
        DATA_DIR.mkdir(exist_ok=True)

    # ── Bootstrap (one-time) ─────────────────────────────────────────────────

    def bootstrap(self, force: bool = False) -> pd.DataFrame:
        """
        Pull 90 days of history and save to disk.
        Skipped if local cache already exists (unless force=True).
        Returns merged DataFrame.
        """
        if MERGED_FILE.exists() and not force:
            log.info("Local cache found — loading from disk")
            return pd.read_parquet(MERGED_FILE)

        log.info("No cache found — bootstrapping 90 days of history...")

        prices  = fetch_caiso_history(BOOTSTRAP_DAYS)
        weather = fetch_weather_history(BOOTSTRAP_DAYS)
        gas     = fetch_gas_prices(self.eia_key)

        prices.to_parquet(PRICE_FILE, index=False)
        weather.to_parquet(WEATHER_FILE, index=False)

        merged = _merge_and_enrich(prices, weather, gas)
        merged.to_parquet(MERGED_FILE, index=False)

        self._gas_cache = gas
        log.info(f"Bootstrap complete: {len(merged)} merged rows")
        return merged

    # ── Real-time update (every 5 min) ────────────────────────────────────────

    def update(self) -> pd.DataFrame:
        """
        Fetch latest CAISO prices, merge with cached weather.
        Appends new rows to local parquet, returns full merged DataFrame.
        """
        # 1. Fetch latest prices (last 2 hours to catch any gaps)
        new_prices = fetch_caiso_latest(lookback_hours=2)

        if new_prices is not None and len(new_prices) > 0:
            # Append to local price history (dedup on datetime)
            if PRICE_FILE.exists():
                existing = pd.read_parquet(PRICE_FILE)
                combined = pd.concat([existing, new_prices], ignore_index=True)
                combined = combined.drop_duplicates("datetime").sort_values("datetime")
            else:
                combined = new_prices
            combined.to_parquet(PRICE_FILE, index=False)
            log.info(f"Appended {len(new_prices)} new price rows")
        else:
            log.warning("No new price data — using cached prices")
            if PRICE_FILE.exists():
                combined = pd.read_parquet(PRICE_FILE)
            else:
                combined = _synthetic_prices(BOOTSTRAP_DAYS)

        # 2. Refresh weather every 60 minutes
        now = datetime.utcnow()
        if (self._last_weather_fetch is None
                or (now - self._last_weather_fetch).seconds > 3600):
            new_weather = fetch_weather_current()
            if new_weather is not None:
                if WEATHER_FILE.exists():
                    existing_w = pd.read_parquet(WEATHER_FILE)
                    combined_w = pd.concat([existing_w, new_weather], ignore_index=True)
                    combined_w = (combined_w
                                  .drop_duplicates("datetime")
                                  .sort_values("datetime"))
                else:
                    combined_w = new_weather
                combined_w.to_parquet(WEATHER_FILE, index=False)
                self._weather_cache = combined_w
                self._last_weather_fetch = now
                log.info("Weather cache refreshed")
        else:
            combined_w = (pd.read_parquet(WEATHER_FILE)
                          if WEATHER_FILE.exists()
                          else _synthetic_weather(BOOTSTRAP_DAYS))

        # 3. Gas prices (daily — refresh once per day is enough)
        if self._gas_cache is None:
            self._gas_cache = fetch_gas_prices(self.eia_key)

        # 4. Merge and enrich
        merged = _merge_and_enrich(combined, combined_w, self._gas_cache)
        merged.to_parquet(MERGED_FILE, index=False)
        return merged

    def get_forecast(self) -> pd.DataFrame:
        """Return 7-day weather forecast DataFrame."""
        return fetch_weather_current() or _synthetic_weather(7)

    def get_gas(self) -> pd.DataFrame:
        if self._gas_cache is None:
            self._gas_cache = fetch_gas_prices(self.eia_key)
        return self._gas_cache

    def get_news(self) -> list:
        """Fetch latest energy news headlines."""
        from pipeline import fetch_energy_news
        return fetch_energy_news()
