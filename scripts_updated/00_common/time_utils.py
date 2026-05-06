"""Time utilities for the DER dataset pipeline."""
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


START_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
TOTAL_ROWS = 604_800
TIMESTEP_S = 1


def build_time_index() -> pd.DatetimeIndex:
    """Return the full 7-day 1-second UTC DatetimeIndex."""
    return pd.date_range(
        start=START_UTC,
        periods=TOTAL_ROWS,
        freq="1s",
        tz="UTC",
        name="timestamp_utc",
    )


def build_time_s_array() -> np.ndarray:
    """Return integer array [0, 1, ..., 604799]."""
    return np.arange(TOTAL_ROWS, dtype=np.int64)


def time_s_to_utc(time_s: int) -> str:
    """Convert integer seconds-since-epoch offset to UTC ISO string."""
    t = START_UTC + pd.Timedelta(seconds=int(time_s))
    return t.isoformat()


def utc_str_to_time_s(utc_str: str) -> int:
    """Convert UTC ISO string to seconds offset from start."""
    t = pd.Timestamp(utc_str, tz="UTC")
    return int((t - START_UTC).total_seconds())


def seconds_of_day(time_s_array: np.ndarray) -> np.ndarray:
    """Return time-of-day in seconds for each element."""
    return time_s_array % 86400


def day_of_week(time_s_array: np.ndarray) -> np.ndarray:
    """Return day index [0..6] for each element."""
    return time_s_array // 86400


def solar_irradiance_pu(time_s_array: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Generate synthetic 1-second solar irradiance [0..1] for 7 days.
    Sunrise ~6 AM, solar noon ~12 PM, sunset ~18 PM.
    Adds realistic cloud-like perturbations.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    tod = seconds_of_day(time_s_array).astype(np.float64)
    day = day_of_week(time_s_array)

    # Base sinusoidal solar profile  (zero outside [6h, 18h])
    sunrise_s = 6 * 3600.0
    sunset_s = 18 * 3600.0
    solar_duration_s = sunset_s - sunrise_s

    base = np.where(
        (tod >= sunrise_s) & (tod <= sunset_s),
        np.sin(np.pi * (tod - sunrise_s) / solar_duration_s),
        0.0,
    )

    # Day-to-day diversity: different peak irradiance per day
    day_peak = np.array([0.92, 0.85, 0.97, 0.78, 0.90, 0.95, 0.82])
    base = base * day_peak[day]

    # Cloud perturbation: low-pass filtered noise
    n = len(time_s_array)
    raw_noise = rng.normal(0, 0.08, n)
    # Simple exponential smoothing to create correlated cloud noise
    alpha = 0.002  # very slow changes (about 500s time constant)
    cloud = np.zeros(n)
    cloud[0] = 0.0
    for i in range(1, n):
        cloud[i] = (1 - alpha) * cloud[i - 1] + alpha * raw_noise[i]

    irr = np.clip(base + cloud * base, 0.0, 1.0)

    # Ensure night is truly zero
    irr = np.where(base <= 0.001, 0.0, irr)
    return irr


def temperature_c(time_s_array: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Synthetic ambient temperature: 5–30°C, daily cycle + noise."""
    if rng is None:
        rng = np.random.default_rng(43)
    tod = seconds_of_day(time_s_array).astype(np.float64)
    day = day_of_week(time_s_array)
    day_base = np.array([15.0, 12.0, 18.0, 10.0, 16.0, 20.0, 14.0])
    base = day_base[day] + 8.0 * np.sin(np.pi * (tod / 86400 - 0.25))
    noise = rng.normal(0, 0.5, len(time_s_array))
    return np.clip(base + noise, -5.0, 45.0)


def load_profile_kw(time_s_array: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """
    Synthetic local load at PCC bus (kW).
    Range: ~25 to ~150 kW with daily and weekly variation.
    """
    if rng is None:
        rng = np.random.default_rng(44)
    tod = seconds_of_day(time_s_array).astype(np.float64)
    day = day_of_week(time_s_array)

    # Weekday vs weekend base load
    weekend_mask = (day >= 5)
    base_kw = np.where(weekend_mask, 55.0, 80.0)

    # Morning ramp-up (7-9 AM), evening peak (5-8 PM), overnight low
    morning_peak = 40.0 * np.clip(np.sin(np.pi * (tod - 6 * 3600) / (4 * 3600)), 0, 1)
    evening_peak = 50.0 * np.clip(np.sin(np.pi * (tod - 15 * 3600) / (5 * 3600)), 0, 1)
    profile = base_kw + morning_peak + evening_peak

    # High-freq noise
    noise = rng.normal(0, 3.0, len(time_s_array))
    return np.clip(profile + noise, 20.0, 160.0)
