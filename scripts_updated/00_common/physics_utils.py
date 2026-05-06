"""
Shared physics utilities used across physical data generation scripts.
Placed here to avoid circular imports between numbered script modules.
"""
import numpy as np


def compute_soc(bess_p_kw: np.ndarray, meta: dict, initial_soc: float = None) -> np.ndarray:
    """Forward-simulate BESS SOC from power timeseries."""
    cap_kwh = meta["bess_capacity_kwh"]
    eff_ch = meta["bess_eff_charge_percent"] / 100.0
    eff_dis = meta["bess_eff_discharge_percent"] / 100.0
    soc_min = meta["bess_soc_min_percent"]
    soc_max = meta["bess_soc_max_percent"]
    if initial_soc is None:
        initial_soc = meta.get("bess_initial_soc_percent", 50.0)

    soc = np.empty(len(bess_p_kw))
    soc[0] = initial_soc
    stored_kwh = initial_soc / 100.0 * cap_kwh

    for i in range(1, len(bess_p_kw)):
        p = bess_p_kw[i - 1]
        dt_h = 1.0 / 3600.0
        if p > 0:
            delta = p * dt_h / eff_dis
            stored_kwh = max(stored_kwh - delta, cap_kwh * soc_min / 100.0)
        elif p < 0:
            delta = abs(p) * dt_h * eff_ch
            stored_kwh = min(stored_kwh + delta, cap_kwh * soc_max / 100.0)
        stored_kwh -= 0.0001 * dt_h * cap_kwh
        stored_kwh = np.clip(stored_kwh, cap_kwh * soc_min / 100.0, cap_kwh * soc_max / 100.0)
        soc[i] = stored_kwh / cap_kwh * 100.0

    return soc


def compute_voltage_unbalance(va: np.ndarray, vb: np.ndarray, vc: np.ndarray):
    """
    Compute three-phase voltage unbalance per IEEE definition.
    Returns (mean_v, unbalance_pu, status_array).
    """
    mean_v = (va + vb + vc) / 3.0
    unb = np.maximum.reduce([np.abs(va - mean_v), np.abs(vb - mean_v), np.abs(vc - mean_v)])
    status = np.where(unb >= 0.030, "critical",
              np.where(unb >= 0.020, "warning",
              np.where(unb >= 0.012, "caution", "normal")))
    return mean_v, unb, status
