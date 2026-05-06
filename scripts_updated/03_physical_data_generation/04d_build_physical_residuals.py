"""
Compute physical residuals (attacked - clean) for anomaly detection.
Writes: data_updated/raw/physical_residuals_improved_7d.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import (
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, RESIDUALS_CSV,
)

DELTA_COLS = [
    ("pv_actual_p_kw", "delta_pv_p_kw"),
    ("pv_actual_q_kvar", "delta_pv_q_kvar"),
    ("bess_actual_p_kw", "delta_bess_p_kw"),
    ("bess_actual_q_kvar", "delta_bess_q_kvar"),
    ("bess_soc_percent", "delta_bess_soc_percent"),
    ("pcc_v_a_pu", "delta_pcc_v_a_pu"),
    ("pcc_v_b_pu", "delta_pcc_v_b_pu"),
    ("pcc_v_c_pu", "delta_pcc_v_c_pu"),
    ("pcc_i_a_amp", "delta_pcc_i_a_amp"),
    ("pcc_i_b_amp", "delta_pcc_i_b_amp"),
    ("pcc_i_c_amp", "delta_pcc_i_c_amp"),
    ("pcc_p_kw", "delta_pcc_p_kw"),
    ("pcc_q_kvar", "delta_pcc_q_kvar"),
]


def main() -> pd.DataFrame:
    if not CLEAN_PHYSICAL_CSV.exists():
        print("ERROR: Clean CSV missing.")
        sys.exit(1)
    if not ATTACKED_PHYSICAL_CSV.exists():
        print("ERROR: Attacked CSV missing.")
        sys.exit(1)

    print("Computing physical residuals (attacked - clean)...")
    clean = pd.read_csv(CLEAN_PHYSICAL_CSV)
    attacked = pd.read_csv(ATTACKED_PHYSICAL_CSV)

    assert len(clean) == len(attacked), (
        f"Row count mismatch: clean={len(clean)}, attacked={len(attacked)}"
    )
    assert (clean["time_s"].values == attacked["time_s"].values).all(), \
        "time_s alignment mismatch between clean and attacked"

    residuals = pd.DataFrame({
        "timestamp_utc": clean["timestamp_utc"],
        "time_s": clean["time_s"],
    })

    for src_col, dst_col in DELTA_COLS:
        if src_col in clean.columns and src_col in attacked.columns:
            residuals[dst_col] = (
                attacked[src_col].values - clean[src_col].values
            )
        else:
            residuals[dst_col] = 0.0

    # Add context columns from attacked dataset
    for col in ["physical_effect_active_flag", "physical_scenario_id",
                "physical_effect_type", "generation_method"]:
        if col in attacked.columns:
            residuals[col] = attacked[col].values

    # Derive anomaly_type from physical_effect_type
    residuals["anomaly_type"] = residuals.get(
        "physical_effect_type",
        pd.Series(["none"] * len(residuals))
    )

    # Scenario_id alias
    if "physical_scenario_id" in residuals.columns:
        residuals["scenario_id"] = residuals["physical_scenario_id"]

    RESIDUALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    residuals.to_csv(RESIDUALS_CSV, index=False)
    print(f"Physical residuals: {len(residuals)} rows, {len(residuals.columns)} columns")
    print(f"Saved: {RESIDUALS_CSV}")

    # Summary of non-zero residuals
    for src_col, dst_col in DELTA_COLS:
        if dst_col in residuals.columns:
            nz = (residuals[dst_col].abs() > 0.01).sum()
            if nz > 0:
                print(f"  {dst_col}: {nz} non-zero rows")
    return residuals


if __name__ == "__main__":
    main()
