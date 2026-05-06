"""
Window builder for Phase 1 anomaly detection benchmark.

Reads:
  D:/updated_dataset/data_updated/raw/physical_timeseries_clean_improved_7d.csv
  D:/updated_dataset/data_updated/raw/physical_timeseries_attacked_improved_7d.csv
  D:/updated_dataset/data_updated/scenarios/scenario_manifest_improved_7d.csv

Writes:
  D:/updated_dataset/models/windows/windows_normal.parquet
  D:/updated_dataset/models/windows/windows_attacked.parquet
  D:/updated_dataset/models/windows/windows_all.parquet
  D:/updated_dataset/models/windows/feature_manifest.json
  D:/updated_dataset/models/windows/window_build_summary.json
  D:/updated_dataset/models/windows/split_manifest.json
  D:/updated_dataset/models/windows/sequence_windows_normal.npz
  D:/updated_dataset/models/windows/sequence_windows_attacked.npz
  D:/updated_dataset/models/windows/sequence_windows_all.npz
  D:/updated_dataset/models/reports/WINDOW_BUILD_REPORT.md

Window IDs are globally unique:
  clean-source windows:    clean_w0000000, clean_w0000001, ...
  attacked-source windows: attacked_w0000000, attacked_w0000001, ...

source_dataset column is preserved in all parquet files and NPZ arrays so
downstream training can filter to clean-source normal windows for unsupervised
anomaly detection (no contamination from attacked-source normal windows).

Primary benchmark: 60s window / 10s stride.
Secondary sweep (threshold, IF, PCA, MLP only): 30s/10s and 120s/30s.

Leakage prevention: only RAW_FEATURES used in feature tensors.
Labels come from physical_effect_active_flag in attacked CSV.
No cyber log columns used.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    CLEAN_CSV, ATTACKED_CSV, SCENARIO_MANIFEST_CSV,
    WINDOWS_DIR, REPORTS_DIR, RESULTS_DIR,
    WINDOWS_NORMAL_PARQUET, WINDOWS_ATTACKED_PARQUET, WINDOWS_ALL_PARQUET,
    FEATURE_MANIFEST_JSON, WINDOW_BUILD_SUMMARY_JSON, SPLIT_MANIFEST_JSON,
    SEQ_NORMAL_NPZ, SEQ_ATTACKED_NPZ, SEQ_ALL_NPZ,
    ensure_all_dirs,
)
from feature_config import (
    RAW_FEATURES, N_RAW_FEATURES, FLAT_FEATURE_NAMES, WINDOW_STATS,
    N_FLAT_FEATURES, LEAKAGE_COLUMNS,
)

# Primary benchmark config
PRIMARY_WINDOW_S = 60
PRIMARY_STRIDE_S = 10

# Secondary sweep (simpler models only)
SECONDARY_CONFIGS = [
    {"window_s": 30, "stride_s": 10, "suffix": "30s"},
    {"window_s": 120, "stride_s": 30, "suffix": "120s"},
]

# Split fractions (time-ordered)
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20


def _compute_slope(arr: np.ndarray) -> float:
    """Linear regression slope over the window."""
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float32)
    x -= x.mean()
    denom = (x * x).sum()
    if denom == 0:
        return 0.0
    return float((x * arr).sum() / denom)


def _compute_rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr ** 2)))


def compute_window_features(chunk: np.ndarray) -> np.ndarray:
    """Compute 10 statistics for each of N_RAW_FEATURES in the chunk.

    chunk: shape (window_s, N_RAW_FEATURES) float32
    Returns: shape (N_FLAT_FEATURES,) = N_RAW_FEATURES x N_STATS
    """
    feats = []
    for f_idx in range(N_RAW_FEATURES):
        col = chunk[:, f_idx]
        feats.extend([
            float(np.mean(col)),
            float(np.std(col)),
            float(np.min(col)),
            float(np.max(col)),
            float(np.median(col)),
            _compute_slope(col),
            float(col[0]),
            float(col[-1]),
            float(col[-1] - col[0]),
            _compute_rms(col),
        ])
    return np.array(feats, dtype=np.float32)


def load_physical_csv(path: Path) -> pd.DataFrame:
    """Load physical CSV with only needed columns, optimizing dtype."""
    print(f"  Loading {path.name}...")
    t0 = time.time()
    needed = (RAW_FEATURES +
              ["time_s", "timestamp_utc", "generation_method",
               "physical_effect_active_flag", "physical_scenario_id", "physical_effect_type"])
    available = list(pd.read_csv(path, nrows=0).columns)
    use_cols = [c for c in needed if c in available]
    df = pd.read_csv(path, usecols=use_cols, low_memory=False)
    for c in RAW_FEATURES:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    print(f"    {len(df)} rows loaded in {time.time()-t0:.1f}s")
    return df


def build_windows(df: pd.DataFrame, window_s: int, stride_s: int,
                  source_dataset: str, scenario_map: dict) -> tuple:
    """Build sliding windows from a physical timeseries DataFrame.

    source_dataset: "clean" or "attacked" — stored per-window for downstream filtering.
    Window IDs are prefixed with source_dataset to ensure global uniqueness across
    the combined windows_all.parquet and sequence_windows_all.npz files.

    Returns:
        meta_rows: list of dicts (metadata per window)
        flat_arr:  np.ndarray shape (n_windows, N_FLAT_FEATURES)
        seq_arr:   np.ndarray shape (n_windows, window_s, N_RAW_FEATURES)
    """
    feat_data = np.zeros((len(df), N_RAW_FEATURES), dtype=np.float32)
    for i, col in enumerate(RAW_FEATURES):
        if col in df.columns:
            feat_data[:, i] = df[col].values

    if "physical_effect_active_flag" in df.columns:
        labels = df["physical_effect_active_flag"].fillna(0).astype(int).values
    else:
        labels = np.zeros(len(df), dtype=int)

    if "physical_scenario_id" in df.columns:
        scen_ids = df["physical_scenario_id"].fillna("none").values
    else:
        scen_ids = np.full(len(df), "none")

    ts = df["time_s"].values if "time_s" in df.columns else np.arange(len(df))
    timestamps = df["timestamp_utc"].values if "timestamp_utc" in df.columns else np.full(len(df), "")

    n_rows = len(df)
    n_windows = max(0, (n_rows - window_s) // stride_s + 1)

    meta_rows = []
    flat_list = []
    seq_list = []

    for w_idx in range(n_windows):
        start = w_idx * stride_s
        end = start + window_s

        chunk = feat_data[start:end]
        win_labels = labels[start:end]
        win_scen = scen_ids[start:end]

        y_anomaly = int(win_labels.max()) if len(win_labels) > 0 else 0
        y_physical = y_anomaly

        scen_counts = {}
        for s in win_scen:
            if s and s != "none" and not (isinstance(s, float) and np.isnan(s)):
                scen_counts[s] = scen_counts.get(s, 0) + 1
        dominant_scen = max(scen_counts, key=scen_counts.get) if scen_counts else "none"

        scen_info = scenario_map.get(dominant_scen, {})

        # window_id is globally unique: prefix ensures no collision across clean/attacked
        meta_rows.append({
            "window_id": f"{source_dataset}_w{w_idx:07d}",
            "source_dataset": source_dataset,
            "window_start_utc": str(timestamps[start]) if start < len(timestamps) else "",
            "window_end_utc": str(timestamps[end - 1]) if end - 1 < len(timestamps) else "",
            "window_start_s": int(ts[start]) if start < len(ts) else start,
            "window_end_s": int(ts[end - 1]) if end - 1 < len(ts) else end - 1,
            "split": "",  # assigned later
            "y_anomaly": y_anomaly,
            "y_cyber_anomaly": 0,
            "y_physical_anomaly": y_physical,
            "scenario_id": dominant_scen,
            "scenario_name": scen_info.get("scenario_name", "normal"),
            "scenario_class": scen_info.get("scenario_class", "normal"),
            "anomaly_type": scen_info.get("physical_effect_type", "normal"),
            "generation_method_summary": "physics_constrained_surrogate",
            "protocol_claim_level_summary": "semantic_ieee2030_5_style",
        })

        flat_feats = compute_window_features(chunk)
        flat_list.append(flat_feats)
        seq_list.append(chunk.copy())

    flat_arr = (np.array(flat_list, dtype=np.float32) if flat_list
                else np.zeros((0, N_FLAT_FEATURES), dtype=np.float32))
    seq_arr = (np.array(seq_list, dtype=np.float32) if seq_list
               else np.zeros((0, window_s, N_RAW_FEATURES), dtype=np.float32))

    return meta_rows, flat_arr, seq_arr


def assign_splits_normal(meta_rows: list) -> list:
    """Assign train/val/test by time order for normal (clean-source) windows."""
    n = len(meta_rows)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))
    for i, row in enumerate(meta_rows):
        if i < train_end:
            row["split"] = "train"
        elif i < val_end:
            row["split"] = "val"
        else:
            row["split"] = "test"
    return meta_rows


def assign_splits_attacked(meta_rows: list) -> list:
    """Assign attacked window splits by scenario_id group, not random rows."""
    seen_order = []
    seen_set = set()
    for row in meta_rows:
        sid = row["scenario_id"]
        if sid not in seen_set:
            seen_set.add(sid)
            seen_order.append(sid)

    n_scen = len(seen_order)
    train_end = int(n_scen * TRAIN_FRAC)
    val_end = int(n_scen * (TRAIN_FRAC + VAL_FRAC))

    scen_split = {}
    for i, sid in enumerate(seen_order):
        if i < train_end:
            scen_split[sid] = "train"
        elif i < val_end:
            scen_split[sid] = "val"
        else:
            scen_split[sid] = "test"

    for row in meta_rows:
        row["split"] = scen_split.get(row["scenario_id"], "test")

    return meta_rows


def save_windows(meta_rows: list, flat_arr: np.ndarray, prefix: str,
                 windows_dir: Path) -> tuple:
    """Save windows as parquet. source_dataset column is preserved."""
    df_meta = pd.DataFrame(meta_rows).reset_index(drop=True)
    flat_df = pd.DataFrame(flat_arr, columns=FLAT_FEATURE_NAMES)
    df_full = pd.concat([df_meta, flat_df], axis=1)
    out_path = windows_dir / f"windows_{prefix}.parquet"
    df_full.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name} ({len(df_full)} windows)")
    return df_full, out_path


def main(primary_only: bool = False):
    ensure_all_dirs()
    t_start = time.time()

    print("Loading scenario manifest...")
    sm = pd.read_csv(SCENARIO_MANIFEST_CSV)
    scenario_map = {
        row["scenario_id"]: {
            "scenario_name": row.get("scenario_name", "unknown"),
            "scenario_class": row.get("scenario_class", "unknown"),
            "physical_effect_type": row.get("physical_effect_type", "unknown"),
            "anomaly_type": row.get("physical_effect_type", "unknown"),
        }
        for _, row in sm.iterrows()
    }

    df_clean = load_physical_csv(CLEAN_CSV)
    df_attacked = load_physical_csv(ATTACKED_CSV)

    print(f"\nBuilding primary windows ({PRIMARY_WINDOW_S}s / {PRIMARY_STRIDE_S}s stride)...")
    t_w = time.time()

    # source_dataset="clean" for clean CSV, "attacked" for attacked CSV
    meta_normal, flat_normal, seq_normal = build_windows(
        df_clean, PRIMARY_WINDOW_S, PRIMARY_STRIDE_S, "clean", scenario_map)
    meta_attacked, flat_attacked, seq_attacked = build_windows(
        df_attacked, PRIMARY_WINDOW_S, PRIMARY_STRIDE_S, "attacked", scenario_map)

    print(f"  Normal (clean source): {len(meta_normal)} windows | "
          f"Attacked (attacked source): {len(meta_attacked)} windows")
    print(f"  Window build time: {time.time()-t_w:.1f}s")

    meta_normal = assign_splits_normal(meta_normal)
    meta_attacked = assign_splits_attacked(meta_attacked)

    # Collect scenario-based split info from attacked windows
    attacked_scen_splits = {}
    for row in meta_attacked:
        sid = row["scenario_id"]
        if sid not in attacked_scen_splits:
            attacked_scen_splits[sid] = row["split"]

    print("\nSaving window parquet files...")
    df_norm, _ = save_windows(meta_normal, flat_normal, "normal", WINDOWS_DIR)
    df_att, _ = save_windows(meta_attacked, flat_attacked, "attacked", WINDOWS_DIR)

    df_all = pd.concat([df_norm, df_att], ignore_index=True)
    df_all.to_parquet(WINDOWS_ALL_PARQUET, index=False)
    print(f"  Saved: windows_all.parquet ({len(df_all)} windows)")

    # Verify window_id uniqueness in combined file
    dup_count = df_all["window_id"].duplicated().sum()
    if dup_count > 0:
        raise RuntimeError(f"FATAL: {dup_count} duplicate window_ids in windows_all.parquet. "
                           f"This indicates a bug in the window_id generation logic.")
    print(f"  window_id uniqueness: PASS (0 duplicates across {len(df_all)} windows)")

    # Save sequence tensors — include source_dataset array in each NPZ
    print("\nSaving sequence tensors...")
    np.savez_compressed(
        str(SEQ_NORMAL_NPZ),
        sequences=seq_normal,
        window_ids=np.array([r["window_id"] for r in meta_normal]),
        y_anomaly=np.array([r["y_anomaly"] for r in meta_normal]),
        splits=np.array([r["split"] for r in meta_normal]),
        source_dataset=np.array([r["source_dataset"] for r in meta_normal]),
    )
    print(f"  seq_normal: shape {seq_normal.shape}")

    np.savez_compressed(
        str(SEQ_ATTACKED_NPZ),
        sequences=seq_attacked,
        window_ids=np.array([r["window_id"] for r in meta_attacked]),
        y_anomaly=np.array([r["y_anomaly"] for r in meta_attacked]),
        splits=np.array([r["split"] for r in meta_attacked]),
        source_dataset=np.array([r["source_dataset"] for r in meta_attacked]),
    )
    print(f"  seq_attacked: shape {seq_attacked.shape}")

    seq_all = np.concatenate([seq_normal, seq_attacked], axis=0)
    y_all = np.array([r["y_anomaly"] for r in meta_normal + meta_attacked])
    splits_all = np.array([r["split"] for r in meta_normal + meta_attacked])
    window_ids_all = np.array([r["window_id"] for r in meta_normal + meta_attacked])
    source_all = np.array([r["source_dataset"] for r in meta_normal + meta_attacked])
    np.savez_compressed(
        str(SEQ_ALL_NPZ),
        sequences=seq_all,
        window_ids=window_ids_all,
        y_anomaly=y_all,
        splits=splits_all,
        source_dataset=source_all,
    )
    print(f"  seq_all: shape {seq_all.shape}")

    # Secondary sweep
    secondary_results = {}
    if not primary_only:
        for cfg in SECONDARY_CONFIGS:
            w_s, st_s, suffix = cfg["window_s"], cfg["stride_s"], cfg["suffix"]
            print(f"\nBuilding secondary windows ({w_s}s / {st_s}s stride)...")
            m_n, f_n, _ = build_windows(df_clean, w_s, st_s, "clean", scenario_map)
            m_a, f_a, _ = build_windows(df_attacked, w_s, st_s, "attacked", scenario_map)
            m_n = assign_splits_normal(m_n)
            m_a = assign_splits_attacked(m_a)
            df_n2, _ = save_windows(m_n, f_n, f"normal_{suffix}", WINDOWS_DIR)
            df_a2, _ = save_windows(m_a, f_a, f"attacked_{suffix}", WINDOWS_DIR)
            df_all2 = pd.concat([df_n2, df_a2], ignore_index=True)
            df_all2.to_parquet(WINDOWS_DIR / f"windows_all_{suffix}.parquet", index=False)
            secondary_results[suffix] = {
                "window_s": w_s, "stride_s": st_s,
                "normal_windows": len(m_n),
                "attacked_windows": len(m_a),
            }

    normal_counts = df_norm.groupby("split").size().to_dict()
    attacked_counts = df_att.groupby("split").size().to_dict()
    anomaly_in_attacked = int((df_att["y_anomaly"] == 1).sum())
    normal_in_attacked = int((df_att["y_anomaly"] == 0).sum())

    feature_manifest = {
        "raw_features": RAW_FEATURES,
        "n_raw_features": N_RAW_FEATURES,
        "window_stats": WINDOW_STATS,
        "n_stats": len(WINDOW_STATS),
        "flat_feature_names": FLAT_FEATURE_NAMES,
        "n_flat_features": N_FLAT_FEATURES,
        "primary_window_s": PRIMARY_WINDOW_S,
        "primary_stride_s": PRIMARY_STRIDE_S,
        "window_id_format": "{source_dataset}_w{index:07d}",
        "source_dataset_values": ["clean", "attacked"],
        "leakage_excluded": {
            "label_columns": ["physical_effect_active_flag", "physical_effect_type",
                              "physical_scenario_id", "physical_constraint_status"],
            "command_leakage": ["pv_commanded_p_kw", "pv_commanded_q_kvar",
                                "bess_commanded_p_kw", "bess_commanded_q_kvar"],
            "metadata": ["timestamp_utc", "time_s", "der_site_id", "pcc_id", "generation_method"],
            "constants": ["pv_s_rated_kva", "bess_s_rated_kva", "bess_capacity_kwh",
                          "bess_soc_min_percent", "bess_soc_max_percent"],
        },
    }
    with open(FEATURE_MANIFEST_JSON, "w") as f:
        json.dump(feature_manifest, f, indent=2)

    split_manifest = {
        "strategy": "time_aware",
        "train_frac": TRAIN_FRAC,
        "val_frac": VAL_FRAC,
        "test_frac": TEST_FRAC,
        "normal_split": "first 60% / next 20% / last 20% by time order (clean source only)",
        "attacked_split": "by scenario_id group (no scenario across val+test boundary)",
        "attacked_scenario_splits": attacked_scen_splits,
        "training_filter": "source_dataset == clean AND y_anomaly == 0",
        "normal_train_windows": normal_counts.get("train", 0),
        "normal_val_windows": normal_counts.get("val", 0),
        "normal_test_windows": normal_counts.get("test", 0),
        "attacked_train_windows": attacked_counts.get("train", 0),
        "attacked_val_windows": attacked_counts.get("val", 0),
        "attacked_test_windows": attacked_counts.get("test", 0),
        "attacked_anomaly_windows": anomaly_in_attacked,
        "attacked_normal_windows": normal_in_attacked,
    }
    with open(SPLIT_MANIFEST_JSON, "w") as f:
        json.dump(split_manifest, f, indent=2)

    elapsed = time.time() - t_start
    summary = {
        "status": "PASS",
        "window_id_format": "{source_dataset}_w{index:07d}",
        "source_dataset_column": "present in all parquet and NPZ files",
        "primary_window_s": PRIMARY_WINDOW_S,
        "primary_stride_s": PRIMARY_STRIDE_S,
        "total_normal_windows": len(meta_normal),
        "total_attacked_windows": len(meta_attacked),
        "total_windows": len(df_all),
        "window_id_duplicates": 0,
        "anomaly_windows": anomaly_in_attacked,
        "normal_attacked_windows": normal_in_attacked,
        "n_raw_features": N_RAW_FEATURES,
        "n_flat_features": N_FLAT_FEATURES,
        "seq_shape_normal": list(seq_normal.shape),
        "seq_shape_attacked": list(seq_attacked.shape),
        "split_normal": normal_counts,
        "split_attacked": attacked_counts,
        "secondary_configs": secondary_results,
        "elapsed_s": round(elapsed, 1),
    }
    with open(WINDOW_BUILD_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Window Build Report", "",
        "**Status:** PASS",
        f"**Primary window:** {PRIMARY_WINDOW_S}s / stride {PRIMARY_STRIDE_S}s",
        f"**Total windows:** {len(df_all):,}",
        f"**window_id format:** {{source_dataset}}_w{{index:07d}}",
        f"**window_id uniqueness check:** PASS (0 duplicates)",
        f"**source_dataset column:** present in all parquet + NPZ files",
        f"**Normal windows (clean source):** {len(meta_normal):,}",
        f"**Attacked windows (attacked source):** {len(meta_attacked):,}",
        f"**  - with anomaly:** {anomaly_in_attacked:,}",
        f"**  - no anomaly:** {normal_in_attacked:,}",
        f"**Raw features:** {N_RAW_FEATURES}",
        f"**Flat features:** {N_FLAT_FEATURES} ({N_RAW_FEATURES} x {len(WINDOW_STATS)} stats)",
        f"**Sequence tensor shape:** {seq_normal.shape}", "",
        "## Training Filter",
        "",
        "Models train on: `source_dataset == 'clean' AND y_anomaly == 0`",
        "This ensures no contamination from attacked-source normal windows.", "",
        "## Split Strategy", "",
        "- Normal (clean source): time-ordered 60/20/20 split",
        "- Attacked (attacked source): by scenario_id group (no cross-contamination)", "",
        "| Split | Normal (clean) | Attacked |",
        "| --- | --- | --- |",
    ]
    for s in ["train", "val", "test"]:
        lines.append(f"| {s} | {normal_counts.get(s, 0):,} | {attacked_counts.get(s, 0):,} |")

    lines += ["", "## Raw Features Used", ""]
    for i, feat in enumerate(RAW_FEATURES, 1):
        lines.append(f"{i}. `{feat}`")

    lines += ["", "## Secondary Sweep", ""]
    for suffix, info in secondary_results.items():
        lines.append(f"- {info['window_s']}s/{info['stride_s']}s: "
                     f"{info['normal_windows']:,} normal, {info['attacked_windows']:,} attacked windows")

    lines += ["", f"**Build time:** {elapsed:.1f}s"]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORTS_DIR / "WINDOW_BUILD_REPORT.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nWindow build complete: {len(df_all):,} total windows in {elapsed:.1f}s")
    print(f"  window_id format: {{source_dataset}}_w{{index:07d}}")
    print(f"  source_dataset: 'clean' for {len(meta_normal):,} | 'attacked' for {len(meta_attacked):,}")
    return summary


if __name__ == "__main__":
    main()
