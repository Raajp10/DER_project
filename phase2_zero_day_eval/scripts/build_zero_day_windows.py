"""
Phase 2 Zero-Day Window Builder.

Reads zero_day_physical_attacked.csv and builds sliding windows using the
exact same feature order, statistics, and window geometry as Phase 1 so
frozen models can score them without any feature-space mismatch.

Inputs:
  models/windows/feature_manifest.json             (Phase 1 feature spec)
  outputs/zero_day_physical_attacked.csv           (compiled zero-day CSV)

Outputs:
  outputs/zero_day_windows.parquet                 (flat features + metadata)
  outputs/zero_day_windows.npz                     (sequence arrays)
  outputs/zero_day_window_build_summary.json
  reports/ZERO_DAY_WINDOW_BUILD_REPORT.md
"""
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PHASE2_ROOT  = Path(r"D:\updated_dataset\phase2_zero_day_eval")
PROJECT_ROOT = Path(r"D:\updated_dataset")
OUTPUTS_DIR  = PHASE2_ROOT / "outputs"
REPORTS_DIR  = PHASE2_ROOT / "reports"
FEATURE_MANIFEST = PROJECT_ROOT / "models" / "windows" / "feature_manifest.json"
ATTACKED_CSV = OUTPUTS_DIR / "zero_day_physical_attacked.csv"

PRIMARY_WINDOW_S = 60
PRIMARY_STRIDE_S = 10

NOW = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

ZD_LABEL_COLS = [
    "zero_day_active_flag",
    "zero_day_label_anomaly",
    "zero_day_label_cyber_anomaly",
    "zero_day_label_physical_anomaly",
    "zero_day_scenario_id",
    "zero_day_scenario_family",
    "zero_day_scenario_class",
    "zero_day_author_model",
]


# ---------------------------------------------------------------------------
# Feature computation (mirrors 00_build_model_windows.py exactly)
# ---------------------------------------------------------------------------

def _slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr), dtype=np.float32)
    x -= x.mean()
    denom = float((x * x).sum())
    if denom == 0.0:
        return 0.0
    return float((x * arr).sum() / denom)


def _rms(arr: np.ndarray) -> float:
    return float(np.sqrt(np.mean(arr.astype(np.float32) ** 2)))


def compute_window_features(chunk: np.ndarray, n_raw: int) -> np.ndarray:
    """10 statistics per raw feature — identical to Phase 1.

    chunk : shape (window_s, n_raw_features) float32
    Returns: shape (n_raw * 10,) float32
    """
    feats = []
    for f_idx in range(n_raw):
        col = chunk[:, f_idx].astype(np.float32)
        feats.extend([
            float(np.mean(col)),
            float(np.std(col)),
            float(np.min(col)),
            float(np.max(col)),
            float(np.median(col)),
            _slope(col),
            float(col[0]),
            float(col[-1]),
            float(col[-1] - col[0]),
            _rms(col),
        ])
    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_zero_day_windows() -> dict:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- load feature manifest ---
    if not FEATURE_MANIFEST.exists():
        print(f"[ERROR] Feature manifest not found: {FEATURE_MANIFEST}")
        sys.exit(1)
    with open(FEATURE_MANIFEST) as f:
        manifest = json.load(f)
    raw_features     = manifest["raw_features"]
    n_raw            = manifest["n_raw_features"]
    flat_feat_names  = manifest["flat_feature_names"]
    n_flat           = manifest["n_flat_features"]
    print(f"[INFO] Feature manifest loaded: {n_raw} raw features, {n_flat} flat features")

    # --- load zero-day attacked CSV ---
    if not ATTACKED_CSV.exists():
        print(f"[ERROR] Zero-day attacked CSV not found: {ATTACKED_CSV}")
        print("[ERROR] Run compile_zero_day_dataset.py first.")
        sys.exit(1)
    print(f"[INFO] Loading zero-day attacked CSV: {ATTACKED_CSV}")
    t0 = time.time()
    needed = raw_features + ["time_s", "timestamp_utc"] + ZD_LABEL_COLS
    avail_cols = list(pd.read_csv(ATTACKED_CSV, nrows=0).columns)
    use_cols = [c for c in needed if c in avail_cols]
    df = pd.read_csv(ATTACKED_CSV, usecols=use_cols, low_memory=False)
    print(f"[INFO] Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # Verify all raw features present
    missing = [f for f in raw_features if f not in df.columns]
    if missing:
        print(f"[ERROR] Missing raw feature columns: {missing}")
        sys.exit(1)

    # Cast raw features to float32
    for col in raw_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

    # Build label arrays
    def _int_col(name: str) -> np.ndarray:
        if name in df.columns:
            return df[name].fillna(0).astype(int).values
        return np.zeros(len(df), dtype=int)

    def _str_col(name: str) -> np.ndarray:
        if name in df.columns:
            return df[name].fillna("").astype(str).values
        return np.full(len(df), "", dtype=object)

    zd_active     = _int_col("zero_day_active_flag")
    zd_anomaly    = _int_col("zero_day_label_anomaly")
    zd_cyber      = _int_col("zero_day_label_cyber_anomaly")
    zd_physical   = _int_col("zero_day_label_physical_anomaly")
    zd_scen_id    = _str_col("zero_day_scenario_id")
    zd_family     = _str_col("zero_day_scenario_family")
    zd_class      = _str_col("zero_day_scenario_class")
    zd_author     = _str_col("zero_day_author_model")

    ts = df["time_s"].values if "time_s" in df.columns else np.arange(len(df))
    timestamps = df["timestamp_utc"].values if "timestamp_utc" in df.columns else np.full(len(df), "")

    # Build feature matrix
    feat_data = np.zeros((len(df), n_raw), dtype=np.float32)
    for i, col in enumerate(raw_features):
        feat_data[:, i] = df[col].values

    # --- sliding windows ---
    n_rows = len(df)
    n_windows = max(0, (n_rows - PRIMARY_WINDOW_S) // PRIMARY_STRIDE_S + 1)
    print(f"[INFO] Building {n_windows:,} windows (window={PRIMARY_WINDOW_S}s, stride={PRIMARY_STRIDE_S}s)")

    meta_rows = []
    flat_list = []
    seq_list  = []

    t1 = time.time()
    for w_idx in range(n_windows):
        start = w_idx * PRIMARY_STRIDE_S
        end   = start + PRIMARY_WINDOW_S

        chunk     = feat_data[start:end]
        w_active  = zd_active[start:end]
        w_anomaly = zd_anomaly[start:end]
        w_cyber   = zd_cyber[start:end]
        w_phys    = zd_physical[start:end]
        w_scen    = zd_scen_id[start:end]
        w_fam     = zd_family[start:end]
        w_cls     = zd_class[start:end]
        w_auth    = zd_author[start:end]

        y_active   = int(w_active.max())
        y_anomaly  = int(w_anomaly.max())
        y_cyber    = int(w_cyber.max())
        y_physical = int(w_phys.max())

        # dominant scenario: most frequent non-empty entry
        scen_counts: dict = {}
        for s in w_scen:
            if s and s not in ("", "MULTIPLE"):
                scen_counts[s] = scen_counts.get(s, 0) + 1
        dom_scen = max(scen_counts, key=scen_counts.get) if scen_counts else ""
        if not dom_scen and y_active:
            dom_scen = "MULTIPLE"

        # dominant family / class / author (most frequent non-empty)
        def _dominant(arr):
            counts: dict = {}
            for v in arr:
                if v:
                    counts[v] = counts.get(v, 0) + 1
            return max(counts, key=counts.get) if counts else ""

        dom_family = _dominant(w_fam)
        dom_class  = _dominant(w_cls)
        dom_author = _dominant(w_auth)

        meta_rows.append({
            "window_id":             f"zero_day_w{w_idx:07d}",
            "source_dataset":        "zero_day",
            "window_start_utc":      str(timestamps[start]) if start < len(timestamps) else "",
            "window_end_utc":        str(timestamps[end - 1]) if end - 1 < len(timestamps) else "",
            "window_start_s":        int(ts[start]) if start < len(ts) else start,
            "window_end_s":          int(ts[end - 1]) if end - 1 < len(ts) else end - 1,
            "zero_day_active":       y_active,
            "y_anomaly":             y_anomaly,
            "y_cyber_anomaly":       y_cyber,
            "y_physical_anomaly":    y_physical,
            "scenario_id":           dom_scen,
            "scenario_family":       dom_family,
            "scenario_class":        dom_class,
            "author_model":          dom_author,
        })

        flat_feats = compute_window_features(chunk, n_raw)
        flat_list.append(flat_feats)
        seq_list.append(chunk.copy())

    print(f"[INFO] Window loop done in {time.time()-t1:.1f}s")

    flat_arr = (np.array(flat_list, dtype=np.float32) if flat_list
                else np.zeros((0, n_flat), dtype=np.float32))
    seq_arr  = (np.array(seq_list, dtype=np.float32) if seq_list
                else np.zeros((0, PRIMARY_WINDOW_S, n_raw), dtype=np.float32))

    meta_df = pd.DataFrame(meta_rows)

    # --- merge flat features into parquet ---
    feat_df = pd.DataFrame(flat_arr, columns=flat_feat_names)
    out_df  = pd.concat([meta_df, feat_df], axis=1)

    parquet_path = OUTPUTS_DIR / "zero_day_windows.parquet"
    out_df.to_parquet(parquet_path, index=False)
    print(f"[INFO] Saved parquet: {parquet_path} ({len(out_df):,} windows)")

    # --- save NPZ ---
    npz_path = OUTPUTS_DIR / "zero_day_windows.npz"
    np.savez_compressed(
        npz_path,
        X_flat=flat_arr,
        X_seq=seq_arr,
        y_anomaly=meta_df["y_anomaly"].values.astype(np.int8),
        y_cyber_anomaly=meta_df["y_cyber_anomaly"].values.astype(np.int8),
        y_physical_anomaly=meta_df["y_physical_anomaly"].values.astype(np.int8),
        zero_day_active=meta_df["zero_day_active"].values.astype(np.int8),
    )
    print(f"[INFO] Saved NPZ: {npz_path}")

    # --- summary ---
    n_active_windows = int(meta_df["zero_day_active"].sum())
    n_anomaly_windows = int(meta_df["y_anomaly"].sum())

    summary = {
        "timestamp":             NOW,
        "total_windows":         n_windows,
        "window_s":              PRIMARY_WINDOW_S,
        "stride_s":              PRIMARY_STRIDE_S,
        "n_raw_features":        n_raw,
        "n_flat_features":       n_flat,
        "n_active_zd_windows":   n_active_windows,
        "n_anomaly_windows":     n_anomaly_windows,
        "flat_shape":            list(flat_arr.shape),
        "seq_shape":             list(seq_arr.shape),
        "outputs": {
            "parquet": str(parquet_path),
            "npz":     str(npz_path),
        },
        "phase1_feature_manifest": str(FEATURE_MANIFEST),
        "feature_order_matches_phase1": True,
    }

    summary_path = OUTPUTS_DIR / "zero_day_window_build_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    _write_report(summary, manifest)
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _write_report(summary: dict, manifest: dict) -> None:
    n_w = summary["total_windows"]
    n_act = summary["n_active_zd_windows"]
    n_anom = summary["n_anomaly_windows"]
    pct_act = 100.0 * n_act / n_w if n_w else 0.0
    pct_anom = 100.0 * n_anom / n_w if n_w else 0.0

    lines = [
        "# ZERO_DAY_WINDOW_BUILD_REPORT",
        "",
        f"Generated: {summary['timestamp']}",
        "",
        "## Window Geometry",
        "",
        f"- Window size : {summary['window_s']} seconds",
        f"- Stride      : {summary['stride_s']} seconds",
        f"- Total windows built: **{n_w:,}**",
        "",
        "## Feature Space",
        "",
        f"- Raw features   : {summary['n_raw_features']} (identical to Phase 1)",
        f"- Flat features  : {summary['n_flat_features']} (raw × 10 stats)",
        f"- Feature order  : matches `{summary['phase1_feature_manifest']}`",
        "",
        "## Label Coverage",
        "",
        f"- Zero-day active windows : {n_act:,} / {n_w:,} ({pct_act:.1f}%)",
        f"- Anomaly-labelled windows: {n_anom:,} / {n_w:,} ({pct_anom:.1f}%)",
        "",
        "## Outputs",
        "",
        f"- `{summary['outputs']['parquet']}`",
        f"  Shape: {summary['flat_shape']}",
        f"- `{summary['outputs']['npz']}`",
        f"  X_seq shape: {summary['seq_shape']}",
        "",
        "## Frozen-Model Compatibility",
        "",
        "These windows are ready for frozen Phase 1 model scoring.  No retraining or",
        "threshold recalibration is performed.  The feature manifest from Phase 1 is",
        "authoritative; any mismatch would surface as a missing-column error above.",
        "",
        "---",
        "*Phase 2 Zero-Day Evaluation — Window build complete*",
    ]

    report_path = REPORTS_DIR / "ZERO_DAY_WINDOW_BUILD_REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[INFO] Wrote window build report: {report_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    summary = build_zero_day_windows()
    print("\n=== WINDOW BUILD SUMMARY ===")
    print(f"Total windows    : {summary['total_windows']:,}")
    print(f"Active ZD windows: {summary['n_active_zd_windows']:,}")
    print(f"Anomaly windows  : {summary['n_anomaly_windows']:,}")
    print(f"Flat shape       : {summary['flat_shape']}")
    print(f"Seq shape        : {summary['seq_shape']}")
    print("============================")
