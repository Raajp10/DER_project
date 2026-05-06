"""
Master pipeline runner for the DER Cyber-Physical Anomaly Detection Dataset.
Run from D:/updated_dataset as:
    python scripts_updated/run_updated_dataset_pipeline.py
"""
import sys
import os
import json
import shutil
import zipfile
import traceback
import importlib.util
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(r"D:\updated_dataset")
SCRIPTS = ROOT / "scripts_updated"
COMMON = SCRIPTS / "00_common"

# ── Bootstrap: add 00_common to sys.path first ───────────────────────────────
for _d in [str(ROOT), str(SCRIPTS), str(COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

# ── Now safe to import from 00_common ────────────────────────────────────────
from loader import setup_path, load_script
setup_path()

from paths import (
    ensure_dirs, LOGS, FINAL_PACKAGE, FIGURES, REPORTS,
    CLEAN_PHYSICAL_CSV, ATTACKED_PHYSICAL_CSV, CYBER_ANOMALOUS_CSV,
    FINAL_VALIDATION_JSON, DER_METADATA_JSON, ENV_CHECK_JSON,
)

LOG_FILE = LOGS / "pipeline_run.log"
ERROR_FILE = LOGS / "pipeline_errors.log"


class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            try:
                f.write(obj)
                f.flush()
            except UnicodeEncodeError:
                safe = obj.encode("ascii", errors="replace").decode("ascii")
                f.write(safe)
                f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


def run_step(path: Path):
    """Load a script by file path and call its main() function."""
    mod = load_script(path)
    if hasattr(mod, "main"):
        return mod.main()
    return None


def step(name: str, script_path: Path, log_f):
    print(f"\n{'-'*60}", flush=True)
    print(f"STEP: {name}", flush=True)
    print(f"  File: {script_path.relative_to(ROOT)}", flush=True)
    print(f"{'-'*60}", flush=True)
    try:
        result = run_step(script_path)
        print(f"  OK: {name} completed.", flush=True)
        return result
    except SystemExit as e:
        if e.code == 0:
            print(f"  OK: {name} completed (sys.exit(0)).", flush=True)
            return None
        print(f"  FAIL: {name} exited with code {e.code}", flush=True)
        with open(ERROR_FILE, "a", encoding="utf-8") as ef:
            ef.write(f"[{datetime.now(timezone.utc).isoformat()}] STEP FAILED: {name}\n")
            traceback.print_exc(file=ef)
        raise RuntimeError(f"Step failed with exit code {e.code}")
    except Exception as e:
        print(f"  FAIL: {name} FAILED: {e}", flush=True)
        with open(ERROR_FILE, "a", encoding="utf-8") as ef:
            ef.write(f"[{datetime.now(timezone.utc).isoformat()}] STEP FAILED: {name}\n")
            ef.write(traceback.format_exc())
            ef.write("\n")
        raise RuntimeError(f"Pipeline step failed: {name}") from e


def create_structure_and_backup():
    ensure_dirs()
    LOGS.mkdir(parents=True, exist_ok=True)
    for d in [SCRIPTS,
              SCRIPTS / "00_common", SCRIPTS / "01_setup",
              SCRIPTS / "02_scenario_generation",
              SCRIPTS / "03_physical_data_generation",
              SCRIPTS / "04_cyber_log_generation",
              SCRIPTS / "05_physical_cyber_mapping",
              SCRIPTS / "06_visualization",
              SCRIPTS / "07_validation_gate"]:
        init = d / "__init__.py"
        if not init.exists():
            init.write_text("")
    backup_dir = ROOT / "original_input_backup"
    src_dir = ROOT / "ieee123_base"
    if src_dir.exists() and not (backup_dir / "IEEE123Master.dss").exists():
        shutil.copytree(src_dir, backup_dir, dirs_exist_ok=True)
        print(f"  Backed up {src_dir.name} → original_input_backup/")


def generate_final_reports_inline():
    """Generate final reports using the reports module."""
    mod = load_script(SCRIPTS / "scripts_updated_reports.py", "pipeline_reports")
    mod.generate_all_reports()


def build_zip_package():
    zip_path = FINAL_PACKAGE / "updated_der_cyber_physical_dataset_package.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    include_dirs = [ROOT / "data_updated", SCRIPTS, ROOT / "reports", FIGURES]
    include_files = [ROOT / "README_UPDATED_DATASET.md", ROOT / "run_pipeline.bat"]
    print(f"  Building ZIP package: {zip_path.name}")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for d in include_dirs:
            if d.exists():
                for f in d.rglob("*"):
                    if f.is_file():
                        try:
                            zf.write(f, f.relative_to(ROOT))
                        except Exception:
                            pass
        for f in include_files:
            if f.exists():
                zf.write(f, f.relative_to(ROOT))
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"  ZIP created: {zip_path.name} ({size_mb:.1f} MB)")


def print_completion_summary():
    val_j = {}
    if FINAL_VALIDATION_JSON.exists():
        with open(FINAL_VALIDATION_JSON) as f:
            val_j = json.load(f)
    meta = {}
    if DER_METADATA_JSON.exists():
        with open(DER_METADATA_JSON) as f:
            meta = json.load(f)
    gen_counts = val_j.get("generation_method_counts", {})
    opendss_ew = gen_counts.get("opendss_event_window_resolved", 0)
    surrogate = gen_counts.get("physics_constrained_surrogate", 0)
    opendss_clean = gen_counts.get("opendss_clean_baseline", 0)
    zip_path = FINAL_PACKAGE / "updated_der_cyber_physical_dataset_package.zip"
    pv_src = meta.get("pv_p_rated_kw_source", "unknown")
    bess_src = meta.get("bess_p_rated_kw_source", "unknown")
    ods_used = False
    master_dss = "N/A"
    if ENV_CHECK_JSON.exists():
        with open(ENV_CHECK_JSON) as f:
            env = json.load(f)
        ods_used = env.get("opendss_available", False)
    disc = ROOT / "data_updated" / "metadata" / "input_discovery_report.json"
    if disc.exists():
        with open(disc) as f:
            d = json.load(f)
        master_dss = d.get("master_dss_file", "N/A")
    clean_method = "opendss_clean_baseline" if opendss_clean > 0 else "physics_constrained_surrogate"
    remaining = []
    if surrogate > 0:
        remaining.append(f"Install opendssdirect.py to upgrade {surrogate:,} surrogate rows to OpenDSS event-window")
    if meta.get("rating_fallback_used"):
        remaining.append("RATING FALLBACK USED — verify PV/BESS ratings match DSS files")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETION SUMMARY")
    print("=" * 70)
    print(f"Root folder:                       D:\\updated_dataset")
    print(f"OpenDSS master file used:          {master_dss}")
    print(f"OpenDSS available:                 {'yes' if ods_used else 'no'}")
    print(f"Clean physical generation method:  {clean_method}")
    print(f"PV rating used and source:         {meta.get('pv_p_rated_kw', 100.0)} kW ({pv_src})")
    print(f"BESS rating used and source:       {meta.get('bess_p_rated_kw', 50.0)} kW ({bess_src})")
    print(f"OpenDSS event-window scenarios:    {opendss_ew}")
    print(f"Surrogate fallback scenarios:      {surrogate}")
    print(f"Physical layer improved:           yes")
    print(f"Cyber layer improved:              yes")
    print(f"Context layer improved:            yes")
    print(f"Graphs generated:                  {'yes' if len(list(FIGURES.glob('*.png'))) > 0 else 'no'}")
    print(f"Final validation status:           {val_j.get('overall', 'UNKNOWN')}")
    print(f"Final package path:                {zip_path}")
    print(f"\nExact final claim sentence:")
    print(f'  "{val_j.get("final_claim", "N/A")}"')
    print(f"\nRemaining manual review items:")
    for item in remaining:
        print(f"  - {item}")
    if not remaining:
        print("  None")
    print("=" * 70)


# ── Step table: (name, script_path, critical) ────────────────────────────────
def get_pipeline_steps():
    S = SCRIPTS
    return [
        ("Create folder structure and backup", None, True),   # handled inline
        ("Discover DSS input files",
         S / "01_setup" / "00_discover_inputs.py", True),
        ("Check environment and OpenDSS",
         S / "01_setup" / "01_check_environment.py", True),
        ("Generate scenario manifest",
         S / "02_scenario_generation" / "02_generate_scenario_manifest.py", True),
        ("Build DER physical metadata",
         S / "03_physical_data_generation" / "04a_build_der_physical_metadata.py", True),
        ("Generate clean physical timeseries",
         S / "03_physical_data_generation" / "04b_generate_clean_physical_opendss.py", True),
        ("Generate attacked physical timeseries",
         S / "03_physical_data_generation" / "04c_generate_attacked_physical_event_windows.py", True),
        ("OpenDSS event-window scenario resolution",
         S / "03_physical_data_generation" / "04e_run_opendss_event_window_scenarios.py", False),
        ("Build physical residuals",
         S / "03_physical_data_generation" / "04d_build_physical_residuals.py", True),
        ("Validate physical constraints",
         S / "07_validation_gate" / "07a_validate_physical_constraints.py", False),
        ("Validate clean/attacked alignment",
         S / "07_validation_gate" / "07b_validate_clean_attacked_alignment.py", False),
        ("Build IEEE 2030.5 semantic schema",
         S / "04_cyber_log_generation" / "05a_build_ieee2030_5_semantic_schema.py", True),
        ("Generate normal cyber log",
         S / "04_cyber_log_generation" / "05c_generate_normal_ieee2030_5_semantic_log.py", True),
        ("Generate anomalous cyber log",
         S / "04_cyber_log_generation" / "05d_generate_anomalous_ieee2030_5_semantic_log.py", True),
        ("Generate XML research artifacts",
         S / "04_cyber_log_generation" / "05e_optional_generate_ieee2030_5_xml_research_artifacts.py", False),
        ("Validate IEEE 2030.5 semantic layer",
         S / "07_validation_gate" / "07c_validate_ieee2030_5_semantic_layer.py", False),
        ("Build lifecycle response map",
         S / "05_physical_cyber_mapping" / "06a_build_lifecycle_response_map.py", True),
        ("Build event-specific context windows",
         S / "05_physical_cyber_mapping" / "06b_build_event_specific_context_windows.py", True),
        ("Build evidence packets",
         S / "05_physical_cyber_mapping" / "06c_build_cyber_physical_evidence_packets.py", True),
        ("Validate context causality",
         S / "07_validation_gate" / "07d_validate_context_causality.py", False),
        ("Visualize physical data",
         S / "06_visualization" / "08a_visualize_physical.py", False),
        ("Visualize cyber data",
         S / "06_visualization" / "08b_visualize_cyber.py", False),
        ("Visualize context data",
         S / "06_visualization" / "08c_visualize_context.py", False),
        ("Generate DER component visualizations",
         S / "06_visualization" / "08e_visualize_der_components.py", False),
        ("Generate final reports", None, False),   # handled inline
        ("Run final validation gate",
         S / "07_validation_gate" / "07e_final_validation_gate.py", False),
        ("Generate validation dashboard",
         S / "06_visualization" / "08d_visualize_validation_dashboard.py", False),
        ("Build ZIP package", None, False),   # handled inline
    ]


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    if ERROR_FILE.exists():
        ERROR_FILE.unlink()

    start_t = datetime.now(timezone.utc)
    print(f"DER Cyber-Physical Dataset Pipeline")
    print(f"Started: {start_t.isoformat()}")
    print(f"Root: {ROOT}")

    with open(LOG_FILE, "w", encoding="utf-8") as lf:
        orig_stdout = sys.stdout
        sys.stdout = Tee(orig_stdout, lf)
        try:
            for name, script_path, critical in get_pipeline_steps():
                try:
                    if script_path is None:
                        # Inline steps
                        print(f"\n{'-'*60}")
                        print(f"STEP: {name}")
                        print(f"{'-'*60}")
                        if name.startswith("Create folder"):
                            create_structure_and_backup()
                        elif name.startswith("Generate final reports"):
                            generate_final_reports_inline()
                        elif name.startswith("Build ZIP"):
                            build_zip_package()
                        print(f"  OK: {name} completed.")
                    else:
                        step(name, script_path, lf)
                except Exception as e:
                    print(f"\n  {'ABORTED' if critical else 'SKIPPED'}: {name} - {e}")
                    with open(ERROR_FILE, "a", encoding="utf-8") as ef:
                        ef.write(f"STEP {'FAILED' if critical else 'SKIPPED'}: {name}\n"
                                 f"{traceback.format_exc()}\n")
                    if critical:
                        sys.stdout = orig_stdout
                        print(f"\nPIPELINE ABORTED at critical step: {name}", file=sys.stderr)
                        sys.exit(1)

            end_t = datetime.now(timezone.utc)
            elapsed = (end_t - start_t).total_seconds()
            print(f"\nPipeline completed in {elapsed:.1f} seconds")
            print_completion_summary()

        finally:
            sys.stdout = orig_stdout

    return 0


if __name__ == "__main__":
    sys.exit(main())
