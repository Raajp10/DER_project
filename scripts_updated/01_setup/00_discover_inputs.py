"""
Discover OpenDSS input files inside D:/updated_dataset.
Writes:
  data_updated/metadata/input_discovery_report.json
  reports/00_input_discovery_report.md
"""
import sys
import json
import re
from pathlib import Path

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import INPUT_DISCOVERY_JSON, REPORTS, METADATA
from validation_utils import write_report

EXCLUDE_DIRS = {
    "original_input_backup", "data_updated", "final_package",
    "logs", "figures", "reports", "scripts_updated",
}
MASTER_NAMES = {
    "ieee123master.dss", "master.dss", "der_qsts_7day.dss",
    "der_qsts_debug_1hour.dss",
}
DER_KEYWORDS = {"pvsystem", "storage", "generator", "der"}


def find_dss_files(root: Path) -> list:
    results = []
    for ext in ["*.dss", "*.DSS"]:
        for p in root.rglob(ext):
            parts = set(pp.lower() for pp in p.relative_to(root).parts)
            if not (parts & EXCLUDE_DIRS):
                results.append(p)
    return sorted(set(results), key=str)


def classify_dss(path: Path) -> dict:
    name_lower = path.name.lower()
    is_master = name_lower in MASTER_NAMES
    content = ""
    try:
        content = path.read_text(errors="replace").lower()
    except Exception:
        pass
    is_der = any(k in content for k in DER_KEYWORDS)
    has_compile = "compile" in content
    has_solve = "solve" in content
    return {
        "path": str(path),
        "name": path.name,
        "is_master_candidate": is_master or (has_compile and has_solve),
        "is_der_file": is_der,
        "has_loadshape": "new loadshape" in content,
        "has_monitor": "new monitor" in content,
    }


def find_ancillary(root: Path) -> dict:
    anc = {"bus_coord": [], "config": [], "csv": []}
    for ext, key in [("*.dat", "bus_coord"), ("*.yaml", "config"), ("*.yml", "config"), ("*.csv", "csv")]:
        for p in root.rglob(ext):
            parts = set(pp.lower() for pp in p.relative_to(root).parts)
            if not (parts & EXCLUDE_DIRS):
                anc[key].append(str(p))
    return anc


def main() -> dict:
    METADATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    dss_files = find_dss_files(ROOT)
    classified = [classify_dss(p) for p in dss_files]
    ancillary = find_ancillary(ROOT)

    master_candidates = [c for c in classified if c["is_master_candidate"]]
    preferred = None
    for mc in master_candidates:
        if "der_qsts_7day" in mc["name"].lower():
            preferred = mc
            break
    if preferred is None and master_candidates:
        preferred = master_candidates[0]
    if preferred is None:
        for mc in classified:
            if "ieee123master" in mc["name"].lower():
                preferred = mc
                break

    master_file = preferred["path"] if preferred else None
    der_files = [c["path"] for c in classified if c["is_der_file"]]

    report = {
        "root": str(ROOT),
        "dss_files_found": len(classified),
        "all_dss_files": classified,
        "master_dss_file": master_file,
        "der_dss_files": der_files,
        "ancillary": ancillary,
        "discovery_status": "SUCCESS" if master_file else "NO_MASTER_FOUND",
    }

    INPUT_DISCOVERY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(INPUT_DISCOVERY_JSON, "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# Input Discovery Report", "",
        f"**Root:** `{ROOT}`",
        f"**DSS files found:** {len(classified)}",
        f"**Master DSS:** `{master_file or 'NOT FOUND'}`", "",
        "## All DSS Files", "",
    ]
    for c in classified:
        lines.append(f"- `{c['path']}` — master={c['is_master_candidate']}, der={c['is_der_file']}")
    lines += ["", f"**Status:** {report['discovery_status']}"]
    write_report(lines, REPORTS / "00_input_discovery_report.md")

    if master_file:
        print(f"OK: Master DSS = {master_file}")
    else:
        print("ERROR: No master DSS file found.")
    return report


if __name__ == "__main__":
    r = main()
    if r["discovery_status"] != "SUCCESS":
        sys.exit(1)
