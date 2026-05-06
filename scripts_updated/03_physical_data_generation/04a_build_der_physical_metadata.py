"""Parse DER ratings from DSS/config files. Produce der_physical_metadata.json."""
import sys
import json
import re
from pathlib import Path

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import DER_METADATA_JSON, REPORTS, METADATA
from config import (
    PV_P_RATED_KW, PV_S_RATED_KVA, BESS_P_RATED_KW, BESS_S_RATED_KVA,
    BESS_CAPACITY_KWH, BESS_SOC_MIN_PERCENT, BESS_SOC_MAX_PERCENT,
    BESS_INITIAL_SOC_PERCENT, BESS_EFF_CHARGE_PERCENT, BESS_EFF_DISCHARGE_PERCENT,
)
from validation_utils import write_report


def parse_dss_ratings(dss_path: Path) -> dict:
    found = {}
    if not dss_path.exists():
        return found
    text = dss_path.read_text(errors="replace")
    lines = text.splitlines()
    current_obj = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("!"):
            continue
        lower = stripped.lower()
        if lower.startswith("new pvsystem"):
            current_obj = "pv"
        elif lower.startswith("new storage"):
            current_obj = "bess"
        elif lower.startswith("new "):
            current_obj = None
        if current_obj == "pv":
            m = re.search(r"\bpmpp\s*=\s*([\d.]+)", lower)
            if m:
                found["pv_p_rated_kw"] = float(m.group(1))
            m = re.search(r"\bkva\s*=\s*([\d.]+)", lower)
            if m:
                found["pv_s_rated_kva"] = float(m.group(1))
        if current_obj == "bess":
            m = re.search(r"\bkwrated\s*=\s*([\d.]+)", lower)
            if m:
                found["bess_p_rated_kw"] = float(m.group(1))
            m = re.search(r"\bkwhrated\s*=\s*([\d.]+)", lower)
            if m:
                found["bess_capacity_kwh"] = float(m.group(1))
            m = re.search(r"\b%reserve\s*=\s*([\d.]+)", lower)
            if m:
                found["bess_soc_min_percent"] = float(m.group(1))
            m = re.search(r"\b%stored\s*=\s*([\d.]+)", lower)
            if m:
                found["bess_initial_soc_percent"] = float(m.group(1))
            m = re.search(r"\b%effcharge\s*=\s*([\d.]+)", lower)
            if m:
                found["bess_eff_charge_percent"] = float(m.group(1))
            m = re.search(r"\b%effdischarge\s*=\s*([\d.]+)", lower)
            if m:
                found["bess_eff_discharge_percent"] = float(m.group(1))
    return found


def parse_yaml_ratings(yaml_path: Path) -> dict:
    found = {}
    if not yaml_path.exists():
        return found
    try:
        import yaml
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
        pv = cfg.get("pv", {})
        bess = cfg.get("bess", {})
        if "rated_kw" in pv:
            found["pv_p_rated_kw"] = float(pv["rated_kw"])
        if "rated_kva" in pv:
            found["pv_s_rated_kva"] = float(pv["rated_kva"])
        if "rated_kw" in bess:
            found["bess_p_rated_kw"] = float(bess["rated_kw"])
        if "rated_kwh" in bess:
            found["bess_capacity_kwh"] = float(bess["rated_kwh"])
        if "min_soc_percent" in bess:
            found["bess_soc_min_percent"] = float(bess["min_soc_percent"])
        if "initial_soc_percent" in bess:
            found["bess_initial_soc_percent"] = float(bess["initial_soc_percent"])
        if "eff_charge_percent" in bess:
            found["bess_eff_charge_percent"] = float(bess["eff_charge_percent"])
        if "eff_discharge_percent" in bess:
            found["bess_eff_discharge_percent"] = float(bess["eff_discharge_percent"])
    except Exception as e:
        print(f"  WARN: YAML parse error: {e}")
    return found


def main() -> dict:
    METADATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    dss_path = ROOT / "ieee123_base" / "DER_site_001.dss"
    yaml_path = ROOT / "ieee123_base" / "configs" / "project_config.yaml"

    dss_ratings = parse_dss_ratings(dss_path)
    yaml_ratings = parse_yaml_ratings(yaml_path)

    defaults = {
        "pv_p_rated_kw": PV_P_RATED_KW, "pv_s_rated_kva": PV_S_RATED_KVA,
        "bess_p_rated_kw": BESS_P_RATED_KW, "bess_s_rated_kva": BESS_S_RATED_KVA,
        "bess_capacity_kwh": BESS_CAPACITY_KWH, "bess_soc_min_percent": BESS_SOC_MIN_PERCENT,
        "bess_soc_max_percent": BESS_SOC_MAX_PERCENT,
        "bess_initial_soc_percent": BESS_INITIAL_SOC_PERCENT,
        "bess_eff_charge_percent": BESS_EFF_CHARGE_PERCENT,
        "bess_eff_discharge_percent": BESS_EFF_DISCHARGE_PERCENT,
    }
    merged = {**defaults}
    sources = {k: "project_config_defaults" for k in defaults}
    for k, v in yaml_ratings.items():
        merged[k] = v
        sources[k] = "project_config_yaml"
    for k, v in dss_ratings.items():
        merged[k] = v
        sources[k] = "dss_file_parsed"

    if "bess_s_rated_kva" not in dss_ratings and "bess_p_rated_kw" in merged:
        merged["bess_s_rated_kva"] = round(merged["bess_p_rated_kw"] * 1.111, 2)
        sources["bess_s_rated_kva"] = "derived_from_bess_p_rated"

    warnings = []
    for bad, key, good in [(450.0, "pv_p_rated_kw", PV_P_RATED_KW),
                           (500.0, "pv_p_rated_kw", PV_P_RATED_KW),
                           (300.0, "bess_p_rated_kw", BESS_P_RATED_KW)]:
        if merged.get(key) == bad:
            merged[key] = good
            sources[key] = "project_config_defaults_override_bad_generic"
            warnings.append(f"WARN: {key}={bad} rejected as generic; reset to {good}")

    critical_rating_keys = ["pv_p_rated_kw", "pv_s_rated_kva", "bess_p_rated_kw", "bess_capacity_kwh"]
    rating_fallback_used = any("defaults" in sources.get(k, "") for k in critical_rating_keys)
    metadata = {
        **merged,
        "pv_p_rated_kw_source": sources.get("pv_p_rated_kw"),
        "pv_s_rated_kva_source": sources.get("pv_s_rated_kva"),
        "bess_p_rated_kw_source": sources.get("bess_p_rated_kw"),
        "bess_s_rated_kva_source": sources.get("bess_s_rated_kva"),
        "rating_source_file": str(dss_path) if dss_path.exists() else str(yaml_path),
        "rating_source_line_or_object": "PVSystem.pv_001 / Storage.bess_001",
        "rating_fallback_used": rating_fallback_used,
        "warnings": warnings,
        "all_sources": sources,
    }

    with open(DER_METADATA_JSON, "w") as f:
        json.dump(metadata, f, indent=2)

    lines = [
        "# DER Physical Metadata", "", "## Rating Audit", "",
        f"- PV rated kW: **{merged['pv_p_rated_kw']} kW** (source: `{sources['pv_p_rated_kw']}`)",
        f"- PV rated kVA: **{merged['pv_s_rated_kva']} kVA**",
        f"- BESS rated kW: **{merged['bess_p_rated_kw']} kW** (source: `{sources['bess_p_rated_kw']}`)",
        f"- BESS capacity kWh: **{merged['bess_capacity_kwh']} kWh**",
        f"- Rating fallback used: `{rating_fallback_used}`", "",
    ]
    if rating_fallback_used:
        lines.append("> **RATING FALLBACK USED — MANUAL REVIEW REQUIRED**")
    write_report(lines, REPORTS / "02_physical_layer_improvement_report.md")

    print(f"DER metadata: PV={merged['pv_p_rated_kw']} kW, "
          f"BESS={merged['bess_p_rated_kw']} kW / {merged['bess_capacity_kwh']} kWh")
    return metadata


if __name__ == "__main__":
    main()
