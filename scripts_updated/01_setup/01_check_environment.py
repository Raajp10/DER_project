"""Check Python environment and OpenDSS availability."""
import sys
import json
import importlib
from pathlib import Path

ROOT = Path(r"D:\updated_dataset")
_COMMON = ROOT / "scripts_updated" / "00_common"
for _d in [str(ROOT), str(_COMMON)]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from paths import ENV_CHECK_JSON, REPORTS, VALIDATION
from validation_utils import write_report


def check_pkg(name: str, import_name: str = None) -> dict:
    import_name = import_name or name
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "unknown")
        return {"package": name, "available": True, "version": str(ver)}
    except ImportError as e:
        return {"package": name, "available": False, "error": str(e)}


def check_opendss(master_dss: str = None) -> dict:
    result = {"engine": None, "available": False, "compile_test": None, "error": None}
    try:
        import opendssdirect as dss
        result["engine"] = "opendssdirect"
        result["available"] = True
        result["version"] = getattr(dss, "__version__", "unknown")
        if master_dss and Path(master_dss).exists():
            try:
                dss.run_command("Clear")
                dss.run_command(f"Compile [{master_dss}]")
                buses = dss.Circuit.NumBuses()
                result["compile_test"] = {
                    "status": "SUCCESS" if buses > 0 else "PARTIAL",
                    "master_dss": master_dss,
                    "num_buses": buses,
                }
            except Exception as e:
                result["compile_test"] = {"status": "FAILED", "error": str(e)}
        return result
    except ImportError:
        pass
    try:
        import dss as dss_ext
        result["engine"] = "dss-python"
        result["available"] = True
        return result
    except ImportError:
        pass
    result["error"] = "Neither opendssdirect nor dss-python installed. Will use physics-constrained surrogate."
    return result


def main() -> dict:
    VALIDATION.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    packages = [
        check_pkg("pandas"), check_pkg("numpy"), check_pkg("matplotlib"),
        check_pkg("pyarrow"), check_pkg("fastparquet"),
        check_pkg("yaml", "yaml"), check_pkg("scipy"),
    ]

    disc_path = ROOT / "data_updated" / "metadata" / "input_discovery_report.json"
    master_dss = None
    if disc_path.exists():
        with open(disc_path) as f:
            disc = json.load(f)
        master_dss = disc.get("master_dss_file")

    ods = check_opendss(master_dss)
    env = {
        "python_version": sys.version,
        "packages": packages,
        "opendss": ods,
        "opendss_available": ods["available"],
        "physical_generation_mode": (
            "opendss_clean_baseline" if ods["available"] else "physics_constrained_surrogate"
        ),
        "surrogate_fallback": not ods["available"],
    }

    with open(ENV_CHECK_JSON, "w") as f:
        json.dump(env, f, indent=2)

    lines = [
        "# Environment Check Report", "",
        f"**Python:** `{sys.version}`", "",
        "## Packages", "",
    ]
    for p in packages:
        icon = "✓" if p["available"] else "✗"
        lines.append(f"- {icon} `{p['package']}` v{p.get('version', 'N/A')}")
    lines += [
        "", "## OpenDSS Engine", "",
        f"- Engine: `{ods.get('engine', 'none')}`",
        f"- Available: `{ods['available']}`",
        "", f"**Physical generation mode:** `{env['physical_generation_mode']}`",
    ]
    if env["surrogate_fallback"]:
        lines.append("> **WARNING:** OpenDSS unavailable. Physics-constrained surrogate will be used.")
    write_report(lines, REPORTS / "00_environment_check.md")

    print(f"OpenDSS available: {ods['available']}")
    print(f"Physical generation mode: {env['physical_generation_mode']}")
    return env


if __name__ == "__main__":
    main()
