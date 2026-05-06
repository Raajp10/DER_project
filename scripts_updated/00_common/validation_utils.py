"""Shared validation helpers for the DER dataset pipeline."""
import numpy as np
import pandas as pd
from pathlib import Path
import json


def check_row_count(df: pd.DataFrame, expected: int, name: str) -> dict:
    ok = len(df) == expected
    return {"check": f"{name}_row_count", "status": "PASS" if ok else "FAIL",
            "expected": expected, "actual": len(df)}


def check_no_duplicates(df: pd.DataFrame, col: str, name: str) -> dict:
    n_dup = df[col].duplicated().sum()
    ok = n_dup == 0
    return {"check": f"{name}_no_duplicate_{col}", "status": "PASS" if ok else "FAIL",
            "duplicates_found": int(n_dup)}


def check_no_gaps(time_s: np.ndarray, name: str, timestep: int = 1) -> dict:
    if len(time_s) < 2:
        return {"check": f"{name}_no_gaps", "status": "PASS", "gaps": 0}
    diffs = np.diff(time_s)
    gaps = int(np.sum(diffs != timestep))
    return {"check": f"{name}_no_gaps", "status": "PASS" if gaps == 0 else "FAIL",
            "gap_count": gaps}


def check_column_exists(df: pd.DataFrame, col: str, name: str) -> dict:
    ok = col in df.columns
    return {"check": f"{name}_col_{col}", "status": "PASS" if ok else "FAIL"}


def check_range(series: pd.Series, lo: float, hi: float, name: str) -> dict:
    violations = int(((series < lo) | (series > hi)).sum())
    return {"check": name, "status": "PASS" if violations == 0 else "FAIL",
            "violations": violations, "range": [lo, hi]}


def check_no_nan(series: pd.Series, name: str) -> dict:
    n = int(series.isna().sum())
    return {"check": f"{name}_no_nan", "status": "PASS" if n == 0 else "FAIL", "nan_count": n}


def save_json(data: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_report(lines: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(str(l) for l in lines))


def aggregate_results(results: list) -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.get("status") == "PASS")
    warned = sum(1 for r in results if r.get("status") == "WARN")
    failed = sum(1 for r in results if r.get("status") == "FAIL")
    return {"total": total, "passed": passed, "warned": warned, "failed": failed,
            "pass_rate": round(passed / total * 100, 1) if total else 0,
            "overall": "PASS" if failed == 0 else "FAIL"}


def format_md_table(rows: list, headers: list) -> str:
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    hdr = "| " + " | ".join(headers) + " |"
    lines = [hdr, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)
