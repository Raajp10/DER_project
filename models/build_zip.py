"""Build the phase1 ZIP package."""
import zipfile
from pathlib import Path

ROOT = Path(r"D:\updated_dataset")
MODELS = ROOT / "models"
zip_path = MODELS / "phase1_model_results_package.zip"

all_files = []

# Scripts
for d in [MODELS / "scripts_common", MODELS / "train", MODELS / "evaluate"]:
    if d.exists():
        for f in sorted(d.glob("*.py")):
            all_files.append((f, Path("phase1/scripts") / d.name / f.name))

# Master scripts
for fname in ["run_phase1_models.py", "run_phase1_models.bat"]:
    f = MODELS / fname
    if f.exists():
        all_files.append((f, Path("phase1") / fname))

# Windows metadata only
win_dir = MODELS / "windows"
if win_dir.exists():
    for ext in ["*.json", "*.csv"]:
        for p in sorted(win_dir.glob(ext)):
            all_files.append((p, Path("phase1/windows") / p.name))

# Results
results_dir = MODELS / "results"
if results_dir.exists():
    for f in sorted(results_dir.rglob("*")):
        if f.is_file() and f.suffix in (".csv", ".json", ".md"):
            all_files.append((f, Path("phase1/results") / f.name))

# Docs
docs_dir = MODELS / "docs"
if docs_dir.exists():
    for f in sorted(docs_dir.glob("*.md")):
        all_files.append((f, Path("phase1/docs") / f.name))

# Figures
figs_dir = MODELS / "figures"
if figs_dir.exists():
    for f in sorted(figs_dir.glob("*.png")):
        all_files.append((f, Path("phase1/figures") / f.name))

# Weight artifacts
weights_dir = MODELS / "weights"
if weights_dir.exists():
    for model_dir in sorted(weights_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        for ext in ["*.json", "*.csv", "*.joblib"]:
            for f in model_dir.glob(ext):
                all_files.append((f, Path("phase1/weights") / model_dir.name / f.name))
        for pt in model_dir.glob("*.pt"):
            if pt.stat().st_size < 200 * 1024 * 1024:
                all_files.append((pt, Path("phase1/weights") / model_dir.name / pt.name))

written = 0
with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    for src, arc in all_files:
        if src.exists():
            zf.write(str(src), str(arc))
            written += 1

size_mb = zip_path.stat().st_size / 1e6
print(f"ZIP: {zip_path}")
print(f"Files: {written}  Size: {size_mb:.1f} MB")
