"""
Path-safe loader for the DER dataset pipeline.
Adds all scripts_updated subdirectories to sys.path so that modules
in directories with numeric prefixes (00_common, 01_setup, etc.)
can be imported by their base filename.
"""
import sys
import importlib.util
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).parent.parent  # scripts_updated/


def setup_path():
    """Add all scripts_updated subdirectories to sys.path."""
    for d in _SCRIPTS_DIR.iterdir():
        if d.is_dir():
            dstr = str(d)
            if dstr not in sys.path:
                sys.path.insert(0, dstr)


def load_script(file_path, module_name=None):
    """
    Load a Python script file (even if it has a numeric-prefixed name)
    and return the module object.
    """
    file_path = Path(file_path)
    module_name = module_name or file_path.stem.replace("-", "_").lstrip("0123456789_")
    if not module_name or module_name[0].isdigit():
        module_name = "mod_" + module_name
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Call setup_path() immediately when this module is imported
setup_path()
