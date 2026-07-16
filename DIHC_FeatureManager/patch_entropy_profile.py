"""Patch DIHC_EntropyProfile.py for SciPy versions where simps was removed."""
from pathlib import Path
import sys

path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("DIHC_FeatureManager/DIHC_EntropyProfile.py")
text = path.read_text(encoding="utf-8")
old = "from scipy.integrate import simps"
new = """try:\n    from scipy.integrate import simpson\nexcept ImportError:  # SciPy < 1.6\n    from scipy.integrate import simps as simpson\n\n# Backward-compatible wrapper for existing calls in this module.\ndef simps(y, x=None, dx=1.0, axis=-1, even=None):\n    return simpson(y, x=x, dx=dx, axis=axis)"""
if old not in text:
    print(f"No '{old}' import found in {path}; no change made.")
else:
    backup = path.with_suffix(path.suffix + ".bak")
    backup.write_text(text, encoding="utf-8")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"Patched {path}")
    print(f"Backup saved as {backup}")
