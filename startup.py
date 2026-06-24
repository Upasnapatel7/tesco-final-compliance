"""
startup.py
==========
Run this instead of `streamlit run dashboard.py` to suppress
the MINGW NumPy warnings on Windows.

Usage:
    python startup.py dashboard
    python startup.py app
"""

import sys
import os
import warnings

# Suppress ALL numpy/scipy runtime warnings before any import
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*MINGW.*")
warnings.filterwarnings("ignore", message=".*exp2.*")
warnings.filterwarnings("ignore", message=".*nextafter.*")
warnings.filterwarnings("ignore", message=".*log10.*")

# Set env var to suppress at C level too
os.environ["PYTHONWARNINGS"] = "ignore::RuntimeWarning"
os.environ["NPY_DISABLE_CPU_FEATURES"] = "AVX512F"

# Add Tesseract to PATH on Windows
tesseract_path = r"C:\Program Files\Tesseract-OCR"
if os.path.exists(tesseract_path) and tesseract_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = tesseract_path + os.pathsep + os.environ.get("PATH", "")

# Import and patch pytesseract path
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except Exception:
    pass

# Now launch streamlit
target = sys.argv[1] if len(sys.argv) > 1 else "dashboard"
if target not in ("dashboard", "app"):
    print(f"Unknown target: {target}. Use 'dashboard' or 'app'")
    sys.exit(1)

script = f"{target}.py"
print(f"Starting {script}...")
print("Open browser at: http://localhost:8501")
print()

from streamlit.web import cli as stcli
sys.argv = ["streamlit", "run", script, "--server.port=8501",
            "--server.headless=false"]
stcli.main()
