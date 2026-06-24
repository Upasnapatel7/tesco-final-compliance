@echo off
echo ========================================
echo  GenAI Creative Compliance Studio v8
echo ========================================
echo.

REM Suppress MINGW numpy warnings
set PYTHONWARNINGS=ignore::RuntimeWarning
set NPY_DISABLE_CPU_FEATURES=AVX512F

REM Add Tesseract to PATH (adjust if installed elsewhere)
set PATH=%PATH%;C:\Program Files\Tesseract-OCR

REM Check which app to run
if "%1"=="dashboard" goto dashboard
if "%1"=="app" goto app

:dashboard
echo Starting Dashboard (recommended)...
echo Open browser at: http://localhost:8501
echo.
python -W ignore -c "import warnings; warnings.filterwarnings('ignore'); import streamlit.web.cli as stcli; import sys; sys.argv=['streamlit','run','dashboard.py','--server.port=8501']; stcli.main()"
goto end

:app
echo Starting Creative Builder...
echo Open browser at: http://localhost:8501
echo.
python -W ignore -c "import warnings; warnings.filterwarnings('ignore'); import streamlit.web.cli as stcli; import sys; sys.argv=['streamlit','run','app.py','--server.port=8501']; stcli.main()"
goto end

:end
