@echo off
setlocal

REM Change to the directory where this script is located
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
cd /d "%SCRIPT_DIR%"

set PYTHONPATH=%SCRIPT_DIR%

echo Starting enotropos...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop
python -m streamlit run winegpt/app.py

endlocal
pause
