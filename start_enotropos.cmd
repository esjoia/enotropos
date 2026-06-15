@echo off
cd /d C:\Users\Carles\Desktop\enotropos
set PYTHONPATH=C:\Users\Carles\Desktop\enotropos
echo Starting enotropos...
echo Open http://localhost:8501 in your browser
echo Press Ctrl+C to stop
python -m streamlit run winegpt/app.py
pause
