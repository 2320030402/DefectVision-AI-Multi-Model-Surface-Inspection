@echo off
setlocal
cd /d "%~dp0backend"
if not exist venv (
  echo Creating Python virtual environment...
  python -m venv venv
)
call venv\Scripts\activate.bat
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 0.0.0.0 --port 8000
