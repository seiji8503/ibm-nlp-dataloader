@echo off
echo Creating virtual environment...
python -m venv .venv
call .venv\Scripts\activate
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt
echo All set! Activate with:
echo call .venv\Scripts\activate
pause
