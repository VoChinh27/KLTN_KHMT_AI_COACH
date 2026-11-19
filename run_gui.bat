@echo off
cd /d %~dp0
call .venv\Scripts\activate.bat 2>nul
python -m gui_app
pause
