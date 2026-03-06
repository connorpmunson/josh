@echo off
timeout /t 5 >nul
cd /d C:\Users\cpmun\josh
call .venv\Scripts\activate
cd /d C:\Users\cpmun\josh\josh_v4
python josh_main.py