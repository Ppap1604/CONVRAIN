@echo off
chcp 65001 > nul
echo =================================================================
echo   CONVRAIN - Convective Rain Nowcast Web Platform
echo =================================================================
echo.
cd /d "%~dp0"
echo กำลังเริ่มต้น Web Server ที่ http://localhost:8000 ...
python run_web.py --port 8000
pause
